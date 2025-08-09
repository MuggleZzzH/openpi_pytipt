import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
import gc
import time
from torch.cuda.amp import GradScaler, autocast

class RLOptimizerPI0_CFG:
    """
    RL optimizer for PI0Policy using an advantage-weighted training scheme,
    inspired by Classifier-Free Guidance (CFG).

    This optimizer replaces traditional PPO by directly using the trajectory-level
    advantage to weight the policy's L2 loss, guiding the model towards
    behaviors from high-reward trajectories.
    """
    
    def __init__(
        self,
        rollout_generator,
        reward_function,
        num_epochs=5,
        batch_size=64,
        gradient_accumulation_steps=1,
        grad_norm_clip=None,
    ):
        """
        Initialize the PI0 CFG-style RL optimizer.
        
        Args:
            rollout_generator: An instance for generating rollout trajectories.
            reward_function: A function to compute rewards for trajectories.
            num_epochs: Number of optimization epochs over the collected data.
            batch_size: Mini-batch size for each optimization step.
            gradient_accumulation_steps: Number of steps to accumulate gradients.
            grad_norm_clip: Value for gradient norm clipping.
        """
        self.rollout_generator = rollout_generator
        self.reward_function = reward_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_norm_clip = grad_norm_clip
        
        # 检查是否在分布式环境中运行
        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
    
    def train_on_rollouts(self, model_adapter, lr, scaler=None, use_amp=False):
        """
        Perform a full optimization cycle: rollout, advantage calculation, and weighted training.
        
        Args:
            model_adapter: A PI0_CFG_Adapter instance.
            lr: The learning rate for this training step.
            scaler: Optional GradScaler for mixed precision training.
            use_amp: Whether to use automatic mixed precision.
            
        Returns:
            A dictionary of optimization metrics.
        """
        policy = model_adapter.get_policy_model()
        
        # 获取优化器 - 现在应该从外部传入，而不是在这里创建
        # 我们假设优化器已经在主训练脚本中创建并附加到模型上
        optimizer = getattr(policy, 'optimizer', None)
        
        if optimizer is None:
            # 如果模型没有附加优化器，我们记录警告但继续执行
            # 这允许在评估模式下运行而不进行参数更新
            if self.rank == 0:
                print("警告: 未找到优化器。将只执行前向传播而不更新参数。")
            return self._collect_rollouts_and_compute_advantage(model_adapter)
        
        # === 步骤 1: 生成轨迹并计算优势 ===
        start_time = time.time()
        rollout_metrics = self._collect_rollouts_and_compute_advantage(model_adapter)
        rollout_time = time.time() - start_time
        
        if not rollout_metrics or 'all_episodes' not in rollout_metrics or not rollout_metrics['all_episodes']:
            if self.rank == 0:
                print("未收集到有效轨迹。跳过优化步骤。")
            return rollout_metrics
        
        all_episodes = rollout_metrics['all_episodes']
        all_advantage = rollout_metrics['all_advantage']
        
        # === 步骤 2: 使用优势加权损失进行优化 ===
        start_time = time.time()
        ppo_dataset_size = len(all_episodes)
        
        all_weighted_losses = []
        total_gradient_steps = 0
        
        # 理论上的梯度步数
        steps_per_epoch = (ppo_dataset_size + self.batch_size - 1) // self.batch_size
        total_expected_steps = self.num_epochs * steps_per_epoch
        
        if self.rank == 0:
            print(f"开始优化: {ppo_dataset_size} 个轨迹, {self.num_epochs} 个周期, "
                  f"预期 {total_expected_steps} 个步骤")
        
        # 清零梯度，准备新的优化步骤
        optimizer.zero_grad()

        for epoch in range(self.num_epochs):
            # 为每个周期生成新的随机索引
            indices = np.random.permutation(ppo_dataset_size)
            
            # 创建进度条（只在主进程上显示）
            epoch_iter = tqdm(
                range(0, ppo_dataset_size, self.batch_size),
                desc=f'优化周期 {epoch+1}/{self.num_epochs}',
                disable=self.rank != 0,
                leave=False
            )
            
            for i in epoch_iter:
                batch_indices = indices[i : i + self.batch_size]
                
                # 创建小批量
                mb_episodes = [all_episodes[j] for j in batch_indices if j < len(all_episodes)]
                mb_advantages = all_advantage[batch_indices]
                
                # 跳过空批次
                if not mb_episodes:
                    continue
                
                # 确保优势在正确的设备上
                if isinstance(mb_advantages, torch.Tensor):
                    mb_advantages = mb_advantages.to(model_adapter.device)
                else:
                    mb_advantages = torch.tensor(mb_advantages, device=model_adapter.device)
                
                try:
                    # 计算加权损失
                    if use_amp and scaler is not None:
                        with autocast():
                            loss = model_adapter.compute_weighted_loss(mb_episodes, mb_advantages)
                            # 对梯度累积进行归一化
                            loss = loss / self.gradient_accumulation_steps
                        
                        # 使用 scaler 进行反向传播
                        scaler.scale(loss).backward()
                    else:
                        # 常规精度训练
                        loss = model_adapter.compute_weighted_loss(mb_episodes, mb_advantages)
                        # 对梯度累积进行归一化
                        loss = loss / self.gradient_accumulation_steps
                        # 反向传播
                        loss.backward()
                    
                    # 记录损失（恢复原始值）
                    original_loss = loss.item() * self.gradient_accumulation_steps
                    all_weighted_losses.append(original_loss)
                    
                except Exception as e:
                    if self.rank == 0:
                        print(f"计算损失时出错: {e}")
                    # 跳过这个批次
                    continue
                
                total_gradient_steps += 1
                
                # 如果达到梯度累积步数，则更新参数
                if total_gradient_steps % self.gradient_accumulation_steps == 0:
                    try:
                        # 在分布式模式下同步梯度
                        if self.is_distributed:
                            for param in policy.parameters():
                                if param.requires_grad and param.grad is not None:
                                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                        
                        # 裁剪梯度
                        if self.grad_norm_clip is not None:
                            if use_amp and scaler is not None:
                                # 使用 scaler 进行梯度裁剪
                                scaler.unscale_(optimizer)
                                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_norm_clip)
                            else:
                                # 常规梯度裁剪
                                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_norm_clip)
                        
                        # 更新优化器
                        if use_amp and scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        
                        # 清零梯度
                        optimizer.zero_grad()
                        
                    except Exception as e:
                        if self.rank == 0:
                            print(f"优化器更新时出错: {e}")
                        # 清零梯度以保证下一步能正常进行
                        optimizer.zero_grad()
                        continue

        # 清理内存
        del all_episodes
        gc.collect()
        
        optimization_time = time.time() - start_time

        # === 步骤 3: 收集并返回指标 ===
        metrics = {
            'mean_weighted_loss': np.mean(all_weighted_losses) if all_weighted_losses else 0.0,
            'mean_advantage': rollout_metrics.get('mean_advantage', 0.0),
            'mean_reward': rollout_metrics.get('mean_reward', 0.0),
            'rollouts_collected': rollout_metrics.get('rollouts_collected', 0),
            'rollout_time_seconds': rollout_time,
            'optimization_time_seconds': optimization_time,
            'total_time_seconds': rollout_time + optimization_time,
            'gradient_steps': total_gradient_steps,
            'actual_epochs': epoch + 1,
        }
        
        # 在所有GPU上减少指标
        metrics = self._reduce_metrics(metrics, model_adapter.device)

        return metrics
    
    def _collect_rollouts_and_compute_advantage(self, model_adapter):
        """
        收集轨迹并计算优势值。
        
        Args:
            model_adapter: 模型适配器实例。
            
        Returns:
            包含轨迹、优势值和基本指标的字典。
        """
        with torch.no_grad():
            if self.rank == 0:
                print('开始生成轨迹...')
            
            try:
                # 生成轨迹
                all_episodes = self.rollout_generator.generate_rollouts()
                
                if not all_episodes:
                    if self.rank == 0:
                        print(f"未收集到有效轨迹。跳过优化步骤。")
                    return None
                
                if self.rank == 0:
                    print(f'完成生成 {len(all_episodes)} 个轨迹。')
                
                # 验证轨迹数据的完整性
                valid_episodes = []
                for i, episode in enumerate(all_episodes):
                    if not isinstance(episode, dict):
                        if self.rank == 0:
                            print(f"警告: 轨迹 {i} 不是字典格式，跳过")
                        continue
                    
                    # 检查必要的键
                    required_keys = ['observations', 'actions', 'task']
                    missing_keys = [key for key in required_keys if key not in episode]
                    if missing_keys:
                        if self.rank == 0:
                            print(f"警告: 轨迹 {i} 缺少键 {missing_keys}，跳过")
                        continue
                    
                    valid_episodes.append(episode)
                
                if not valid_episodes:
                    if self.rank == 0:
                        print("没有有效的轨迹数据，跳过优化步骤")
                    return None
                
                all_episodes = valid_episodes
                
                # 获取设备信息 - 确保从模型适配器正确获取
                if hasattr(model_adapter, 'device'):
                    device = model_adapter.device
                elif hasattr(model_adapter, 'policy') and hasattr(model_adapter.policy, 'device'):
                    device = model_adapter.policy.device
                else:
                    # 从模型参数推断设备
                    try:
                        device = next(model_adapter.get_policy_model().parameters()).device
                    except (StopIteration, AttributeError):
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        if self.rank == 0:
                            print(f"无法推断设备，使用默认设备: {device}")
                
                if self.rank == 0:
                    print(f"使用设备: {device}")
                
                # 计算奖励
                all_scores = []
                for i, episode in enumerate(all_episodes):
                    try:
                        score = self.reward_function.compute_reward(i, episode, None)
                        all_scores.append(float(score))  # 确保是Python float
                    except Exception as e:
                        if self.rank == 0:
                            print(f"计算轨迹 {i} 的奖励时出错: {e}")
                        all_scores.append(0.0)  # 默认奖励
                
                # 确保奖励张量正确创建
                try:
                    rlhf_reward = torch.tensor(all_scores, device=device, dtype=torch.float32)
                except Exception as e:
                    if self.rank == 0:
                        print(f"创建奖励张量时出错: {e}")
                        print(f"奖励分数: {all_scores}")
                    # 创建默认奖励张量
                    rlhf_reward = torch.zeros(len(all_episodes), device=device, dtype=torch.float32)
                
                # 计算留一法优势 - 增强版本，防止NaN值
                num_rollouts = len(all_episodes)
                rollouts_per_batch = self.rollout_generator.num_envs
                
                try:
                    # 检查奖励张量是否包含NaN或Inf
                    if torch.isnan(rlhf_reward).any() or torch.isinf(rlhf_reward).any():
                        if self.rank == 0:
                            print("警告: 检测到奖励中的NaN或Inf值，使用零值替换")
                        rlhf_reward = torch.nan_to_num(rlhf_reward, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    if num_rollouts % rollouts_per_batch != 0 or rollouts_per_batch <= 1:
                        # 简化的优势: reward - mean(reward)
                        if self.rank == 0:
                            print("警告: 轨迹数量不均匀或批次大小过小。使用简化的优势计算。")
                        reward_mean = rlhf_reward.mean()
                        # 防止均值为NaN的情况
                        if torch.isnan(reward_mean):
                            reward_mean = torch.tensor(0.0, device=rlhf_reward.device)
                        advantage = rlhf_reward - reward_mean
                    else:
                        num_batches = num_rollouts // rollouts_per_batch
                        rlhf_reward_reshaped = rlhf_reward.reshape(num_batches, rollouts_per_batch)
                        
                        # 防止除零错误
                        if rollouts_per_batch <= 1:
                            if self.rank == 0:
                                print("警告: rollouts_per_batch <= 1，使用简化优势计算")
                            advantage = rlhf_reward - rlhf_reward.mean()
                        else:
                            baseline = (rlhf_reward_reshaped.sum(1, keepdim=True) - rlhf_reward_reshaped) / (rollouts_per_batch - 1)
                            # 检查baseline是否包含NaN
                            if torch.isnan(baseline).any():
                                if self.rank == 0:
                                    print("警告: baseline计算产生NaN，使用简化方法")
                                advantage = rlhf_reward - rlhf_reward.mean()
                            else:
                                advantage = rlhf_reward_reshaped - baseline
                                advantage = advantage.flatten()
                    
                    # 最终NaN检查和清理
                    if torch.isnan(advantage).any() or torch.isinf(advantage).any():
                        if self.rank == 0:
                            print("警告: 优势计算产生NaN/Inf值，使用安全替换")
                        advantage = torch.nan_to_num(advantage, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    all_advantage = advantage
                    
                except Exception as e:
                    if self.rank == 0:
                        print(f"计算优势值时出错: {e}")
                        import traceback
                        traceback.print_exc()
                    # 使用安全的零优势作为备用方案
                    all_advantage = torch.zeros_like(rlhf_reward)

                # 在所有GPU上同步优势值（仅在分布式模式下）- 增强错误处理
                if self.is_distributed:
                    try:
                        # 首先广播轨迹数量，以便所有进程可以调整其张量大小
                        num_rollouts_tensor = torch.tensor([num_rollouts], device=device)
                        dist.all_reduce(num_rollouts_tensor, op=dist.ReduceOp.MAX)
                        max_rollouts = int(num_rollouts_tensor.item())
                        
                        # 如果本地轨迹数量小于最大值，则填充
                        if num_rollouts < max_rollouts:
                            padding = torch.zeros(max_rollouts - num_rollouts, device=device, dtype=all_advantage.dtype)
                            all_advantage = torch.cat([all_advantage, padding])
                            rlhf_reward = torch.cat([rlhf_reward, padding])
                        
                        # 在同步前检查NaN值
                        if torch.isnan(all_advantage).any():
                            if self.rank == 0:
                                print("警告: 同步前检测到优势中的NaN值")
                            all_advantage = torch.nan_to_num(all_advantage, nan=0.0)
                        
                        if torch.isnan(rlhf_reward).any():
                            if self.rank == 0:
                                print("警告: 同步前检测到奖励中的NaN值")
                            rlhf_reward = torch.nan_to_num(rlhf_reward, nan=0.0)
                        
                        # 同步所有进程的优势值和奖励
                        dist.all_reduce(all_advantage, op=dist.ReduceOp.SUM)
                        dist.all_reduce(rlhf_reward, op=dist.ReduceOp.SUM)
                        
                        # 恢复原始大小
                        all_advantage = all_advantage[:num_rollouts]
                        rlhf_reward = rlhf_reward[:num_rollouts]
                        
                        # 同步后再次检查NaN值
                        if torch.isnan(all_advantage).any() or torch.isnan(rlhf_reward).any():
                            if self.rank == 0:
                                print("警告: 分布式同步后检测到NaN值，进行清理")
                            all_advantage = torch.nan_to_num(all_advantage, nan=0.0)
                            rlhf_reward = torch.nan_to_num(rlhf_reward, nan=0.0)
                            
                    except Exception as e:
                        if self.rank == 0:
                            print(f"分布式同步时出错: {e}")
                            import traceback
                            traceback.print_exc()
                        # 同步失败时使用本地值
                
            except Exception as e:
                if self.rank == 0:
                    print(f"收集轨迹和计算优势时出现未预期错误: {e}")
                    import traceback
                    traceback.print_exc()
                return None
        
        return {
            'all_episodes': all_episodes,
            'all_advantage': all_advantage,
            'mean_advantage': all_advantage.mean().item(),
            'mean_reward': rlhf_reward.mean().item(),
            'rollouts_collected': len(rlhf_reward),
        }
    
    def _reduce_metrics(self, metrics, device):
        """
        在分布式环境中统一指标。修复了可能的数据类型问题。
        
        Args:
            metrics: 指标字典。
            device: 设备。
            
        Returns:
            减少后的指标字典。
        """
        if not self.is_distributed:
            return metrics
        
        reduced_metrics = {}
        for key, value in metrics.items():
            try:
                # 跳过非数值指标
                if isinstance(value, str) or value is None:
                    reduced_metrics[key] = value
                    continue
                
                # 转换为张量
                if not isinstance(value, torch.Tensor):
                    tensor_value = torch.tensor(float(value), device=device, dtype=torch.float32)
                else:
                    tensor_value = value.to(device, dtype=torch.float32)
                
                # 进行 all_reduce 操作
                dist.all_reduce(tensor_value, op=dist.ReduceOp.AVG)
                
                # 转换回 Python 数值
                reduced_metrics[key] = tensor_value.item()
                
            except Exception as e:
                if self.rank == 0:
                    print(f"减少指标 {key} 时出错: {e}，使用原始值")
                reduced_metrics[key] = value
        
        return reduced_metrics