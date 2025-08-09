import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch.nn.functional as F
import json
from pathlib import Path
from pi0.modeling_pi0 import PI0Policy
# from lerobot.common.utils.utils import get_safe_dtype  # 暂时注释掉，使用本地实现

# Assuming the base interface is in a shared location
from pi0.ript.algos.rl_optimizers.model_interface import RLModelInterface

# 本地实现get_safe_dtype函数
def get_safe_dtype(tensor):
    """Get safe dtype for tensor operations"""
    if tensor.dtype in [torch.float16, torch.bfloat16]:
        return torch.float32
    return tensor.dtype


class PI0_CFG_Adapter(RLModelInterface):
    """
    Adapter for PI0Policy to work with the RIPT framework using an
    advantage-weighted loss, inspired by Classifier-Free Guidance.

    This adapter computes a weighted L2 loss from the policy's forward pass,
    which serves as a proxy for policy gradient updates, bypassing the need
    for explicit log_prob calculations or a value network.
    """

    def __init__(self, policy: PI0Policy, norm_stats_path: Optional[str] = None, **kwargs):
        """
        Initialize the adapter with a PI0Policy instance.
        Args:
            policy: An instance of PI0Policy.
            norm_stats_path: Path to norm_stats.json file for normalization. 
                           If None, will try to find it automatically.
        """
        # The base model is the PI0Policy itself.
        super().__init__(model=policy, **kwargs)
        self.policy = policy
        
        # 视频收集相关
        self.video_frames = {}  # {episode_idx: [frames]}
        self.video_save_enabled = True
        
        # 加载归一化统计信息
        self._load_norm_stats(norm_stats_path)
        
    def _load_norm_stats(self, norm_stats_path: Optional[str] = None):
        """Load normalization statistics from norm_stats.json"""
        if norm_stats_path is None:
            # 尝试在常见位置找到norm_stats.json
            possible_paths = [
                "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch/norm_stats.json",
                "/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json", 
                "./norm_stats.json"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    norm_stats_path = path
                    break
        
        if norm_stats_path and Path(norm_stats_path).exists():
            print(f"Loading norm_stats from: {norm_stats_path}")
            with open(norm_stats_path) as f:
                norm_stats = json.load(f)
                
            # 提取状态和动作的归一化参数
            self.state_mean = np.array(norm_stats["norm_stats"]["state"]["mean"][:8], dtype=np.float32)
            self.state_std = np.array(norm_stats["norm_stats"]["state"]["std"][:8], dtype=np.float32)
            self.action_mean = np.array(norm_stats["norm_stats"]["actions"]["mean"][:7], dtype=np.float32)
            self.action_std = np.array(norm_stats["norm_stats"]["actions"]["std"][:7], dtype=np.float32)
            
            print(f"✅ Loaded normalization stats:")
            print(f"  State mean shape: {self.state_mean.shape}")
            print(f"  State std shape: {self.state_std.shape}")
            print(f"  Action mean shape: {self.action_mean.shape}")
            print(f"  Action std shape: {self.action_std.shape}")
        else:
            print("⚠️  Warning: No norm_stats.json found, using identity normalization")
            # 使用单位归一化（不进行归一化）
            self.state_mean = np.zeros(8, dtype=np.float32)
            self.state_std = np.ones(8, dtype=np.float32)
            self.action_mean = np.zeros(7, dtype=np.float32)
            self.action_std = np.ones(7, dtype=np.float32)
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using loaded statistics"""
        return (state - self.state_mean[:len(state)]) / (self.state_std[:len(state)] + 1e-6)
    
    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action using loaded statistics"""
        return action * (self.action_std + 1e-6) + self.action_mean

    def process_episodes(
        self,
        episodes: List[Dict[str, Any]],
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Process a list of episode dictionaries into a single batch usable by PI0Policy.
        优化版本，更好地处理 Pi0LiberoRunner 生成的 episode 数据。
        
        Args:
            episodes: A list of episode dictionaries collected from the runner.
            device: The device to place tensors on.
        Returns:
            A batch dictionary ready for `policy.forward()`.
        """
        if device is None:
            device = self.device

        # Validate input
        if not episodes:
            raise ValueError("Empty episodes list provided")

        all_obs_states = []
        all_obs_images = []
        all_actions = []
        all_tasks = []

        # Process each episode
        for i, episode in enumerate(episodes):
            try:
                # 验证 episode 结构
                required_keys = ['observations', 'actions', 'task']
                missing_keys = [key for key in required_keys if key not in episode]
                if missing_keys:
                    raise KeyError(f"Episode {i} missing keys: {missing_keys}")
                
                # 获取 episode 数据
                observations = episode['observations']
                actions = episode['actions'] 
                tasks = episode['task']
                
                if not observations:
                    raise ValueError(f"Episode {i} has empty observations")
                    
                # 处理每个时间步的数据
                episode_states = []
                episode_images = []
                episode_actions = []
                
                # 取样本：确保固定的序列长度
                max_steps = min(len(observations), len(actions))
                # 🔧 修复：分离扩散步数和动作序列长度
                # 使用 n_action_steps 作为动作序列长度，而不是 num_steps（扩散去噪步数）
                target_seq_len = getattr(self.policy.config, 'n_action_steps', 50)  
                diffusion_steps = getattr(self.policy.config, 'num_steps', 10)
                print(f"Using target_seq_len={target_seq_len} from PI0 config (n_action_steps)")
                print(f"Diffusion denoising steps={diffusion_steps} from PI0 config (num_steps)")
                
                # 为了生成完整的调试视频，我们需要保存所有步骤的图像
                # 但对于模型训练，仍然使用固定的序列长度
                debug_save_all_steps = True  # 启用完整轨迹调试保存
                
                if debug_save_all_steps:
                    # 调试模式：保存所有步骤的图像数据
                    sample_indices = list(range(max_steps))
                    print(f"调试模式：保存完整轨迹 {max_steps} 步用于视频生成")
                elif max_steps >= target_seq_len:
                    # 如果有足够的步数，取最后 target_seq_len 步
                    sample_indices = list(range(max_steps - target_seq_len, max_steps))
                else:
                    # 如果步数不足，取所有步数并补齐
                    sample_indices = list(range(max_steps))
                
                for step_idx in sample_indices:
                    # 处理观测数据
                    if step_idx < len(observations):
                        step_obs = observations[step_idx]
                        
                        # 处理状态数据
                        state = self._extract_state_from_obs(step_obs, i, step_idx)
                        episode_states.append(state)
                        
                        # 处理图像数据
                        image = self._extract_image_from_obs(step_obs, i, step_idx)
                        episode_images.append(image)
                    else:
                        # 如果观测数据不足，使用默认值
                        episode_states.append(np.zeros(7, dtype=np.float32))
                        episode_images.append(np.ones((224, 224, 3), dtype=np.uint8) * 128)
                    
                    # 处理动作数据
                    if step_idx < len(actions):
                        step_action = actions[step_idx]
                        if isinstance(step_action, list) and len(step_action) > 0:
                            # 取第一个环境的动作（对于单环境或第一个环境）
                            action = np.array(step_action[0], dtype=np.float32)
                        else:
                            action = np.array(step_action, dtype=np.float32)
                        
                        # 确保动作是 7 维的（LIBERO 格式）
                        if action.size != 7:
                            if action.size > 7:
                                action = action[:7]
                            else:
                                padded_action = np.zeros(7, dtype=np.float32)
                                padded_action[:action.size] = action
                                action = padded_action
                        
                        episode_actions.append(action)
                    else:
                        episode_actions.append(np.zeros(7, dtype=np.float32))
                
                # 对整个 episode 的数据做聚合，确保固定的序列长度
                if episode_states:
                    if debug_save_all_steps:
                        # 调试模式：保存所有图像用于视频生成，但只用最后的状态和动作用于训练
                        print(f"调试：保存了 {len(episode_images)} 张图像用于视频生成")
                        
                        # 对于模型训练，仍使用最后的状态和最后target_seq_len个动作
                        final_state = np.array(episode_states[-1])  # 取最后一个状态
                        final_image = episode_images[-1]  # 取最后一个图像用于模型输入
                        
                        # 确保动作序列长度为 target_seq_len（用于模型训练）
                        if len(episode_actions) >= target_seq_len:
                            # 取最后 target_seq_len 个动作用于训练
                            final_action = np.array(episode_actions[-target_seq_len:])
                        else:
                            # 不足的话进行填充
                            padded_actions = []
                            for i in range(target_seq_len):
                                if i < len(episode_actions):
                                    padded_actions.append(episode_actions[i])
                                else:
                                    # 用最后一个动作填充
                                    padded_actions.append(episode_actions[-1] if episode_actions else np.zeros(7, dtype=np.float32))
                            final_action = np.array(padded_actions)
                        
                        print(f"调试：训练数据 - 状态形状: {final_state.shape}, 图像形状: {final_image.shape}, 动作形状: {final_action.shape}")
                    else:
                        # 原始模式：只处理采样的步骤
                        final_state = np.array(episode_states[-1])  # 取最后一个状态
                        final_image = episode_images[-1]  # 取最后一个图像
                        
                        # 确保动作序列长度为 target_seq_len
                        if len(episode_actions) >= target_seq_len:
                            final_action = np.array(episode_actions[:target_seq_len])  # 取前 target_seq_len 个
                        else:
                            # 不足的话进行填充
                            padded_actions = []
                            for i in range(target_seq_len):
                                if i < len(episode_actions):
                                    padded_actions.append(episode_actions[i])
                                else:
                                    # 用最后一个动作填充
                                    padded_actions.append(episode_actions[-1] if episode_actions else np.zeros(7, dtype=np.float32))
                            final_action = np.array(padded_actions)
                else:
                    final_state = np.zeros(7, dtype=np.float32)
                    final_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
                    final_action = np.zeros((target_seq_len, 7), dtype=np.float32)
                
                all_obs_states.append(final_state)
                all_obs_images.append(final_image)
                all_actions.append(final_action)
                
                # 在调试模式下，生成episode视频
                if debug_save_all_steps and len(episode_images) > 10:
                    try:
                        self._generate_episode_video(episode_images, i, tasks[0] if tasks else "unknown_task")
                    except Exception as video_e:
                        print(f"生成episode视频失败: {video_e}")
                
                # 处理任务描述
                if tasks:
                    task_str = tasks[0] if isinstance(tasks, list) else str(tasks)
                else:
                    task_str = "default task"
                all_tasks.append(task_str)
                
            except Exception as e:
                print(f"Error processing episode {i}: {e}")
                print(f"Episode keys: {episode.keys() if isinstance(episode, dict) else 'Not a dict'}")
                # 使用默认值继续
                all_obs_states.append(np.zeros(7, dtype=np.float32))
                all_obs_images.append(np.ones((224, 224, 3), dtype=np.uint8) * 128)
                all_actions.append(np.zeros((1, 7), dtype=np.float32))
                all_tasks.append("default task")

        try:
            # 🔧 关键修复：将7维LIBERO动作填充到32维PI0动作
            padded_actions = []
            for action_seq in all_actions:
                # action_seq shape: (sequence_length, 7)
                seq_len = action_seq.shape[0]
                # 创建 (sequence_length, 32) 的零填充动作
                padded_action_seq = np.zeros((seq_len, 32), dtype=np.float32)
                # 将前7维复制过去
                padded_action_seq[:, :7] = action_seq
                padded_actions.append(padded_action_seq)
            
            # Create the final batch dictionary with nested structure expected by PI0
            batch = {
                'state': torch.from_numpy(np.stack(all_obs_states)).to(device, dtype=torch.float32),
                'image': {
                    'base_0_rgb': torch.from_numpy(np.stack(all_obs_images)).to(device, dtype=torch.uint8)
                },
                'action': torch.from_numpy(np.stack(padded_actions)).to(device, dtype=torch.float32),
                'prompt': all_tasks,
            }
            
            # Validate batch dimensions
            batch_size = len(episodes)
            assert batch['state'].shape[0] == batch_size
            assert batch['image']['base_0_rgb'].shape[0] == batch_size
            assert batch['action'].shape[0] == batch_size
            assert len(batch['prompt']) == batch_size
            
            print(f"Processed batch - states: {batch['state'].shape}, "
                  f"images: {batch['image']['base_0_rgb'].shape}, "
                  f"actions: {batch['action'].shape} (padded to 32 dims)")
            
            return batch
            
        except Exception as e:
            print(f"Error creating batch: {e}")
            print(f"States shapes: {[s.shape for s in all_obs_states]}")
            print(f"Images shapes: {[img.shape for img in all_obs_images]}")
            print(f"Actions shapes: {[a.shape for a in all_actions]}")
            raise

    def compute_weighted_loss(
        self,
        episodes: List[Dict[str, Any]],
        advantages: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Computes the advantage-weighted L2 loss for a batch of trajectories.
        This is the core of the CFG-style training update.
        Enhanced with comprehensive NaN/Inf detection and error handling.

        Args:
            episodes: A list of episode dictionaries.
            advantages: A tensor of shape (batch_size,) containing the advantage for each episode.
            device: The device to place tensors on.

        Returns:
            A scalar tensor representing the final weighted loss for backpropagation.
        """
        if device is None:
            device = self.device

        # Validate inputs
        if not episodes:
            print("❌ 空的episode列表，返回零损失")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        if advantages is None or len(advantages) == 0:
            print("❌ 空的优势张量，返回零损失")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        if len(episodes) != len(advantages):
            print(f"❌ episodes数量({len(episodes)})与advantages数量({len(advantages)})不匹配")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Check for NaN/Inf in advantages before processing
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            print("⚠️ 检测到优势中的NaN/Inf值，进行清理")
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1.0, neginf=-1.0)

        # 关键修改：为每个轨迹分别计算损失，以获得正确的形状
        print(f"开始为 {len(episodes)} 个轨迹分别计算损失")
        per_trajectory_losses = []
        
        # 添加进度条以更清晰地显示处理进度
        from tqdm import tqdm
        
        try:
            for i, episode in tqdm(enumerate(episodes), total=len(episodes), desc="计算轨迹损失", leave=False):
                # 将单个轨迹转换为批次格式
                single_batch = self.process_episodes([episode], device)
                
                # 只在有问题时才输出详细调试信息
                debug_needed = False
                for key, value in single_batch.items():
                    if torch.is_tensor(value):
                        # Check for NaN/Inf in input tensors
                        if torch.isnan(value).any() or torch.isinf(value).any():
                            if not debug_needed:  # 只输出一次header
                                print(f"\n⚠️ 轨迹 {i}: 检测到异常值")
                                debug_needed = True
                            print(f"  - 张量 {key} 含有NaN/Inf值")
                            if key == 'observation.image':
                                single_batch[key] = torch.nan_to_num(value, nan=0.5, posinf=1.0, neginf=0.0)
                            else:
                                single_batch[key] = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 为单个轨迹计算损失
                # 🔧 Debug: 检查传递给PI0的batch结构
                if i == 0:  # 只打印第一个轨迹的调试信息
                    print(f"传递给PI0的batch结构:")
                    for key, value in single_batch.items():
                        if isinstance(value, dict):
                            print(f"  {key}: dict with keys {list(value.keys())}")
                            for k2, v2 in value.items():
                                print(f"    {k2}: {type(v2)} {getattr(v2, 'shape', 'N/A')}")
                        else:
                            print(f"  {key}: {type(value)} {getattr(value, 'shape', 'N/A')}")
                
                policy_outputs = self.policy.forward(single_batch)
                
                # 处理不同的policy输出格式
                if isinstance(policy_outputs, tuple):
                    single_loss = policy_outputs[0]
                elif isinstance(policy_outputs, torch.Tensor):
                    single_loss = policy_outputs
                elif isinstance(policy_outputs, dict) and 'loss' in policy_outputs:
                    single_loss = policy_outputs['loss']
                else:
                    print(f"❌ 轨迹 {i}: 未知的策略输出格式: {type(policy_outputs)}")
                    single_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                if single_loss is None:
                    print(f"❌ 轨迹 {i}: 策略返回None损失")
                    single_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                # 确保损失是标量
                if single_loss.dim() > 0:
                    single_loss = single_loss.mean()
                
                # 检查NaN/Inf
                if torch.isnan(single_loss) or torch.isinf(single_loss):
                    print(f"⚠️ 轨迹 {i}: 检测到NaN/Inf损失，替换为0")
                    single_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                per_trajectory_losses.append(single_loss)
                # 只保留异常情况的日志输出，正常情况静默处理
                
            # 将列表转换为张量
            per_trajectory_loss = torch.stack(per_trajectory_losses)  # Shape: (num_trajectories,)
            
            print(f"✓ 轨迹损失计算完成 - 形状: {per_trajectory_loss.shape}")
            if len(per_trajectory_losses) <= 10:  # 只有轨迹数较少时才显示详细值
                print(f"  损失值: {[f'{x.item():.4f}' for x in per_trajectory_loss]}")
            else:
                print(f"  损失范围: [{per_trajectory_loss.min().item():.4f}, {per_trajectory_loss.max().item():.4f}]")
            
        except Exception as e:
            print(f"❌ 分别计算轨迹损失时出错: {e}")
            import traceback
            traceback.print_exc()
            # 创建安全的默认损失
            per_trajectory_loss = torch.zeros(len(episodes), device=device, requires_grad=True)

        # 4. Weight the loss by the advantage
        # advantages tensor shape is (batch_size,)
        print(f"优势形状: {advantages.shape}, 值: {advantages}")
        
        try:
            # 现在形状应该匹配了！
            if per_trajectory_loss.shape[0] != advantages.shape[0]:
                print(f"❌ 损失和优势形状仍然不匹配: {per_trajectory_loss.shape[0]} vs {advantages.shape[0]}")
                print("这不应该发生，但为了安全起见进行截断")
                min_size = min(per_trajectory_loss.shape[0], advantages.shape[0])
                per_trajectory_loss = per_trajectory_loss[:min_size]
                advantages = advantages[:min_size]
            else:
                print(f"✅ 损失和优势形状匹配: {per_trajectory_loss.shape[0]} vs {advantages.shape[0]}")
            
            advantages = advantages.to(device)
            weighted_loss = per_trajectory_loss * advantages
            
            # Check for NaN/Inf in weighted loss
            if torch.isnan(weighted_loss).any() or torch.isinf(weighted_loss).any():
                print("⚠️ 加权损失产生NaN/Inf值，进行清理")
                weighted_loss = torch.nan_to_num(weighted_loss, nan=0.0, posinf=10.0, neginf=-10.0)
            
            print(f"✓ 加权损失: {weighted_loss}")
        
        except Exception as e:
            print(f"❌ 计算加权损失时出错: {e}")
            weighted_loss = torch.zeros(len(episodes), device=device, requires_grad=True)

        # 5. Return the mean loss for the batch
        try:
            final_loss = weighted_loss.mean()
            
            # Final NaN/Inf check
            if torch.isnan(final_loss) or torch.isinf(final_loss):
                print("⚠️ 最终损失为NaN/Inf，使用零损失")
                final_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            print(f"✓ 最终损失: {final_loss.item()}")
            return final_loss
        
        except Exception as e:
            print(f"❌ 计算最终损失时出错: {e}")
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def compute_act_logits(self, model, episodes: List[Dict[str, Any]], device: Optional[torch.device] = None):
        """
        This method is required by the RLModelInterface, but for our CFG-style
        optimizer, we use `compute_weighted_loss` instead.
        We can raise an error or return a placeholder.
        """
        raise NotImplementedError(
            "compute_act_logits is not used for PI0_CFG_Adapter. "
            "Use compute_weighted_loss instead."
        )

    @property
    def device(self):
        return next(self.policy.parameters()).device

    def get_policy_model(self):
        """Return the policy model, which is the PI0Policy instance itself."""
        return self.policy
        
    def _extract_state_from_obs(self, obs, episode_idx, step_idx):
        """从观测中提取状态信息并应用归一化"""
        try:
            # 🔧 修复：处理SubprocVectorEnv返回的numpy.array包装的观测
            if isinstance(obs, np.ndarray) and obs.dtype == object:
                if obs.size == 1:
                    obs = obs.item()  # 提取单个元素
                elif obs.size > 0:
                    obs = obs[0]  # 取第一个元素
            
            if isinstance(obs, list) and len(obs) > 0:
                obs = obs[0]  # 取第一个环境的观测
            
            if not isinstance(obs, dict) or not obs:
                return np.zeros(7, dtype=np.float32)
            
            # 优先使用 end-effector 信息
            if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
                eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
                eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
                
                if eef_pos.size != 3:
                    eef_pos = np.zeros(3, dtype=np.float32)
                if eef_quat.size != 4:
                    eef_quat = np.array([0, 0, 0, 1], dtype=np.float32)
                
                # 转换四元数为轴角
                try:
                    from pi0.ript.utils.pi0_libero_utils import quat2axisangle
                    axis_angle = quat2axisangle(eef_quat).astype(np.float32)
                except Exception:
                    axis_angle = np.zeros(3, dtype=np.float32)
                
                # 获取 gripper 状态
                gripper_qpos = 0.0
                if "robot0_gripper_qpos" in obs:
                    try:
                        gripper_qpos = float(obs["robot0_gripper_qpos"][0])
                    except (IndexError, TypeError, ValueError):
                        gripper_qpos = 0.0
                
                # 构造未归一化的状态 (按照原始demo的格式)
                unnorm_state = np.concatenate([eef_pos[:3], axis_angle[:3], [gripper_qpos]]).astype(np.float32)
                
                # 应用归一化（就像原始demo中的做法）
                state = self.normalize_state(unnorm_state)
                
            # 备用方案：使用关节位置
            elif "robot0_joint_pos" in obs:
                joint_pos = np.array(obs["robot0_joint_pos"], dtype=np.float32)
                if joint_pos.size >= 7:
                    unnorm_state = joint_pos[:7]
                else:
                    unnorm_state = np.zeros(7, dtype=np.float32)
                    unnorm_state[:joint_pos.size] = joint_pos
                
                # 应用归一化 
                state = self.normalize_state(unnorm_state)
            else:
                state = np.zeros(7, dtype=np.float32)
            
            # 处理异常值
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            return state.astype(np.float32)
            
        except Exception as e:
            print(f"提取状态时出错 (episode {episode_idx}, step {step_idx}): {e}")
            return np.zeros(7, dtype=np.float32)
    
    def _extract_image_from_obs(self, obs, episode_idx, step_idx):
        """从观测中提取图像信息"""
        try:
            # 🔧 修复：处理SubprocVectorEnv返回的numpy.array包装的观测
            if isinstance(obs, np.ndarray) and obs.dtype == object:
                if obs.size == 1:
                    obs = obs.item()  # 提取单个元素
                elif obs.size > 0:
                    obs = obs[0]  # 取第一个元素
            
            if isinstance(obs, list) and len(obs) > 0:
                obs = obs[0]  # 取第一个环境的观测
            
            if isinstance(obs, dict) and "agentview_image" in obs:
                img = obs["agentview_image"]
                if isinstance(img, np.ndarray) and img.size > 0:
                    # 🔧 修复图像格式检查：处理CHW和HWC两种格式
                    if img.ndim == 3:
                        # 检查是CHW格式 (3, H, W) 还是HWC格式 (H, W, 3)
                        if img.shape[0] == 3 and img.shape[-1] != 3:  # CHW格式
                            # 转换CHW → HWC
                            img = img.transpose(1, 2, 0)
                        elif img.shape[-1] != 3:  # 既不是CHW也不是HWC
                            print(f"✗ 未知图像格式: {img.shape}")
                            raise ValueError(f"Unexpected image format: {img.shape}")
                        
                        # 现在img应该是HWC格式
                        # 静默处理图像（减少冗余日志）
                        
                        # 为模型推理准备图像（确保接收正确方向的图像）
                        # 根据您的需求，确保模型接收的是正着的图像
                        model_img = img.copy()  # 直接使用原始图像，不进行旋转
                        
                        # 收集图像用于视频生成（需要旋转以显示正确方向）
                        video_img = img[::-1, ::-1]  # 180度旋转用于视频显示
                        self._collect_image_for_video(video_img, episode_idx, step_idx)
                        
                        # 返回用于模型推理的图像（已进行180度旋转）
                        # 如果图像尺寸不是 224x224，需要调整大小
                        if model_img.shape[:2] != (224, 224):
                            try:
                                from skimage.transform import resize
                                model_img = resize(model_img, (224, 224), preserve_range=True).astype(np.uint8)
                            except ImportError:
                                # 如果没有 skimage，使用简单的插值
                                import cv2
                                model_img = cv2.resize(model_img, (224, 224))
                        return model_img
                    else:
                        print(f"✗ 图像维度异常: {img.shape}")
                else:
                    print(f"✗ 图像数据无效: type={type(img)}, size={getattr(img, 'size', 'N/A')}")
            else:
                available_keys = list(obs.keys()) if isinstance(obs, dict) else "非字典类型"
                print(f"✗ 观测中无agentview_image键, 可用键: {available_keys}")
            
            # 如果没有有效图像，返回占位符
            print(f"✗ 使用占位符图像 (episode {episode_idx}, step {step_idx})")
            return np.ones((224, 224, 3), dtype=np.uint8) * 128
            
        except Exception as e:
            print(f"提取图像时出错 (episode {episode_idx}, step {step_idx}): {e}")
            return np.ones((224, 224, 3), dtype=np.uint8) * 128

    def _collect_image_for_video(self, img, episode_idx, step_idx):
        """收集图像用于视频生成"""
        if not self.video_save_enabled:
            return
            
        try:
            if episode_idx not in self.video_frames:
                self.video_frames[episode_idx] = []
            
            # 保存原始方向的图像用于视频
            self.video_frames[episode_idx].append(img.copy())
            
        except Exception as e:
            print(f"收集视频帧时出错: {e}")

    def finalize_episode_video(self, episode_idx, task_description="default_task"):
        """完成episode时生成视频"""
        if not self.video_save_enabled or episode_idx not in self.video_frames:
            return None
            
        frames = self.video_frames[episode_idx]
        if len(frames) > 0:
            video_path = self._generate_episode_video(frames, episode_idx, task_description)
            # 清理已使用的帧
            del self.video_frames[episode_idx]
            return video_path
        return None

    def _generate_episode_video(self, episode_images, episode_idx, task_description):
        """从episode图像生成视频"""
        try:
            import os
            import imageio
            from datetime import datetime
            
            # 创建视频保存目录
            debug_dir = os.path.join(os.getcwd(), "ript", "debug_images")
            video_dir = os.path.join(debug_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            
            # 生成视频文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_str = str(task_description).replace(" ", "_").replace("/", "_")[:30]
            video_path = os.path.join(video_dir, f"episode_{episode_idx}_{timestamp}_{task_str}.mp4")
            
            # 确保所有图像尺寸一致
            processed_images = []
            for img in episode_images:
                if isinstance(img, np.ndarray) and img.size > 0:
                    if img.ndim == 3 and img.shape[-1] == 3:
                        # 调整到统一尺寸
                        if img.shape[:2] != (224, 224):
                            try:
                                from skimage.transform import resize
                                img = resize(img, (224, 224), preserve_range=True).astype(np.uint8)
                            except ImportError:
                                import cv2
                                img = cv2.resize(img, (224, 224))
                        
                        # 图像已经在pi0_libero_utils.py中处理过180度旋转
                        # 但是为了视频显示正确，我们需要使用原始方向的图像
                        processed_images.append(img)  # 使用原始方向的图像
            
            if len(processed_images) >= 5:  # 至少5帧才生成视频
                # 保存视频
                imageio.mimsave(video_path, processed_images, fps=10)
                print(f"✓ Episode视频已生成: {video_path} ({len(processed_images)} 帧)")
                return video_path
            else:
                print(f"⚠️ Episode {episode_idx} 图像数量不足，跳过视频生成 ({len(processed_images)} 帧)")
                return None
                
        except Exception as e:
            print(f"生成episode视频时出错: {e}")
            return None