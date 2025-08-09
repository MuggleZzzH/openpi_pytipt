#!/usr/bin/env python3
"""
第2阶段：智能采样和批处理训练 (8_train_with_epochs.py)
基于7_train_with_rl_core.py，添加完整的RIPT智能采样和梯度累积逻辑

核心新增功能：
1. 初始状态数据集和智能采样
2. 多轮轨迹收集与过滤（跳过全0/全1状态）
3. 梯度累积机制（N个batch后才更新）
4. 多训练迭代循环
5. 统计跟踪和状态管理

使用方法:
    cd /zhaohan/ZJH/openpi_pytorch
    python 8_train_with_epochs.py
    
    # 自定义配置
    python 8_train_with_epochs.py --training_iterations 3 --rollouts_per_batch 4 --gradient_accumulation_steps 4
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
from typing import List, Dict, Any, Tuple

# 修复tokenizers并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pi0 import PI0Policy
from pi0.ript.env.pi0_libero_runner import LIBEROEnvRunner
from pi0.ript.reward_function import BinarySuccessReward
from pi0.ript.algos.rl_optimizers.pi0_cfg_interface import PI0_CFG_Adapter

class AdvancedTrainingConfig:
    """高级训练配置类，支持完整的RIPT功能"""
    def __init__(self):
        # 模型配置
        self.model_path = "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch"
        self.norm_stats_path = "/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json"
        
        # 环境配置
        self.benchmark_name = "libero_spatial"
        self.task_id = 8
        self.num_parallel_envs = 1
        self.max_episode_length = 200
        
        # 🎯 完整RIPT训练配置
        self.training_iterations = 2    # 主训练循环数
        self.rollouts_per_batch = 2     # 每个初始状态收集的轨迹数
        self.target_batches_per_iteration = 2  # 每轮训练收集的有效batch数
        self.gradient_accumulation_steps = 2   # 梯度累积步数
        self.num_epochs = 2             # 每轮收集数据的优化epoch数
        
        # 强化学习配置
        self.learning_rate = 1e-5
        self.optimizer_type = "AdamW"
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        
        # 🧠 智能采样配置
        self.enable_smart_filtering = True      # 启用智能过滤
        self.skip_uniform_results = True        # 跳过全0或全1的结果
        self.max_sampling_attempts = 20         # 最大采样尝试次数
        self.success_rate_tolerance = 0.1       # 成功率容忍度
        
        # 统计和跟踪
        self.enable_state_tracking = True       # 启用状态跟踪
        self.stats_save_freq = 2               # 统计保存频率
        
        # 🎯 增强采样配置 (新增)
        self.enable_rollout_stats_tracking = True  # 启用rollout统计跟踪
        self.rollout_skip_threshold = 3            # 跳过阈值
        self.rollout_stats_path = f"./rollout_stats_task{self.task_id}.json"  # 统计文件路径
        self.state_history_window = 20             # 状态历史窗口大小
        
        # 调试配置
        self.save_videos = True
        self.video_dir = "rollout_videos_advanced"
        self.debug_sampling = True
        self.log_training_details = True

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PI0 RIPT Advanced Training with Smart Sampling")
    parser.add_argument("--training_iterations", type=int, default=2, help="Number of training iterations")
    parser.add_argument("--rollouts_per_batch", type=int, default=2, help="Rollouts per initial state batch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--target_batches", type=int, default=2, help="Target batches per iteration")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--debug_sampling", type=bool, default=True, help="Enable sampling debugging")
    return parser.parse_args()

def compute_state_hash(state_array: np.ndarray) -> str:
    """计算状态数组的哈希值用于追踪"""
    return hashlib.sha256(np.ascontiguousarray(state_array).tobytes()).hexdigest()[:16]

def create_optimizer(policy, config):
    """创建优化器（与7脚本一致）"""
    print(f"\n=== 创建优化器 ===")
    print(f"优化器类型: {config.optimizer_type}")
    print(f"学习率: {config.learning_rate}")
    print(f"权重衰减: {config.weight_decay}")
    
    if config.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
    elif config.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=0.0
        )
    else:
        raise ValueError(f"不支持的优化器类型: {config.optimizer_type}")
    
    # 检查可训练参数
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"训练参数比例: {trainable_params/total_params:.2%}")
    
    return optimizer

class InitialStateDataset:
    """多样化的初始状态数据集，支持RIPT智能采样"""
    def __init__(self, num_states=50, state_dim=8):
        """创建多样化的初始状态数据集"""
        # 创建多样化的初始状态
        self.states = []
        np.random.seed(None)  # 使用真随机种子以获得不同的状态
        
        for i in range(num_states):
            # 生成更多样化的初始状态
            base_state = np.zeros(state_dim, dtype=np.float32)
            
            # 添加不同程度的随机扰动
            if i < num_states // 3:
                # 小扰动状态
                noise = np.random.normal(0, 0.05, state_dim).astype(np.float32)
            elif i < 2 * num_states // 3:
                # 中等扰动状态
                noise = np.random.normal(0, 0.15, state_dim).astype(np.float32)
            else:
                # 大扰动状态
                noise = np.random.normal(0, 0.25, state_dim).astype(np.float32)
            
            state = base_state + noise
            self.states.append(state)
        
        print(f"✓ 创建了包含 {len(self.states)} 个多样化初始状态的数据集")
        print(f"  状态分布：小扰动({num_states//3}个), 中扰动({num_states//3}个), 大扰动({num_states - 2*num_states//3}个)")
    
    def sample_batch(self, batch_size=1):
        """随机采样一批不重复的初始状态"""
        if batch_size > len(self.states):
            # 如果需要的状态数超过可用状态，则允许重复
            indices = np.random.choice(len(self.states), batch_size, replace=True)
        else:
            # 不重复采样
            indices = np.random.choice(len(self.states), batch_size, replace=False)
        
        sampled_states = np.array([self.states[i] for i in indices])
        return sampled_states

# 导入增强智能采样系统
from pi0.ript.utils.enhanced_smart_sampling import EnhancedSmartSampler, enhanced_collect_smart_batches

# 保留原有的SmartSampler作为备用（已被EnhancedSmartSampler替代）
class SmartSampler:
    """智能采样器 - RIPT的核心创新 (已被EnhancedSmartSampler替代)"""
    def __init__(self, config):
        self.config = config
        self.state_stats = {}  # 记录每个状态的统计信息
        self.sampling_history = []  # 采样历史
        
    def is_batch_useful(self, rollouts: List[Tuple]) -> bool:
        """判断一个batch的轨迹是否有用（不是全0或全1）"""
        if not rollouts:
            return False
            
        successes = [r[0] for r in rollouts]  # 提取success标志
        
        # 检查是否全成功或全失败
        all_success = all(s == True for s in successes)
        all_failure = all(s == False for s in successes)
        
        if all_success:
            if self.config.debug_sampling:
                print(f"    跳过：全部成功的batch (成功率=100%)")
            return False
        elif all_failure:
            if self.config.debug_sampling:
                print(f"    跳过：全部失败的batch (成功率=0%)")
            return False
        else:
            success_rate = sum(successes) / len(successes)
            if self.config.debug_sampling:
                print(f"    ✓ 有用的batch：成功率={success_rate:.2f}")
            return True
    
    def update_state_stats(self, state_hash: str, rollouts: List[Tuple]):
        """更新状态统计信息"""
        if not self.config.enable_state_tracking:
            return
            
        successes = [r[0] for r in rollouts]
        
        if state_hash not in self.state_stats:
            self.state_stats[state_hash] = {
                'attempts': 0,
                'successes': [],
                'last_success_rate': 0.0
            }
        
        self.state_stats[state_hash]['attempts'] += 1
        self.state_stats[state_hash]['successes'].extend(successes)
        
        # 只保留最近的记录
        if len(self.state_stats[state_hash]['successes']) > 20:
            self.state_stats[state_hash]['successes'] = self.state_stats[state_hash]['successes'][-20:]
        
        # 计算最新成功率
        recent_successes = self.state_stats[state_hash]['successes']
        self.state_stats[state_hash]['last_success_rate'] = sum(recent_successes) / len(recent_successes)

def collect_smart_batches(env_runner, reward_function, init_dataset, sampler, config, iteration_idx):
    """智能收集有用的轨迹batches"""
    print(f"\n--- 智能轨迹收集 (迭代 {iteration_idx + 1}) ---")
    print(f"目标: 收集 {config.target_batches_per_iteration} 个有用的batches")
    print(f"每个batch包含 {config.rollouts_per_batch} 个轨迹")
    
    # 切换到推理模式收集轨迹
    env_runner.policy.eval()
    
    collected_batches = []
    total_attempts = 0
    
    while len(collected_batches) < config.target_batches_per_iteration:
        if total_attempts >= config.max_sampling_attempts:
            print(f"⚠️ 达到最大采样次数 {config.max_sampling_attempts}，停止收集")
            break
            
        total_attempts += 1
        
        # 1. 随机采样不同的初始状态
        init_states = init_dataset.sample_batch(config.num_parallel_envs)
        state_hash = compute_state_hash(init_states)
        
        if config.debug_sampling:
            print(f"\n尝试 {total_attempts}: 使用初始状态 {state_hash}")
            print(f"    状态值: {init_states[0][:4]}")  # 显示前4个值
        
        # 2. 在该初始状态下运行多个轨迹
        try:
            state_rollouts = []
            for rollout_idx in range(config.rollouts_per_batch):
                if config.debug_sampling:
                    print(f"    轨迹 {rollout_idx + 1}/{config.rollouts_per_batch} 从状态 {state_hash[:8]}...")
                
                # 为每个轨迹单独调用环境
                rollout_generator = env_runner.run_policy_in_env(
                    env_name=config.task_id,
                    all_init_states=init_states
                )
                
                try:
                    rollout_batch = list(rollout_generator)
                    if rollout_batch:
                        state_rollouts.extend(rollout_batch)
                    else:
                        if config.debug_sampling:
                            print(f"      轨迹 {rollout_idx + 1} 失败：无数据")
                        break
                except StopIteration:
                    if config.debug_sampling:
                        print(f"      轨迹 {rollout_idx + 1} 完成")
                    break
            
            if not state_rollouts:
                if config.debug_sampling:
                    print(f"    无效：没有收集到轨迹")
                continue
                
        except Exception as e:
            if config.debug_sampling:
                print(f"    错误：轨迹收集失败 - {e}")
            continue
        
        # 3. 智能过滤：检查是否有用
        if config.enable_smart_filtering:
            if not sampler.is_batch_useful(state_rollouts):
                sampler.update_state_stats(state_hash, state_rollouts)
                continue  # 跳过这个batch
        
        # 4. 转换为episode格式并计算奖励
        episodes = []
        for success, total_reward, episode_data in state_rollouts:
            episode = {
                'success': success,
                'total_reward': total_reward,
                **episode_data
            }
            
            # 使用奖励函数计算奖励
            try:
                computed_reward = reward_function.compute_reward(len(episodes), episode, None)
                episode['computed_reward'] = computed_reward
            except Exception as e:
                episode['computed_reward'] = 0.0
            
            episodes.append(episode)
        
        # 5. 收集有用的batch
        collected_batches.append(episodes)
        sampler.update_state_stats(state_hash, state_rollouts)
        
        if config.debug_sampling:
            success_count = sum(ep['success'] for ep in episodes)
            success_rate = success_count/len(episodes)
            print(f"    ✓ 收集batch {len(collected_batches)}: {len(episodes)} 个episodes, "
                  f"成功率={success_rate:.2f} ({'有用' if 0 < success_rate < 1 else '均匀'})")
    
    print(f"\n✓ 智能采样完成:")
    print(f"  总尝试次数: {total_attempts}")
    print(f"  收集有用batches: {len(collected_batches)}")
    print(f"  总episodes: {sum(len(batch) for batch in collected_batches)}")
    print(f"  采样效率: {len(collected_batches)/total_attempts:.2%}")
    
    return collected_batches

def train_with_gradient_accumulation(policy, optimizer, cfg_adapter, all_episodes, advantages, config):
    """使用梯度累积进行训练"""
    print(f"\n--- 梯度累积训练 ---")
    print(f"总episodes: {len(all_episodes)}")
    print(f"梯度累积步数: {config.gradient_accumulation_steps}")
    print(f"训练epochs: {config.num_epochs}")
    
    # 切换到训练模式
    policy.train()
    
    total_steps = 0
    all_losses = []
    
    for epoch in range(config.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
        
        # 创建episode批次
        episode_batches = []
        batch_size = max(1, len(all_episodes) // config.gradient_accumulation_steps)
        
        for i in range(0, len(all_episodes), batch_size):
            batch_episodes = all_episodes[i:i + batch_size]
            batch_advantages = advantages[i:i + len(batch_episodes)]
            episode_batches.append((batch_episodes, batch_advantages))
        
        print(f"创建了 {len(episode_batches)} 个批次，平均每批 {batch_size} episodes")
        
        # 梯度累积循环
        optimizer.zero_grad()
        epoch_losses = []
        
        for step, (batch_episodes, batch_advantages) in enumerate(episode_batches):
            try:
                # 计算batch损失
                batch_loss = cfg_adapter.compute_weighted_loss(batch_episodes, batch_advantages)
                
                # 归一化损失（梯度累积的关键）
                normalized_loss = batch_loss / config.gradient_accumulation_steps
                
                # 反向传播（累积梯度）
                normalized_loss.backward()
                
                batch_loss_value = batch_loss.item()
                epoch_losses.append(batch_loss_value)
                
                if config.log_training_details:
                    print(f"  Step {step + 1}: batch_loss={batch_loss_value:.6f}, "
                          f"normalized_loss={normalized_loss.item():.6f}")
                
                # 每N步或最后一步进行参数更新
                if (step + 1) % config.gradient_accumulation_steps == 0 or (step + 1) == len(episode_batches):
                    # 梯度裁剪
                    if config.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                        if config.log_training_details:
                            print(f"    梯度范数: {grad_norm:.6f}")
                    
                    # 参数更新
                    optimizer.step()
                    optimizer.zero_grad()
                    total_steps += 1
                    
                    if config.log_training_details:
                        print(f"    ✓ 参数更新 (总步数: {total_steps})")
                        
            except Exception as e:
                print(f"    ❌ Step {step + 1} 训练出错: {e}")
                continue
        
        # Epoch统计
        if epoch_losses:
            epoch_mean_loss = np.mean(epoch_losses)
            all_losses.extend(epoch_losses)
            print(f"✓ Epoch {epoch + 1} 平均损失: {epoch_mean_loss:.6f}")
        else:
            print(f"⚠️ Epoch {epoch + 1} 没有有效损失")
    
    return {
        'total_steps': total_steps,
        'total_losses': all_losses,
        'mean_loss': np.mean(all_losses) if all_losses else 0.0
    }

def main():
    print("=== 第2阶段：智能采样和批处理训练 ===")
    print("完整的RIPT智能采样、梯度累积和多轮训练")
    print()
    
    # 解析参数和配置
    args = parse_args()
    config = AdvancedTrainingConfig()
    
    # 应用命令行参数
    config.training_iterations = args.training_iterations
    config.rollouts_per_batch = args.rollouts_per_batch
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.target_batches_per_iteration = args.target_batches
    config.learning_rate = args.lr
    config.debug_sampling = args.debug_sampling
    
    print(f"🎯 高级训练配置:")
    print(f"  训练迭代数: {config.training_iterations}")
    print(f"  每batch轨迹数: {config.rollouts_per_batch}")
    print(f"  梯度累积步数: {config.gradient_accumulation_steps}")
    print(f"  目标batch数: {config.target_batches_per_iteration}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  智能采样: {config.enable_smart_filtering}")
    print(f"  任务ID: {config.task_id}")
    print("")
    
    # 加载PI0模型
    print(f"加载PI0模型: {config.model_path}")
    policy = PI0Policy.from_pretrained(config.model_path)
    device = policy.config.device
    print(f"使用设备: {device}")
    
    # 创建优化器
    optimizer = create_optimizer(policy, config)
    
    # 创建CFG适配器
    print(f"\n=== 创建CFG适配器 ===")
    os.environ["PI0_DEBUG_SAVE_VIDEO"] = "false"  # 避免重复视频
    cfg_adapter = PI0_CFG_Adapter(policy, norm_stats_path=config.norm_stats_path)
    print("✓ CFG适配器创建成功")
    
    # 创建环境运行器
    print("\n=== 初始化环境运行器 ===")
    env_runner = LIBEROEnvRunner(
        policy=policy,
        benchmark_name=config.benchmark_name,
        rollouts_per_env=1,
        num_parallel_envs=config.num_parallel_envs,
        max_episode_length=config.max_episode_length,
        task_names_to_use=[config.task_id],
        config=config.__dict__,
        rank=0,
        world_size=1,
        norm_stats_path=config.norm_stats_path
    )
    
    # 创建组件
    reward_function = BinarySuccessReward()
    init_dataset = InitialStateDataset()
    # 使用增强版智能采样器
    smart_sampler = EnhancedSmartSampler(config)
    
    # 创建视频目录
    video_dir = Path(config.video_dir)
    video_dir.mkdir(exist_ok=True)
    
    # 🚀 主训练循环开始
    print(f"\n=== 开始智能采样训练循环 ===")
    print(f"将进行 {config.training_iterations} 轮完整的训练迭代")
    print("🔍 使用RIPT智能采样：自动选择不同初始状态，过滤全0/全1结果")
    
    all_training_metrics = []
    
    for iteration in range(config.training_iterations):
        print(f"\n" + "="*60)
        print(f"🔄 训练迭代 {iteration + 1}/{config.training_iterations}")
        print("="*60)
        
        # 第1步：增强智能采样收集数据
        print(f"🎯 开始增强智能采样数据收集...")
        collected_batches = enhanced_collect_smart_batches(
            env_runner, reward_function, init_dataset, 
            smart_sampler, config, iteration
        )
        
        if not collected_batches:
            print("⚠️ 智能采样未收集到有用的数据，跳过本轮训练")
            continue
        
        # 第2步：合并所有episodes
        all_episodes = []
        for batch in collected_batches:
            all_episodes.extend(batch)
        
        if not all_episodes:
            print("⚠️ 没有有效的episodes，跳过本轮训练")
            continue
        
        # 第3步：计算优势值
        print(f"\n--- 优势计算 ---")
        all_rewards = [ep['total_reward'] for ep in all_episodes]
        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
        
        # Leave-One-Out优势计算
        advantages = []
        for i in range(len(all_rewards)):
            other_rewards = torch.cat([rewards_tensor[:i], rewards_tensor[i+1:]])
            baseline = other_rewards.mean() if len(other_rewards) > 0 else 0.0
            advantage = rewards_tensor[i] - baseline
            advantages.append(advantage.item())
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        
        print(f"Episodes总数: {len(all_episodes)}")
        print(f"平均奖励: {rewards_tensor.mean().item():.4f}")
        print(f"成功率: {sum(ep['success'] for ep in all_episodes) / len(all_episodes):.2f}")
        print(f"优势方差: {advantages.var().item():.6f}")
        
        # 第4步：梯度累积训练
        training_metrics = train_with_gradient_accumulation(
            policy, optimizer, cfg_adapter, all_episodes, advantages, config
        )
        
        # 第5步：记录迭代指标
        iteration_metrics = {
            'iteration': iteration + 1,
            'collected_batches': len(collected_batches),
            'total_episodes': len(all_episodes),
            'mean_reward': rewards_tensor.mean().item(),
            'success_rate': sum(ep['success'] for ep in all_episodes) / len(all_episodes),
            'advantage_variance': advantages.var().item(),
            **training_metrics
        }
        
        all_training_metrics.append(iteration_metrics)
        
        print(f"\n✓ 迭代 {iteration + 1} 完成:")
        print(f"  收集batches: {iteration_metrics['collected_batches']}")
        print(f"  总episodes: {iteration_metrics['total_episodes']}")
        print(f"  平均奖励: {iteration_metrics['mean_reward']:.4f}")
        print(f"  成功率: {iteration_metrics['success_rate']:.2f}")
        print(f"  平均损失: {iteration_metrics['mean_loss']:.6f}")
        print(f"  训练步数: {iteration_metrics['total_steps']}")
    
    # 训练总结
    print(f"\n" + "="*60)
    print("🎉 完整训练循环结束！")
    print("="*60)
    
    print("\n✅ 成功实现的RIPT核心功能:")
    print("1. ✓ 智能初始状态采样")
    print("2. ✓ 全0/全1过滤机制")
    print("3. ✓ 多轮轨迹收集") 
    print("4. ✓ 梯度累积训练")
    print("5. ✓ 多训练迭代循环")
    print("6. ✓ 统计跟踪和管理")
    
    if all_training_metrics:
        print(f"\n📊 训练轨迹总结:")
        print("Iter | Batches | Episodes | Reward | Success | Loss    | Steps")
        print("-----|---------|----------|--------|---------|---------|------")
        for m in all_training_metrics:
            print(f"{m['iteration']:4d} | {m['collected_batches']:7d} | "
                  f"{m['total_episodes']:8d} | {m['mean_reward']:6.3f} | "
                  f"{m['success_rate']:7.2f} | {m['mean_loss']:7.4f} | {m['total_steps']:5d}")
        
        # 训练趋势分析
        first_loss = all_training_metrics[0]['mean_loss']
        last_loss = all_training_metrics[-1]['mean_loss']
        if first_loss > 0:
            loss_change = ((last_loss - first_loss) / first_loss) * 100
            print(f"\n📈 训练趋势:")
            print(f"  初始损失: {first_loss:.6f}")
            print(f"  最终损失: {last_loss:.6f}")
            print(f"  损失变化: {loss_change:+.2f}%")
    
    # 保存训练结果
    results_file = f"advanced_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': config.__dict__,
            'training_metrics': all_training_metrics,
            'sampler_stats': smart_sampler.state_stats,
            'final_summary': {
                'total_iterations': len(all_training_metrics),
                'total_episodes': sum(m['total_episodes'] for m in all_training_metrics),
                'total_training_steps': sum(m['total_steps'] for m in all_training_metrics),
                'final_success_rate': all_training_metrics[-1]['success_rate'] if all_training_metrics else 0.0
            }
        }, f, indent=2)
    
    print(f"\n💾 完整训练结果已保存到: {results_file}")
    print("\n🎯 第2阶段完成！现在具备了完整的RIPT智能采样和批处理能力")

if __name__ == "__main__":
    main()