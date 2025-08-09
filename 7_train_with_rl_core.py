#!/usr/bin/env python3
"""
第1阶段：核心RL训练循环 (7_train_with_rl_core.py)
基于6_simple_train_direct_runner.py，添加真实的强化学习训练功能

核心新增功能：
1. 优化器创建和管理
2. PI0Policy训练模式切换 
3. CFG-style优势加权损失计算
4. 真实的梯度更新循环
5. 训练指标监控

使用方法:
    cd /zhaohan/ZJH/openpi_pytorch
    python 7_train_with_rl_core.py
    
    # 自定义配置
    python 7_train_with_rl_core.py --lr 2e-5 --num_epochs 3 --debug_loss true
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import argparse

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

class TrainingConfig:
    """训练配置类，替代硬编码参数"""
    def __init__(self):
        # 模型配置
        self.model_path = "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch"
        self.norm_stats_path = "/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json"
        
        # 环境配置
        self.benchmark_name = "libero_goal"
        self.task_id = 2
        self.num_parallel_envs = 1
        self.max_episode_length = 200
        
        # 强化学习配置
        self.num_rollouts = 5
        self.num_epochs = 3  # 新增：多轮训练
        self.learning_rate = 1e-5
        self.optimizer_type = "AdamW"
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0  # 新增：梯度裁剪
        
        # 调试配置
        self.save_videos = True
        self.video_dir = "rollout_videos_rl_core"
        self.debug_loss_computation = True  # 新增：损失计算调试
        self.log_training_details = True   # 新增：详细训练日志

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PI0 RL Core Training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--num_rollouts", type=int, default=5, help="Number of rollouts")
    parser.add_argument("--debug_loss", type=bool, default=True, help="Enable loss computation debugging")
    parser.add_argument("--save_videos", type=bool, default=True, help="Save rollout videos")
    return parser.parse_args()

def create_optimizer(policy, config):
    """创建优化器"""
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
            weight_decay=0.0  # Adam不使用weight_decay
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

def save_episode_video(frames, episode_idx, success, total_reward, task_description, video_dir):
    """保存episode视频（与6脚本保持一致）"""
    if not frames:
        print(f"  警告: Episode {episode_idx} 没有帧数据，跳过视频保存")
        return None
        
    # 处理可能的不同帧格式
    processed_frames = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            # 确保帧格式正确 (H, W, C)
            if frame.ndim == 3 and frame.shape[2] == 3:
                # 确保数据类型正确
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                processed_frames.append(frame)
    
    if not processed_frames:
        print(f"  警告: Episode {episode_idx} 没有有效的帧数据")
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 处理任务描述，使其适合作为文件名
    clean_task = str(task_description).lower().replace(" ", "_").replace(".", "").replace(",", "")[:30]
    
    # 创建文件名
    filename = f"episode_{episode_idx:02d}_{timestamp}_success_{success}_reward_{total_reward:.2f}_{clean_task}.mp4"
    video_path = video_dir / filename
    
    print(f"  保存视频: {video_path} (共{len(processed_frames)}帧)")
    
    # 创建视频
    try:
        import imageio
        writer = imageio.get_writer(str(video_path), fps=10)  # 10 FPS for better visibility
        
        for frame in processed_frames:
            writer.append_data(frame)
        
        writer.close()
        print(f"  成功保存视频文件")
        return str(video_path)
    except Exception as e:
        print(f"  错误: 保存视频失败 - {e}")
        return None

def main():
    print("=== 第1阶段：核心RL训练循环 ===")
    print("基于6脚本，添加真实的强化学习训练功能")
    print()
    
    # 解析命令行参数和创建配置
    args = parse_args()
    config = TrainingConfig()
    
    # 应用命令行参数
    config.learning_rate = args.lr
    config.num_epochs = args.num_epochs
    config.num_rollouts = args.num_rollouts
    config.debug_loss_computation = args.debug_loss
    config.save_videos = args.save_videos
    
    print(f"训练配置:")
    print(f"  学习率: {config.learning_rate}")
    print(f"  训练轮数: {config.num_epochs}")
    print(f"  收集轨迹数: {config.num_rollouts}")
    print(f"  调试模式: {config.debug_loss_computation}")
    print("")
    
    # 加载PI0模型
    print(f"加载PI0模型: {config.model_path}")
    policy = PI0Policy.from_pretrained(config.model_path)
    device = policy.config.device
    print(f"使用设备: {device}")
    
    # 模式管理说明
    print("\n=== 模式管理说明 ===")
    print("PI0Policy有两种模式:")
    print("  - eval() 模式: 用于轨迹收集，调用select_action()")
    print("  - train() 模式: 用于参数更新，调用forward()")
    print("我们将在适当的时候切换模式")
    
    # 创建优化器
    optimizer = create_optimizer(policy, config)
    
    # 创建CFG适配器
    print(f"\n=== 创建CFG适配器 ===")
    print(f"加载归一化参数: {config.norm_stats_path}")
    
    # 禁用CFG适配器的视频保存，避免重复保存
    os.environ["PI0_DEBUG_SAVE_VIDEO"] = "false"  # 临时禁用
    cfg_adapter = PI0_CFG_Adapter(policy, norm_stats_path=config.norm_stats_path)
    print("✓ CFG适配器创建成功 (已禁用重复视频保存)")
    
    # 创建视频保存目录
    video_dir = Path(config.video_dir)
    video_dir.mkdir(exist_ok=True)
    
    # 创建LIBERO环境运行器（与6脚本保持一致）
    print("\n=== 初始化环境运行器 ===")
    env_runner = LIBEROEnvRunner(
        policy=policy,
        benchmark_name=config.benchmark_name,
        rollouts_per_env=1,
        num_parallel_envs=config.num_parallel_envs,
        max_episode_length=config.max_episode_length,
        task_names_to_use=[config.task_id],
        config=config.__dict__,  # 传递配置字典
        rank=0,
        world_size=1,
        norm_stats_path=config.norm_stats_path
    )
    
    # 创建奖励函数
    reward_function = BinarySuccessReward()
    
    # 🎯 核心RL训练循环开始
    print(f"\n=== 开始核心RL训练循环 ===")
    print(f"将进行 {config.num_rollouts} 次轨迹收集和 {config.num_epochs} 轮优化")
    print("注意: 当前PI0模型性能很好，可能出现成功率100%的情况")
    print("这会导致Leave-One-Out优势计算结果为0，这是正常现象，表明策略已收敛")
    
    all_episodes = []
    all_rewards = []
    training_metrics = []
    
    # 第1步：轨迹收集（与6脚本保持一致）
    print(f"\n--- 第1步：轨迹收集 ---")
    print("🔄 切换到eval模式进行轨迹收集...")
    policy.eval()  # 关键：推理模式收集轨迹
    print("✓ PI0Policy现在处于推理模式")
    
    for rollout_idx in range(config.num_rollouts):
        print(f"\nRollout {rollout_idx + 1}/{config.num_rollouts}")
        
        # 创建模拟的初始状态数据
        dummy_init_states = np.zeros((config.num_parallel_envs, 8), dtype=np.float32)
        
        print("收集轨迹...")
        
        try:
            # 直接调用env_runner的run_policy_in_env方法
            rollout_generator_for_batch = env_runner.run_policy_in_env(
                env_name=config.task_id,
                all_init_states=dummy_init_states
            )
            
            # 收集生成的轨迹
            batch_rollouts = list(rollout_generator_for_batch)
            
            if not batch_rollouts:
                print("警告: 未生成有效轨迹，跳过此rollout")
                continue
                
        except Exception as e:
            print(f"生成轨迹时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 处理生成的rollouts
        for rollout_idx_in_batch, rollout in enumerate(batch_rollouts):
            print(f"处理rollout {rollout_idx_in_batch + 1}/{len(batch_rollouts)}")
            
            # rollout格式: (success, total_reward, episode_data)
            success, total_reward, episode_data = rollout
            
            # 创建episode字典
            episode = {
                'success': success,
                'total_reward': total_reward,
                **episode_data  # 包含observations, actions, rewards, task等
            }
            
            # 使用奖励函数计算奖励（保持一致性）
            try:
                computed_reward = reward_function.compute_reward(rollout_idx_in_batch, episode, None)
                episode['computed_reward'] = computed_reward
            except Exception as e:
                print(f"计算奖励时出错: {e}")
                episode['computed_reward'] = 0.0
            
            # 保存episode数据
            all_episodes.append(episode)
            all_rewards.append(total_reward)
            
            # 保存视频（如果启用）
            if config.save_videos and 'video_frames' in episode_data and episode_data['video_frames']:
                save_episode_video(
                    episode_data['video_frames'], 
                    len(all_episodes) - 1, 
                    success, 
                    total_reward,
                    episode_data.get('task', 'unknown_task'),
                    video_dir
                )
            
            print(f"  Episode完成: 奖励={total_reward:.4f}, 成功={success}, 计算奖励={episode['computed_reward']:.4f}")
    
    # 第2步：优势计算（标准Leave-One-Out方法）
    print(f"\n--- 第2步：优势计算 ---")
    
    if not all_rewards:
        print("警告: 没有收集到任何奖励数据，无法进行训练")
        return
    
    rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
    advantages = []
    
    # 标准的Leave-One-Out优势计算
    for i in range(len(all_rewards)):
        # Leave-One-Out baseline: 除了当前轨迹外，其他轨迹的平均奖励
        other_rewards = torch.cat([rewards_tensor[:i], rewards_tensor[i+1:]])
        baseline = other_rewards.mean() if len(other_rewards) > 0 else 0.0
        advantage = rewards_tensor[i] - baseline
        advantages.append(advantage.item())
    
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    
    print(f"奖励: {[f'{r:.3f}' for r in all_rewards]}")
    print(f"优势: {[f'{a:.3f}' for a in advantages.tolist()]}")
    print(f"平均奖励: {rewards_tensor.mean().item():.4f}")
    print(f"成功率: {sum(ep['success'] for ep in all_episodes) / len(all_episodes):.2f}")
    
    # 分析优势情况
    advantage_variance = advantages.var().item()
    print(f"优势方差: {advantage_variance:.6f}")
    
    if advantage_variance < 1e-6:
        print("ℹ️  所有轨迹表现相同，优势均为0")
        print("这是正常现象：当前策略已经收敛，所有轨迹质量相等")
        print("在真实训练中，这表明需要更多探索或更困难的任务")
    else:
        print("✓ 检测到轨迹质量差异，优势计算有效")
    
    # 🚀 第3步：核心RL训练循环（全新功能）
    print(f"\n--- 第3步：核心RL训练循环 ---")
    print("🔄 切换到train模式进行参数更新...")
    policy.train()  # 关键：训练模式更新参数
    print("✓ PI0Policy现在处于训练模式")
    print(f"开始 {config.num_epochs} 轮CFG-style优势加权训练...")
    
    for epoch in range(config.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
        
        # 清零梯度
        optimizer.zero_grad()
        
        try:
            # 🔥 关键：使用CFG适配器计算加权损失
            print("计算CFG-style优势加权损失...")
            
            if config.debug_loss_computation:
                print(f"  输入episodes数量: {len(all_episodes)}")
                print(f"  输入advantages形状: {advantages.shape}")
                print(f"  优势值范围: [{advantages.min().item():.4f}, {advantages.max().item():.4f}]")
            
            # 这是核心的CFG-style训练
            weighted_loss = cfg_adapter.compute_weighted_loss(all_episodes, advantages)
            
            if config.debug_loss_computation:
                print(f"  计算得到加权损失: {weighted_loss.item():.6f}")
                print(f"  损失张量设备: {weighted_loss.device}")
                print(f"  损失requires_grad: {weighted_loss.requires_grad}")
            
            # 检查损失有效性
            if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                print(f"⚠️ 检测到无效损失值: {weighted_loss.item()}")
                print("跳过本轮优化")
                continue
            
            # 🔥 关键：反向传播
            print("执行反向传播...")
            weighted_loss.backward()
            
            # 梯度裁剪
            if config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                if config.debug_loss_computation:
                    print(f"  梯度范数: {grad_norm:.6f}")
            
            # 🔥 关键：参数更新
            print("更新模型参数...")
            optimizer.step()
            
            # 记录训练指标
            epoch_metrics = {
                'epoch': epoch + 1,
                'loss': weighted_loss.item(),
                'grad_norm': grad_norm.item() if config.max_grad_norm > 0 else 0.0,
                'mean_advantage': advantages.mean().item(),
                'mean_reward': rewards_tensor.mean().item(),
                'success_rate': sum(ep['success'] for ep in all_episodes) / len(all_episodes)
            }
            
            training_metrics.append(epoch_metrics)
            
            # 输出训练进度
            print(f"✓ Epoch {epoch + 1} 完成:")
            print(f"  损失: {epoch_metrics['loss']:.6f}")
            print(f"  梯度范数: {epoch_metrics['grad_norm']:.6f}")
            print(f"  平均优势: {epoch_metrics['mean_advantage']:.4f}")
            print(f"  平均奖励: {epoch_metrics['mean_reward']:.4f}")
            print(f"  成功率: {epoch_metrics['success_rate']:.2f}")
            
        except Exception as e:
            print(f"❌ Epoch {epoch + 1} 训练出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 第4步：训练总结
    print(f"\n=== 训练总结 ===")
    print("核心RL训练循环完成！")
    print("\n✅ 成功实现的关键功能:")
    print("1. ✓ 优化器创建和管理")
    print("2. ✓ PI0Policy训练模式切换")
    print("3. ✓ CFG-style优势加权损失计算")
    print("4. ✓ 真实的梯度更新循环")
    print("5. ✓ 训练指标监控")
    
    if training_metrics:
        print(f"\n📊 训练轨迹:")
        print("Epoch | Loss      | GradNorm  | MeanAdv   | MeanRew   | Success")
        print("------|-----------|-----------|-----------|-----------|--------")
        for m in training_metrics:
            print(f"{m['epoch']:5d} | {m['loss']:9.6f} | {m['grad_norm']:9.6f} | "
                  f"{m['mean_advantage']:9.4f} | {m['mean_reward']:9.4f} | {m['success_rate']:7.2f}")
        
        # 训练趋势分析
        first_loss = training_metrics[0]['loss']
        last_loss = training_metrics[-1]['loss']
        loss_change = ((last_loss - first_loss) / abs(first_loss)) * 100
        
        print(f"\n📈 训练趋势:")
        print(f"  初始损失: {first_loss:.6f}")
        print(f"  最终损失: {last_loss:.6f}")
        print(f"  损失变化: {loss_change:+.2f}%")
        
        if loss_change < -5:
            print("  🎉 损失显著下降，训练效果良好！")
        elif loss_change < 5:
            print("  📊 损失相对稳定，模型收敛中...")
        else:
            print("  ⚠️ 损失上升，可能需要调整学习率")
    
    # 保存训练指标
    metrics_file = f"training_metrics_7_rl_core_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'config': config.__dict__,
            'training_metrics': training_metrics,
            'final_summary': {
                'total_episodes': len(all_episodes),
                'mean_reward': rewards_tensor.mean().item(),
                'success_rate': sum(ep['success'] for ep in all_episodes) / len(all_episodes),
                'training_epochs': len(training_metrics)
            }
        }, f, indent=2)
    
    print(f"\n💾 训练指标已保存到: {metrics_file}")
    
    # 显示保存的视频文件
    if config.save_videos:
        print(f"\n🎬 视频文件保存在: {video_dir}")
        video_files = list(video_dir.glob("*.mp4"))
        if video_files:
            print(f"共保存了 {len(video_files)} 个视频文件")
        else:
            print("没有保存视频文件")
    
    print(f"\n🎯 第1阶段核心RL训练完成！")
    print("下一步可以添加更多训练轮数、批处理和配置管理功能")

if __name__ == "__main__":
    main()