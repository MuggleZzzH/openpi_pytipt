#!/usr/bin/env python3
"""
直接使用run_policy_in_env的简化版RIPT训练脚本
跳过RolloutGenerator，直接调用LIBEROEnvRunner.run_policy_in_env函数
最简洁的结构，展示runner的核心功能

使用方法:
    cd /zhaohan/ZJH/openpi_pytorch
    python 6_simple_train_direct_runner.py
    
输出:
    - rollout_videos_direct/ 目录中保存所有episode的视频文件
    - 控制台输出训练过程和统计信息
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

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

def create_simple_config():
    """创建简化的配置，匹配4_simple_train_ript.py的设置"""
    return {
        'norm_stats_path': "/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json",
        'benchmark_name': "libero_goal",
        'task_id': 2,  # 与4脚本保持一致，使用task_id=2
        'rollouts_per_env': 1,
        'num_parallel_envs': 1,
        'max_episode_length': 200,  # 与4脚本的max_steps_per_rollout=200保持一致
        'num_rollouts': 5,  # 与4脚本保持一致
    }

def main():
    print("=== 直接使用run_policy_in_env的简化版RIPT训练脚本 ===")
    print("跳过RolloutGenerator，直接调用LIBEROEnvRunner.run_policy_in_env")
    print()
    
    # 创建配置
    config = create_simple_config()
    
    # 加载模型
    PATH_TO_PI_MODEL = "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch"
    print(f"加载PI0模型: {PATH_TO_PI_MODEL}")
    policy = PI0Policy.from_pretrained(PATH_TO_PI_MODEL)
    
    # 使用与参考实现一致的设备获取方式
    device = policy.config.device
    print(f"使用设备: {device}")
    
    print(f"加载归一化参数: {config['norm_stats_path']}")
    print(f"基准环境: {config['benchmark_name']}")
    print(f"任务ID: {config['task_id']} (与4_simple_train_ript.py保持一致)")
    
    # 创建视频保存目录
    video_dir = Path("rollout_videos_direct")
    video_dir.mkdir(exist_ok=True)
    
    # 创建LIBERO环境运行器
    print("\n=== 初始化环境运行器 ===")
    env_runner = LIBEROEnvRunner(
        policy=policy,
        benchmark_name=config['benchmark_name'],
        rollouts_per_env=config['rollouts_per_env'],
        num_parallel_envs=config['num_parallel_envs'],
        max_episode_length=config['max_episode_length'],
        task_names_to_use=[config['task_id']],  # 使用task_id替代具体任务名
        config=config,
        rank=0,
        world_size=1,
        norm_stats_path=config['norm_stats_path']
    )
    
    # 创建奖励函数
    reward_function = BinarySuccessReward()
    
    # 开始简单的训练循环
    print("\n=== 开始简化训练循环 ===")
    
    all_episodes = []
    all_rewards = []
    
    for rollout_idx in range(config['num_rollouts']):
        print(f"\n--- Rollout {rollout_idx + 1}/{config['num_rollouts']} ---")
        
        # 创建模拟的初始状态数据（与RolloutGenerator中的格式一致）
        # 这里只需要提供形状正确的数据，不需要真实的初始状态值
        dummy_init_states = np.zeros((config['num_parallel_envs'], 8), dtype=np.float32)
        
        print("直接调用run_policy_in_env生成轨迹...")
        
        try:
            # 直接调用env_runner的run_policy_in_env方法
            # 这是核心：跳过RolloutGenerator，直接使用runner
            rollout_generator_for_batch = env_runner.run_policy_in_env(
                env_name=config['task_id'],  # 使用task_id作为环境名
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
            
            # 创建episode字典（与5脚本格式保持一致）
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
            
            # 保存视频（如果有视频数据）
            if 'video_frames' in episode_data and episode_data['video_frames']:
                save_episode_video(
                    episode_data['video_frames'], 
                    len(all_episodes) - 1, 
                    success, 
                    total_reward,
                    episode_data.get('task', 'unknown_task'),
                    video_dir
                )
            
            print(f"  Episode完成: 奖励={total_reward:.4f}, 成功={success}, 计算奖励={episode['computed_reward']:.4f}")
    
    # 计算优势值（与4和5脚本保持一致）
    print("\n=== 计算优势值 ===")
    
    if not all_rewards:
        print("警告: 没有收集到任何奖励数据")
        return
    
    rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
    advantages = []
    
    for i in range(len(all_rewards)):
        # Leave-One-Out baseline
        other_rewards = torch.cat([rewards_tensor[:i], rewards_tensor[i+1:]])
        baseline = other_rewards.mean() if len(other_rewards) > 0 else 0.0
        advantage = rewards_tensor[i] - baseline
        advantages.append(advantage.item())
    
    advantages = torch.tensor(advantages, dtype=torch.float32)
    
    print(f"奖励: {[f'{r:.3f}' for r in all_rewards]}")
    print(f"优势: {[f'{a:.3f}' for a in advantages.tolist()]}")
    print(f"平均奖励: {rewards_tensor.mean().item():.4f}")
    print(f"成功率: {sum(ep['success'] for ep in all_episodes) / len(all_episodes):.2f}")
    
    # 简化的"训练"步骤（与4和5脚本保持一致）
    print("\n=== 模拟训练步骤 ===")
    
    # 处理episodes数据用于训练
    for i, (episode, advantage) in enumerate(zip(all_episodes, advantages)):
        print(f"Episode {i}: advantage={advantage:.3f}, 将用于加权训练")
        
        # 这里可以加入实际的训练逻辑
        # 比如：调用 policy.forward() 计算损失，然后反向传播
        # 但为了简化，我们只是演示数据流
        
        if 'actions' in episode and len(episode['actions']) > 0:
            first_action = episode['actions'][0]
            print(f"  第一个动作形状: {np.array(first_action).shape}")
    
    print("\n=== 训练完成 ===")
    print("这是一个直接使用run_policy_in_env的简化版本，展示了：")
    print("1. 跳过RolloutGenerator，直接使用LIBEROEnvRunner")
    print("2. 直接调用run_policy_in_env方法")
    print("3. 收集和处理轨迹数据")
    print("4. 计算优势值")
    print("5. 准备训练数据")
    print("6. 保存rollout视频用于分析")
    print("这展示了runner的核心功能，比5脚本更直接")
    
    # 显示保存的视频文件
    print(f"\n=== 视频文件保存在: {video_dir} ===")
    video_files = list(video_dir.glob("*.mp4"))
    if video_files:
        print(f"共保存了 {len(video_files)} 个视频文件:")
        for video_file in sorted(video_files):
            print(f"  - {video_file.name}")
    else:
        print("没有保存视频文件")

def save_episode_video(frames, episode_idx, success, total_reward, task_description, video_dir):
    """保存单个episode的视频（与5脚本保持一致）"""
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

if __name__ == "__main__":
    main()