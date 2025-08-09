#!/usr/bin/env python3
"""
简化版RIPT训练脚本 - 不使用wrapper，直接调用policy
最正确的结构，不使用try-catch容错
包含视频输出功能，保存每个rollout的MP4视频

使用方法:
    cd /zhaohan/ZJH/openpi_pytorch
    python 4_simple_train_ript.py
    
输出:
    - rollout_videos/ 目录中保存所有episode的视频文件
    - 控制台输出训练过程和统计信息
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import gym
from cleandiffuser.env import libero
import robosuite.utils.transform_utils as T
import imageio

# 修复tokenizers并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pi0 import PI0Policy

def save_episode_video(frames, episode_idx, success, total_reward, task_description, video_dir):
    """保存单个episode的视频"""
    if not frames:
        print(f"  警告: Episode {episode_idx} 没有帧数据，跳过视频保存")
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 处理任务描述，使其适合作为文件名
    clean_task = task_description.lower().replace(" ", "_").replace(".", "").replace(",", "")[:30]
    
    # 创建文件名
    filename = f"episode_{episode_idx:02d}_{timestamp}_success_{success}_reward_{total_reward:.2f}_{clean_task}.mp4"
    video_path = video_dir / filename
    
    print(f"  保存视频: {video_path} (共{len(frames)}帧)")
    
    # 检查第一帧的格式
    first_frame = frames[0]
    print(f"  帧格式: shape={first_frame.shape}, dtype={first_frame.dtype}")
    
    # 创建视频
    writer = imageio.get_writer(str(video_path), fps=10)  # 10 FPS for better visibility
    
    frame_count = 0
    for frame in frames:
        # 确保帧格式正确
        if frame.ndim != 3 or frame.shape[2] != 3:
            print(f"  警告: 跳过无效帧，形状={frame.shape}")
            continue
            
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        
        writer.append_data(frame)
        frame_count += 1
    
    writer.close()
    print(f"  成功保存 {frame_count} 帧到视频文件")
    return str(video_path)

def main():
    print("=== 简化版RIPT训练脚本 ===")
    print("直接调用policy，不使用wrapper")
    print()
    
    # 加载模型
    PATH_TO_PI_MODEL = "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch"
    print(f"加载PI0模型: {PATH_TO_PI_MODEL}")
    policy = PI0Policy.from_pretrained(PATH_TO_PI_MODEL)
    
    # 使用与参考实现一致的设备获取方式
    device = policy.config.device
    print(f"使用设备: {device}")
    
    # 强制使用统一的归一化参数路径
    norm_stats_path = "/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json"
    print(f"加载归一化参数: {norm_stats_path}")
    
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)
    
    state_mean = np.array(norm_stats["norm_stats"]["state"]["mean"][:8], dtype=np.float32)
    state_std = np.array(norm_stats["norm_stats"]["state"]["std"][:8], dtype=np.float32)
    action_mean = np.array(norm_stats["norm_stats"]["actions"]["mean"][:7], dtype=np.float32)
    action_std = np.array(norm_stats["norm_stats"]["actions"]["std"][:7], dtype=np.float32)
    
    print(f"State mean shape: {state_mean.shape}")
    print(f"Action mean shape: {action_mean.shape}")
    
    # 创建环境
    print("创建LIBERO环境...")
    env = gym.make(
        "libero-goal-v0",
        task_id=2,
        image_size=224,
        camera_names=["agentview", "robot0_eye_in_hand"],
        seed=0,
    )
    
    # 重置环境
    obs = env.reset()
    
    # 执行热机步骤
    dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])
    for _ in range(20):
        obs, _, _, _ = env.step(dummy_action)
    
    print(f"任务描述: {env.task_description}")
    print(f"初始End-effector位置: {obs['robot0_eef_pos']}")
    
    # 开始简单的训练循环
    print("\n=== 开始简化训练循环 ===")
    
    num_rollouts = 5  # 简化：只做5个rollout
    max_steps_per_rollout = 200  # 每个rollout最多50步
    
    all_episodes = []
    all_rewards = []
    
    # 创建视频保存目录
    video_dir = Path("rollout_videos")
    video_dir.mkdir(exist_ok=True)
    
    for rollout_idx in range(num_rollouts):
        print(f"\n--- Rollout {rollout_idx + 1}/{num_rollouts} ---")
        
        # 重置环境
        obs = env.reset()
        for _ in range(20):  # 热机
            obs, _, _, _ = env.step(dummy_action)
        
        episode_observations = []
        episode_actions = []
        episode_rewards = []
        episode_frames = []  # 用于保存视频帧
        total_reward = 0.0
        done = False
        step = 0
        
        # 与参考实现一致：循环直到done或达到步数限制
        while not done and step < max_steps_per_rollout:
            # 准备观测数据
            unnorm_state = np.concatenate([
                obs["robot0_eef_pos"],
                T.quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ], dtype=np.float32)
            
            # 添加调试输出（与参考实现一致）
            if step == 0 and rollout_idx == 0:
                print(">> gripper_qpos shape:", np.asarray(obs["robot0_gripper_qpos"]).shape)
            
            state = (unnorm_state - state_mean) / (state_std + 1e-6)
            
            # 注意：这里的图像处理要与2_test_pi0_on_libero.py完全一致
            # policy输入的图像不使用transpose
            base_0_rgb = obs["agentview_image"][:, :, ::-1].copy()
            left_wrist_0_rgb = obs["robot0_eye_in_hand_image"][:, :, ::-1].copy()
            
            observation = {
                "image": {
                    "base_0_rgb": torch.from_numpy(base_0_rgb).to(device)[None],
                    "left_wrist_0_rgb": torch.from_numpy(left_wrist_0_rgb).to(device)[None],
                },
                "state": torch.from_numpy(state).to(device)[None],
                "prompt": [env.task_description],
            }
            
            # 直接调用policy进行推理（与参考实现完全一致）
            action = policy.select_action(observation)[0, :, :7]
            action = action.cpu().numpy()
            action = action * (action_std + 1e-6) + action_mean
            action[:, :6] += unnorm_state[None, :6]
            
            # 为了保持与训练逻辑的兼容性，重命名为action_denorm
            action_denorm = action
            
            # 记录观测和动作
            episode_observations.append(obs.copy())
            episode_actions.append(action_denorm.copy())
            
            # 记录动作执行前的初始帧
            frame = obs["agentview_image"][:, :, ::-1].transpose(1, 2, 0).copy()
            if frame.ndim == 3 and frame.shape[2] == 3:
                episode_frames.append(frame)
            
            # 执行动作序列 - 参考2_test_pi0_on_libero.py的方式
            # 原实现执行整个50步的动作序列
            for i in range(min(action_denorm.shape[0], max_steps_per_rollout - step)):
                current_action = action_denorm[i, :7]
                obs, reward, done, info = env.step(current_action)
                
                # 记录每步的视频帧
                frame = obs["agentview_image"][:, :, ::-1].transpose(1, 2, 0).copy()
                if frame.ndim == 3 and frame.shape[2] == 3:
                    episode_frames.append(frame)
                
                episode_rewards.append(reward)
                total_reward += reward
                step += 1
                
                if step % 10 == 0:
                    print(f"  Step {step}: reward={reward:.4f}, total_reward={total_reward:.4f}")
                
                if done or step >= max_steps_per_rollout:
                    break
        
        # 添加最后一帧（如果环境没有结束）
        if not done:
            final_frame = obs["agentview_image"][:, :, ::-1].transpose(1, 2, 0).copy()
            if final_frame.ndim == 3 and final_frame.shape[2] == 3:
                episode_frames.append(final_frame)
        
        # 计算episode成功率（简化：基于总奖励）
        success = total_reward > 0.5  # 简化的成功判断
        
        episode_data = {
            'observations': episode_observations,
            'actions': episode_actions,
            'rewards': episode_rewards,
            'task': [env.task_description],
            'success': success,
            'total_reward': total_reward,
            'length': len(episode_rewards)
        }
        
        all_episodes.append(episode_data)
        all_rewards.append(total_reward)
        
        # 保存该episode的视频
        if episode_frames:
            save_episode_video(episode_frames, rollout_idx, success, total_reward, env.task_description, video_dir)
        
        print(f"  Episode完成: 长度={len(episode_rewards)}, 总奖励={total_reward:.4f}, 成功={success}")
    
    # 计算优势值（简化：使用Leave-One-Out baseline）
    print("\n=== 计算优势值 ===")
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
    
    # 简化的"训练"步骤（实际上只是演示数据处理）
    print("\n=== 模拟训练步骤 ===")
    
    # 处理episodes数据用于训练
    for i, (episode, advantage) in enumerate(zip(all_episodes, advantages)):
        print(f"Episode {i}: advantage={advantage:.3f}, 将用于加权训练")
        
        # 这里可以加入实际的训练逻辑
        # 比如：调用 policy.forward() 计算损失，然后反向传播
        # 但为了简化，我们只是演示数据流
        
        if len(episode['actions']) > 0:
            first_action = episode['actions'][0]
            print(f"  第一个动作形状: {first_action.shape}")
    
    print("\n=== 训练完成 ===")
    print("这是一个简化的演示版本，展示了：")
    print("1. 直接调用policy.select_action()进行推理")
    print("2. 收集轨迹数据")
    print("3. 计算优势值")
    print("4. 准备训练数据")
    print("5. 保存rollout视频用于分析")
    print("实际的训练步骤（梯度更新）需要根据具体需求实现")
    
    # 显示保存的视频文件
    print(f"\n=== 视频文件保存在: {video_dir} ===")
    video_files = list(video_dir.glob("*.mp4"))
    if video_files:
        print(f"共保存了 {len(video_files)} 个视频文件:")
        for video_file in sorted(video_files):
            print(f"  - {video_file.name}")
    else:
        print("没有保存视频文件")
    
    # 清理
    env.close()

if __name__ == "__main__":
    main()