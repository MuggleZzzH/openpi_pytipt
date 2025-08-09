#!/usr/bin/env python3
"""
基于2_test_pi0_on_libero.py逻辑，生成LIBERO任务1-9的演示视频
"""
import json
import os
from pathlib import Path

import gym
import imageio
import numpy as np
import robosuite.utils.transform_utils as T
import torch
from cleandiffuser.env import libero  # noqa: F401
from termcolor import cprint

# 修复tokenizers并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pi0 import PI0Policy

# 路径配置
PATH_TO_PI_MODEL = "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch"
PATH_TO_JAX_PI_MODEL = "/zhaohan/ZJH/openpi_pytorch/lerobot_dataset"

# 创建视频保存目录
video_dir = Path("libero_task_videos")
video_dir.mkdir(exist_ok=True)

# 加载模型
cprint("Loading PI0 model...", "green")
policy = PI0Policy.from_pretrained(PATH_TO_PI_MODEL)

# 加载归一化统计
device = policy.config.device
norm_stats_path = Path(PATH_TO_JAX_PI_MODEL) / "norm_stats.json"
with open(norm_stats_path) as f:
    norm_stats = json.load(f)

state_mean = np.array(norm_stats["norm_stats"]["state"]["mean"][:8], dtype=np.float32)
state_std = np.array(norm_stats["norm_stats"]["state"]["std"][:8], dtype=np.float32)
action_mean = np.array(norm_stats["norm_stats"]["actions"]["mean"][:7], dtype=np.float32)
action_std = np.array(norm_stats["norm_stats"]["actions"]["std"][:7], dtype=np.float32)

# Dummy动作用于环境预热
dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])

# 循环生成任务1-9的视频
for task_id in range(1, 10):  # 1到9
    cprint(f"\n=== 生成任务 {task_id} 视频 ===", "cyan")
    
    # 创建环境
    env = gym.make(
        "libero-goal-v0",
        task_id=task_id,
        image_size=224,
        camera_names=["agentview", "robot0_eye_in_hand"],
        seed=0,
    )

    # 重置环境
    o = env.reset()
    
    # 重要：执行一些dummy步骤，因为模拟器在开始时会掉落物体
    for _ in range(20):
        o, r, d, i = env.step(dummy_action)

    frames = []
    cprint(f"开始录制任务 {task_id}: {env.task_description}", "green")
    
    # 🔧 使用与2_test_pi0_on_libero.py完全相同的逻辑
    while not d:
        # 构造状态
        unnorm_state = np.concatenate([
            o["robot0_eef_pos"],
            T.quat2axisangle(o["robot0_eef_quat"]),
            o["robot0_gripper_qpos"],
        ], dtype=np.float32)
        
        # 归一化状态
        state = (unnorm_state - state_mean) / (state_std + 1e-6)

        # 处理图像（BGR -> RGB）
        base_0_rgb = o["agentview_image"][:, :, ::-1].copy()
        left_wrist_0_rgb = o["robot0_eye_in_hand_image"][:, :, ::-1].copy()

        # 构造观测
        observation = {
            "image": {
                "base_0_rgb": torch.from_numpy(base_0_rgb).to(device)[None],
                "left_wrist_0_rgb": torch.from_numpy(left_wrist_0_rgb).to(device)[None],
            },
            "state": torch.from_numpy(state).to(device)[None],
            "prompt": [env.task_description],
        }
        
        # 策略推理（获得50个动作）
        action = policy.select_action(observation)[0, :, :7]
        action = action.cpu().numpy()
        
        # 反归一化动作
        action = action * (action_std + 1e-6) + action_mean
        
        # 应用状态偏移（前6维）
        action[:, :6] += unnorm_state[None, :6]

        # ✅ 关键：执行50个动作步骤
        for i in range(50):
            o, r, d, _ = env.step(action[i, :7])
            # 保存帧（RGB格式）
            frames.append(o["agentview_image"][:, :, ::-1].transpose(1, 2, 0).copy())
            if d:
                break

    env.close()

    # 保存视频
    video_path = video_dir / f"task_{task_id:02d}.mp4"
    writer = imageio.get_writer(str(video_path), fps=30)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
    cprint(f"✅ 任务 {task_id} 视频已保存: {video_path}", "yellow")
    cprint(f"   任务描述: {env.task_description}", "blue")
    cprint(f"   帧数: {len(frames)}", "blue")

cprint(f"\n🎉 所有9个任务视频生成完成！保存在: {video_dir}", "green")