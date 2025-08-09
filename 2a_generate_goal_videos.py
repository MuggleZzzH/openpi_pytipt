#!/usr/bin/env python3
"""
generate_goal_videos.py

一次性为 LIBERO GOAL 基准的 9 个任务各录制一段 PI0 演示视频。
- 复用 2_test_pi0_on_libero.py 的推理流程
- 每个任务保存为 pi0_goal_taskXX.mp4（XX 为两位索引）
- 视频保存目录： pi0/ript/debug_images/goal_videos
"""
import os
import json
from pathlib import Path
import gym
import imageio
import numpy as np
import torch
import robosuite.utils.transform_utils as T
from termcolor import cprint
from cleandiffuser.env import libero  # noqa: F401

# 关闭 tokenizer 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pi0 import PI0Policy

# ----------------- 路径配置 -----------------
PATH_TO_PI_MODEL = "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch"
PATH_TO_JAX_PI_MODEL = "/zhaohan/ZJH/openpi_pytorch/lerobot_dataset"
SAVE_DIR = Path("pi0/ript/debug_images/goal_videos")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- 加载模型 -----------------
cprint("Loading PI0 model...", "green")
policy = PI0Policy.from_pretrained(PATH_TO_PI_MODEL)

# ----------------- 加载归一化统计 -----------------
with open(Path(PATH_TO_JAX_PI_MODEL) / "norm_stats.json") as f:
    norm_stats = json.load(f)
state_mean = np.array(norm_stats["norm_stats"]["state"]["mean"][:8], dtype=np.float32)
state_std = np.array(norm_stats["norm_stats"]["state"]["std"][:8], dtype=np.float32)
action_mean = np.array(norm_stats["norm_stats"]["actions"]["mean"][:7], dtype=np.float32)
action_std = np.array(norm_stats["norm_stats"]["actions"]["std"][:7], dtype=np.float32)

device = policy.config.device

# ----------------- 任务列表 -----------------
try:
    from libero.benchmark import LIBERO_GOAL_TASKS as TASK_LIST
except ImportError:
    TASK_LIST = [
        "LIBERO_GOAL_move_the_blue_mug_to_the_target_pose",
        "LIBERO_GOAL_move_the_green_bowl_to_the_target_pose",
        "LIBERO_GOAL_move_the_red_plate_to_the_target_pose",
        "LIBERO_GOAL_move_the_yellow_bottle_to_the_target_pose",
        "LIBERO_GOAL_move_the_black_can_to_the_target_pose",
        "LIBERO_GOAL_move_the_black_bowl_to_the_target_pose",
        "LIBERO_GOAL_move_the_silver_pot_to_the_target_pose",
        "LIBERO_GOAL_move_the_pink_cup_to_the_target_pose",
        "LIBERO_GOAL_move_the_white_box_to_the_target_pose",
    ]

# ----------------- Dummy 动作用于热机 -----------------
dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])

# ----------------- 主循环 -----------------
for task_id, task_name in enumerate(TASK_LIST):
    cprint(f"\n=== Task {task_id}: {task_name} ===", "cyan")

    env = gym.make(
        "libero-goal-v0",
        task_id=task_id,
        image_size=224,
        camera_names=["agentview", "robot0_eye_in_hand"],
        seed=0,
    )

    # reset & warm-up (20 steps)
    obs, done = env.reset(), False
    for _ in range(20):
        obs, _, _, _ = env.step(dummy_action)

    frames = []
    while not done:
        # -------------- 构造 PI0 观测 --------------
        unnorm_state = np.concatenate([
            obs["robot0_eef_pos"],
            T.quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ], dtype=np.float32)
        state = (unnorm_state - state_mean) / (state_std + 1e-6)

        base_rgb = obs["agentview_image"][..., ::-1].copy()
        wrist_rgb = obs["robot0_eye_in_hand_image"][..., ::-1].copy()

        observation = {
            "image": {
                "base_0_rgb": torch.from_numpy(base_rgb).to(device)[None],
                "left_wrist_0_rgb": torch.from_numpy(wrist_rgb).to(device)[None],
            },
            "state": torch.from_numpy(state).to(device)[None],
            "prompt": [task_name],
        }

        action = policy.select_action(observation)[0, :, :7]
        action = action.cpu().numpy() * (action_std + 1e-6) + action_mean
        action[:, :6] += unnorm_state[None, :6]

        for a in action:
            obs, _, done, _ = env.step(a[:7])
            frames.append(obs["agentview_image"].transpose(1, 2, 0)[..., ::-1])
            if done:
                break

    env.close()

    # 保存视频
    vid_path = SAVE_DIR / f"pi0_goal_task{task_id:02d}.mp4"
    writer = imageio.get_writer(str(vid_path), fps=30)
    for f in frames:
        writer.append_data(f)
    writer.close()
    cprint(f"✓ Saved video → {vid_path}", "yellow")

cprint("\nAll 9 videos generated!", "green") 