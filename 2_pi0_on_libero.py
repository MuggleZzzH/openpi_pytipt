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

PATH_TO_PI_MODEL = (
    "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch"
)
#PATH_TO_JAX_PI_MODEL = "/zhaohan/ZJH/openpi/assets/pi0_libero/zhaohan/ZJH/openpi/physical-intelligence/libero"
PATH_TO_JAX_PI_MODEL = "/zhaohan/ZJH/openpi_pytorch/lerobot_dataset"

# load model
cprint("Loading PI0 model...", "green")
policy = PI0Policy.from_pretrained(PATH_TO_PI_MODEL)

# load normalization stats
device = policy.config.device
norm_stats_path = (
    Path(PATH_TO_JAX_PI_MODEL) / "norm_stats.json"
)
with open(norm_stats_path) as f:
    norm_stats = json.load(f)
state_mean = np.array(norm_stats["norm_stats"]["state"]["mean"][:8], dtype=np.float32)
state_std = np.array(norm_stats["norm_stats"]["state"]["std"][:8], dtype=np.float32)
action_mean = np.array(
    norm_stats["norm_stats"]["actions"]["mean"][:7], dtype=np.float32
)
action_std = np.array(norm_stats["norm_stats"]["actions"]["std"][:7], dtype=np.float32)

# create environment
# ** Change `env_name` and `task_id` to test different environments and tasks **
cprint("Creating Libero environment...", "green")
env = gym.make(
    "libero-goal-v0",  # from ["libero-goal-v0", "libero-object-v0", "libero-spatial-v0", "libero-10-v0", "libero-90-v0"],
    task_id=1,  # task id from 0 to 9
    image_size=224,  # image size (height, width)
    camera_names=["agentview", "robot0_eye_in_hand"],  # camera names
    seed=0,  # random seed
)

# reset environment
o = env.reset()
# important: do some `dummy` steps because the simulator drops object at the beginning
dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])
for _ in range(20):
    o, r, d, i = env.step(dummy_action)

wrote_stats = False  # 只在第一次循环时写出调试信息

frames = []
cprint("Starting demo...", "green")
while not d:
    unnorm_state = np.concatenate(
        [
            o["robot0_eef_pos"],
            T.quat2axisangle(o["robot0_eef_quat"]),
            o["robot0_gripper_qpos"],
        ],
        dtype=np.float32,
    )
    print(">> gripper_qpos shape:", np.asarray(o["robot0_gripper_qpos"]).shape)
    state = (unnorm_state - state_mean) / (state_std + 1e-6)

    base_0_rgb = o["agentview_image"][:, :, ::-1].copy()
    left_wrist_0_rgb = o["robot0_eye_in_hand_image"][:, :, ::-1].copy()

    observation = {
        "image": {
            "base_0_rgb": torch.from_numpy(base_0_rgb).to(device)[None],
            "left_wrist_0_rgb": torch.from_numpy(left_wrist_0_rgb).to(device)[None],
        },
        "state": torch.from_numpy(state).to(device)[None],
        "prompt": [env.task_description],
    }
    action = policy.select_action(observation)[0, :, :7]
    action = action.cpu().numpy()
    action = action * (action_std + 1e-6) + action_mean

    action[:, :6] += unnorm_state[None, :6]

    # 在获得 action 后保存调试信息（仅执行一次）
    if not wrote_stats:
        # 记录 observation 中各 key 的 shape（若数据量小则记录具体值）
        obs_info = {}
        for key, val in o.items():
            arr = np.asarray(val)
            info = {"shape": list(arr.shape)}
            if arr.size <= 30:  # 小数组直接记录具体值，避免图像等大数组膨胀
                info["values"] = arr.tolist()
            obs_info[key] = info

        debug_info = {
            "state_mean": state_mean.tolist(),
            "state_mean_shape": list(state_mean.shape),
            "state_std": state_std.tolist(),
            "state_std_shape": list(state_std.shape),
            "action_mean": action_mean.tolist(),
            "action_mean_shape": list(action_mean.shape),
            "action_std": action_std.tolist(),
            "action_std_shape": list(action_std.shape),
            "first_unnorm_state": unnorm_state.tolist(),
            "first_unnorm_state_shape": list(unnorm_state.shape),
            "first_state": state.tolist(),
            "first_state_shape": list(state.shape),
            "observation_info": obs_info,
            "action": action.tolist(),
            "action_shape": list(action.shape),
        }
        with open("pi0_debug_stats.json", "w") as fp:
            json.dump(debug_info, fp, indent=2, ensure_ascii=False)
        wrote_stats = True

    for i in range(50):
        o, r, d, _ = env.step(action[i, :7])
        frames.append(o["agentview_image"][:, :, ::-1].transpose(1, 2, 0).copy())
        if d:
            break

# save video
writer = imageio.get_writer("pi0_libero_demo1.mp4", fps=30)
for frame in frames:
    writer.append_data(frame)
writer.close()
