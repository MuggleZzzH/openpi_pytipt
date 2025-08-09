import os
import sys
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from datetime import datetime
import imageio
from pathlib import Path
import json

# 导入LIBERO相关的变换函数 —— 若失败直接抛异常，与官方脚本保持一致
try:
    import robosuite.utils.transform_utils as T
except ImportError as e:
    raise ImportError("robosuite is required for LIBEROEnvRunner but not found") from e

# 调试设置
DEBUG_SAVE_IMAGES = False  # 禁用单独图像保存，只保留视频
DEBUG_SAVE_VIDEO = os.environ.get("PI0_DEBUG_SAVE_VIDEO", "true").lower() in ("true", "1", "yes")
DEBUG_IMAGE_DIR = os.environ.get("PI0_DEBUG_IMAGE_DIR", "ript/debug_images")

class LIBEROEnvRunner:
    """LIBERO环境运行器，提供PI0策略在LIBERO环境中的运行机制"""
    
    def __init__(self, policy=None, benchmark_name=None, rollouts_per_env=1, 
                 num_parallel_envs=1, max_episode_length=200, task_names_to_use=None,
                 config=None, rank=0, world_size=1, norm_stats_path=None):
        """初始化LIBERO环境运行器"""
        # 存储参数
        self.policy = policy
        self.benchmark_name = benchmark_name
        self.rollouts_per_env = rollouts_per_env
        self.num_parallel_envs = num_parallel_envs
        self.max_episode_length = max_episode_length
        self.task_names_to_use = task_names_to_use or []
        self.task_names = task_names_to_use or []  # 兼容性别名，供rollout_generator使用
        self.max_steps = max_episode_length if max_episode_length is not None else 500
        
        # ✅ 存储分布式训练参数
        self.config = config
        self.rank = rank
        self.world_size = world_size
        
        # 调试选项
        self.debug_save_images = DEBUG_SAVE_IMAGES
        self.debug_save_video = DEBUG_SAVE_VIDEO
        self.debug_image_dir = DEBUG_IMAGE_DIR
        
        # 加载归一化统计信息
        self._load_norm_stats(norm_stats_path)
        
    def _load_norm_stats(self, norm_stats_path: Optional[str] = None):
        """Load normalization statistics from norm_stats.json"""
        if norm_stats_path is None:
            # 尝试在常见位置找到norm_stats.json
            possible_paths = [
                "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch/norm_stats.json",
                "/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json", 
                "./checkpoints/pi0_libero_pytorch/norm_stats.json",
                "./norm_stats.json"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    norm_stats_path = path
                    break
        
        if norm_stats_path and Path(norm_stats_path).exists():
            print(f"LIBEROEnvRunner: Loading norm_stats from: {norm_stats_path}")
            with open(norm_stats_path) as f:
                norm_stats = json.load(f)
                
            # 提取状态和动作的归一化参数
            self.state_mean = np.array(norm_stats["norm_stats"]["state"]["mean"][:8], dtype=np.float32)
            self.state_std = np.array(norm_stats["norm_stats"]["state"]["std"][:8], dtype=np.float32)
            self.action_mean = np.array(norm_stats["norm_stats"]["actions"]["mean"][:7], dtype=np.float32)
            self.action_std = np.array(norm_stats["norm_stats"]["actions"]["std"][:7], dtype=np.float32)
            
            print(f"✅ LIBEROEnvRunner: Loaded normalization stats for action post-processing")
        else:
            raise FileNotFoundError("norm_stats.json not found in expected locations; cannot proceed")


    
    def construct_pi0_observation(self, obs, task_description):
        """构造PI0需要的观测格式，完全模仿2_test_pi0_on_libero.py的做法"""
        # 获取设备
        device = self.policy.config.device if hasattr(self.policy, 'config') else 'cuda:0'
        
        # 状态处理
        import robosuite.utils.transform_utils as T
        
        axis_angle = T.quat2axisangle(obs["robot0_eef_quat"])
            
        unnorm_state = np.concatenate([
            obs["robot0_eef_pos"],                    # 3D: end-effector position
            axis_angle, # 3D: rotation as axis-angle  
            obs["robot0_gripper_qpos"],               # 2D: gripper joint positions
        ], dtype=np.float32)
        print(">> gripper_qpos shape:", np.asarray(obs["robot0_gripper_qpos"]).shape)
        # 状态归一化
        state = (unnorm_state - self.state_mean) / (self.state_std + 1e-6)
        
        # 图像处理
        base_0_rgb = obs["agentview_image"][:, :, ::-1].copy()
        left_wrist_0_rgb = obs["robot0_eye_in_hand_image"][:, :, ::-1].copy()
        
        # 构造观测格式
        observation = {
            "image": {
                "base_0_rgb": torch.from_numpy(base_0_rgb).to(device)[None],
                "left_wrist_0_rgb": torch.from_numpy(left_wrist_0_rgb).to(device)[None],
            },
            "state": torch.from_numpy(state).to(device)[None],
            "prompt": [task_description],
        }
        
        return observation
    
    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action using loaded statistics"""
        return action * (self.action_std + 1e-6) + self.action_mean
    
    def get_unnormalized_state(self, obs) -> np.ndarray:
        """从观测中提取未归一化的状态（用于动作偏移）"""
        try:
            if isinstance(obs, list) and len(obs) > 0:
                obs = obs[0]  # 取第一个环境的观测
            
            if not isinstance(obs, dict) or not obs:
                raise RuntimeError("Required observation dict missing or empty")
            
            # 使用 end-effector 信息
            if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
                eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
                eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
                
                if eef_pos.size != 3 or eef_quat.size != 4:
                    raise RuntimeError("Observation fields robot0_eef_pos or robot0_eef_quat have incorrect dimensions")
                
                # 转换四元数为轴角
                try:
                    axis_angle = T.quat2axisangle(eef_quat).astype(np.float32)
                except Exception as e:
                    raise RuntimeError(f"Failed to convert quaternion to axis-angle: {e}")
                
                # 获取 gripper 状态
                if "robot0_gripper_qpos" not in obs:
                    raise RuntimeError("Observation missing required field robot0_gripper_qpos")
                try:
                    gripper_qpos = float(obs["robot0_gripper_qpos"][0])
                except (IndexError, TypeError, ValueError) as e:
                    raise RuntimeError(f"Invalid robot0_gripper_qpos value: {e}")
                
                # 构造未归一化的状态
                unnorm_state = np.concatenate([eef_pos[:3], axis_angle[:3], [gripper_qpos]]).astype(np.float32)
                return unnorm_state
                
            # 官方脚本要求上述关键字段均存在，若缺失则立即报错
            else:
                raise RuntimeError("Observation missing required end-effector fields")
                
        except Exception as e:
            raise
    
    def make_env(self, env_name: str):
        """创建LIBERO环境"""
        try:
            import gym
            from cleandiffuser.env import libero  # 确保环境注册
            
            # 使用与参考实现2_test_pi0_on_libero.py完全相同的环境配置
            benchmark_to_env_id = {
                'libero_spatial': 'libero-spatial-v0',
                'libero_object': 'libero-object-v0',
                'libero_goal': 'libero-goal-v0',
                'libero_10': 'libero-10-v0',
                'libero_90': 'libero-90-v0'
            }

            if self.benchmark_name not in benchmark_to_env_id:
                raise ValueError(f"Unknown benchmark_name: {self.benchmark_name}")

            env_id = benchmark_to_env_id[self.benchmark_name]
            # 🔧 修复：根据任务名称动态确定task_id，与2_test_pi0_on_libero.py保持一致
            # 对于libero_goal，使用task_id=1 (与2_test_pi0_on_libero.py保持一致)
            if self.benchmark_name == "libero_goal":
                task_id = 1  # 与2_test_pi0_on_libero.py保持一致
            elif self.benchmark_name == "libero_spatial":
                task_id = 0  # 第一个spatial任务
            else:
                task_id = 0  # 其他benchmark的默认任务
            
            # 创建环境
            env = gym.make(
                env_id,
                task_id=task_id,
                image_size=224,  # 匹配PI0的输入尺寸
                camera_names=["agentview", "robot0_eye_in_hand"],  # 匹配原始demo
                seed=0,  # 使用与参考实现相同的随机种子
            )
            
            # 获取任务描述
            if hasattr(env, 'task_description'):
                task_description = env.task_description
            else:
                task_description = env_name
                
            return env, task_description
            
        except Exception as e:
            raise RuntimeError(f"创建环境失败: {e}") from e
        
    def run_policy_in_env(self, env_name, all_init_states=None, debug_save_video=None):
        """在环境中运行策略，生成轨迹"""
        # 使用环境变量或参数控制调试
        save_video = debug_save_video if debug_save_video is not None else self.debug_save_video
        
        # 对每个初始状态运行一次episode
        for i, init_state in enumerate(all_init_states):
            rollout_images = []
            
            # 创建环境
            env, task_description = self.make_env(env_name)
            
            # 设置初始状态并获取初始观测
            obs = env.reset()
            
            # 热机步骤
            dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])
            for warmup_step in range(20):
                obs, _, _, _ = env.step(dummy_action)
            
            step = 0
            done = False
            total_reward = 0
            success = False
            
            # 收集初始观测图像用于视频
            if save_video:
                initial_img = obs["agentview_image"][:, :, ::-1].transpose(1, 2, 0).copy()
                rollout_images.append(initial_img)
            
            # 收集轨迹
            observations = [obs]
            actions = []
            rewards = []
            dones = []
            infos = []
            
            try:
                action_buffer = None
                action_index = 0
                
                # 执行策略直到完成或达到最大步数
                while not done and step < self.max_steps:
                    # 只在需要时推理（动作队列为空时）
                    if action_buffer is None or action_index >= action_buffer.shape[0]:
                        # 使用正确的PI0观测格式
                        pi0_observation = self.construct_pi0_observation(obs, task_description)
                        
                        # 选择动作
                        raw_action = self.policy.select_action(pi0_observation)
                        action = raw_action[0, :, :7]  # shape: (50, 7)
                        
                        # 转换为numpy并处理维度
                        if isinstance(action, torch.Tensor):
                            action_after_cpu = action.cpu().numpy()
                        else:
                            action_after_cpu = action
                        
                        # 反归一化动作
                        action_buffer = action_after_cpu * (self.action_std + 1e-6) + self.action_mean
                        
                        # 获取当前未归一化状态用于偏移
                        import robosuite.utils.transform_utils as T
                        unnorm_state = np.concatenate([
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ], dtype=np.float32)
                        
                        # 应用状态偏移（前6维）
                        action_buffer[:, :6] += unnorm_state[None, :6]
                        
                        # 重置动作索引
                        action_index = 0
                    
                    # 从动作队列中取出当前动作执行
                    current_action = action_buffer[action_index, :7]
                    
                    # 执行单步动作
                    next_obs, reward, done, info = env.step(current_action)
                    
                    # 记录轨迹数据
                    observations.append(next_obs)
                    actions.append(current_action)
                    rewards.append(reward)
                    dones.append(done)
                    infos.append(info)
                    
                    # 更新累计奖励
                    total_reward += reward
                    
                    # 收集图像用于视频
                    if save_video:
                        frame_img = next_obs["agentview_image"][:, :, ::-1].transpose(1, 2, 0).copy()
                        rollout_images.append(frame_img)
                    
                    # 更新状态和计数器
                    obs = next_obs
                    step += 1
                    action_index += 1
                    
                    # 检查成功状态 - 优先使用info中的success，否则基于奖励判断（与4脚本保持一致）
                    if info.get("success", False):
                        success = True
                    # 如果info中没有success字段，使用与4_simple_train_ript.py相同的判断逻辑
                    elif total_reward > 0.5:
                        success = True
                    
                    # 如果任务完成，提前退出
                    if done:
                        break
            
            except Exception as e:
                print(f"执行环境步骤时出错: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                # 关闭环境并释放资源
                try:
                    env.close()
                except:
                    pass
            
            # 保存整个轨迹的视频
            if save_video and rollout_images:
                try:
                    # 创建视频目录
                    video_dir = Path("pi0/ript/debug_images/videos")
                    video_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 生成视频文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    task_str = task_description.replace(" ", "_")[:30] if isinstance(task_description, str) else f"episode_{i}"
                    success_str = "success" if success else "failure"
                    video_path = video_dir / f"{timestamp}_{task_str}_{success_str}.mp4"
                    
                    # 保存视频
                    writer = imageio.get_writer(str(video_path), fps=30)
                    for frame in rollout_images:
                        writer.append_data(frame)
                    writer.close()
                    
                    print(f"✅ 已保存视频: {video_path}")
                except Exception as e:
                    print(f"保存视频出错: {e}")
            
            # 收集episode数据
            episode_data = {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "infos": infos,
                "task": task_description,
            }
            
            # 返回轨迹结果
            yield (success, total_reward, episode_data)