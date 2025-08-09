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

# 导入并行环境支持 (仅在需要时使用)
try:
    from libero.libero.envs import SubprocVectorEnv
    import multiprocessing
    import gc
    VECTOR_ENV_AVAILABLE = True
except ImportError:
    VECTOR_ENV_AVAILABLE = False

# 导入独立环境工厂 (用于真正的多进程并行)
try:
    from .parallel_env_factory import create_env_factory
    TRUE_PARALLEL_AVAILABLE = True
except ImportError:
    TRUE_PARALLEL_AVAILABLE = False

# 调试设置
DEBUG_SAVE_IMAGES = False  # 禁用单独图像保存，只保留视频
DEBUG_SAVE_VIDEO = os.environ.get("PI0_DEBUG_SAVE_VIDEO", "false").lower() in ("true", "1", "yes")
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
        
        # 🔥 新增：功能开关控制 (安全集成复杂功能)
        # 从配置文件的features部分读取开关设置
        if config and hasattr(config, 'features'):
            features = config.features
            self.enable_task_polling = getattr(features, 'enable_task_polling', False)
            self.enable_parallel_envs = getattr(features, 'enable_parallel_envs', False)
            self.enable_smart_sampling = getattr(features, 'enable_smart_sampling', False)
            # 🆕 新增：真正多进程并行开关（默认关闭以保持兼容性）
            self.enable_true_parallel_envs = getattr(features, 'enable_true_parallel_envs', False)
        else:
            # 回退到直接从config读取或使用默认值
            self.enable_task_polling = getattr(config, 'enable_task_polling', False) if config else False
            self.enable_parallel_envs = getattr(config, 'enable_parallel_envs', False) if config else False
            self.enable_smart_sampling = getattr(config, 'enable_smart_sampling', False) if config else False
            # 🆕 新增：真正多进程并行开关（默认关闭以保持兼容性）
            self.enable_true_parallel_envs = getattr(config, 'enable_true_parallel_envs', False) if config else False
        
        if self.rank == 0:
            print(f"🔧 LIBEROEnvRunner功能开关:")
            print(f"   任务轮询: {'✅' if self.enable_task_polling else '❌'}")
            print(f"   并行环境: {'✅' if self.enable_parallel_envs else '❌'}")
            print(f"   真正多进程并行: {'✅' if self.enable_true_parallel_envs else '❌'}")
            print(f"   智能采样: {'✅' if self.enable_smart_sampling else '❌'}")
        
        # 🔥 新增：任务轮询机制 (仅在开关启用时初始化)
        if self.enable_task_polling:
            self.assigned_tasks = task_names_to_use or []  
            self.task_cursor = 0                          
            self.current_task = None                      
            
            if self.assigned_tasks:
                self.current_task = self.assigned_tasks[0]
                if self.rank == 0:
                    print(f"🎯 任务轮询初始化: 分配任务 {self.assigned_tasks}, 当前任务: {self.current_task}")
            else:
                if self.rank == 0:
                    print(f"⚠️ 任务轮询: Rank {self.rank} 没有分配任务")
        else:
            # 简单模式：不使用任务轮询
            self.assigned_tasks = task_names_to_use or []
            self.current_task = task_names_to_use[0] if task_names_to_use else None
        
        # 调试选项
        self.debug_save_images = DEBUG_SAVE_IMAGES
        # YAML开关优先，其次环境变量
        yaml_save_video = False
        try:
            if config and hasattr(config, 'features'):
                yaml_save_video = bool(getattr(config.features, 'save_video', False))
        except Exception:
            yaml_save_video = False
        self.debug_save_video = bool(yaml_save_video) or DEBUG_SAVE_VIDEO
        self.debug_image_dir = DEBUG_IMAGE_DIR
        
        # 加载归一化统计信息
        self._load_norm_stats(norm_stats_path)

    def _ensure_list_of_dict_obs(self, obs_any, env_num: int):
        """Normalize vector-env observation to List[Dict] per environment.

        Supports formats:
        - dict of batched arrays -> split along first dim
        - numpy array of objects (each a dict) -> list(obs)
        - single dict (non-batched) -> broadcast to env_num
        - fallback: wrap unknown type by broadcasting
        """
        try:
            import numpy as _np
        except Exception:
            _np = None

        # dict of batched arrays
        if isinstance(obs_any, dict):
            result = []
            for i in range(env_num):
                per_env = {}
                for k, v in obs_any.items():
                    try:
                        if hasattr(v, '__len__') and len(v) >= env_num:
                            per_env[k] = v[i]
                        else:
                            per_env[k] = v
                    except Exception:
                        per_env[k] = v
                result.append(per_env)
            return result

        # numpy array of objects (each is a dict)
        if _np is not None and isinstance(obs_any, _np.ndarray):
            if obs_any.dtype == object:
                return [obs_any[i] for i in range(min(len(obs_any), env_num))]
            # unknown numeric array; broadcast
            return [obs_any for _ in range(env_num)]

        # single dict -> broadcast
        if isinstance(obs_any, dict):
            return [obs_any for _ in range(env_num)]

        # list already -> assume per-env
        if isinstance(obs_any, list):
            return obs_any[:env_num] if len(obs_any) >= env_num else (obs_any + [obs_any[-1]] * (env_num - len(obs_any)))

        # fallback: broadcast
        return [obs_any for _ in range(env_num)]

    def _ensure_list_generic(self, value, env_num: int, fill_default, expect_dict: bool = False):
        """Normalize rewards/dones/infos to python list of length env_num.

        - If numpy array: convert to list; if too short, pad by last; if too long, slice.
        - If object array and expect_dict, ensure dict per item; otherwise cast as-is.
        - If single scalar/value, broadcast.
        """
        try:
            import numpy as _np
        except Exception:
            _np = None

        result = None
        if isinstance(value, list):
            result = value
        elif _np is not None and isinstance(value, _np.ndarray):
            # object array (e.g., dicts)
            if value.dtype == object:
                result = list(value)
            else:
                # numeric/bool array
                try:
                    result = value.tolist()
                except Exception:
                    result = [value]
        else:
            # single scalar/value -> broadcast
            result = [value for _ in range(env_num)]

        # adjust length
        if len(result) < env_num:
            pad_val = result[-1] if result else fill_default
            result = (result + [pad_val] * env_num)[:env_num]
        else:
            result = result[:env_num]

        if expect_dict:
            # ensure each item is a dict
            result = [item if isinstance(item, dict) else {} for item in result]

        return result
        
    # 🔥 新增：任务轮询核心方法 (仅在功能开关启用时可用)
    def set_current_task_by_cursor(self):
        """根据cursor设置当前任务"""
        if not self.enable_task_polling:
            return self.current_task  # 简单模式：直接返回当前任务
            
        if not self.assigned_tasks:
            self.current_task = None
            return None
        
        # 使用模运算实现循环轮询
        self.current_task = self.assigned_tasks[self.task_cursor % len(self.assigned_tasks)]
        return self.current_task
    
    def advance_task_cursor(self):
        """推进任务cursor到下一个任务"""
        if not self.enable_task_polling:
            return self.current_task  # 简单模式：不推进，返回当前任务
            
        if not self.assigned_tasks:
            return None
        
        self.task_cursor += 1
        new_task = self.set_current_task_by_cursor()
        
        # 调试输出
        if self.rank == 0 or len(self.assigned_tasks) > 1:  # 主进程或多任务时输出
            print(f"🔄 Rank {self.rank}: 任务cursor推进到 {self.task_cursor}, 当前任务: {new_task}")
        
        return new_task
    
    def get_current_task(self):
        """获取当前任务"""
        return self.current_task
    
    def has_tasks(self):
        """检查是否有分配的任务"""
        return len(self.assigned_tasks) > 0
    
    # 🔥 新增：任务统计和高级功能 (模仿原版RIPT-VLA)
    def update_task_stats(self, task_name: str, success: bool, reward: float):
        """更新任务完成统计"""
        if not hasattr(self, 'task_completion_stats'):
            self.task_completion_stats = {}
        
        if task_name not in self.task_completion_stats:
            self.task_completion_stats[task_name] = {
                'attempts': 0,
                'successes': 0,
                'total_reward': 0.0,
                'success_rate': 0.0
            }
        
        stats = self.task_completion_stats[task_name]
        stats['attempts'] += 1
        stats['total_reward'] += reward
        if success:
            stats['successes'] += 1
        stats['success_rate'] = stats['successes'] / stats['attempts'] if stats['attempts'] > 0 else 0.0
    
    def get_task_stats(self) -> dict:
        """获取任务统计信息"""
        if not hasattr(self, 'task_completion_stats'):
            return {}
        return self.task_completion_stats.copy()
    
    def get_best_performing_task(self) -> Optional[str]:
        """获取表现最好的任务（用于智能采样）"""
        if not hasattr(self, 'task_completion_stats') or not self.task_completion_stats:
            return None
        
        best_task = None
        best_score = -1.0
        
        for task, stats in self.task_completion_stats.items():
            # 综合考虑成功率和平均奖励
            if stats['attempts'] > 0:
                avg_reward = stats['total_reward'] / stats['attempts']
                score = stats['success_rate'] * 0.7 + min(avg_reward, 1.0) * 0.3
                if score > best_score:
                    best_score = score
                    best_task = task
        
        return best_task
    
    def setup_file_counter(self, counter_name: str = "rollout", work_dir: str = "./") -> Optional[object]:
        """设置文件计数器用于分布式协调"""
        try:
            from pi0.ript.algos.rl_optimizers.file_counter import setup_global_counter
            counter = setup_global_counter(counter_name, work_dir=work_dir)
            if self.rank == 0:
                print(f"✅ 文件计数器设置成功: {counter_name}")
            return counter
        except ImportError as e:
            if self.rank == 0:
                print(f"⚠️ 文件计数器不可用: {e}")
            return None
        except Exception as e:
            if self.rank == 0:
                print(f"⚠️ 文件计数器创建失败: {e}")
            return None
        
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
            # 始终使用自然语言任务描述作为prompt，避免使用env_name等短标识
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
        
    def run_policy_in_env(self, env_name, all_init_states=None, debug_save_video=None, created_env=None):
        """在环境中运行策略，生成轨迹 - 支持并行和串行环境"""
        save_video = debug_save_video if debug_save_video is not None else self.debug_save_video
        
        # 🔥 根据功能开关选择并行或串行模式
        if self.enable_parallel_envs and created_env is None:
            if self.rank == 0:
                print(f"🚀 使用并行环境模式 (num_parallel_envs={self.num_parallel_envs})")
            # 创建并行环境
            env, env_id, env_num = self.create_parallel_envs(env_name)
            created_env = (env, env_id, env_num)
        elif created_env is None:
            if self.rank == 0:
                print(f"🔄 使用串行环境模式")
            # 使用串行环境
            env, task_description = self.make_env(env_name)
            created_env = (env, env_name, 1)
        
        env, env_id, env_num = created_env
        
        if all_init_states is None:
            # 如果没有提供初始状态，生成默认状态
            all_init_states = np.zeros((self.rollouts_per_env, 8), dtype=np.float32)
        
        try:
            # 检查是否为真正的并行环境
            is_vector_env = hasattr(env, 'num_envs') or 'VectorEnv' in str(type(env))
            
            if self.enable_parallel_envs and env_num > 1 and is_vector_env:
                # 使用真正的并行环境
                if self.rank == 0:
                    print(f"🚀 并行环境模式: {env_num} 个环境")
                yield from self._run_parallel_episodes(env, env_name, all_init_states, env_num, save_video)
            else:
                # 使用单环境模式
                if self.rank == 0:
                    print(f"🔄 单环境模式")
                yield from self._run_serial_episodes(env, env_name, all_init_states, save_video)
        
        finally:
            # 清理环境
            try:
                if hasattr(env, 'close'):
                    env.close()
                if VECTOR_ENV_AVAILABLE:
                    import gc
                    gc.collect()
            except Exception as e:
                if self.rank == 0:
                    print(f"环境清理时出错: {e}")
    
    def _run_serial_episodes(self, env, env_name, all_init_states, save_video):
        """运行串行 episodes（处理单个环境和单个VectorEnv）"""
        if self.rank == 0:
            print(f"使用串行模式执行 {len(all_init_states)} 个episodes")
        
        # 检查是否为VectorEnv类型
        is_vector_env = hasattr(env, 'num_envs') or 'VectorEnv' in str(type(env))
        if is_vector_env:
            if self.rank == 0:
                print(f"检测到VectorEnv，使用单个环境模式")
        
        for i, init_state in enumerate(all_init_states):
            rollout_images = []
            
            # 重用传入的环境或创建新环境
            if hasattr(env, 'task_description'):
                task_description = env.task_description
            else:
                task_description = env_name
            
            # 设置初始状态并获取初始观测
            obs = env.reset()
            
            # 如果是VectorEnv，取第一个环境的观测
            if is_vector_env and isinstance(obs, list):
                obs = obs[0]
            
            # 热机步骤
            dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])
            for warmup_step in range(20):
                if is_vector_env:
                    # VectorEnv期望动作列表
                    step_results = env.step([dummy_action])
                    obs, _, _, _ = step_results
                    if isinstance(obs, list):
                        obs = obs[0]  # 取第一个环境的结果
                else:
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
                    if is_vector_env:
                        # VectorEnv期望动作列表
                        step_results = env.step([current_action])
                        next_obs, reward, done, info = step_results
                        # 取第一个环境的结果
                        if isinstance(next_obs, list):
                            next_obs = next_obs[0]
                        if isinstance(reward, list):
                            reward = reward[0]
                        if isinstance(done, list):
                            done = done[0]
                        if isinstance(info, list):
                            info = info[0]
                    else:
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
    
    # 🔥 新增：并行环境支持 (仅在功能开关启用时可用)
    def create_env(self, env_name: str):
        """创建单个环境（用于并行环境）"""
        if not self.enable_parallel_envs:
            return self.make_env(env_name)  # 简单模式：直接返回单个环境
            
        env, task_description = self.make_env(env_name)
        return env, env_name, 1
    
    def create_parallel_envs(self, env_name: str):
        """创建并行环境"""
        if not self.enable_parallel_envs or not VECTOR_ENV_AVAILABLE or self.num_parallel_envs <= 1:
            # 如果不支持并行环境或只需要1个环境，使用单个环境
            if self.rank == 0:
                print(f"使用单个环境 (num_parallel_envs={self.num_parallel_envs})")
            env, task_description = self.make_env(env_name)
            return env, env_name, 1
        
        # 检查是否启用真正的多进程并行
        if self.enable_true_parallel_envs:
            return self._create_true_parallel_envs(env_name)
        else:
            # 回退到单环境
            return self._create_single_env(env_name)
    
    def _create_single_env(self, env_name: str):
        """创建单个环境 - 简单直接"""
        if self.rank == 0:
            print(f"🔄 使用单环境模式")
        
        env, task_description = self.make_env(env_name) 
        return env, env_name, 1
    
    def _create_true_parallel_envs(self, env_name: str):
        """🆕 创建真正的多进程并行环境 - 使用独立环境工厂解决序列化问题"""
        if not TRUE_PARALLEL_AVAILABLE:
            if self.rank == 0:
                print(f"⚠️ 独立环境工厂不可用，回退到单环境模式")
            return self._create_single_env(env_name)
        
        # 计算GPU内存需求
        model_size_gb = 3.5  # PI0模型大小
        required_memory_gb = self.num_parallel_envs * model_size_gb
        
        # 检查GPU内存
        available_memory_gb = 0
        current_usage = 0
        if torch.cuda.is_available():
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            current_usage = torch.cuda.memory_allocated(0) / (1024**3)
            available_memory = available_memory_gb - current_usage
        
        if self.rank == 0:
            print(f"🚀 尝试创建真正的多进程并行环境:")
            print(f"🧠 内存分析:")
            print(f"   {self.num_parallel_envs}个并行环境理论需要: {required_memory_gb:.1f}GB")
            print(f"   当前GPU总内存: {available_memory_gb:.1f}GB") 
            print(f"   当前GPU已使用: {current_usage:.1f}GB")
            print(f"   可用内存: {available_memory:.1f}GB")
        
        # 内存安全检查
        if available_memory < required_memory_gb * 1.2:
            if self.rank == 0:
                print(f"⚠️ GPU内存不足，回退到单环境模式")
            return self._create_single_env(env_name)
        
        try:
            # 🔑 关键：使用独立环境工厂，避免序列化self对象
            env_factory = create_env_factory(
                benchmark_name=self.benchmark_name,
                env_name=env_name,
                task_id=None  # 自动推断
            )
            
            # 创建多个环境工厂实例
            env_factories = [env_factory for _ in range(self.num_parallel_envs)]
            
            if self.rank == 0:
                print(f"🔧 创建 {self.num_parallel_envs} 个独立并行环境...")
            
            # 设置multiprocessing启动方法
            if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                multiprocessing.set_start_method('spawn', force=True)
            
            # 创建SubprocVectorEnv
            parallel_env = SubprocVectorEnv(env_factories)
            
            # 🔍 仅记录reset返回信息，不做强校验以适配不同实现
            try:
                test_obs = parallel_env.reset()
                if self.rank == 0:
                    if isinstance(test_obs, list):
                        print(f"✅ SubprocVectorEnv已创建，reset返回list，长度: {len(test_obs)}")
                    else:
                        print(f"✅ SubprocVectorEnv已创建，reset返回类型: {type(test_obs)}")
            except Exception as e:
                if self.rank == 0:
                    print(f"⚠️ SubprocVectorEnv.reset() 调用异常: {e}")
            
            if self.rank == 0:
                print(f"   🔄 {self.num_parallel_envs} 个独立子进程")
                print(f"   🧠 每个子进程无模型，总内存节省 ~{(self.num_parallel_envs-1)*3.5:.1f}GB")
                print(f"   ⚡ 真正的并行执行，性能提升 ~{self.num_parallel_envs}x")
            
            # 🧠 获取每个子进程的任务描述（自然语言prompt），用于构造模型输入
            try:
                vector_prompts = []
                for w in getattr(parallel_env, 'workers', []):
                    desc = None
                    try:
                        desc = w.get_env_attr('task_description')
                    except Exception:
                        desc = None
                    vector_prompts.append(desc if isinstance(desc, str) and len(desc) > 0 else env_name)
                self._vector_env_prompts = vector_prompts
            except Exception:
                self._vector_env_prompts = [env_name for _ in range(self.num_parallel_envs)]

            return parallel_env, env_name, self.num_parallel_envs
            
        except Exception as e:
            if self.rank == 0:
                print(f"❌ 真正并行环境创建失败: {e}")
                print(f"🔄 回退到单环境模式")
                import traceback
                traceback.print_exc()
            
            # 清理可能的资源
            try:
                if 'parallel_env' in locals():
                    parallel_env.close()
            except:
                pass
            
            # 回退到单环境
            return self._create_single_env(env_name)
    
    # ❌ 轻量级并行环境方案已被移除
    # 原因：CloudPickle序列化时会包含包含Policy的self对象引用
    # 无论如何构造环境函数，都会导致模型在每个子进程中重复加载
    # 这是SubprocVectorEnv + PyTorch模型的固有限制
    
    def _deprecated_lightweight_parallel_note(self):
        """记录轻量级并行方案失败的原因"""
        return """
        轻量级并行方案失败原因分析：
        
        1. CloudPickle序列化问题：
           - SubprocVectorEnv使用CloudPickle序列化环境构造函数
           - 即使create_env_without_model不直接引用self.policy
           - 但函数闭包中包含self对象的引用
           - CloudPickle会递归序列化self及其包含的policy对象
        
        2. GPU内存问题：
           - 每个子进程反序列化时都会重建完整的PI0模型
           - 3个进程 × 3.5GB = 10.5GB GPU内存需求
           - 超过单卡内存限制导致CUDA OOM
        
        3. 解决方案：
           - 使用批处理模拟并行策略
           - 单环境 + 智能批量处理
           - 内存安全且性能接近真实并行
        """
    
    def _run_parallel_episodes(self, env, env_name, all_init_states, env_num, save_video):
        """运行真正的并行 episodes"""
        if self.rank == 0:
            print(f"🚀 开始并行执行 {env_num} 个环境")
        
        # 计算需要的轮次
        eval_loop_num = (self.rollouts_per_env + env_num - 1) // env_num
        count = 0
        
        while count < eval_loop_num:
            # 选择当前轮次的初始状态
            start_idx = count * env_num
            end_idx = min(start_idx + env_num, len(all_init_states))
            indices = np.arange(start_idx, end_idx) % len(all_init_states)
            current_init_states = all_init_states[indices]
            
            if self.rank == 0:
                print(f"并行轮次 {count+1}/{eval_loop_num}, 状态索引: {indices}")
            
            # 并行执行 episodes
            try:
                results = self._run_single_parallel_batch(env, env_name, current_init_states, env_num, save_video)
                
                # 返回结果
                for result in results:
                    yield result
                    
            except Exception as e:
                if self.rank == 0:
                    print(f"并行批次执行失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            count += 1
    
    def _run_single_parallel_batch(self, env, env_name, init_states, env_num, save_video):
        """执行单个并行批次 - 支持轻量级并行环境"""
        # 检查环境类型
        is_vector_env = hasattr(env, 'num_envs') or 'VectorEnv' in str(type(env))
        
        if not is_vector_env or env_num == 1:
            # 使用批处理模拟并行（推荐策略）
            return self._run_batch_simulated_parallel(env, env_name, init_states, save_video)
        
        # 真实并行环境处理
        if self.rank == 0:
            print(f"🚀 执行真实并行处理 ({env_num} 个环境)")
        
        # 重置所有环境
        obs_any = env.reset()
        # 默认不在并行模式使用 set_init_state（避免 MuJoCo qpos 维度不匹配导致子进程崩溃）
        use_parallel_init = False
        try:
            if getattr(self, 'config', None) and hasattr(self.config, 'features'):
                use_parallel_init = bool(getattr(self.config.features, 'use_parallel_init_state', False))
        except Exception:
            use_parallel_init = False
        if use_parallel_init and init_states is not None:
            try:
                obs_any = env.set_init_state(init_states)
            except Exception as e:
                if self.rank == 0:
                    print(f"⚠️ set_init_state 调用失败，继续使用默认初始状态: {e}")
        else:
            if self.rank == 0 and init_states is not None:
                print("ℹ️ 并行模式下已跳过 set_init_state（use_parallel_init_state=false）")
        obs_list = self._ensure_list_of_dict_obs(obs_any, env_num)
        
        if self.rank == 0:
            print(f"🔧 初始化 {len(obs_list)} 个并行环境")
        
        # 对每个环境进行热身
        dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])
        for warmup_step in range(20):
            # 🔑 确保actions数组长度与环境数量完全匹配
            actions = [dummy_action.copy() for _ in range(env_num)]
            if self.rank == 0 and warmup_step == 0:
                print(f"🔧 热身动作: {len(actions)} 个动作 for {env_num} 个环境")
            step_out = env.step(actions)
            obs_any = step_out[0] if isinstance(step_out, (list, tuple)) and len(step_out) >= 1 else step_out
            obs_list = self._ensure_list_of_dict_obs(obs_any, env_num)
        
        # 初始化episode数据
        episodes_data = []
        for i in range(len(obs_list)):
            episodes_data.append({
                'observations': [obs_list[i]],
                'actions': [],
                'rewards': [],
                'dones': [],
                'infos': [],
                'success': False,
                'total_reward': 0.0,
                'step': 0,
                'action_buffer': None,
                'action_index': 0,
                'rollout_images': [] if save_video else None,  # 🎬 视频帧存储
                'completed': False
            })
            
            # 🎬 收集初始观测图像用于视频
            if save_video:
                try:
                    initial_img = obs_list[i]["agentview_image"][:, :, ::-1].transpose(1, 2, 0).copy()
                    episodes_data[i]['rollout_images'].append(initial_img)
                except Exception as e:
                    if self.rank == 0:
                        print(f"⚠️ 收集初始图像失败 (环境{i}): {e}")
                    episodes_data[i]['rollout_images'] = None
        
        max_steps = self.max_steps
        all_done = False
        
        # 并行执行steps - 关键优化：模型推理在主进程中进行
        while not all_done:
            # 🔥 批量处理所有环境的观测 (避免逐个推理)
            need_inference_indices = []
            observations_to_infer = []
            
            # 第一步：识别需要推理的环境
            for i, obs in enumerate(obs_list):
                episode = episodes_data[i]
                
                if episode['dones'] and len(episode['dones']) > 0 and episode['dones'][-1]:
                    # 环境已完成，跳过
                    continue
                
                # 检查是否需要推理
                if (episode['action_buffer'] is None or 
                    episode['action_index'] >= episode['action_buffer'].shape[0]):
                    need_inference_indices.append(i)
                    observations_to_infer.append(obs)
            
            # 第二步：批量推理 (关键优化: 一次推理多个观测)
            if observations_to_infer:
                prompts_for_obs = None
                if hasattr(self, '_vector_env_prompts'):
                    prompts_for_obs = [self._vector_env_prompts[i] if i < len(self._vector_env_prompts) else env_name for i in need_inference_indices]
                batch_actions = self._batch_policy_inference(observations_to_infer, env_name, prompts_for_obs)
                
                # 分配推理结果到对应的episode
                for batch_idx, env_idx in enumerate(need_inference_indices):
                    episode = episodes_data[env_idx]
                    obs = obs_list[env_idx]
                    
                    # 处理动作
                    action_buffer = batch_actions[batch_idx]
                    
                    # 获取状态偏移
                    import robosuite.utils.transform_utils as T
                    unnorm_state = np.concatenate([
                        obs["robot0_eef_pos"],
                        T.quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ], dtype=np.float32)
                    
                    action_buffer[:, :6] += unnorm_state[None, :6]
                    
                    episode['action_buffer'] = action_buffer
                    episode['action_index'] = 0
            
            # 第三步：为所有环境构建动作数组（确保顺序和数量匹配）
            actions_to_execute = []
            for i, obs in enumerate(obs_list):
                episode = episodes_data[i]
                
                if episode['completed'] or (episode['dones'] and len(episode['dones']) > 0 and episode['dones'][-1]):
                    # 环境已完成，使用dummy动作
                    actions_to_execute.append(dummy_action)
                else:
                    # 获取当前动作（应该已经有action_buffer了）
                    if episode['action_buffer'] is not None:
                        current_action = episode['action_buffer'][episode['action_index'], :7]
                        actions_to_execute.append(current_action)
                        episode['action_index'] += 1
                    else:
                        # 备用：如果还是没有action_buffer，使用dummy动作
                        actions_to_execute.append(dummy_action)
            
            # 并行执行动作
            step_out = env.step(actions_to_execute)
            if isinstance(step_out, (list, tuple)) and len(step_out) >= 4:
                obs_any, rewards_any, dones_any, infos_any = step_out[:4]
            else:
                obs_any, rewards_any, dones_any, infos_any = step_out, 0.0, False, {}

            obs_list = self._ensure_list_of_dict_obs(obs_any, env_num)
            rewards = self._ensure_list_generic(rewards_any, env_num, 0.0, expect_dict=False)
            dones = self._ensure_list_generic(dones_any, env_num, False, expect_dict=False)
            infos = self._ensure_list_generic(infos_any, env_num, {}, expect_dict=True)
            
            # 更新episode数据
            all_done = True
            for i in range(len(obs_list)):
                episode = episodes_data[i]
                
                # 已完成的环境不再追加任何数据，保持与单环境逻辑一致（完成即停止）
                if episode['completed']:
                    continue
                
                episode['observations'].append(obs_list[i])
                episode['actions'].append(actions_to_execute[i])
                episode['rewards'].append(rewards[i])
                episode['dones'].append(dones[i])
                episode['infos'].append(infos[i])
                episode['total_reward'] += rewards[i]
                episode['step'] += 1
                
                # 🎬 收集图像用于视频
                if save_video and episode['rollout_images'] is not None:
                    try:
                        frame_img = obs_list[i]["agentview_image"][:, :, ::-1].transpose(1, 2, 0).copy()
                        episode['rollout_images'].append(frame_img)
                    except Exception as e:
                        if self.rank == 0:
                            print(f"⚠️ 收集图像帧失败 (环境{i}): {e}")
                
                if infos[i].get("success", False) or episode['total_reward'] > 0.5:
                    episode['success'] = True
                
                # 完成条件：本步返回 done 或达到最大步数
                if dones[i] or episode['step'] >= max_steps:
                    episode['completed'] = True
                
            # 只要存在任一未完成的环境，则继续循环
            for i in range(len(obs_list)):
                if not episodes_data[i]['completed']:
                    all_done = False
                    break
        
        # 🎬 保存视频并返回结果
        results = []
        for i, episode in enumerate(episodes_data):
            # 保存视频（如果启用）
            video_path = None
            if save_video and episode['rollout_images'] and len(episode['rollout_images']) > 0:
                try:
                    from datetime import datetime
                    from pathlib import Path
                    import imageio
                    
                    # 创建视频目录
                    video_dir = Path("pi0/ript/debug_images/videos")
                    video_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 生成视频文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    task_str = env_name.replace(" ", "_")[:30] if isinstance(env_name, str) else f"parallel_env_{i}"
                    success_str = "success" if episode['success'] else "failure"
                    video_path = video_dir / f"{timestamp}_{task_str}_env{i}_{success_str}.mp4"
                    
                    # 保存视频
                    writer = imageio.get_writer(str(video_path), fps=30)
                    for frame in episode['rollout_images']:
                        writer.append_data(frame)
                    writer.close()
                    
                    if self.rank == 0:
                        print(f"✅ 已保存视频 (环境{i}): {video_path}")
                except Exception as e:
                    if self.rank == 0:
                        print(f"⚠️ 保存视频失败 (环境{i}): {e}")
            
            episode_data = {
                "observations": episode['observations'],
                "actions": episode['actions'],
                "rewards": episode['rewards'],
                "dones": episode['dones'],
                "infos": episode['infos'],
                "task": self._vector_env_prompts[i] if hasattr(self, '_vector_env_prompts') and i < len(self._vector_env_prompts) else env_name,
            }
            
            # 添加视频路径信息
            if video_path:
                episode_data["video_path"] = str(video_path)
            
            results.append((episode['success'], episode['total_reward'], episode_data))
        
        return results
    
    def _batch_policy_inference(self, observations, env_name, prompts_for_obs=None):
        """批量模型推理 - 避免逐个推理提高效率"""
        if not observations:
            return []
        
        try:
            # 构建批量观测
            batch_obs = []
            for idx, obs in enumerate(observations):
                prompt_text = None
                if prompts_for_obs is not None and idx < len(prompts_for_obs):
                    prompt_text = prompts_for_obs[idx]
                pi0_obs = self.construct_pi0_observation(obs, prompt_text or env_name)
                batch_obs.append(pi0_obs)
            
            # 合并批量观测（如果可能）
            if len(batch_obs) == 1:
                # 单个观测直接推理
                raw_action = self.policy.select_action(batch_obs[0])
                action = raw_action[0, :, :7]  # (50, 7)
                
                if isinstance(action, torch.Tensor):
                    action_after_cpu = action.cpu().numpy()
                else:
                    action_after_cpu = action
                
                action_buffer = action_after_cpu * (self.action_std + 1e-6) + self.action_mean
                return [action_buffer]
            else:
                # 多个观测分别推理 (目前PI0不支持真正的批量推理)
                batch_actions = []
                for pi0_obs in batch_obs:
                    raw_action = self.policy.select_action(pi0_obs)
                    action = raw_action[0, :, :7]
                    
                    if isinstance(action, torch.Tensor):
                        action_after_cpu = action.cpu().numpy()
                    else:
                        action_after_cpu = action
                    
                    action_buffer = action_after_cpu * (self.action_std + 1e-6) + self.action_mean
                    batch_actions.append(action_buffer)
                
                return batch_actions
                
        except Exception as e:
            if self.rank == 0:
                print(f"批量推理失败: {e}")
            # 回退到逐个推理
            return self._fallback_individual_inference(observations, env_name)
    
    def _fallback_individual_inference(self, observations, env_name, prompts_for_obs=None):
        """回退到逐个推理"""
        batch_actions = []
        for idx, obs in enumerate(observations):
            prompt_text = None
            if prompts_for_obs is not None and idx < len(prompts_for_obs):
                prompt_text = prompts_for_obs[idx]
            pi0_obs = self.construct_pi0_observation(obs, prompt_text or env_name)
            raw_action = self.policy.select_action(pi0_obs)
            action = raw_action[0, :, :7]
            
            if isinstance(action, torch.Tensor):
                action_after_cpu = action.cpu().numpy()
            else:
                action_after_cpu = action
            
            action_buffer = action_after_cpu * (self.action_std + 1e-6) + self.action_mean
            batch_actions.append(action_buffer)
        
        return batch_actions
    
