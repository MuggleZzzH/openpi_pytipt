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
        
        # ✅ 存储分布式训练参数（需要先赋值rank，后续代码会用到）
        self.config = config
        self.rank = rank
        self.world_size = world_size
        
        # 🔥 新增：评估时可关闭预热
        self.disable_warmup = False
        
        # 🔥 新增：初始化benchmark以获取真实基准状态池，与RIPT-VLA完全对齐
        self.benchmark = None
        if benchmark_name:
            try:
                from libero.libero.benchmark import get_benchmark
                self.benchmark = get_benchmark(benchmark_name.lower())()
                if self.rank == 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"✅ Benchmark初始化成功: {benchmark_name} -> {type(self.benchmark).__name__}")
            except Exception as e:
                if self.rank == 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"⚠️ Benchmark初始化失败: {e}")
                    logger.debug(f"   将使用合成初始状态池作为回退")
                self.benchmark = None
        
        # 🔥 使用RIPT-VLA官方的任务最大步数设置（基于训练数据统计）
        TASK_MAX_STEPS = {
            'libero_spatial': 220,  # longest training demo has 193 steps
            'libero_object': 280,   # longest training demo has 254 steps
            'libero_goal': 240,     # longest training demo has 211 steps
            'libero_long': 400,     # longest training demo has 372 steps
        }
        
        if max_episode_length is not None:
            self.max_steps = max_episode_length
        elif self.benchmark_name and self.benchmark_name.lower() in TASK_MAX_STEPS:
            self.max_steps = TASK_MAX_STEPS[self.benchmark_name.lower()]
            if self.rank == 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"🎯 使用官方任务限制: {self.benchmark_name} → {self.max_steps}步")
        else:
            self.max_steps = 300  # 安全默认值（libero_goal的限制）
            if self.rank == 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"⚠️ 未知benchmark {self.benchmark_name}，使用默认限制: {self.max_steps}步")
        
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
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"🔧 LIBEROEnvRunner功能开关:")
            logger.debug(f"   任务轮询: {'✅' if self.enable_task_polling else '❌'}")
            logger.debug(f"   并行环境: {'✅' if self.enable_parallel_envs else '❌'}")
            logger.debug(f"   真正多进程并行: {'✅' if self.enable_true_parallel_envs else '❌'}")
            logger.debug(f"   智能采样: {'✅' if self.enable_smart_sampling else '❌'}")
        
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
        
        # 🔥 新增：文件计数器支持
        self.file_counter = None
        
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
    
    def to_hwc_hmirror(self, arr: np.ndarray) -> np.ndarray:
        """
        统一图像处理函数，与"2_pi0_on_libero.py"完全对齐
        先规范到HWC格式，再做水平镜像（不做通道交换）
        """
        if isinstance(arr, np.ndarray) and arr.ndim == 3:
            # CHW -> HWC（如果需要）
            if arr.shape[0] == 3 and arr.shape[-1] != 3:
                arr = arr.transpose(1, 2, 0)
            # 水平镜像（翻转宽度维）
            return arr[:, ::-1, :].copy()
        return arr
    
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
            if counter:
                self.file_counter = counter  # 🔥 保存到实例变量
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

    def get_task_init_states(self, task_id=0):
        """
        获取任务的可用初始状态池
        
        Args:
            task_id: 任务ID
            
        Returns:
            list: 可用的初始状态列表，每个状态是一个8维numpy数组
        """
        # 方案1: 从benchmark获取（如果可用）
        if hasattr(self, 'benchmark') and self.benchmark is not None:
            try:
                return self.benchmark.get_task_init_states(task_id)
            except:
                pass
        
        # 方案2: 生成固定的可复现初始状态池
        import numpy as np
        np.random.seed(42 + task_id)  # 确保可复现
        num_states = 100  # 生成100个候选初始状态
        init_states = []
        
        for i in range(num_states):
            # 生成一个8维的初始状态（3 pos + 3 rot + 2 gripper）
            state = np.random.randn(8).astype(np.float32)
            state[:3] *= 0.1  # 位置范围适中
            state[3:6] *= 0.5  # 旋转范围适中
            state[6:8] = np.array([0.0, 0.0])  # 夹爪初始闭合
            init_states.append(state)
        
        return init_states
    
    def _compute_state_hash(self, state):
        """计算状态的哈希值"""
        import hashlib
        state_bytes = np.array(state).astype(np.float32).tobytes()
        return hashlib.sha256(state_bytes).hexdigest()
    
    def _extract_state_from_obs(self, obs):
        """从观测中提取8维状态用于哈希计算"""
        try:
            if isinstance(obs, dict):
                # 提取位置、姿态和夹爪状态
                pos = np.array(obs.get("robot0_eef_pos", [0, 0, 0]), dtype=np.float32)[:3]
                quat = np.array(obs.get("robot0_eef_quat", [0, 0, 0, 1]), dtype=np.float32)[:4]
                gripper = np.array(obs.get("robot0_gripper_qpos", [0, 0]), dtype=np.float32)[:2]
                
                # 转换四元数到轴角
                try:
                    import robosuite.utils.transform_utils as T
                    axis_angle = T.quat2axisangle(quat).astype(np.float32)
                except Exception:
                    axis_angle = np.zeros(3, np.float32)
                
                # 组合成8维状态
                state = np.concatenate([pos, axis_angle, gripper])
                return state[:8]  # 确保是8维
            else:
                # 如果obs不是字典，返回零状态
                return np.zeros(8, np.float32)
        except Exception as e:
            print(f"⚠️ 提取状态时出错: {e}")
            return np.zeros(8, np.float32)


    
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
        # 状态归一化
        state = (unnorm_state - self.state_mean) / (self.state_std + 1e-6)
        
        # 图像处理 - 使用统一的to_hwc_hmirror函数，与"2_pi0_on_libero.py"完全对齐
        base_0_rgb = self.to_hwc_hmirror(obs["agentview_image"])
        left_wrist_0_rgb = self.to_hwc_hmirror(obs["robot0_eye_in_hand_image"])
        
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
                
                # 获取 gripper 状态（🔥 修复：保持8维一致性）
                if "robot0_gripper_qpos" not in obs:
                    raise RuntimeError("Observation missing required field robot0_gripper_qpos")
                try:
                    gripper_qpos_raw = obs["robot0_gripper_qpos"]
                    gripper_qpos = np.array(gripper_qpos_raw, dtype=np.float32)
                    if gripper_qpos.size < 2:
                        # 如果gripper状态不足2维，填充为2维
                        full_gripper = np.zeros(2, dtype=np.float32)
                        full_gripper[:gripper_qpos.size] = gripper_qpos.flatten()[:gripper_qpos.size]
                        gripper_qpos = full_gripper
                    else:
                        gripper_qpos = gripper_qpos[:2]  # 取前2维，保持和construct_pi0_observation一致
                except (IndexError, TypeError, ValueError) as e:
                    raise RuntimeError(f"Invalid robot0_gripper_qpos value: {e}")
                
                # 🔥 修复：返回8维状态，与construct_pi0_observation完全一致
                unnorm_state = np.concatenate([eef_pos[:3], axis_angle[:3], gripper_qpos]).astype(np.float32)
                return unnorm_state  # 8维：3+3+2
                
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
            # 🔧 动态任务映射：使用任务名推断task_id，支持多任务训练
            try:
                from libero.libero.benchmark import get_benchmark
                benchmark = get_benchmark(self.benchmark_name.lower())()
                task_names = benchmark.get_task_names()
                if isinstance(env_name, str) and env_name in task_names:
                    task_id = task_names.index(env_name)
                    print(f"✅ 动态映射: {env_name} -> task_id={task_id}")
                else:
                    # 回退：保持与旧逻辑一致（确保健壮性）
                    if self.benchmark_name == "libero_goal":
                        task_id = 1
                    elif self.benchmark_name == "libero_spatial":
                        task_id = 0
                    else:
                        task_id = 0
                    print(f"⚠️ 映射回退: {env_name} -> task_id={task_id} (默认)")
            except Exception as e:
                # 回退：保持与旧逻辑一致
                if self.benchmark_name == "libero_goal":
                    task_id = 1
                elif self.benchmark_name == "libero_spatial":
                    task_id = 0
                else:
                    task_id = 0
                print(f"⚠️ 映射失败: {e} -> task_id={task_id} (回退)")
            
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
    def run(self, policy, n_video=0, do_tqdm=False, save_video_fn=None, run_env_names=None, render=False):
        """
        🎯 与原版RIPT对齐的评估接口：遍历所有任务，聚合结果
        
        Args:
            policy: 待评估的策略模型
            n_video: 每个任务保存的视频数量
            do_tqdm: 是否显示进度条
            save_video_fn: 视频保存回调函数
            run_env_names: 要运行的任务名列表（None则运行所有配置的任务）
            render: 是否渲染（用于视频保存）
            
        Returns:
            dict: 包含overall和per-task成功率的评估结果
        """
        from tqdm import tqdm
        import numpy as np
        
        # 确定要运行的任务列表
        if run_env_names is None:
            env_names = self.task_names  # 使用初始化时配置的任务列表
        else:
            env_names = run_env_names
            
        if not env_names:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("⚠️ 没有指定要评估的任务")
            return {}
        
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"🎯 开始评估 {len(env_names)} 个任务，每个任务 {self.rollouts_per_env} 个rollout")
        
        # 聚合所有任务的结果
        successes, per_env_any_success, rewards = [], [], []
        per_env_success_rates, per_env_rewards = {}, {}
        videos = {}
        
        # 遍历所有任务
        for env_name in tqdm(env_names, disable=not do_tqdm, desc="评估任务"):
            logger.debug(f"🔄 运行任务: {env_name}")
            any_success = False
            env_succs, env_rews, env_video = [], [], []
            
            # 获取该任务的基准初始状态池
            task_init_states = None
            if hasattr(self, 'benchmark') and self.benchmark is not None:
                try:
                    # 从任务名映射到task_id
                    if hasattr(self.benchmark, 'get_task_names'):
                        task_names = self.benchmark.get_task_names()
                        if env_name in task_names:
                            task_id = task_names.index(env_name)
                            task_init_states = self.benchmark.get_task_init_states(task_id)
                            logger.debug(f"  ✅ 获取到 {len(task_init_states)} 个基准初始状态 (task_id={task_id})")
                except Exception as e:
                    logger.debug(f"  ⚠️ 获取基准状态失败: {e}")
                    task_init_states = None
            
            # 运行该任务的rollouts，使用正确的eval_loop_num逻辑
            eval_loop_num = (self.rollouts_per_env + self.num_parallel_envs - 1) // self.num_parallel_envs
            logger.debug(f"  📊 评估轮数: {eval_loop_num} (总rollout={self.rollouts_per_env}, 并行={self.num_parallel_envs})")
            
            rollout_count = 0
            for loop_idx in range(eval_loop_num):
                # 计算这一轮要用的初始状态索引
                start_idx = loop_idx * self.num_parallel_envs
                end_idx = min(start_idx + self.num_parallel_envs, self.rollouts_per_env)
                batch_size = end_idx - start_idx
                
                if batch_size <= 0:
                    break
                
                # 准备这一轮的初始状态
                batch_init_states = None
                if task_init_states is not None:
                    indices = np.arange(start_idx, end_idx) % len(task_init_states)
                    batch_init_states = [task_init_states[i] for i in indices]
                    logger.debug(f"    批次 {loop_idx+1}: 使用基准状态索引 {indices}")
                
                # 运行这一批rollout
                try:
                    rollouts = self.run_policy_in_env(
                        env_name, 
                        all_init_states=batch_init_states,
                        debug_save_video=render and (rollout_count < n_video)
                    )
                    
                    # 收集结果，确保不超过rollouts_per_env总数
                    batch_count = 0
                    for success, total_reward, episode in rollouts:
                        if rollout_count >= self.rollouts_per_env:
                            break  # 确保总数不超过rollouts_per_env
                            
                        any_success = any_success or success
                        successes.append(success)
                        env_succs.append(success)
                        env_rews.append(total_reward)
                        rewards.append(total_reward)
                        
                        # 视频保存
                        if rollout_count < n_video and render and episode and 'render' in episode:
                            if save_video_fn is not None:
                                video_hwc = np.array(episode['render'])
                                video_chw = video_hwc.transpose((0, 3, 1, 2))
                                save_video_fn(video_chw, env_name, rollout_count)
                            else:
                                env_video.extend(episode['render'])
                        
                        rollout_count += 1
                        batch_count += 1
                        
                    logger.debug(f"    批次 {loop_idx+1}: 完成 {batch_count} 个rollout")
                    
                except Exception as e:
                    logger.debug(f"    ❌ 批次 {loop_idx+1} 执行失败: {e}")
                    # 填充失败结果
                    for _ in range(batch_size):
                        if rollout_count >= self.rollouts_per_env:
                            break
                        successes.append(False)
                        env_succs.append(False)
                        env_rews.append(0.0)
                        rewards.append(0.0)
                        rollout_count += 1
            
            # 计算该任务的统计
            per_env_success_rates[env_name] = np.mean(env_succs) if env_succs else 0.0
            per_env_rewards[env_name] = np.mean(env_rews) if env_rews else 0.0
            per_env_any_success.append(any_success)
            
            # 关键信息：每任务成功率（控制台可见）
            logger.info(f"任务 {env_name}: {per_env_success_rates[env_name]:.2%} ({sum(env_succs)}/{len(env_succs)})")
            
            # 处理视频
            if len(env_video) > 0:
                try:
                    import wandb
                    video_hwc = np.array(env_video)
                    video_chw = video_hwc.transpose((0, 3, 1, 2))
                    videos[env_name] = wandb.Video(video_chw, fps=10)
                except:
                    pass  # 忽略wandb相关错误
        
        # 构建输出结构，与原版ript对齐
        output = {}
        output['rollout'] = {
            'overall_success_rate': np.mean(successes) if successes else 0.0,
            'overall_average_reward': np.mean(rewards) if rewards else 0.0,
            'environments_solved': int(np.sum(per_env_any_success)),
            'rollout_count': len(successes),
        }
        output['rollout_success_rate'] = per_env_success_rates.copy()
        
        if len(videos) > 0:
            output['rollout_videos'] = videos
            
        # 关键信息：总体成功率（控制台可见）
        logger.info(f"🎉 评估完成: 总体成功率 {output['rollout']['overall_success_rate']:.2%}")
        return output
        
    def run_policy_in_env(self, env_name, all_init_states=None, debug_save_video=None, created_env=None):
        """在环境中运行策略，生成轨迹 - 支持并行和串行环境"""
        save_video = debug_save_video if debug_save_video is not None else self.debug_save_video
        
        # 🔥 根据功能开关选择并行或串行模式
        if self.enable_parallel_envs and created_env is None:
            if self.rank == 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"🚀 使用并行环境模式 (num_parallel_envs={self.num_parallel_envs})")
            # 创建并行环境
            env, env_id, env_num = self.create_parallel_envs(env_name, all_init_states)
            created_env = (env, env_id, env_num)
        elif created_env is None:
            if self.rank == 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"🔄 使用串行环境模式")
            # 使用串行环境
            env, task_description = self.make_env(env_name)
            created_env = (env, env_name, 1)
        
        env, env_id, env_num = created_env
        
        if all_init_states is None:
            # 与 2_pi0_on_libero.py 对齐：不设置环境初始状态，只做普通 reset
            all_init_states = [None for _ in range(self.rollouts_per_env)]
            print(f"ℹ️ 未提供初始状态，将进行普通reset（不调用set_init_state）")
        
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
        
        episodes = []
        for i, target_init_state in enumerate(all_init_states):
            rollout_images = []
            
            # 重用传入的环境或创建新环境
            if hasattr(env, 'task_description'):
                task_description = env.task_description
            else:
                task_description = env_name
            
            # 🔥 关键修改：尝试设置初始状态
            if target_init_state is not None:
                try:
                    # 🔥 简化：使用原版RIPT的调用方式
                    # 确保状态是连续的numpy数组
                    if isinstance(target_init_state, np.ndarray):
                        target_init_state = np.ascontiguousarray(target_init_state.astype(np.float64))
                    
                    # 先reset，再set_init_state（与原版RIPT完全一致）
                    obs = env.reset()
                    obs = env.set_init_state(target_init_state)
                    print(f"✅ 串行模式：状态设置成功 {i}，状态维度: {target_init_state.shape if hasattr(target_init_state, 'shape') else 'N/A'}")
                except Exception as e:
                    print(f"⚠️ 设置初始状态失败: {e}，回退到随机reset")
                    obs = env.reset()
                    if is_vector_env and isinstance(obs, list):
                        actual_obs = obs[0]
                    else:
                        actual_obs = obs
                    # 🔥 保持92维状态：不使用8维截断，而是生成92维零状态
                    print(f"⚠️ set_init_state失败，使用92维零状态回退")
                    target_init_state = np.zeros(92, dtype=np.float64)
            else:
                # 如果没有指定初始状态，使用默认reset
                obs = env.reset()
                if is_vector_env and isinstance(obs, list):
                    actual_obs = obs[0]
                else:
                    actual_obs = obs
                # 🔥 保持92维状态：使用92维零状态而不是8维截断
                print(f"⚠️ 未指定初始状态，使用92维零状态")
                target_init_state = np.zeros(92, dtype=np.float64)
            
            # 如果是VectorEnv，取第一个环境的观测
            if is_vector_env and isinstance(obs, list):
                obs = obs[0]
            
            # 计算初始状态哈希
            init_hash = self._compute_state_hash(target_init_state)
            
            # 热机步骤
            if not getattr(self, "disable_warmup", False):
                dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])
                for _ in range(20):
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
                initial_img = self.to_hwc_hmirror(obs["agentview_image"])
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
                        
                        # 选择动作 - 强制要求YAML中配置CFG参数
                        cfg_scale = None
                        
                        # 🔥 读取YAML配置（支持对象模式和dict模式）
                        if self.config:
                            # 对象模式
                            if hasattr(self.config, 'algo') and hasattr(self.config.algo, 'collection_cfg_scale'):
                                cfg_scale = self.config.algo.collection_cfg_scale
                            # dict模式
                            elif isinstance(self.config, dict):
                                cfg_scale = self.config.get('algo', {}).get('collection_cfg_scale', None)
                        
                        # 🔥 如果YAML没写，直接抛错，不再回退默认值
                        if cfg_scale is None:
                            raise ValueError(
                                "❌ 未在YAML的algo.collection_cfg_scale中找到CFG scale配置！\n"
                                "请在配置文件中添加：\n"
                                "algo:\n"
                                "  collection_cfg_scale: 1.1  # 或其他值"
                            )
                        
                        # 🔧 （可选）警告policy内置值被忽略
                        if hasattr(self, 'policy') and hasattr(self.policy, 'default_cfg_scale'):
                            policy_cfg = getattr(self.policy, 'default_cfg_scale', None)
                            if policy_cfg is not None:
                                print(f"⚠️ policy.default_cfg_scale={policy_cfg} 被忽略，使用YAML中的 {cfg_scale}")
                        
                        # 确认使用的CFG scale
                        print(f"✅ 使用CFG Scale: {cfg_scale}")
                        
                        raw_action = self.policy.select_action(pi0_observation, cfg_scale=cfg_scale)
                        action = raw_action[0, :, :7]  # shape: (50, 7)
                        
                        # 转换为numpy并处理维度
                        if isinstance(action, torch.Tensor):
                            action_after_cpu = action.cpu().numpy()
                        else:
                            action_after_cpu = action
                        
                        # 反归一化动作
                        action_buffer = action_after_cpu * (self.action_std + 1e-6) + self.action_mean
                        
                        # 🔥 关键修复：窗口一致的残差基准
                        # 获取当前未归一化状态用于偏移（整个50步序列用同一个基准）
                        import robosuite.utils.transform_utils as T
                        unnorm_state = np.concatenate([
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ], dtype=np.float32)
                        
                        # 对整个动作序列应用同一个状态偏移（前6维：位置+旋转）
                        action_buffer[:, :6] += unnorm_state[None, :6]
                        
                        # 重置动作索引（注意：在整个序列执行期间，不再更新基准）
                        action_index = 0
                    
                    # 从动作队列中取出当前动作执行
                    current_action = action_buffer[action_index, :7]
                    
                    # 🔧 动作裁剪：确保动作在合法范围内
                    if hasattr(env, 'action_space'):
                        action_space = env.action_space
                        if is_vector_env and hasattr(action_space, 'spaces') and len(action_space.spaces) > 0:
                            # VectorEnv使用第一个环境的动作空间
                            sub_space = action_space.spaces[0]
                            current_action = np.clip(current_action, sub_space.low, sub_space.high)
                        elif hasattr(action_space, 'low') and hasattr(action_space, 'high'):
                            # 单环境动作空间
                            current_action = np.clip(current_action, action_space.low, action_space.high)
                        else:
                            # 使用默认范围[-1, 1]
                            current_action = np.clip(current_action, -1, 1)
                    else:
                        # 没有action_space，使用默认范围[-1, 1]
                        current_action = np.clip(current_action, -1, 1)
                    
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
                        frame_img = self.to_hwc_hmirror(next_obs["agentview_image"])
                        rollout_images.append(frame_img)
                    
                    # 更新状态和计数器
                    obs = next_obs
                    step += 1
                    action_index += 1
                    
                    # 检查成功状态 - 优先使用info中的success，否则基于奖励判断（与4脚本保持一致）
                    # 修复：处理可能的NumPy数组success值
                    success_value = info.get("success", False)
                    if hasattr(success_value, 'shape') and len(getattr(success_value, 'shape', [])) > 0:
                        if hasattr(success_value, 'any'):
                            success_value = success_value.any()  # 任一元素为真则为真
                        else:
                            success_value = bool(success_value)  # 兜底转换
                    
                    if success_value:
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
                # 注意：环境关闭由run_policy_in_env的finally块统一处理，避免重复关闭
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
                "init_state": target_init_state,  # 🔥 添加初始状态
                "init_hash": init_hash,           # 🔥 添加初始状态哈希
                "temp_init_hash": init_hash,      # 🔥 添加临时哈希用于后续迁移
            }
            
            # 返回轨迹结果
            episodes.append(episode_data)
        
        # 返回所有episodes
        for i, episode_data in enumerate(episodes):
            # 计算最终成功率和奖励
            success = episode_data.get('rewards', [0])[-1] if episode_data.get('rewards') else False
            success = bool(success > 0) if isinstance(success, (int, float)) else bool(success)
            total_reward = sum(episode_data.get('rewards', []))
            yield (success, total_reward, episode_data)
    
    # 🔥 新增：并行环境支持 (仅在功能开关启用时可用)
    def create_env(self, env_name: str):
        """创建单个环境（用于并行环境）"""
        if not self.enable_parallel_envs:
            return self.make_env(env_name)  # 简单模式：直接返回单个环境
            
        env, task_description = self.make_env(env_name)
        return env, env_name, 1
    
    def create_parallel_envs(self, env_name: str, all_init_states=None):
        """创建并行环境"""
        if not self.enable_parallel_envs or not VECTOR_ENV_AVAILABLE or self.num_parallel_envs <= 1:
            # 如果不支持并行环境或只需要1个环境，使用单个环境
            if self.rank == 0:
                print(f"使用单个环境 (num_parallel_envs={self.num_parallel_envs})")
            env, task_description = self.make_env(env_name)
            return env, env_name, 1
        
        # 检查是否启用真正的多进程并行
        if self.enable_true_parallel_envs:
            return self._create_true_parallel_envs(env_name, all_init_states)
        else:
            # 回退到单环境
            return self._create_single_env(env_name)
    
    def _create_single_env(self, env_name: str):
        """创建单个环境 - 简单直接"""
        if self.rank == 0:
            print(f"🔄 使用单环境模式")
        
        env, task_description = self.make_env(env_name) 
        return env, env_name, 1
    
    def _create_true_parallel_envs(self, env_name: str, all_init_states=None):
        """🆕 创建真正的多进程并行环境 - 使用独立环境工厂解决序列化问题"""
        import logging
        logger = logging.getLogger(__name__)
        
        if not TRUE_PARALLEL_AVAILABLE:
            if self.rank == 0:
                logger.warning(f"⚠️ 独立环境工厂不可用，回退到单环境模式")
            return self._create_single_env(env_name)
        

        
        # 🔥 重要：子进程不加载模型，模型只在主进程中！
        

        
        if self.rank == 0:
            logger.debug(f"🚀 尝试创建真正的多进程并行环境:")

        try:
            # 🔑 关键：使用独立环境工厂，避免序列化self对象
            # 🔥 关键修复：使用固定初始状态ID确保并行环境同步

            # 从配置中读取同步设置
            sync_config = getattr(self.config, 'features', {}).get('parallel_env_sync', {})
            sync_enabled = sync_config.get('enabled', True)
            fixed_init_state_id = sync_config.get('fixed_init_state_id', 0) if sync_enabled else None

            # 🔥 修复：为每个worker创建带有不同worker_idx的工厂
            env_factories = []
            for worker_idx in range(self.num_parallel_envs):
                worker_env_factory = create_env_factory(
                    benchmark_name=self.benchmark_name,
                    env_name=env_name,
                    task_id=None,  # 自动推断
                    fixed_init_state_id=fixed_init_state_id,  # 🔥 新增：固定初始状态ID
                    init_states_array=all_init_states,  # 🔥 传递初始状态数组
                    worker_idx=worker_idx  # 🔥 每个worker有不同的索引
                )
                env_factories.append(worker_env_factory)
            
            if self.rank == 0:
                logger.debug(f"🔧 创建 {self.num_parallel_envs} 个独立并行环境...")
                if sync_enabled and fixed_init_state_id is not None:
                    if fixed_init_state_id == -1:
                        logger.debug("🎲 启用智能随机模式，每次重置随机选择初始状态")
                    else:
                        logger.debug(f"🔒 启用同步模式，固定初始状态ID: {fixed_init_state_id}")
                else:
                    logger.debug("🎲 使用完全随机初始状态模式")

            # 设置multiprocessing启动方法
            if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                multiprocessing.set_start_method('spawn', force=True)

            # 🔥 设置静默环境变量避免子进程重复输出
            prev_silent = os.environ.get("PI0_SILENT_IMPORT")
            os.environ["PI0_SILENT_IMPORT"] = "1"
            
            try:
                # 创建SubprocVectorEnv
                parallel_env = SubprocVectorEnv(env_factories)
            finally:
                # 恢复原始环境变量设置
                if prev_silent is None:
                    os.environ.pop("PI0_SILENT_IMPORT", None)
                else:
                    os.environ["PI0_SILENT_IMPORT"] = prev_silent
            
            # 🔍 验证并行环境初始状态同步性
            try:
                test_obs = parallel_env.reset()
                if self.rank == 0:
                    if isinstance(test_obs, list):
                        logger.info(f"✅ 创建并行环境: {len(test_obs)}个worker")

                        # 🔥 新增：验证初始状态同步性
                        verify_sync = sync_config.get('verify_sync', True)
                        if verify_sync:
                            sync_verified = self._verify_parallel_env_sync(test_obs)
                            if sync_verified:
                                logger.debug("✅ 并行环境初始状态同步验证通过")
                            else:
                                logger.warning("⚠️ 并行环境初始状态可能不同步")
                                if sync_enabled:
                                    logger.debug("   建议检查SyncedInitStateWrapper是否正常工作")
                        else:
                            logger.debug("ℹ️ 跳过同步验证（verify_sync=false）")
                    else:
                        logger.debug(f"✅ SubprocVectorEnv已创建，reset返回类型: {type(test_obs)}")
            except Exception as e:
                if self.rank == 0:
                    logger.warning(f"⚠️ SubprocVectorEnv.reset() 调用异常: {e}")
            
            if self.rank == 0:
                logger.debug(f"   🔄 {self.num_parallel_envs} 个独立子进程")
                logger.debug(f"   🧠 每个子进程无模型，总内存节省 ~{(self.num_parallel_envs-1)*3.5:.1f}GB")
                logger.debug(f"   ⚡ 真正的并行执行，性能提升 ~{self.num_parallel_envs}x")
            
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
            # 始终生成 env_num 个索引，环形取模，避免空索引
            if isinstance(all_init_states, list):
                total = len(all_init_states)
                if total == 0:
                    current_init_states = None
                    indices = []
                else:
                    indices = (np.arange(env_num) + start_idx) % total
                    current_init_states = [all_init_states[i] for i in indices]
            else:
                total = len(all_init_states)
                if total == 0:
                    current_init_states = None
                    indices = []
                else:
                    indices = (np.arange(env_num) + start_idx) % total
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
                if self.rank == 0:
                    print(f"🔧 并行状态设置配置: use_parallel_init_state = {use_parallel_init}")
        except Exception as e:
            use_parallel_init = False
            if self.rank == 0:
                print(f"⚠️ 读取并行状态配置失败: {e}")
                print(f"   config存在: {getattr(self, 'config', None) is not None}")
                if getattr(self, 'config', None):
                    print(f"   features存在: {hasattr(self.config, 'features')}")
                    if hasattr(self.config, 'features'):
                        print(f"   features内容: {self.config.features}")

        if self.rank == 0:
            print(f"🔧 最终决定: use_parallel_init = {use_parallel_init}")
        if use_parallel_init and init_states is not None:
            try:
                # 🔥 修复：确保状态格式正确
                # 🔥 直接处理状态，确保维度正确
                if isinstance(init_states, list) and len(init_states) > 0:
                    # 确保有足够的状态给每个环境
                    if len(init_states) < env_num:
                        # 重复状态以填满所有环境
                        processed_states = []
                        for i in range(env_num):
                            state_idx = i % len(init_states)
                            processed_states.append(init_states[state_idx])
                    else:
                        processed_states = init_states[:env_num]
                elif isinstance(init_states, np.ndarray):
                    # 如果是numpy数组，确保形状正确
                    if init_states.ndim == 1:
                        # 单个状态，复制给所有环境
                        processed_states = [init_states] * env_num
                    elif init_states.ndim == 2:
                        # 多个状态
                        if len(init_states) < env_num:
                            # 重复状态
                            processed_states = []
                            for i in range(env_num):
                                state_idx = i % len(init_states)
                                processed_states.append(init_states[state_idx])
                        else:
                            processed_states = init_states[:env_num]
                    else:
                        # 3D或更高维，取第一个状态并复制
                        first_state = init_states[0, 0] if init_states.ndim == 3 else init_states[0]
                        processed_states = [first_state] * env_num
                else:
                    processed_states = None
                
                if self.rank == 0 and processed_states is not None:
                    print(f"🔧 处理后的并行状态: 形状={np.array(processed_states).shape}, 类型={np.array(processed_states).dtype}")

                # 🔥 修复：确保状态数据能正确序列化到子进程
                if processed_states is not None and len(processed_states) > 0:
                    # 🔥 修复：使用原始RIPT的正确方式 - 通过reset方法传递初始状态
                    # 确保状态数据格式正确（numpy数组格式，与原始RIPT保持一致）
                    if isinstance(processed_states, list):
                        # 转换为numpy数组，保持[env_num, state_dim]格式
                        init_states_array = np.array(processed_states, dtype=np.float64)
                    else:
                        init_states_array = processed_states
                    
                    # 🔥 关键修复：使用reset方法而不是set_init_state方法
                    obs_any = env.reset(init_states=init_states_array)
                    if self.rank == 0:
                        print(f"✅ 并行状态设置成功，状态数量: {len(init_states_array)}")
                else:
                    # 没有初始状态时，使用普通reset
                    obs_any = env.reset()
            except Exception as e:
                if self.rank == 0:
                    print(f"⚠️ 并行状态设置失败: {e}")
                    print(f"   状态类型: {type(init_states)}")
                    if hasattr(init_states, 'shape'):
                        print(f"   状态形状: {init_states.shape}")
                    print(f"   状态数据类型: {init_states.dtype if hasattr(init_states, 'dtype') else 'N/A'}")
                # 重新reset环境
                obs_any = env.reset()
        else:
            if self.rank == 0:
                if init_states is not None:
                    print("ℹ️ 并行模式下跳过状态设置，使用随机初始化（避免MuJoCo状态格式不兼容）")
                else:
                    print("ℹ️ 并行模式下使用随机初始化")
        obs_list = self._ensure_list_of_dict_obs(obs_any, env_num)
        
        if self.rank == 0:
            print(f"🔧 初始化 {len(obs_list)} 个并行环境")
        
        # 对每个环境进行热身
        if not getattr(self, "disable_warmup", False):
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
                    initial_img = self.to_hwc_hmirror(obs_list[i]["agentview_image"])
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
                    
                    # 处理动作与状态偏移
                    action_buffer = batch_actions[batch_idx]
                    
                    # 🔥 关键修复：窗口一致的残差基准
                    # 获取推理时刻的状态偏移（整个50步动作序列使用同一个基准）
                    import robosuite.utils.transform_utils as T
                    unnorm_state = np.concatenate([
                        obs["robot0_eef_pos"],
                        T.quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ], dtype=np.float32)
                    
                    # 对整个动作序列应用同一个状态偏移（前6维：位置+旋转）
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
            
            # 🔧 动作裁剪（并行路径可配置禁用）
            disable_clip = False
            try:
                if self.config and hasattr(self.config, 'features'):
                    disable_clip = bool(getattr(self.config.features, 'disable_action_clipping', False))
                elif isinstance(self.config, dict):
                    disable_clip = bool(self.config.get('features', {}).get('disable_action_clipping', False))
            except Exception:
                disable_clip = False

            if not disable_clip:
                if hasattr(env, 'action_space'):
                    action_space = env.action_space
                    if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
                        # 单环境动作空间
                        action_low = action_space.low
                        action_high = action_space.high
                        actions_to_execute = [np.clip(action, action_low, action_high) for action in actions_to_execute]
                    elif hasattr(action_space, 'spaces'):
                        # 向量环境，每个子环境有独立的动作空间
                        clipped_actions = []
                        for i, action in enumerate(actions_to_execute):
                            if i < len(action_space.spaces):
                                sub_space = action_space.spaces[i]
                                clipped_action = np.clip(action, sub_space.low, sub_space.high)
                            else:
                                # 使用默认范围[-1, 1]
                                clipped_action = np.clip(action, -1, 1)
                            clipped_actions.append(clipped_action)
                        actions_to_execute = clipped_actions
                    else:
                        # 使用默认范围[-1, 1]进行裁剪
                        actions_to_execute = [np.clip(action, -1, 1) for action in actions_to_execute]
                else:
                    # 没有action_space信息，使用默认范围[-1, 1]
                    actions_to_execute = [np.clip(action, -1, 1) for action in actions_to_execute]
            
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
                        frame_img = self.to_hwc_hmirror(obs_list[i]["agentview_image"])
                        episode['rollout_images'].append(frame_img)
                    except Exception as e:
                        if self.rank == 0:
                            print(f"⚠️ 收集图像帧失败 (环境{i}): {e}")
                
                # 修复：处理可能的NumPy数组success值
                success_value = infos[i].get("success", False)
                if hasattr(success_value, 'shape') and len(getattr(success_value, 'shape', [])) > 0:
                    if hasattr(success_value, 'any'):
                        success_value = success_value.any()  # 任一元素为真则为真
                    else:
                        success_value = bool(success_value)  # 兜底转换
                
                if success_value or episode['total_reward'] > 0.5:
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
                    # 优先使用真实任务描述（从子进程env上获取并在reset时缓存到 _vector_env_prompts）
                    if hasattr(self, '_vector_env_prompts') and i < len(self._vector_env_prompts):
                        task_label = self._vector_env_prompts[i]
                    else:
                        task_label = env_name
                    task_str = (task_label or "").replace(" ", "_")[:60] if isinstance(task_label, str) else f"parallel_env_{i}"
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
            # 单个观测直接推理 - 强制要求YAML中配置CFG参数
            cfg_scale = None
            if self.config:
                # 对象模式
                if hasattr(self.config, 'algo') and hasattr(self.config.algo, 'collection_cfg_scale'):
                    cfg_scale = self.config.algo.collection_cfg_scale
                # dict模式
                elif isinstance(self.config, dict):
                    cfg_scale = self.config.get('algo', {}).get('collection_cfg_scale', None)
            
            if cfg_scale is None:
                raise ValueError("❌ 未在YAML的algo.collection_cfg_scale中找到CFG scale配置！")
            
            print(f"✅ 使用CFG Scale (单观测): {cfg_scale}")
            raw_action = self.policy.select_action(batch_obs[0], cfg_scale=cfg_scale)
            action = raw_action[0, :, :7]  # (50, 7)
            
            if isinstance(action, torch.Tensor):
                action_after_cpu = action.cpu().numpy()
            else:
                action_after_cpu = action
            
            action_buffer = action_after_cpu * (self.action_std + 1e-6) + self.action_mean
            return [action_buffer]
        else:
            # 🚀 优化：尝试真正的批推理，失败时回退到循环推理
            try:
                batch_observation = self._stack_pi0_observations(batch_obs)
                
                # 🔥 强制要求YAML中配置CFG参数（批推理）
                cfg_scale = None
                if self.config:
                    # 对象模式
                    if hasattr(self.config, 'algo') and hasattr(self.config.algo, 'collection_cfg_scale'):
                        cfg_scale = self.config.algo.collection_cfg_scale
                    # dict模式
                    elif isinstance(self.config, dict):
                        cfg_scale = self.config.get('algo', {}).get('collection_cfg_scale', None)
                
                if cfg_scale is None:
                    raise ValueError("❌ 未在YAML的algo.collection_cfg_scale中找到CFG scale配置！")
                
                print(f"✅ 使用CFG Scale (批推理): {cfg_scale}")
                
                # 一次性批推理 - 这是核心优化点
                raw_actions = self.policy.select_action(batch_observation, cfg_scale=cfg_scale)
                # raw_actions shape: (B, T, 7)
                
                # 分解批量结果
                batch_actions = []
                for i in range(len(batch_obs)):
                    action = raw_actions[i, :, :7]  # (50, 7)
                    
                    if isinstance(action, torch.Tensor):
                        action_after_cpu = action.cpu().numpy()
                    else:
                        action_after_cpu = action
                    
                    action_buffer = action_after_cpu * (self.action_std + 1e-6) + self.action_mean
                    batch_actions.append(action_buffer)
                
                if self.rank == 0:
                    print(f"🚀 批推理成功：{len(batch_obs)} envs -> 1次GPU调用")
                return batch_actions
                
            except Exception as e:
                if self.rank == 0:
                    print(f"⚠️ 批推理失败，回退到循环推理: {e}")
                # 回退到原有的循环推理
            
            # 🔄 回退路径：循环推理（原有逻辑）
            batch_actions = []
            for pi0_obs in batch_obs:
                cfg_scale = getattr(self.config, 'collection_cfg_scale', None)
            if cfg_scale is None and self.config and hasattr(self.config, 'algo'):
                cfg_scale = getattr(self.config.algo, 'collection_cfg_scale', None)
            if cfg_scale is None:
                print(f"⚠️ 未找到collection_cfg_scale配置，请在YAML中设置")
                cfg_scale = 1.5  # 临时回退
                raw_action = self.policy.select_action(pi0_obs, cfg_scale=cfg_scale)
                action = raw_action[0, :, :7]
                
                if isinstance(action, torch.Tensor):
                    action_after_cpu = action.cpu().numpy()
                else:
                    action_after_cpu = action
                
                action_buffer = action_after_cpu * (self.action_std + 1e-6) + self.action_mean
                batch_actions.append(action_buffer)
            
            return batch_actions
    
    def _stack_pi0_observations(self, batch_obs):
        """将多个PI0观测堆叠成批量观测
        
        输入: [obs1, obs2, ...] 每个obs是单独的PI0观测字典
        输出: 批量观测字典，所有张量的batch维度堆叠
        """
        if not batch_obs:
            raise ValueError("batch_obs不能为空")
        
        # 获取第一个观测作为模板
        template_obs = batch_obs[0]
        batch_size = len(batch_obs)
        
        # 构建批量观测字典
        batched_observation = {}
        
        # 处理图像
        if "image" in template_obs:
            batched_observation["image"] = {}
            for img_key in template_obs["image"]:
                # 🔥 修复：每个obs["image"][img_key]已经是[1, C, H, W]，需要先squeeze再stack
                img_tensors = [obs["image"][img_key].squeeze(0) for obs in batch_obs]  # 去掉batch维度1
                batched_observation["image"][img_key] = torch.stack(img_tensors, dim=0)  # -> [B, C, H, W]
        
        # 处理状态  
        if "state" in template_obs:
            # 🔥 修复：每个obs["state"]已经是[1, state_dim]，需要先squeeze再stack
            state_tensors = [obs["state"].squeeze(0) for obs in batch_obs]  # 去掉batch维度1
            batched_observation["state"] = torch.stack(state_tensors, dim=0)  # -> [B, state_dim]
        
        # 处理提示文本
        if "prompt" in template_obs:
            # 合并所有提示文本到列表
            batched_observation["prompt"] = [obs["prompt"][0] if isinstance(obs["prompt"], list) else obs["prompt"] 
                                           for obs in batch_obs]
        
        # 处理语言tokens（如果存在）
        if "lang_tokens" in template_obs:
            # 🔥 修复：每个obs["lang_tokens"]可能已经有batch维度，需要squeeze
            lang_tokens_list = [obs["lang_tokens"].squeeze(0) if obs["lang_tokens"].dim() > 1 else obs["lang_tokens"] for obs in batch_obs]
            batched_observation["lang_tokens"] = torch.stack(lang_tokens_list, dim=0)
        
        if "lang_masks" in template_obs:
            # 🔥 修复：每个obs["lang_masks"]可能已经有batch维度，需要squeeze
            lang_masks_list = [obs["lang_masks"].squeeze(0) if obs["lang_masks"].dim() > 1 else obs["lang_masks"] for obs in batch_obs]
            batched_observation["lang_masks"] = torch.stack(lang_masks_list, dim=0)
        
        return batched_observation
    
    def _fallback_individual_inference(self, observations, env_name, prompts_for_obs=None):
        """回退到逐个推理"""
        batch_actions = []
        for idx, obs in enumerate(observations):
            prompt_text = None
            if prompts_for_obs is not None and idx < len(prompts_for_obs):
                prompt_text = prompts_for_obs[idx]
            pi0_obs = self.construct_pi0_observation(obs, prompt_text or env_name)
            cfg_scale = None
            if hasattr(self, 'policy') and hasattr(self.policy, 'default_cfg_scale'):
                cfg_scale = getattr(self.policy, 'default_cfg_scale', None)
            if cfg_scale is None:
                cfg_scale = getattr(self.config, 'collection_cfg_scale', None) if self.config else None
            if cfg_scale is None:
                cfg_scale = 1.5
            raw_action = self.policy.select_action(pi0_obs, cfg_scale=cfg_scale)
            action = raw_action[0, :, :7]
            
            if isinstance(action, torch.Tensor):
                action_after_cpu = action.cpu().numpy()
            else:
                action_after_cpu = action
            
            action_buffer = action_after_cpu * (self.action_std + 1e-6) + self.action_mean
            batch_actions.append(action_buffer)
        
        return batch_actions
    
    def run_policy_in_env_batch(self, env_name, init_states, batch_size=None):
        """
        批量执行多个episode，确保每个使用指定的初始状态
        
        Args:
            env_name: 环境名称
            init_states: 初始状态列表，长度应为batch_size
            batch_size: 批量大小，如果None则使用init_states的长度
        
        Returns:
            list: episode列表，长度为batch_size
        """
        if batch_size is None:
            batch_size = len(init_states) if init_states else 8
        
        # 确保init_states数量正确
        if init_states is not None and len(init_states) != batch_size:
            # 复制或截断到正确大小
            if len(init_states) < batch_size:
                init_states = init_states * (batch_size // len(init_states) + 1)
            init_states = init_states[:batch_size]
        
        # 优先使用并行环境（如果可用且配置允许）
        if self._can_use_parallel_envs(batch_size):
            return self._run_parallel_episodes_true(
                env=None,  # 将在内部创建
                policy=self.policy_wrapper,
                max_steps=self.max_steps,
                init_states=init_states,
                num_episodes=batch_size
            )
        else:
            # 回退到串行执行
            if self.rank == 0:
                print(f"📋 批量执行回退到串行模式（batch_size={batch_size}）")
            
            # 创建环境
            env = self._create_single_env(env_name)
            
            # 使用串行模式执行
            episodes = []
            for success, total_reward, episode_data in self._run_serial_episodes(
                env, env_name, init_states, save_video=False
            ):
                # 添加成功率和总奖励信息
                episode_data['success'] = success
                episode_data['total_reward'] = total_reward
                episodes.append(episode_data)
            
            # 清理环境
            try:
                env.close()
            except:
                pass
            
            return episodes

    def _verify_parallel_env_sync(self, obs_list):
        """
        验证并行环境的初始状态同步性

        Args:
            obs_list: 并行环境reset返回的观测列表

        Returns:
            bool: True表示同步，False表示不同步
        """
        if not isinstance(obs_list, list) or len(obs_list) < 2:
            return True  # 单环境或无效输入，认为同步

        try:
            # 提取每个环境的状态
            states = []
            hashes = []

            for i, obs in enumerate(obs_list):
                state = self._extract_state_from_obs(obs)
                state_hash = self._compute_state_hash(state)
                states.append(state)
                hashes.append(state_hash)
                print(f"   环境 {i}: 状态哈希 = {state_hash}")

            # 检查同步性
            unique_hashes = set(hashes)
            is_synced = len(unique_hashes) == 1

            if is_synced:
                print(f"   🔒 所有 {len(obs_list)} 个环境状态完全同步")
            else:
                print(f"   ⚠️ 发现 {len(unique_hashes)} 个不同状态，环境未完全同步")
                # 详细报告不同状态的分布
                for j, unique_hash in enumerate(unique_hashes):
                    indices = [i for i, h in enumerate(hashes) if h == unique_hash]
                    print(f"     状态 {j+1}: {unique_hash} (环境 {indices})")

            return is_synced

        except Exception as e:
            print(f"   ❌ 同步验证失败: {e}")
            return False

    def _can_use_parallel_envs(self, batch_size):
        """检查是否可以使用并行环境"""
        # 简化判断逻辑
        features_config = getattr(self.config, 'features', {})
        enable_parallel = features_config.get('enable_parallel_envs', False)
        enable_true_parallel = features_config.get('enable_true_parallel_envs', False)

        return enable_parallel and enable_true_parallel and batch_size > 1

    def _is_mujoco_state(self, state_data):
        """检查是否为MuJoCo状态向量（与原版RIPT对齐）"""
        try:
            if isinstance(state_data, dict):
                return False  # 观测字典不是MuJoCo状态

            if hasattr(state_data, 'shape'):
                # 检查是否为高维状态向量（MuJoCo状态通常>50维，包括92维）
                if len(state_data.shape) == 1 and state_data.shape[0] >= 50:
                    return True
                elif len(state_data.shape) == 2 and state_data.shape[-1] >= 50:
                    return True

            return False
        except:
            return False
    


    def _process_init_states_for_parallel(self, init_states, env_num):
        """处理初始状态以适配并行环境（修复MuJoCo格式问题）"""
        try:
            if init_states is None:
                return None
            if isinstance(init_states, (list, tuple)) and len(init_states) == 0:
                return None

            # 转换为numpy数组
            if isinstance(init_states, torch.Tensor):
                init_states = init_states.cpu().numpy()

            # 确保是numpy数组
            if not isinstance(init_states, np.ndarray):
                init_states = np.array(init_states)

            # 🔥 关键修复：确保数据类型为float64（MuJoCo期望）
            if init_states.dtype != np.float64:
                init_states = init_states.astype(np.float64)

            # 确保形状正确
            if init_states.ndim == 1:
                # 单个状态，复制给所有环境
                init_states = np.tile(init_states, (env_num, 1))
            elif init_states.ndim == 2:
                if init_states.shape[0] == 1:
                    # 单个状态，复制给所有环境
                    init_states = np.tile(init_states[0], (env_num, 1))
                elif init_states.shape[0] != env_num:
                    # 状态数量不匹配，使用第一个状态
                    init_states = np.tile(init_states[0], (env_num, 1))
            elif init_states.ndim == 3:
                # 🔥 修复：处理3D数组 (batch, seq, dim) -> 取第一个batch的第一个状态
                if init_states.shape[0] == 1:
                    # 只有一个batch，取第一个状态并复制给所有环境
                    first_state = init_states[0, 0]  # 取第一个batch的第一个状态
                    init_states = np.tile(first_state, (env_num, 1))
                else:
                    # 多个batch，取第一个batch的第一个状态
                    first_state = init_states[0, 0]
                    init_states = np.tile(first_state, (env_num, 1))

            print(f"🔧 处理后的并行状态: 形状={init_states.shape}, 类型={init_states.dtype}")
            return init_states

        except Exception as e:
            print(f"⚠️ 状态处理失败: {e}")
            return None
    
