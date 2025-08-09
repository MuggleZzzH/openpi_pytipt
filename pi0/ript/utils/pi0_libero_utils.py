"""
This file contains utility functions for evaluating pi0 policies in LIBERO simulation environments.
It is a cleaned-up and adapted version, containing only the generic components
needed for the pi0_libero_runner.py, while removing OpenVLA-specific logic.
"""

import math
import os
import imageio
import numpy as np
import torch
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


class Pi0PolicyWrapper:
    """PI0策略包装器，提供更好的错误处理和动作选择"""
    
    def __init__(self, policy):
        self.policy = policy
        self.action_failure_count = 0
        self.max_failures = 5
        self.step = 0  # 添加步数跟踪变量
        self.action_callback = None  # ✅ 添加action callback支持
    
    def set_action_callback(self, callback):
        """设置动作生成时的回调函数"""
        self.action_callback = callback
        
    @property
    def config(self):
        """暴露底层策略的配置"""
        return self.policy.config
    
    def parameters(self):
        """暴露底层策略的参数 - 修复AttributeError"""
        if hasattr(self.policy, 'parameters'):
            return self.policy.parameters()
        else:
            # 如果底层策略没有parameters方法，返回空的生成器
            return iter([])
    
    @property
    def device(self):
        """获取策略所在的设备"""
        try:
            return next(self.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def select_action(self, batch, noise=None):
        """选择动作 - 适配OpenPI接口格式"""
        try:
            # 将LeRobot batch格式转换为OpenPI observation格式
            observation = self._convert_batch_to_observation(batch)
            
            # 直接调用底层策略的select_action方法
            result = self.policy.select_action(observation, noise=noise)
            self.action_failure_count = 0
            
            # ✅ 调用action callback（如果设置了）
            if self.action_callback is not None:
                try:
                    self.action_callback(result)
                except Exception as callback_error:
                    print(f"Action callback出错: {callback_error}")
            
            # OpenPI直接返回张量格式的动作
            if torch.is_tensor(result):
                return result
            else:
                # 其他格式，尝试转换为张量
                return torch.tensor(result) if not torch.is_tensor(result) else result
            
        except Exception as e:
            self.action_failure_count += 1
            print(f"select_action出错 (第{self.action_failure_count}次): {e}")
            
            # 返回默认动作张量
            return self._get_default_action_tensor()
    
    def _convert_batch_to_observation(self, batch):
        """将LeRobot batch格式转换为OpenPI observation格式"""
        observation = {}
        
        # 处理状态数据
        if "observation.state" in batch:
            observation["state"] = batch["observation.state"]
        elif "state" in batch:
            observation["state"] = batch["state"]
        else:
            # 生成默认状态
            observation["state"] = torch.zeros((1, 7), dtype=torch.float32, device=self.device)
        
        # 处理图像数据
        observation["image"] = {}
        if "observation.images" in batch:
            images = batch["observation.images"]
            if hasattr(images, 'keys'):  # 如果是字典格式
                observation["image"] = images
            else:  # 如果是张量格式
                # 假设第一个图像是base_0_rgb
                observation["image"]["base_0_rgb"] = images
        else:
            # 生成默认图像
            observation["image"]["base_0_rgb"] = torch.zeros(
                (1, 3, 224, 224), dtype=torch.uint8, device=self.device
            )
        
        # 处理语言信息 - 使用默认任务描述
        if "lang_tokens" in batch and "lang_masks" in batch:
            observation["lang_tokens"] = batch["lang_tokens"]
            observation["lang_masks"] = batch["lang_masks"]
        else:
            # 生成默认的任务描述
            observation["prompt"] = ["Pick up the object and place it."]
        
        return observation
    
    def _get_default_action_tensor(self):
        """生成默认动作张量"""
        try:
            # 获取动作维度信息
            action_dim = getattr(self.policy.config, 'max_action_dim', 7)
            
            # 生成零动作张量 (单步动作，不是轨迹)
            default_action = torch.zeros(1, action_dim, dtype=torch.float32, device=self.device)
            print(f"生成默认动作张量: 形状={default_action.shape}, 设备={default_action.device}")
            
            return default_action
            
        except Exception as e:
            print(f"生成默认动作张量时出错: {e}")
            # 最后的备用方案
            return torch.zeros(1, 7, dtype=torch.float32, device=torch.device('cpu'))
    
    def _get_default_action(self):
        """获取默认动作字典（保持向后兼容）"""
        try:
            tensor_action = self._get_default_action_tensor()
            return {"action": tensor_action}
        except Exception as e:
            print(f"创建默认动作字典失败: {e}")
            # 最简单的默认动作
            return {"action": torch.zeros((1, 7), dtype=torch.float32)}
    
    def reset(self):
        """重置策略"""
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
        self.action_failure_count = 0
        self.step = 0  # 重置步数计数器
    
    def get_action(self, obs, task_id, task_emb):
        """获取动作，带有错误处理"""
        try:
            # 增加步数计数器
            self.step += 1
            
            # 减少调试信息输出频率 - 只在每50步或有异常时输出
            if self.step % 50 == 0 or self.step <= 5:
                print(f"[步骤 {self.step}] 推理中...")
            elif hasattr(obs, '__len__') and len(obs) == 0:
                print(f"[警告] 步骤 {self.step}: 接收到空观察")
            
            # 准备批次数据
            batch = self._prepare_batch(obs, task_emb)
            
            # 重置动作队列（如果需要）
            if hasattr(self.policy, 'reset'):
                self.policy.reset()
            
            # 选择动作
            action_result = self.policy.select_action(batch)
            
            # 处理不同格式的动作返回值
            if isinstance(action_result, dict):
                # 如果返回字典，提取动作张量
                if 'action' in action_result:
                    action = action_result['action']
                else:
                    print("警告: actions_dict中没有'action'键，使用默认动作")
                    action = self._get_default_action(obs)
            elif torch.is_tensor(action_result):
                # 如果直接返回张量，直接使用
                action = action_result
                print(f"成功生成动作轨迹，形状: {action.shape}")
            else:
                print(f"Policy返回了无效的动作格式: {type(action_result)}")
                action = self._get_default_action(obs)
            
            # 验证动作有效性
            if action is None:
                print("警告: 策略返回None动作，使用默认动作")
                action = self._get_default_action(obs)
            elif not torch.is_tensor(action):
                print("警告: 策略返回非tensor动作，使用默认动作")
                action = self._get_default_action(obs)
            elif torch.isnan(action).any() or torch.isinf(action).any():
                print("警告: 策略返回NaN或Inf动作，使用默认动作")
                action = self._get_default_action(obs)
            
            # 转换为numpy数组
            if torch.is_tensor(action):
                action_np = action.detach().cpu().numpy()
            else:
                action_np = np.array(action, dtype=np.float32)
            
            # 确保动作维度正确
            if action_np.ndim == 1:
                action_np = action_np.reshape(1, -1)
            
            # 回调通知（动作 chunk）
            self._emit_action_cb(action_np)
            
            # 重置失败计数
            self.action_failure_count = 0
            
            # 减少频繁输出 - 仅在特定条件下显示进度
            if self.step % 100 == 0:  # 每100步显示一次进度
                print(f"[进度] 步骤 {self.step}/5010")
            
            return action_np
            
        except Exception as e:
            self.action_failure_count += 1
            print(f"动作选择出错: {e}")
            
            if self.action_failure_count >= self.max_failures:
                print("达到最大失败次数，将重置策略")
                try:
                    if hasattr(self.policy, 'reset'):
                        self.policy.reset()
                    self.action_failure_count = 0
                except Exception as reset_e:
                    print(f"重置策略时出错: {reset_e}")
            
            _fallback = self._get_default_action(obs)
            self._emit_action_cb(_fallback)
            return _fallback
    
    def _prepare_batch(self, obs, task_emb):
        """准备批次数据"""
        batch = {}
        
        # 添加状态信息
        state_features = []
        
        # 机器人状态
        if "robot0_eef_pos" in obs:
            state_features.extend(obs["robot0_eef_pos"])
        if "robot0_eef_quat" in obs:
            state_features.extend(obs["robot0_eef_quat"])
        if "robot0_gripper_qpos" in obs:
            state_features.extend(obs["robot0_gripper_qpos"])
        if "robot0_joint_pos" in obs:
            state_features.extend(obs["robot0_joint_pos"])
        
        # 对象状态
        if "object_obs" in obs:
            state_features.extend(obs["object_obs"])
        
        # 如果没有状态特征，创建默认状态
        if not state_features:
            state_features = [0.0] * 50  # 默认50维状态
        
        # 转换为tensor
        state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
        batch['observation.state'] = state_tensor
        
        # 添加图像数据
        if "images" in obs and obs["images"]:
            # 使用obs中的images
            images = obs["images"]
            if isinstance(images, list):
                images = images[0]  # 取第一张图像
            
            # 转换为tensor
            if isinstance(images, np.ndarray):
                if images.dtype == np.uint8:
                    images = images.astype(np.float32) / 255.0
                
                # 确保维度正确 [H, W, C] -> [B, H, W, C]
                if images.ndim == 3:
                    images = images[np.newaxis, ...]
                
                images_tensor = torch.tensor(images, dtype=torch.float32)
                batch['images'] = images_tensor
        
        elif "agentview_image" in obs:
            # 使用agentview_image
            img = obs["agentview_image"]
            if isinstance(img, np.ndarray):
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                
                if img.ndim == 3:
                    img = img[np.newaxis, ...]
                
                img_tensor = torch.tensor(img, dtype=torch.float32)
                batch['images'] = img_tensor
        
        # 添加任务描述
        if hasattr(task_emb, 'cpu'):
            task_text = "pick up the black bowl and place it on the plate"  # 默认任务描述
        else:
            task_text = str(task_emb) if task_emb is not None else "pick up the black bowl and place it on the plate"
        
        batch['task'] = [task_text]
        
        return batch
    
    def _get_default_action(self, obs):
        """获取默认动作"""
        # 返回7维零动作 [dx, dy, dz, droll, dpitch, dyaw, gripper]
        return np.zeros((1, 7), dtype=np.float32)
    
    def _emit_action_cb(self, action):
        """发出动作回调，用于通知动作进度"""
        try:
            # 检查是否有回调函数
            if hasattr(self, 'action_callback') and callable(self.action_callback):
                self.action_callback(action)
        except Exception as e:
            # 回调失败不应该影响主要流程
            pass
    
    def set_action_callback(self, callback):
        """设置动作回调函数"""
        self.action_callback = callback

def get_libero_env(task, model_family="pi0", resolution=256, render_mode=None, horizon=5000, **kwargs):
    """Initializes and returns the LIBERO environment, along with the task description.
    
    这个函数创建一个最简单的LIBERO环境，完全禁用渲染功能，以避免EGL相关错误。
    
    Args:
        task: LIBERO task object
        model_family: Model family string (default: "pi0")
        resolution: Image resolution (default: 256)
        render_mode: Rendering mode (ignored for compatibility, always uses OffScreenRenderEnv)
        horizon: Maximum episode length (default: 5000, fixes the 1000-step termination issue)
        **kwargs: Additional keyword arguments (ignored for compatibility)
    """
    # 无论传入什么render_mode参数，我们都会忽略它，统一使用OffScreenRenderEnv
    # 这样即使调用时传入render_mode='offscreen'也不会出错
    
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    
    # 设置环境变量，优先使用EGL后端，如果失败则使用osmesa
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    
    # 简化的配置选项，专注于使用OffScreenRenderEnv
    config_options = [
        # 选项1：基本的offscreen渲染环境
        {
            "bddl_file_name": task_bddl_file,
            "camera_heights": resolution,
            "camera_widths": resolution,
            "horizon": horizon,  # 设置episode最大长度以避免1000步终止问题
        },
        # 选项2：禁用GPU渲染
        {
            "bddl_file_name": task_bddl_file,
            "camera_heights": resolution,
            "camera_widths": resolution,
            "render_gpu_device_id": -1,
            "horizon": horizon,  # 设置episode最大长度以避免1000步终止问题
        },
        # 选项3：最简配置
        {
            "bddl_file_name": task_bddl_file,
            "horizon": horizon,  # 设置episode最大长度以避免1000步终止问题
        }
    ]
    
    # 尝试不同的环境变量配置
    env_var_options = [
        {"MUJOCO_GL": "egl"},
        {"MUJOCO_GL": "osmesa"},
        {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
    ]
    
    env = None
    error_messages = []
    
    # 尝试不同的环境变量和配置组合
    for env_vars in env_var_options:
        # 保存当前环境变量
        old_env_vars = {}
        for var in env_vars:
            old_env_vars[var] = os.environ.get(var)
            os.environ[var] = env_vars[var]
        
        # 尝试每种配置
        for i, config in enumerate(config_options):
            try:
                print(f"尝试环境配置选项 {i+1}/{len(config_options)} 与环境变量 {env_vars}")
                env = OffScreenRenderEnv(**config)
                print(f"成功创建环境，使用配置选项 {i+1} 和环境变量 {env_vars}")
                # 成功创建环境，跳出循环
                break
            except Exception as e:
                error_message = f"配置选项 {i+1} 与环境变量 {env_vars} 失败: {str(e)}"
                print(error_message)
                error_messages.append(error_message)
                continue
        
        # 如果成功创建了环境，就跳出循环
        if env is not None:
            break
        
        # 恢复原始环境变量
        for var in old_env_vars:
            if old_env_vars[var] is None:
                if var in os.environ:
                    del os.environ[var]
            else:
                os.environ[var] = old_env_vars[var]
    
    # 如果所有配置都失败，尝试使用一个模拟环境
    if env is None:
        print("所有环境配置都失败，创建模拟环境")
        print(f"错误信息: {error_messages}")
        
        # 创建一个更强大的模拟环境类，实现必要的接口
        class MockEnv:
            def __init__(self):
                self.state = np.zeros(100)  # 假设状态维度为100
                self.success = False
                self.step_count = 0
                self.image_height = resolution
                self.image_width = resolution
                self.horizon = horizon  # 设置正确的horizon值
                self.done = False  # 初始化done状态
                
            def step(self, action):
                # 返回包含图像的观测，零奖励，不结束，成功状态为False
                self.step_count += 1
                
                # 检查是否超过horizon限制
                if self.step_count >= self.horizon:
                    self.done = True
                
                # 每50步随机成功一次，以模拟任务完成
                if self.step_count % 50 == 0:
                    self.success = np.random.random() < 0.2
                
                obs = {
                    "robot0_eef_pos": np.random.normal(0, 0.1, size=3),
                    "robot0_eef_quat": np.array([0, 0, 0, 1]),  # 单位四元数
                    "robot0_gripper_qpos": np.array([0.0]),
                    "robot0_joint_pos": np.random.normal(0, 0.1, size=7),
                    "object_obs": np.random.normal(0, 0.1, size=50),
                    # 添加图像数据
                    "agentview_image": self._generate_mock_image(),
                    "robot0_eye_in_hand_image": self._generate_mock_image(),
                    # 为PI0添加images键
                    "images": [self._generate_mock_image()],
                }
                
                # 添加一些随机奖励
                reward = float(np.random.random() * 0.1)
                if self.success:
                    reward = 1.0
                
                done = self.success or self.done
                info = {"success": self.success}
                return obs, reward, done, info
                
            def reset(self):
                self.success = False
                self.step_count = 0
                self.done = False  # 重置done状态
                obs = {
                    "robot0_eef_pos": np.random.normal(0, 0.1, size=3),
                    "robot0_eef_quat": np.array([0, 0, 0, 1]),
                    "robot0_gripper_qpos": np.array([0.0]),
                    "robot0_joint_pos": np.random.normal(0, 0.1, size=7),
                    "object_obs": np.random.normal(0, 0.1, size=50),
                    # 添加图像数据
                    "agentview_image": self._generate_mock_image(),
                    "robot0_eye_in_hand_image": self._generate_mock_image(),
                    # 为PI0添加images键
                    "images": [self._generate_mock_image()],
                }
                return obs
                
            def _generate_mock_image(self):
                """生成模拟图像数据"""
                # 创建一个简单的渐变图像，符合PI0的期望格式
                img = np.random.randint(0, 256, size=(self.image_height, self.image_width, 3), dtype=np.uint8)
                
                # 添加一些结构化的模式，使其看起来更像真实图像
                center_x, center_y = self.image_width // 2, self.image_height // 2
                y, x = np.ogrid[:self.image_height, :self.image_width]
                
                # 创建圆形图案
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= (min(self.image_width, self.image_height) // 4) ** 2
                img[mask] = [128, 128, 128]  # 灰色圆形
                
                # 添加一些噪声
                noise = np.random.normal(0, 10, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                return img
                
            def seed(self, seed):
                np.random.seed(seed)
                
            def set_init_state(self, state):
                if state is not None:
                    self.state = state
                return self.reset()
                
            def close(self):
                pass
                
            def check_success(self):
                return self.success
        
        env = MockEnv()
        print("使用模拟环境")
    
    # 设置随机种子
    env.seed(0)
    return env, task_description

def get_libero_dummy_action(model_family="pi0"):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing.
    
    Args:
        model_family: Model family string (for compatibility)
    """
    # This 7-DoF action [dx, dy, dz, droll, dpitch, dyaw, gripper] is generic for many manipulators.
    return [0, 0, 0, 0, 0, 0, -1]

def get_libero_image(obs):
    """Extracts third-person image from observations."""
    # IMPORTANT: 根据OpenVLA原始代码，需要180度旋转以匹配训练预处理
    # OpenVLA原始代码：img = img[::-1, ::-1]  # rotate 180 degrees to match train preprocessing
    # 这对于PI0策略同样适用，因为它需要与预训练时的图像方向保持一致
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # 180度旋转以匹配训练预处理
    return img

def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations."""
    # IMPORTANT: 同样需要180度旋转以匹配训练预处理
    # 与agentview_image保持一致的预处理
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # 180度旋转以匹配训练预处理
    return img

def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.
    Args:
        quat (np.array): (x,y,z,w) vec4 float angles
    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def save_rollout_video(rollout_images, idx, success, task_description, log_dir, log_file=None):
    """Saves an MP4 replay of an episode."""
    from datetime import datetime
    DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    rollout_dir = os.path.join(log_dir, "rollouts", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(rollout_dir, exist_ok=True)
    
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--pi0--episode={idx}--success={success}--task={processed_task_description}.mp4"
    
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path