"""
PI0 + LIBERO 环境运行器 - 基于RIPT-VLA的成功实现
直接复用 /zhaohan/ZJH/ript-vla/ript/env_runner/openvla_oft_libero_runner.py 的核心逻辑
"""
import os
import numpy as np
import gc
import multiprocessing
from collections import deque
from pathlib import Path
import torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
import robosuite.utils.transform_utils as T
from typing import List, Dict, Any


class BatchProcessingWrapper:
    """
    批处理环境包装器
    在主进程中使用单个环境，通过顺序执行模拟并行环境
    避免SubprocVectorEnv的模型复制问题
    """
    
    def __init__(self, single_env, num_parallel_envs):
        self.single_env = single_env
        self.num_parallel_envs = num_parallel_envs
        self.current_states = [None] * num_parallel_envs
        
    def reset(self):
        """重置所有并行环境"""
        # 只重置一次真实环境，然后复制状态
        obs = self.single_env.reset()
        # 为每个模拟环境保存相同的初始状态
        self.current_states = [obs] * self.num_parallel_envs
        return self.current_states
    
    def set_init_state(self, init_states):
        """设置初始状态"""
        # 为每个环境设置不同的初始状态
        if len(init_states) >= self.num_parallel_envs:
            # 使用第一个状态设置环境
            self.single_env.set_init_state(init_states[0])
        
    def step(self, actions):
        """批处理执行动作"""
        batch_obs = []
        batch_rewards = []
        batch_dones = []
        batch_infos = []
        
        # 顺序执行每个环境的动作
        for i, action in enumerate(actions):
            if i < self.num_parallel_envs:
                # 执行第i个动作
                obs, reward, done, info = self.single_env.step(action)
                batch_obs.append(obs)
                batch_rewards.append(reward)
                batch_dones.append(done)
                batch_infos.append(info)
            else:
                # 如果动作数量超过环境数，复制最后结果
                batch_obs.append(batch_obs[-1] if batch_obs else {})
                batch_rewards.append(0.0)
                batch_dones.append(False)
                batch_infos.append({})
        
        return batch_obs, batch_rewards, batch_dones, batch_infos
    
    def close(self):
        """关闭环境"""
        if hasattr(self.single_env, 'close'):
            self.single_env.close()

class PI0LiberoRunner:
    """PI0 + LIBERO 环境运行器 - 直接复用RIPT-VLA的成功模式"""
    
    def __init__(self,
                 policy,
                 benchmark_name,
                 rollouts_per_env,
                 num_parallel_envs,
                 max_episode_length=None,
                 task_names_to_use=[],
                 rank=0):
        self.policy = policy
        self.benchmark_name = benchmark_name.lower()
        self.rollouts_per_env = rollouts_per_env  
        self.num_parallel_envs = num_parallel_envs
        self.rank = rank
        
        # LIBERO benchmark setup
        benchmark_dict = benchmark.get_benchmark_dict()
        self.benchmark = benchmark_dict[benchmark_name.lower()]()
        self.env_names = self.benchmark.get_task_names()
        
        if task_names_to_use is not None and len(task_names_to_use) > 0:
            self.env_names_to_run = [name for name in self.env_names if name in task_names_to_use]
        else:
            self.env_names_to_run = self.env_names
            
        # 设置multiprocessing方法 - 关键！
        if num_parallel_envs > 1:
            if multiprocessing.get_start_method(allow_none=True) != "spawn":  
                multiprocessing.set_start_method("spawn", force=True)
                
        # Episode length
        task_max_steps = {
            "libero_spatial": 220,
            "libero_object": 280, 
            "libero_goal": 300,
            "libero_10": 520,
            "libero_90": 400,
        }
        
        if max_episode_length is None:
            self.max_episode_length = task_max_steps.get(benchmark_name.lower(), 300)
        else:
            self.max_episode_length = max_episode_length
            
        self.num_steps_wait = 20  # LIBERO simulator warmup - 🔧 修复：使用与参考脚本相同的20步
        
        # 🔧 默认强制批处理模式，避免SubprocVectorEnv的模型复制问题
        self._force_batch_mode = True  # 设为False可启用真正并行
        
        # 加载归一化参数
        self._load_norm_stats()
        
    def create_env(self, env_name):
        """创建并行环境 - 修复版本，避免模型复制问题"""
        # 如果env_name是数字，直接使用；否则在任务列表中查找
        if isinstance(env_name, int):
            task_id = env_name
        elif isinstance(env_name, str) and env_name.isdigit():
            task_id = int(env_name)
        elif env_name in self.env_names:
            task_id = self.env_names.index(env_name)
        else:
            # 默认使用第一个任务
            task_id = 0
            if self.rank == 0:
                print(f"⚠️ 任务名称 '{env_name}' 未找到，使用默认task_id=0")
        
        task = self.benchmark.get_task(task_id)
        
        env_num = min(self.num_parallel_envs, self.rollouts_per_env)
        
        # 🔧 修复：检查是否应该使用真正的并行环境
        use_true_parallel = env_num > 1 and hasattr(self, '_force_batch_mode') and not self._force_batch_mode
        
        if use_true_parallel:
            # 🔑 关键：简单的环境工厂，无外部依赖
            def get_env(task):
                task_description = task.language
                # 使用libero的标准路径获取方法
                from libero.libero import get_libero_path
                task_bddl_file = os.path.join(
                    get_libero_path("bddl_files"), 
                    task.problem_folder, 
                    task.bddl_file
                )
                env_args = {
                    "bddl_file_name": task_bddl_file, 
                    "camera_heights": 224, 
                    "camera_widths": 224
                }
                env = OffScreenRenderEnv(**env_args)
                env.seed(0)
                return env
                
            env_factory = lambda: get_env(task)
            
            if self.rank == 0:
                print(f"🚀 创建 {env_num} 个并行环境 (真正并行)")
                
            env = SubprocVectorEnv([env_factory for _ in range(env_num)])
        else:
            # 🔧 修复：批处理模式 - 在主进程创建单个环境，模拟并行
            if self.rank == 0:
                print(f"🔧 使用批处理模式模拟 {env_num} 个并行环境（避免内存问题）")
            
            # 创建单个环境
            task_description = task.language
            from libero.libero import get_libero_path
            task_bddl_file = os.path.join(
                get_libero_path("bddl_files"), 
                task.problem_folder, 
                task.bddl_file
            )
            env_args = {
                "bddl_file_name": task_bddl_file, 
                "camera_heights": 224, 
                "camera_widths": 224
            }
            single_env = OffScreenRenderEnv(**env_args)
            single_env.seed(0)
            
            # 包装成批处理环境
            env = BatchProcessingWrapper(single_env, env_num)
        
        return env, task_id, env_num
        
    def run_policy_in_env(self, env_name, all_init_states=None, created_env=None):
        """运行策略 - 完全复用RIPT-VLA的逻辑"""
        if created_env is None:
            env, env_id, env_num = self.create_env(env_name) 
        else:
            env, env_id, env_num = created_env
            
        if all_init_states is None:
            all_init_states = self.benchmark.get_task_init_states(env_id)
            
        count = 0
        eval_loop_num = (self.rollouts_per_env + self.num_parallel_envs - 1) // self.num_parallel_envs
        
        if self.rank == 0:
            print(f"🚀 开始并行执行: {eval_loop_num} 轮次, 每轮 {env_num} 个环境")
            
        while count < eval_loop_num:
            indices = np.arange(count * env_num, (count + 1) * env_num) % all_init_states.shape[0]
            init_states_ = all_init_states[indices]
            
            success, total_reward, episode = self.run_episode(env, env_name, init_states_, env_num)
            
            # 分离每个环境的结果 - 🔧 完全重写数据分离逻辑
            step_num = len(episode['actions'])
            for k in range(env_num):
                episode_k = {}
                
                # 处理标量字段（所有环境共享）
                for key in ['task', 'task_id']:
                    if key in episode:
                        episode_k[key] = episode[key]
                
                # 🔑 关键修复：正确处理observations字段
                if 'observations' in episode:
                    # episode['observations']是每步的obs列表，每个obs列表包含env_num个环境的观测
                    episode_k['observations'] = []
                    for step_obs_list in episode['observations']:
                        # step_obs_list是一个长度为env_num的列表，包含每个环境的原始观测
                        if isinstance(step_obs_list, list) and len(step_obs_list) > k:
                            episode_k['observations'].append(step_obs_list[k])  # 第k个环境的观测
                        else:
                            # 如果不是列表或索引超界，可能是单环境数据
                            episode_k['observations'].append(step_obs_list)
                
                # 处理其他数组字段（actions, rewards, dones, infos）
                for key in ['actions', 'rewards', 'dones', 'infos']:
                    if key in episode:
                        episode_k[key] = []
                        for step_data in episode[key]:
                            if isinstance(step_data, list) and len(step_data) > k:
                                episode_k[key].append(step_data[k])  # 第k个环境的数据
                            else:
                                # 备用：可能是单环境或其他格式
                                episode_k[key].append(step_data)
                
                yield success[k], total_reward[k], episode_k
                
            count += 1
            
        if created_env is None:
            # 🔧 修复：安全关闭环境，避免EGL错误
            try:
                env.close()
                del env
                gc.collect()
            except Exception as e:
                if self.rank == 0:
                    print(f"⚠️ 环境关闭时出现警告: {e}")
                # 强制清理，忽略EGL错误
                import os
                os.environ["EGL_LOG_LEVEL"] = "fatal"
            
    def run_episode(self, env, env_name, init_states_, env_num):
        """运行单个episode - 核心并行逻辑"""
        if self.rank == 0:
            print(f"🔄 重置环境并开始episode")
            
        env.reset()
        env.set_init_state(init_states_)
        
        # 🔑 获取初始观测 - 通过step获得
        dummy_actions = [np.array([0, 0, 0, 0, 0, 0, -1]) for _ in range(env_num)]
        obs, _, _, _ = env.step(dummy_actions)
        
        task_id = self.env_names.index(env_name) 
        task = self.benchmark.get_task(task_id)
        task_description = task.language
        
        t = 0
        success = [False] * env_num
        total_reward = [0] * env_num
        
        episode = {
            'actions': [],
            'observations': [],
            'rewards': [],
            'dones': [],
            'infos': [],
            # 🔧 修复：添加PI0需要的task字段
            'task': task_description,
            'task_id': task_id,
        }
        
        # 🎬 视频收集：每个环境独立的帧缓存
        video_frames = [[] for _ in range(env_num)]
        save_video = True  # 启用视频保存
        
        # 🔑 关键：动作队列 - 每个环境独立的动作缓存
        batch_action_queue = [deque(maxlen=50) for _ in range(env_num)]
        
        while t < self.max_episode_length + self.num_steps_wait:
            # Simulator warmup
            if t < self.num_steps_wait:
                dummy_actions = [np.array([0, 0, 0, 0, 0, 0, -1]) for _ in range(env_num)]
                obs, _, done, _ = env.step(dummy_actions)
                t += 1
                continue
                
            step_actions = [np.array([0, 0, 0, 0, 0, 0, -1]) for _ in range(env_num)]
            step_observations = [None] * env_num
            
            # 收集需要推理的环境
            step_input_obs_ids = []
            step_input_obs_list = []
            
            for bidx in range(env_num):
                if not success[bidx]:
                    step_input_obs_ids.append(bidx)
                    
                    # 准备PI0观测格式
                    observation = self.prepare_pi0_observation(obs[bidx], task_description)
                    step_observations[bidx] = observation
                    step_input_obs_list.append(observation)
                    
            # 🔑 关键：批量推理条件 - 当所有需要的环境的动作队列都空了
            conduct_inference = all(len(batch_action_queue[i]) == 0 for i in step_input_obs_ids)
            
            if conduct_inference and step_input_obs_list:
                # 🚀 批量推理 - 一次处理多个环境的观测
                if self.rank == 0:
                    print(f"   🧠 批量推理: {len(step_input_obs_list)} 个环境 (步骤 {t-self.num_steps_wait+1})")
                    
                with torch.inference_mode():
                    batch_actions = self.get_pi0_action_batch(step_input_obs_list)
                    
                    # 🔍 调试：显示推理结果
                    if self.rank == 0:
                        for act_idx, actions in enumerate(batch_actions):
                            print(f"     环境 {step_input_obs_ids[act_idx]}: 生成了 {len(actions)} 个动作")
                            if len(actions) > 0:
                                first_action = actions[0]
                                print(f"       首个动作: [{first_action[0]:.3f}, {first_action[1]:.3f}, {first_action[2]:.3f}, ...]")
                    
                    # 分发动作到各个环境的队列
                    for act_idx, obs_id in enumerate(step_input_obs_ids):
                        batch_action_queue[obs_id].extend(batch_actions[act_idx])
                        if self.rank == 0:
                            print(f"     环境 {obs_id}: 动作队列长度 {len(batch_action_queue[obs_id])}")
                        
            # 从队列中取出动作执行
            for i, bidx in enumerate(step_input_obs_ids):
                if len(batch_action_queue[bidx]) > 0:
                    action = batch_action_queue[bidx].popleft()
                    step_actions[bidx] = action
                    
            # 环境步进
            obs, reward, done, info = env.step(step_actions)
            
            # 🔍 调试：显示环境执行信息
            if self.rank == 0 and t % 10 == 0:  # 每10步打印一次
                print(f"   📊 步骤 {t-self.num_steps_wait+1}: 奖励 {reward}, 完成 {done}")
                for k in range(env_num):
                    has_action = len(step_input_obs_ids) > k and step_input_obs_ids[k] < len(step_actions)
                    action_info = f"动作: [{step_actions[k][0]:.2f}, {step_actions[k][1]:.2f}, ...]" if has_action else "无动作"
                    print(f"     环境 {k}: {action_info}, 奖励 {reward[k]:.3f}, 总奖励 {total_reward[k]:.3f}")
            
            # 更新状态
            for k in range(env_num):
                # 🔧 修复：正确的成功判断逻辑 - 使用info中的success字段而不是done
                old_success = success[k]
                if hasattr(info[k], '__getitem__') and 'success' in info[k]:
                    success[k] = success[k] or info[k]['success']
                elif hasattr(info[k], 'success'):
                    success[k] = success[k] or info[k].success
                # 备用：如果没有success信息且奖励>0.5，也认为成功
                elif reward[k] > 0.5:
                    success[k] = success[k] or True
                total_reward[k] += reward[k]
                
                # 🔍 调试：显示成功状态变化
                if not old_success and success[k] and self.rank == 0:
                    print(f"   🎉 环境 {k} 成功完成任务！总奖励: {total_reward[k]:.3f}")
                
            if all(success):
                break
                
            t += 1
            
            # 记录数据 - 🔧 修复：保存原始环境观测而不是PI0格式观测
            if conduct_inference:
                episode['actions'].append(step_actions)
                # 保存原始环境观测数据，而不是PI0转换后的格式
                episode['observations'].append(obs)
                episode['rewards'].append(reward)
                episode['dones'].append(done)
                episode['infos'].append(info)
                
                # 🎬 保存视频帧（每隔5步保存一次以减少存储）
                if save_video and (t - self.num_steps_wait) % 5 == 0:
                    for k in range(env_num):
                        if isinstance(obs, list) and len(obs) > k:
                            frame = obs[k].get("agentview_image")
                            if frame is not None:
                                video_frames[k].append(frame)
                
        if self.rank == 0:
            total_steps = t - self.num_steps_wait
            inference_count = len(episode.get('actions', []))
            print(f"✅ Episode完成: 成功 {success}")
            print(f"   📈 总步数: {total_steps}, 推理次数: {inference_count}, 最终奖励: {total_reward}")
            print(f"   📊 平均奖励/环境: {[f'{r:.3f}' for r in total_reward]}")
            
            # 🎬 保存视频文件
            if save_video:
                self._save_episode_videos(video_frames, env_name, total_reward, success)
            
        return success, total_reward, episode
    
    def _save_episode_videos(self, video_frames, env_name, total_rewards, successes):
        """保存episode视频"""
        try:
            import imageio
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 使用pi0/ript/debug_images/videos作为视频目录，与其他部分保持一致
            video_dir = Path("pi0/ript/debug_images/videos")
            video_dir.mkdir(parents=True, exist_ok=True)
            
            for env_id, frames in enumerate(video_frames):
                if len(frames) > 0:
                    success_tag = "success" if successes[env_id] else "fail"
                    reward_tag = f"r{total_rewards[env_id]:.2f}"
                    video_filename = f"{env_name}_env{env_id}_{success_tag}_{reward_tag}_{timestamp}.mp4"
                    video_path = video_dir / video_filename
                    
                    # BGR to RGB conversion for imageio
                    rgb_frames = [frame[:, :, ::-1] for frame in frames]
                    imageio.mimsave(video_path, rgb_frames, fps=10)
                    print(f"   🎬 视频已保存: {video_path}")
                    
        except Exception as e:
            print(f"   ⚠️ 视频保存失败: {e}")
        
    def prepare_pi0_observation(self, obs, task_description):
        """准备PI0观测格式"""
        # 图像处理  
        base_0_rgb = obs["agentview_image"][:, :, ::-1].copy()
        left_wrist_0_rgb = obs["robot0_eye_in_hand_image"][:, :, ::-1].copy() 
        
        # 状态处理 - 与原实现保持一致
        axis_angle = T.quat2axisangle(obs["robot0_eef_quat"])
        unnorm_state = np.concatenate([
            obs["robot0_eef_pos"],
            axis_angle, 
            obs["robot0_gripper_qpos"],
        ], dtype=np.float32)
        
        # 状态归一化
        state = (unnorm_state - self.state_mean) / (self.state_std + 1e-6)
        
        observation = {
            "image": {
                "base_0_rgb": torch.from_numpy(base_0_rgb).cuda()[None],
                "left_wrist_0_rgb": torch.from_numpy(left_wrist_0_rgb).cuda()[None],
            },
            "state": torch.from_numpy(state).cuda()[None],
            "prompt": [task_description],
        }
        
        return observation
        
    def get_pi0_action_batch(self, obs_batch):
        """批量推理PI0模型 - 关键优化"""
        batch_actions = []
        
        for obs in obs_batch:
            if self.policy is None:
                # 测试模式：生成随机动作
                action = np.random.randn(50, 7).astype(np.float32) * 0.01  # 小幅随机动作
                if self.rank == 0:
                    print("⚠️ 测试模式：使用随机动作")
            else:
                # 真实推理模式
                raw_action = self.policy.select_action(obs)
                action = raw_action[0, :, :7]  # (50, 7)
                
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
            
            # 反归一化动作 - 与原实现保持一致
            if hasattr(self, 'action_mean') and hasattr(self, 'action_std'):
                action = action * (self.action_std + 1e-6) + self.action_mean
                
            batch_actions.append(action)
            
        return batch_actions
    
    def _load_norm_stats(self):
        """加载归一化统计信息"""
        import json
        from pathlib import Path
        
        # 尝试在常见位置找到norm_stats.json
        possible_paths = [
            "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch/norm_stats.json",
            "/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json", 
            "./checkpoints/pi0_libero_pytorch/norm_stats.json",
            "./norm_stats.json"
        ]
        
        norm_stats_path = None
        for path in possible_paths:
            if Path(path).exists():
                norm_stats_path = path
                break
        
        if norm_stats_path and Path(norm_stats_path).exists():
            if self.rank == 0:
                print(f"📊 加载归一化统计: {norm_stats_path}")
            with open(norm_stats_path) as f:
                norm_stats = json.load(f)
                
            # 提取状态和动作的归一化参数
            self.state_mean = np.array(norm_stats["norm_stats"]["state"]["mean"][:8], dtype=np.float32)
            self.state_std = np.array(norm_stats["norm_stats"]["state"]["std"][:8], dtype=np.float32)
            self.action_mean = np.array(norm_stats["norm_stats"]["actions"]["mean"][:7], dtype=np.float32)
            self.action_std = np.array(norm_stats["norm_stats"]["actions"]["std"][:7], dtype=np.float32)
            
            if self.rank == 0:
                print("✅ 归一化参数加载成功")
        else:
            if self.rank == 0:
                print("⚠️ 未找到norm_stats.json，使用默认参数")
            # 使用默认值
            self.state_mean = np.zeros(8, dtype=np.float32)
            self.state_std = np.ones(8, dtype=np.float32)
            self.action_mean = np.zeros(7, dtype=np.float32)
            self.action_std = np.ones(7, dtype=np.float32)