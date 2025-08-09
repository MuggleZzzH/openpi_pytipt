from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import gc
import os
import time
from tqdm import tqdm
import json
import hashlib
import pickle
from pathlib import Path
import imageio
from datetime import datetime

# Set multiprocessing start method to 'spawn' instead of 'fork' to prevent memory copying
import multiprocessing
if hasattr(multiprocessing, 'set_start_method'):
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        print("Failed to set multiprocessing start method to 'spawn', it may already be set")


# 全局配置常量
DEBUG_SAVE_IMAGES = True  # 可通过环境变量控制
DEBUG_SAVE_VIDEO = True
DEBUG_IMAGE_DIR = os.environ.get("PI0_DEBUG_IMAGE_DIR", "ript/debug_images")


def compute_hash_from_numpy(state_numpy: np.ndarray) -> str:
    """Computes a SHA256 hash for a numpy array."""
    # Ensure the array is contiguous in memory before getting its bytes
    return hashlib.sha256(np.ascontiguousarray(state_numpy).tobytes()).hexdigest()


class RolloutGenerator:
    """
    Generates rollouts for reinforcement learning algorithms.
    Manages the interaction between the model and environment to collect experience.
    """

    def __init__(
        self,
        env_runner: Any,
        rollouts_per_env: int,
        num_envs: int,
        max_steps: int,
        agent_gpus: List[int],
        init_state_dataloader: Any,
        init_state_dataset: Any,
        enable_dynamic_sampling: bool = False,
        rollout_skip_threshold: int = 3,
        enable_rollout_stats_tracking: bool = False,
        rollout_stats_path: Optional[str] = None,
        use_val_init: bool = False,
        max_history_per_state: int = 100,  # 每个状态保留的最大历史记录长度
        sync_stats_every_n_steps: int = 10,  # 每隔多少步同步一次统计数据
    ):
        """
        Initialize the rollout generator.
        
        Args:
            env_runner: Environment runner for executing rollouts.
            init_state_dataloader: Dataloader for initial states.
            init_state_dataset: The dataset for initial states.
            rollouts_per_env: Number of rollouts per environment instance.
            num_envs: Number of parallel environments.
            max_steps: Max steps per episode.
            agent_gpus: List of GPU ids for agents.
            enable_dynamic_sampling: If True, discard rollouts that are all success or all fail.
            rollout_skip_threshold: How many times to skip a mastered state before removing it.
            enable_rollout_stats_tracking: If True, track success rates and skip mastered states.
            rollout_stats_path: Path to load/save rollout statistics (JSON file).
            use_val_init: If True, force sampling of initial states from the validation set.
            max_history_per_state: Maximum history length to keep per initial state.
            sync_stats_every_n_steps: How often to synchronize statistics across processes.
        """
        self.env_runner = env_runner
        self.init_state_dataloader = init_state_dataloader
        self.init_state_dataset = init_state_dataset
        self.rollouts_per_env = rollouts_per_env
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.agent_gpus = agent_gpus
        self.enable_dynamic_sampling = enable_dynamic_sampling
        self.enable_rollout_stats_tracking = enable_rollout_stats_tracking
        self.max_history_per_state = max_history_per_state
        self.sync_stats_every_n_steps = sync_stats_every_n_steps

        self.rollout_stats = {}  # key: init_hash, value: list of success
        self.rollout_skip_cnt = {}  # key: init_hash, value: number of rounds skipped
        self.rollout_skip_threshold = rollout_skip_threshold
        self.rollout_stats_path = rollout_stats_path
        self.use_val_init = use_val_init
        
        # 检查是否在分布式环境中运行
        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.device = torch.device(f"cuda:{self.agent_gpus[0]}" if self.agent_gpus else "cpu")
        
        # 统计计数器
        self.step_counter = 0
        
        # 创建临时文件目录用于进程间通信
        if self.is_distributed:
            self.temp_dir = Path(os.path.dirname(self.rollout_stats_path or ".")) / "temp_sync"
            if self.rank == 0:
                os.makedirs(self.temp_dir, exist_ok=True)
            # 确保所有进程看到目录创建
            if self.is_distributed:
                dist.barrier()
        
        # 加载统计数据
        self._load_stats()

    def _load_stats(self):
        """加载轨迹统计数据，如果在分布式环境中，则只由主进程加载，然后广播给所有进程"""
        if self.rollout_stats_path and os.path.exists(self.rollout_stats_path):
            if self.rank == 0:
                print(f"从 {self.rollout_stats_path} 加载现有轨迹统计数据")
                try:
                    with open(self.rollout_stats_path, 'r') as f:
                        self.rollout_stats = json.load(f)
                except json.JSONDecodeError:
                    print(f"警告: 无法解析JSON文件 {self.rollout_stats_path}，将从头开始")
                    self.rollout_stats = {}
                
                # 初始化已加载状态的跳过计数
                for init_hash in self.rollout_stats:
                    self.rollout_skip_cnt[init_hash] = 0
            
            # 在分布式环境中同步统计数据
            if self.is_distributed:
                self._sync_stats_across_processes()
        else:
            if self.rank == 0:
                print("未找到现有轨迹统计数据。从头开始。")

    def _sync_stats_across_processes(self):
        """在分布式环境中同步统计数据"""
        if not self.is_distributed:
            return
        
        if self.rank == 0:
            # 主进程将数据保存到临时文件
            temp_file = self.temp_dir / "rollout_stats_temp.pkl"
            with open(temp_file, 'wb') as f:
                pickle.dump((self.rollout_stats, self.rollout_skip_cnt), f)
        
        # 同步所有进程，确保文件已写入
        dist.barrier()
        
        if self.rank != 0:
            # 非主进程从临时文件加载数据
            temp_file = self.temp_dir / "rollout_stats_temp.pkl"
            with open(temp_file, 'rb') as f:
                self.rollout_stats, self.rollout_skip_cnt = pickle.load(f)
        
        # 再次同步所有进程
        dist.barrier()

    def generate_rollouts(self) -> List[Dict[str, Any]]:
        """
        Generate rollout episodes for RL optimization.
            
        Returns:
            A list of episode dictionaries.
        """
        self.step_counter += 1
        collected_rollouts = []
        
        # 重置数据加载器迭代器，以确保每个训练步骤使用新的随机顺序
        if hasattr(self.init_state_dataloader, 'sampler') and hasattr(self.init_state_dataloader.sampler, 'set_epoch'):
            # 对于DistributedSampler，在每个epoch设置不同的种子
            epoch = self.step_counter
            self.init_state_dataloader.sampler.set_epoch(epoch)
        
        # 创建一个迭代器
        dataloader_iter = iter(self.init_state_dataloader)
        
        # 创建进度条（仅在主进程上显示）
        pbar = None
        if self.rank == 0:
            pbar = tqdm(
                total=self.rollouts_per_env * self.num_envs,
                desc=f"进程 {self.rank} 生成轨迹",
                disable=False
            )
        
        # 循环直到收集到足够的有效轨迹
        while len(collected_rollouts) < self.rollouts_per_env * self.num_envs:
            try:
                # 从训练集数据加载器获取一批初始状态
                init_states_train = next(dataloader_iter)
            except StopIteration:
                # 如果数据集耗尽，则重置迭代器
                dataloader_iter = iter(self.init_state_dataloader)
                init_states_train = next(dataloader_iter)

            # --- 选择使用哪些初始状态 ---
            if self.use_val_init:
                # 如果使用验证初始状态，我们需要直接从基准获取它们
                task_name_to_run = self.env_runner.task_names[0] if self.env_runner.task_names else "default_task"
                task_id = self.env_runner.benchmark.get_task_names().index(task_name_to_run)
                val_states = self.env_runner.benchmark.get_task_init_states(task_id, split='test')
                
                # 从验证状态中随机采样
                indices = np.random.choice(len(val_states), self.num_envs)
                init_states = torch.from_numpy(val_states[indices])
                if self.rank == 0:
                    print(f"使用任务 {task_name_to_run} 的验证初始状态")
            else:
                # 默认使用来自训练集数据加载器的状态
                init_states = init_states_train

            # --- 自适应采样逻辑 ---
            init_hash = compute_hash_from_numpy(init_states.numpy())
            
            # 1. 如果该状态已经"掌握"，则跳过
            if self.enable_rollout_stats_tracking and init_hash in self.rollout_stats:
                # 检查最近N次轨迹是否都成功
                recent_successes = self.rollout_stats[init_hash][-self.num_envs:]
                if len(recent_successes) >= self.num_envs and all(s == 1 for s in recent_successes):
                    if self.rank == 0:
                        print(f"由于持续成功而跳过初始状态 {init_hash[:8]}...")
                    self.rollout_skip_cnt[init_hash] = self.rollout_skip_cnt.get(init_hash, 0) + 1
                    
                    # 如果跳过太多次，则从统计数据中移除以允许稍后重新评估
                    if self.rollout_skip_cnt.get(init_hash, 0) > self.rollout_skip_threshold:
                        if self.rank == 0:
                            print(f"在跳过太多次后从统计数据中移除初始哈希 {init_hash[:8]}...")
                        del self.rollout_stats[init_hash]
                    continue # 移至下一批初始状态
            
            # --- 执行轨迹 ---
            # run_policy_in_env 是一个生成器，产生轨迹
            try:
                rollout_generator_for_batch = self.env_runner.run_policy_in_env(
                    env_name=self.env_runner.task_names[0] if self.env_runner.task_names else "default_task", # 为简单起见，暂时假设一个任务
                    all_init_states=init_states.numpy()
                )

                batch_rollouts = list(rollout_generator_for_batch)
                
                # 检查是否获得了有效的轨迹
                if not batch_rollouts:
                    if self.rank == 0:
                        print(f"警告: 从初始状态 {init_hash[:8]} 获得空轨迹，跳过...")
                    continue
                
            except EOFError as e:
                if self.rank == 0:
                    print(f"多进程通信错误 (EOFError): {e}")
                    print("这通常表明子进程崩溃，跳过当前批次")
                continue
                
            except Exception as e:
                if self.rank == 0:
                    print(f"轨迹生成过程中出现错误: {e}")
                    print(f"错误类型: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                continue
            
            # 验证轨迹数据的完整性
            try:
                batch_successes = []
                for i, rollout in enumerate(batch_rollouts):
                    if not isinstance(rollout, (tuple, list)) or len(rollout) < 3:
                        if self.rank == 0:
                            print(f"警告: 轨迹 {i} 格式不正确: {rollout}")
                        continue
                    
                    success, total_reward, episode_data = rollout
                    batch_successes.append(int(success))
                
                if not batch_successes:
                    if self.rank == 0:
                        print("警告: 没有有效的成功状态，跳过当前批次")
                    continue
                    
            except Exception as e:
                if self.rank == 0:
                    print(f"验证轨迹数据时出错: {e}")
                continue
            
            # 2. 更新统计数据
            if self.enable_rollout_stats_tracking:
                if init_hash not in self.rollout_stats:
                    self.rollout_stats[init_hash] = []
                    self.rollout_skip_cnt[init_hash] = 0
                self.rollout_stats[init_hash].extend(batch_successes)
                
                # 限制每个初始状态存储的历史记录长度，以防止内存无限增长
                if len(self.rollout_stats[init_hash]) > self.max_history_per_state:
                    self.rollout_stats[init_hash] = self.rollout_stats[init_hash][-self.max_history_per_state:]

            # 3. 如果结果不多样（全部成功或全部失败），则丢弃
            if self.enable_dynamic_sampling and (all(s == 1 for s in batch_successes) or all(s == 0 for s in batch_successes)):
                if self.rank == 0:
                    print(f"由于成功率一致 {batch_successes} 而丢弃来自 {init_hash[:8]} 的轨迹")
                continue # 移至下一批初始状态

            # --- 收集有效轨迹 ---
            # 将轨迹（元组）转换为字典
            episodes = [{'success': r[0], 'total_reward': r[1], **r[2]} for r in batch_rollouts]
            collected_rollouts.extend(episodes)
            
            # 更新进度条
            if pbar is not None:
                pbar.update(len(episodes))
                pbar.set_postfix({'collected': len(collected_rollouts)})
        
        # 关闭进度条
        if pbar is not None:
            pbar.close()
        
        # 定期在分布式环境中同步统计数据
        if self.is_distributed and self.enable_rollout_stats_tracking and self.step_counter % self.sync_stats_every_n_steps == 0:
            self._merge_stats_from_all_processes()
        
        # 定期保存统计数据
        if self.rank == 0 and self.enable_rollout_stats_tracking and self.step_counter % self.sync_stats_every_n_steps == 0:
            self.save_stats()
            
        return collected_rollouts
    
    def _merge_stats_from_all_processes(self):
        """合并所有进程的统计数据"""
        if not self.is_distributed:
            return
        
        # 每个进程将自己的统计数据保存到一个临时文件
        temp_file = self.temp_dir / f"rollout_stats_rank_{self.rank}.pkl"
        with open(temp_file, 'wb') as f:
            pickle.dump(self.rollout_stats, f)
        
        # 同步所有进程，确保所有文件都已写入
        dist.barrier()
        
        # 主进程读取所有临时文件并合并数据
        if self.rank == 0:
            merged_stats = self.rollout_stats.copy()
            
            for r in range(1, self.world_size):
                other_file = self.temp_dir / f"rollout_stats_rank_{r}.pkl"
                if os.path.exists(other_file):
                    with open(other_file, 'rb') as f:
                        other_stats = pickle.load(f)
                    
                    # 合并统计数据
                    for init_hash, successes in other_stats.items():
                        if init_hash not in merged_stats:
                            merged_stats[init_hash] = []
                        merged_stats[init_hash].extend(successes)
                        
                        # 限制历史记录长度
                        if len(merged_stats[init_hash]) > self.max_history_per_state:
                            merged_stats[init_hash] = merged_stats[init_hash][-self.max_history_per_state:]
            
            # 将合并后的数据保存到主文件
            merged_file = self.temp_dir / "merged_stats.pkl"
            with open(merged_file, 'wb') as f:
                pickle.dump(merged_stats, f)
        
        # 同步所有进程，确保主文件已写入
        dist.barrier()
        
        # 所有进程从主文件加载合并后的数据
        if self.is_distributed:
            merged_file = self.temp_dir / "merged_stats.pkl"
            with open(merged_file, 'rb') as f:
                self.rollout_stats = pickle.load(f)
            
            # 重置跳过计数
            self.rollout_skip_cnt = {init_hash: 0 for init_hash in self.rollout_stats}
        
        # 最后同步一次
        dist.barrier()
    
    def save_stats(self):
        """将轨迹统计数据保存到文件（如果提供了路径）"""
        if self.rollout_stats_path and self.rollout_stats:
            if self.rank == 0: # 只从主进程保存
                print(f"保存轨迹统计数据到 {self.rollout_stats_path}")
                # 确保目录存在
                os.makedirs(os.path.dirname(self.rollout_stats_path), exist_ok=True)
                
                # 在分布式环境中，确保我们有所有进程的最新统计数据
                if self.is_distributed:
                    self._merge_stats_from_all_processes()
                
                # 保存统计数据
                try:
                    with open(self.rollout_stats_path, 'w') as f:
                        json.dump(self.rollout_stats, f, indent=4)
                except Exception as e:
                    print(f"保存统计数据时出错: {e}")
    
    def cleanup(self):
        """清理轨迹生成器资源"""
        # 保存最终统计数据
        self.save_stats()
        
        # 在退出前进行任何必要的清理
        if self.is_distributed:
            # 确保所有进程都完成了保存
            dist.barrier()
            
            # 清理临时文件
            if self.rank == 0 and hasattr(self, 'temp_dir'):
                import shutil
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    print(f"清理临时文件时出错: {e}")
            
            # 最后同步一次
            dist.barrier()
            
        # 清理环境资源
        if hasattr(self, 'env_runner'):
            try:
                # 使用安全的环境清理
                if hasattr(self.env_runner, 'cleanup'):
                    self.env_runner.cleanup()
                elif hasattr(self.env_runner, 'close'):
                    self.env_runner.close()
                    
                # 等待子进程终止
                time.sleep(0.1)
                
                # 删除环境引用
                del self.env_runner
                
            except Exception as e:
                print(f"清理环境时出错: {e}")
                
        # 强制垃圾回收
        gc.collect()
        
        print("RolloutGenerator 清理完成")