#!/usr/bin/env python3
"""
第10阶段：分布式训练系统升级 (10_train_with_distributed.py)
基于9_train_with_config.py，添加完整的分布式训练支持

核心升级功能：
1. 标准PyTorch分布式训练 (DDP)
2. 任务分片和负载均衡
3. 分布式数据采样
4. 梯度同步和聚合
5. 分布式统计数据同步
6. 多GPU环境优化

使用方法:
    # 单机多GPU训练
    cd /zhaohan/ZJH/openpi_pytorch
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nproc_per_node=4 \
        10_train_with_distributed.py \
        --config_path pi0/ript/config/distributed_train_pi0.yaml
    
    # 多机训练
    # Node 0:
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
        --master_addr=MASTER_IP --master_port=12355 \
        10_train_with_distributed.py --config_path ...
    
    # Node 1:
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
        --master_addr=MASTER_IP --master_port=12355 \
        10_train_with_distributed.py --config_path ...
"""

import os
import sys
import json
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Union
import yaml
import time
import gc

# 修复tokenizers并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入OmegaConf用于配置管理
try:
    from omegaconf import OmegaConf, DictConfig
    OMEGACONF_AVAILABLE = True
    print("✓ 使用OmegaConf进行配置管理")
except ImportError:
    OMEGACONF_AVAILABLE = False
    print("⚠️ OmegaConf未安装，回退到基础YAML解析")

from pi0 import PI0Policy
from pi0.ript.env.pi0_libero_runner import LIBEROEnvRunner
from pi0.ript.reward_function import BinarySuccessReward
from pi0.ript.algos.rl_optimizers.pi0_cfg_interface import PI0_CFG_Adapter

# 导入增强智能采样系统
try:
    from pi0.ript.utils.enhanced_smart_sampling import EnhancedSmartSampler, enhanced_collect_smart_batches
    ENHANCED_SAMPLING_AVAILABLE = True
    print("✓ 增强智能采样系统已导入")
except ImportError:
    ENHANCED_SAMPLING_AVAILABLE = False
    print("⚠️ 增强智能采样系统未找到，将使用基础采样")
    
    # 基础采样实现作为后备
    class EnhancedSmartSampler:
        def __init__(self, config):
            self.config = config
        
        def smart_sample_init_state(self, init_dataset):
            # 🔥 使用环形索引而非随机采样
            if hasattr(init_dataset, 'get_states_for_envs'):
                # 新的环形索引方法
                init_states = init_dataset.get_states_for_envs(
                    self.config.num_parallel_envs, 
                    self.config.rloo_batch_size
                )
            else:
                # 后备：使用原有随机采样
                init_states = init_dataset.sample_batch(self.config.num_parallel_envs)
            
            state_hash = self.compute_state_hash(init_states)
            return init_states, state_hash
        
        def compute_state_hash(self, state_array):
            import hashlib
            return hashlib.sha256(np.ascontiguousarray(state_array).tobytes()).hexdigest()[:16]
    
    def enhanced_collect_smart_batches(env_runner, reward_function, init_dataset, sampler, config, iteration_idx):
        # 简化的采样实现
        print(f"使用基础采样模式 (迭代 {iteration_idx + 1})")
        collected_batches = []
        
        for i in range(config.target_batches_per_iteration):
            init_states, _ = sampler.smart_sample_init_state(init_dataset)
            
            try:
                rollout_generator = env_runner.run_policy_in_env(
                    env_name=config.task_id,
                    all_init_states=init_states
                )
                
                rollout_batch = list(rollout_generator)
                if rollout_batch:
                    episodes = []
                    for success, total_reward, episode_data in rollout_batch:
                        episode = {
                            'success': success,
                            'total_reward': total_reward,
                            **episode_data
                        }
                        episodes.append(episode)
                    
                    if episodes:
                        collected_batches.append(episodes)
            except Exception as e:
                print(f"采样失败: {e}")
                continue
        
        return collected_batches

# 分布式工具函数
def setup_distributed():
    """初始化分布式训练环境"""
    # 如果默认进程组已经初始化，直接返回当前信息，避免重复初始化错误
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # 如果 LOCAL_RANK 环境变量不存在，则尝试从当前设备获取
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        return True, rank, world_size, local_rank, device

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # torchrun 可能已经初始化了进程组，再次检查
        if not dist.is_initialized():
            # 设置超时时间，防止死锁
            timeout = timedelta(seconds=10800)  # 3小时
            
            # 初始化分布式进程组
            dist.init_process_group(
                backend='nccl',
                timeout=timeout,
                rank=rank,
                world_size=world_size
            )
        
        # 设置当前进程的GPU
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        return True, rank, world_size, local_rank, device
    else:
        return False, 0, 1, 0, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def sync_across_processes(tensor, device):
    """在所有进程间同步张量数据"""
    if dist.is_initialized():
        tensor = tensor.to(device)
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor

def gather_object_across_processes(obj):
    """在所有进程间收集对象数据"""
    if dist.is_initialized():
        world_size = dist.get_world_size()
        gathered_objects = [None] * world_size
        dist.all_gather_object(gathered_objects, obj)
        return gathered_objects
    else:
        return [obj]

def load_config_from_yaml(config_path: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    从YAML文件加载配置，支持OmegaConf和基础YAML
    
    Args:
        config_path: YAML配置文件路径
        overrides: 配置覆盖列表，格式如 ["training.lr=1e-5", "algo.batch_size=32"]
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    print(f"📋 加载配置文件: {config_path}")
    
    if OMEGACONF_AVAILABLE:
        # 使用OmegaConf加载配置
        config = OmegaConf.load(config_path)
        
        # 应用命令行覆盖
        if overrides:
            print(f"🔧 应用配置覆盖:")
            for override in overrides:
                print(f"  {override}")
                key, value = override.split('=', 1)
                # 尝试转换为正确的类型
                try:
                    # 尝试数字转换
                    if '.' in value and value.replace('.', '').replace('-', '').isdigit():
                        value = float(value)
                    elif value.replace('-', '').isdigit():
                        value = int(value)
                    elif value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                except:
                    pass  # 保持字符串类型
                
                # 使用OmegaConf的正确方法设置嵌套键
                keys = key.split('.')
                target = config
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                target[keys[-1]] = value
        
        # 转换为普通字典
        config_dict = OmegaConf.to_container(config, resolve=True)
        
    else:
        # 使用基础YAML加载
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 应用简单的配置覆盖
        if overrides:
            print(f"🔧 应用配置覆盖 (基础模式):")
            for override in overrides:
                print(f"  {override}")
                key, value = override.split('=', 1)
                # 简单的嵌套键处理
                keys = key.split('.')
                target = config_dict
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                
                # 类型转换
                try:
                    if '.' in value and value.replace('.', '').replace('-', '').isdigit():
                        value = float(value)
                    elif value.replace('-', '').isdigit():
                        value = int(value)
                    elif value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                except:
                    pass
                
                target[keys[-1]] = value
    
    print("✓ 配置文件加载成功")
    return config_dict

def validate_distributed_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证和补充分布式配置参数
    
    Args:
        config: 原始配置字典
        
    Returns:
        验证后的配置字典
    """
    print("🔍 验证分布式配置参数...")
    
    # 设置分布式相关的默认值
    distributed_defaults = {
        # 分布式配置
        'distributed': {
            'enabled': True,
            'backend': 'nccl',
            'timeout_seconds': 10800,
            'find_unused_parameters': False,
            'bucket_cap_mb': 25,
        },
        
        # 数据并行配置
        'data_parallel': {
            'sync_every_n_steps': 10,
            'enable_gradient_checkpointing': False,
            'async_data_loading': True,
        },
        
        # 任务分片配置
        'task_distribution': {
            'enable_task_sharding': True,
            'balance_tasks_across_gpus': True,
            'min_tasks_per_gpu': 1,
        }
    }
    
    # 递归合并默认值
    def merge_defaults(target: Dict, source: Dict) -> Dict:
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict) and isinstance(target[key], dict):
                target[key] = merge_defaults(target[key], value)
        return target
    
    config = merge_defaults(config, distributed_defaults)
    
    print("✓ 分布式配置验证通过")
    return config

class DistributedTrainingRunner:
    """分布式训练运行器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化分布式环境
        self.is_distributed, self.rank, self.world_size, self.local_rank, self.device = setup_distributed()
        
        if self.rank == 0:
            print(f"🌐 分布式训练初始化:")
            print(f"  World Size: {self.world_size}")
            print(f"  Backend: {config.get('distributed', {}).get('backend', 'nccl')}")
            print(f"  Device: {self.device}")
        
        # 设置随机种子（确保不同进程使用不同种子）
        seed = config.get('training', {}).get('seed', 42)
        torch.manual_seed(seed + self.rank)
        np.random.seed(seed + self.rank)
        
        self.setup_directories()
    
    def setup_directories(self):
        """创建必要的输出目录"""
        output_dir = Path(self.config['output_dir'])
        
        # 创建实验特定的输出目录
        exp_name = self.config.get('exp_name', 'pi0_distributed_experiment')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_output_dir = output_dir / f"{exp_name}_{timestamp}_rank{self.rank}"
        
        # 只在主进程创建目录
        if self.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 同步所有进程
        if self.is_distributed:
            dist.barrier()
        
        # 其他进程也创建自己的目录
        if self.rank != 0:
            self.exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建视频目录
        video_dir = self.config.get('enhanced_sampling', {}).get('video_dir', f'rollout_videos_distributed_rank{self.rank}')
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(exist_ok=True)
        
        if self.rank == 0:
            print(f"📁 实验输出目录: {self.exp_output_dir}")
            print(f"📁 视频保存目录: {self.video_dir}")
    
    def get_distributed_tasks(self):
        """根据任务配置和进程数分配任务"""
        all_tasks = self.config.get('task', {}).get('task_names_to_use', None)
        
        if all_tasks is None:
            # 使用单个任务ID
            task_id = self.config.get('task', {}).get('task_name', 8)
            all_tasks = [task_id]
        
        # 任务分片：将任务分配给不同的GPU
        if self.config.get('task_distribution', {}).get('enable_task_sharding', True):
            rank_to_tasks = {rank_i: [] for rank_i in range(self.world_size)}
            for task_i, task_name in enumerate(all_tasks):
                rank_to_tasks[task_i % self.world_size].append(task_name)
            local_tasks = rank_to_tasks[self.rank]
        else:
            # 所有进程处理相同任务
            local_tasks = all_tasks
        
        if self.rank == 0:
            print(f"🎯 任务分片配置:")
            for rank_i in range(self.world_size):
                tasks = rank_to_tasks[rank_i] if self.config.get('task_distribution', {}).get('enable_task_sharding', True) else all_tasks
                print(f"  Rank {rank_i}: {tasks}")
        
        return local_tasks
    
    def create_distributed_optimizer(self, policy):
        """创建分布式优化器"""
        optimizer_config = self.config['training']['optimizer']
        lr = self.config['algo']['lr']
        
        if self.rank == 0:
            print(f"\n=== 创建分布式优化器 ===")
            print(f"优化器类型: {optimizer_config['type']}")
            print(f"学习率: {lr}")
            print(f"权重衰减: {optimizer_config.get('weight_decay', 0.0)}")
        
        if optimizer_config['type'] == "AdamW":
            optimizer = torch.optim.AdamW(
                policy.parameters(),
                lr=lr,
                weight_decay=optimizer_config.get('weight_decay', 0.01),
                betas=(optimizer_config.get('beta1', 0.9), optimizer_config.get('beta2', 0.999))
            )
        elif optimizer_config['type'] == "Adam":
            optimizer = torch.optim.Adam(
                policy.parameters(),
                lr=lr,
                weight_decay=0.0
            )
        elif optimizer_config['type'] == "SGD":
            optimizer = torch.optim.SGD(
                policy.parameters(),
                lr=lr,
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_config['type']}")
        
        # 检查可训练参数
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        
        if self.rank == 0:
            print(f"总参数量: {total_params:,}")
            print(f"可训练参数: {trainable_params:,}")
            print(f"训练参数比例: {trainable_params/total_params:.2%}")
        
        return optimizer
    
    def wrap_model_for_distributed(self, policy):
        """将模型包装为分布式训练"""
        if self.is_distributed:
            # 使用DistributedDataParallel包装模型
            ddp_config = self.config.get('distributed', {})
            
            # 模型已经在正确的设备上，直接包装
            policy = DDP(
                policy,  # 已经在目标设备上
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=ddp_config.get('find_unused_parameters', False),
                bucket_cap_mb=ddp_config.get('bucket_cap_mb', 25),
            )
            
            if self.rank == 0:
                print("✓ 模型已包装为DistributedDataParallel")
        else:
            # 非分布式模式确保模型在正确设备上
            if not next(policy.parameters()).device == self.device:
                policy = policy.to(self.device)
        
        return policy
    
    def create_distributed_sampler_config(self):
        """创建分布式采样器的配置对象"""
        class DistributedSamplerConfig:
            def __init__(self, config_dict, rank, world_size):
                # 基础配置
                task_config = config_dict.get('task', {})
                algo_config = config_dict.get('algo', {})
                sampling_config = config_dict.get('enhanced_sampling', {})
                
                # 任务配置
                self.benchmark_name = task_config.get('benchmark_name', 'libero_spatial')
                self.task_id = task_config.get('task_name', 8)
                self.num_parallel_envs = task_config.get('num_parallel_envs', 1)
                self.rollouts_per_env = task_config.get('rollouts_per_env', 1)
                self.max_episode_length = task_config.get('max_episode_length', 200)
                
                # 分布式采样配置 - 根据进程数调整batch大小
                self.rollouts_per_batch = algo_config.get('rloo_batch_size', 4)
                self.target_batches_per_iteration = max(1, algo_config.get('data_batch_size', 2) // world_size)
                self.gradient_accumulation_steps = algo_config.get('gradient_accumulation_steps', 1)
                self.num_epochs = algo_config.get('num_epochs', 4)
                
                # 增强采样配置
                self.enable_smart_filtering = sampling_config.get('enable_smart_filtering', True)
                self.max_sampling_attempts = sampling_config.get('max_sampling_attempts', 20)
                self.debug_sampling = sampling_config.get('debug_sampling', True)
                self.save_videos = sampling_config.get('save_videos', True)
                self.video_dir = sampling_config.get('video_dir', f'rollout_videos_distributed_rank{rank}')
                
                # 智能状态跟踪配置
                self.enable_rollout_stats_tracking = algo_config.get('enable_rollout_stats_tracking', True)
                self.rollout_skip_threshold = algo_config.get('rollout_skip_threshold', 3)
                self.rollout_stats_path = algo_config.get('rollout_stats_path', f'./rollout_stats_rank{rank}.json')
                self.state_history_window = 20
                
                # 分布式特定配置
                self.rank = rank
                self.world_size = world_size
                self.sync_every_n_steps = config_dict.get('data_parallel', {}).get('sync_every_n_steps', 10)
        
        return DistributedSamplerConfig(self.config, self.rank, self.world_size)
    
    def distributed_collect_batches(self, env_runner, reward_function, init_dataset, sampler, config, iteration):
        """🔥 精确控制数据收集数量的分布式采样"""
        target_episodes = getattr(config, 'data_batch_size', 4)
        current_task = env_runner.get_current_task()
        
        if self.rank == 0:
            print(f"🎯 分布式采样 (迭代 {iteration + 1}): 目标收集 {target_episodes} 条轨迹")
            print(f"   当前任务: {current_task}")
        
        collected_episodes = []
        attempt = 0
        max_attempts = getattr(config, 'max_sampling_attempts', 10)
        
        # 🔥 精确控制：收集到data_batch_size条就停止
        while len(collected_episodes) < target_episodes and attempt < max_attempts:
            remaining_needed = target_episodes - len(collected_episodes)
            
            if self.rank == 0:
                print(f"    尝试 {attempt + 1}: 还需要 {remaining_needed} 条轨迹")
            
            # 环形索引选择初始状态
            try:
                init_states, state_hash = sampler.smart_sample_init_state(init_dataset)
                
                if self.rank == 0:
                    print(f"    🔍 选择状态哈希: {state_hash[:8]}")
                
                # 运行策略收集轨迹
                rollout_generator = env_runner.run_policy_in_env(
                    env_name=current_task,  # 🔥 使用当前任务
                    all_init_states=init_states,
                    debug_save_video=False  # 避免过多视频文件
                )
                
                # 收集本批次的轨迹
                batch_episodes = []
                for success, total_reward, episode_data in rollout_generator:
                    episode = {
                        'success': success,
                        'total_reward': total_reward,
                        'task_name': current_task,  # 记录任务信息
                        **episode_data
                    }
                    batch_episodes.append(episode)
                    
                    # 🔥 精确控制：达到目标数量就停止
                    if len(collected_episodes) + len(batch_episodes) >= target_episodes:
                        break
                
                collected_episodes.extend(batch_episodes)
                
                if self.rank == 0:
                    print(f"    ✓ 本批次收集 {len(batch_episodes)} 条轨迹 "
                          f"(累计: {len(collected_episodes)}/{target_episodes})")
                
                # 更新采样器统计信息
                if hasattr(sampler, 'update_state_stats'):
                    sampler.update_state_stats(state_hash, [ep['success'] for ep in batch_episodes])
            
            except Exception as e:
                if self.rank == 0:
                    print(f"    ❌ 采样失败 (尝试 {attempt + 1}): {e}")
            
            attempt += 1
        
        # 🔥 裁剪到精确数量
        if len(collected_episodes) > target_episodes:
            collected_episodes = collected_episodes[:target_episodes]
            if self.rank == 0:
                print(f"    ✂️ 裁剪到精确数量: {len(collected_episodes)}")
        
        # 如果需要，同步统计信息
        if config.sync_every_n_steps > 0 and (iteration + 1) % config.sync_every_n_steps == 0:
            self.sync_sampling_stats(sampler)
        
        if self.rank == 0:
            success_count = sum(1 for ep in collected_episodes if ep['success'])
            print(f"✅ Rank {self.rank} 收集完成: {len(collected_episodes)} 条轨迹 "
                  f"(成功率: {success_count}/{len(collected_episodes)})")
        
        return [collected_episodes] if collected_episodes else []
    
    def sync_sampling_stats(self, sampler):
        """同步采样统计信息"""
        if not self.is_distributed:
            return
        
        if self.rank == 0:
            print("🔄 同步采样统计信息...")
        
        # 收集所有进程的状态统计
        if hasattr(sampler, 'state_stats'):
            all_stats = gather_object_across_processes(sampler.state_stats)
            
            if self.rank == 0:
                # 合并统计信息
                merged_stats = {}
                for stats in all_stats:
                    for state_hash, state_info in stats.items():
                        if state_hash not in merged_stats:
                            merged_stats[state_hash] = {
                                'attempts': 0,
                                'successes': [],
                                'last_success_rate': 0.0
                            }
                        
                        merged_stats[state_hash]['attempts'] += state_info.get('attempts', 0)
                        merged_stats[state_hash]['successes'].extend(state_info.get('successes', []))
                        
                        # 限制历史记录长度
                        if len(merged_stats[state_hash]['successes']) > 50:
                            merged_stats[state_hash]['successes'] = merged_stats[state_hash]['successes'][-50:]
                        
                        # 重新计算成功率
                        if merged_stats[state_hash]['successes']:
                            merged_stats[state_hash]['last_success_rate'] = (
                                sum(merged_stats[state_hash]['successes']) / 
                                len(merged_stats[state_hash]['successes'])
                            )
                
                # 广播合并后的统计信息
                broadcast_stats = [merged_stats]
            else:
                broadcast_stats = [None]
            
            # 广播统计信息给所有进程
            dist.broadcast_object_list(broadcast_stats, src=0)
            merged_stats = broadcast_stats[0]
            
            # 更新本地统计信息
            if hasattr(sampler, 'state_stats'):
                sampler.state_stats = merged_stats
    
    def distributed_gradient_accumulation(self, policy, optimizer, cfg_adapter, all_episodes, advantages, config):
        """分布式梯度累积训练"""
        if self.rank == 0:
            print(f"🌐 开始分布式梯度累积训练...")
            print(f"  本地episodes: {len(all_episodes)}")
            print(f"  梯度累积步数: {config.gradient_accumulation_steps}")
            print(f"  训练epochs: {config.num_epochs}")
        
        # 获取原始模型（如果被DDP包装）
        model = policy.module if hasattr(policy, 'module') else policy
        model.train()
        
        total_steps = 0
        all_losses = []
        
        for epoch in range(config.num_epochs):
            if self.rank == 0:
                print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===(Rank {self.rank})")
            
            # 创建episode批次
            episode_batches = []
            batch_size = max(1, len(all_episodes) // config.gradient_accumulation_steps)
            
            for i in range(0, len(all_episodes), batch_size):
                batch_episodes = all_episodes[i:i + batch_size]
                batch_advantages = advantages[i:i + len(batch_episodes)]
                episode_batches.append((batch_episodes, batch_advantages))
            
            if self.rank == 0:
                print(f"创建了 {len(episode_batches)} 个批次，平均每批 {batch_size} episodes")
            
            # 梯度累积循环
            optimizer.zero_grad()
            epoch_losses = []
            
            for step, (batch_episodes, batch_advantages) in enumerate(episode_batches):
                try:
                    # 计算batch损失
                    batch_loss = cfg_adapter.compute_weighted_loss(batch_episodes, batch_advantages)
                    
                    # 归一化损失（梯度累积的关键）
                    normalized_loss = batch_loss / config.gradient_accumulation_steps
                    
                    # 反向传播（累积梯度）
                    normalized_loss.backward()
                    
                    batch_loss_value = batch_loss.item()
                    epoch_losses.append(batch_loss_value)
                    
                    # 每N步或最后一步进行参数更新
                    if (step + 1) % config.gradient_accumulation_steps == 0 or (step + 1) == len(episode_batches):
                        # 梯度裁剪
                        if self.config['algo']['grad_norm_clip'] > 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                policy.parameters(), 
                                self.config['algo']['grad_norm_clip']
                            )
                        
                        # 参数更新 (DDP会自动同步梯度)
                        optimizer.step()
                        optimizer.zero_grad()
                        total_steps += 1
                        
                except Exception as e:
                    if self.rank == 0:
                        print(f"    ❌ Step {step + 1} 训练出错: {e}")
                    continue
            
            # Epoch统计
            if epoch_losses:
                epoch_mean_loss = np.mean(epoch_losses)
                all_losses.extend(epoch_losses)
                if self.rank == 0:
                    print(f"✓ Epoch {epoch + 1} 平均损失: {epoch_mean_loss:.6f}")
            else:
                if self.rank == 0:
                    print(f"⚠️ Epoch {epoch + 1} 没有有效损失")
        
        return {
            'total_steps': total_steps,
            'total_losses': all_losses,
            'mean_loss': np.mean(all_losses) if all_losses else 0.0,
            'rank': self.rank
        }
    
    def aggregate_training_metrics(self, local_metrics):
        """聚合分布式训练指标"""
        if not self.is_distributed:
            return local_metrics
        
        # 收集所有进程的指标
        all_metrics = gather_object_across_processes(local_metrics)
        
        if self.rank == 0:
            # 聚合指标
            aggregated_metrics = {}
            
            # 数值指标求平均
            numeric_keys = ['mean_loss', 'total_steps']
            for key in numeric_keys:
                values = [m.get(key, 0) for m in all_metrics if m.get(key) is not None]
                aggregated_metrics[key] = np.mean(values) if values else 0.0
            
            # 列表指标合并
            all_losses = []
            for m in all_metrics:
                if 'total_losses' in m:
                    all_losses.extend(m['total_losses'])
            aggregated_metrics['total_losses'] = all_losses
            aggregated_metrics['aggregated_mean_loss'] = np.mean(all_losses) if all_losses else 0.0
            
            # 总步数求和
            aggregated_metrics['total_training_steps'] = sum(m.get('total_steps', 0) for m in all_metrics)
            
            return aggregated_metrics
        else:
            return local_metrics
    
    def run_distributed_training(self):
        """执行完整的分布式训练流程"""
        if self.rank == 0:
            print("=== 第10阶段：分布式训练系统 ===")
            print("支持多GPU分布式训练和任务分片")
            print()
        
        # 获取分布式任务分配
        local_tasks = self.get_distributed_tasks()
        
        # 显示关键配置
        if self.rank == 0:
            print(f"🎯 分布式训练配置摘要:")
            print(f"  模型路径: {self.config['policy_path']}")
            print(f"  基准测试: {self.config['task']['benchmark_name']}")
            print(f"  本地任务: {local_tasks}")
            print(f"  训练步数: {self.config['training']['num_train_steps']}")
            print(f"  学习率: {self.config['algo']['lr']}")
            print(f"  批次大小: {self.config['algo']['rloo_batch_size']}")
            print(f"  世界大小: {self.world_size}")
            print(f"  梯度累积: {self.config['algo']['gradient_accumulation_steps']}")
            print("")
        
        # 加载PI0模型 - 优化内存使用
        if self.rank == 0:
            print(f"加载PI0模型: {self.config['policy_path']}")
        
        # 清理内存并设置设备
        torch.cuda.empty_cache()
        torch.cuda.set_device(self.local_rank)
        
        # 直接在目标GPU上加载模型以节省内存
        with torch.cuda.device(self.local_rank):
            policy = PI0Policy.from_pretrained(self.config['policy_path'])
            # 立即移动到目标GPU
            policy = policy.to(self.device)
        
        if self.rank == 0:
            print(f"✓ 模型已加载到 {self.device}")
        
        # 包装为分布式模型
        policy = self.wrap_model_for_distributed(policy)
        
        if self.rank == 0:
            print(f"使用设备: {self.device}")
        
        # 创建优化器
        optimizer = self.create_distributed_optimizer(policy)
        
        # 创建CFG适配器
        if self.rank == 0:
            print(f"\n=== 创建CFG适配器 ===")
        
        os.environ["PI0_DEBUG_SAVE_VIDEO"] = "false"  # 避免重复视频
        cfg_adapter = PI0_CFG_Adapter(policy, norm_stats_path=self.config['norm_stats_path'])
        
        if self.rank == 0:
            print("✓ CFG适配器创建成功")
        
        # 创建环境运行器
        if self.rank == 0:
            print("\\n=== 初始化分布式环境运行器 ===")
        
        sampler_config = self.create_distributed_sampler_config()
        
        env_runner = LIBEROEnvRunner(
            policy=policy,
            benchmark_name=sampler_config.benchmark_name,
            rollouts_per_env=sampler_config.rollouts_per_env,
            num_parallel_envs=sampler_config.num_parallel_envs,
            max_episode_length=sampler_config.max_episode_length,
            task_names_to_use=local_tasks,
            config=self.config,
            rank=self.rank,
            world_size=self.world_size,
            norm_stats_path=self.config['norm_stats_path']
        )
        
        # 创建组件
        reward_function = BinarySuccessReward()
        init_dataset = self.create_initial_state_dataset()
        smart_sampler = EnhancedSmartSampler(sampler_config)
        
        # 创建视频目录
        video_dir = Path(sampler_config.video_dir)
        video_dir.mkdir(exist_ok=True)
        
        # 同步所有进程，确保初始化完成
        if self.is_distributed:
            dist.barrier()
        
        # 🚀 主分布式训练循环开始
        if self.rank == 0:
            print(f"\\n=== 开始分布式训练循环 ===")
            print(f"将进行 {self.config['training']['num_train_steps']} 轮完整的训练迭代")
            print("🔍 使用分布式RIPT智能采样：多GPU并行数据收集")
        
        all_training_metrics = []
        
        for iteration in range(self.config['training']['num_train_steps']):
            if self.rank == 0:
                print(f"\\n" + "="*60)
                print(f"🔄 分布式训练迭代 {iteration + 1}/{self.config['training']['num_train_steps']}")
                print("="*60)
            
            # 🔥 新增：设置当前迭代的任务
            current_task = env_runner.set_current_task_by_cursor()
            
            if current_task is None:  # 该rank没有分配任务
                if self.rank == 0:
                    print(f"⚠️ Rank {self.rank} 没有分配任务，跳过迭代 {iteration + 1}")
                # 推进cursor以保持同步
                env_runner.advance_task_cursor()
                continue
            
            print(f"🎯 Rank {self.rank} 迭代 {iteration + 1}: 处理任务 {current_task}")
            
            # 第1步：分布式智能采样收集数据（只处理当前任务）
            collected_batches = self.distributed_collect_batches(
                env_runner, reward_function, init_dataset, 
                smart_sampler, sampler_config, iteration
            )
            
            if not collected_batches:
                if self.rank == 0:
                    print("⚠️ 智能采样未收集到有用的数据，跳过本轮训练")
                continue
            
            # 第2步：合并所有episodes
            all_episodes = []
            for batch in collected_batches:
                all_episodes.extend(batch)
            
            if not all_episodes:
                if self.rank == 0:
                    print("⚠️ 没有有效的episodes，跳过本轮训练")
                continue
            
            # 第3步：计算优势值
            if self.rank == 0:
                print(f"\\n--- 分布式优势计算 ---")
            
            all_rewards = [ep['total_reward'] for ep in all_episodes]
            rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
            
            # Leave-One-Out优势计算
            advantages = []
            for i in range(len(all_rewards)):
                other_rewards = torch.cat([rewards_tensor[:i], rewards_tensor[i+1:]])
                baseline = other_rewards.mean() if len(other_rewards) > 0 else 0.0
                advantage = rewards_tensor[i] - baseline
                advantages.append(advantage.item())
            
            advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            
            if self.rank == 0:
                print(f"Rank {self.rank} - Episodes总数: {len(all_episodes)}")
                print(f"Rank {self.rank} - 平均奖励: {rewards_tensor.mean().item():.4f}")
                print(f"Rank {self.rank} - 成功率: {sum(ep['success'] for ep in all_episodes) / len(all_episodes):.2f}")
                print(f"Rank {self.rank} - 优势方差: {advantages.var().item():.6f}")
            
            # 第4步：分布式梯度累积训练
            training_metrics = self.distributed_gradient_accumulation(
                policy, optimizer, cfg_adapter, all_episodes, advantages, sampler_config
            )
            
            # 第5步：聚合训练指标
            aggregated_metrics = self.aggregate_training_metrics(training_metrics)
            
            # 第6步：记录迭代指标（只在主进程）
            if self.rank == 0:
                iteration_metrics = {
                    'iteration': iteration + 1,
                    'collected_batches': len(collected_batches),
                    'total_episodes': len(all_episodes),
                    'mean_reward': rewards_tensor.mean().item(),
                    'success_rate': sum(ep['success'] for ep in all_episodes) / len(all_episodes),
                    'advantage_variance': advantages.var().item(),
                    'world_size': self.world_size,
                    **aggregated_metrics
                }
                
                all_training_metrics.append(iteration_metrics)
                
                print(f"\\n✓ 分布式迭代 {iteration + 1} 完成:")
                print(f"  收集batches: {iteration_metrics['collected_batches']}")
                print(f"  总episodes: {iteration_metrics['total_episodes']}")
                print(f"  平均奖励: {iteration_metrics['mean_reward']:.4f}")
                print(f"  成功率: {iteration_metrics['success_rate']:.2f}")
                print(f"  聚合平均损失: {iteration_metrics.get('aggregated_mean_loss', 0.0):.6f}")
                print(f"  总训练步数: {iteration_metrics.get('total_training_steps', 0)}")
            
            # 🔥 新增：迭代结束后推进任务cursor
            next_task = env_runner.advance_task_cursor()
            if self.rank == 0 and next_task:
                print(f"📝 下次迭代将处理任务: {next_task}")
        
        # 训练总结
        if self.rank == 0:
            self.save_distributed_training_results(all_training_metrics)
            self.print_distributed_training_summary(all_training_metrics)
        
        return all_training_metrics
    
    def create_initial_state_dataset(self):
        """创建分布式初始状态数据集"""
        class DistributedInitialStateDataset:
            def __init__(self, num_states=50, state_dim=8, rank=0, world_size=1):
                self.rank = rank
                self.world_size = world_size
                self.states = []
                
                # 🔥 新增：环形索引相关属性
                self.init_state_cursor = 0  # 环形索引游标
                
                # 确保不同进程使用不同的随机种子
                np.random.seed(42 + rank)
                
                for i in range(num_states):
                    base_state = np.zeros(state_dim, dtype=np.float32)
                    
                    if i < num_states // 3:
                        noise = np.random.normal(0, 0.05, state_dim).astype(np.float32)
                    elif i < 2 * num_states // 3:
                        noise = np.random.normal(0, 0.15, state_dim).astype(np.float32)
                    else:
                        noise = np.random.normal(0, 0.25, state_dim).astype(np.float32)
                    
                    state = base_state + noise
                    self.states.append(state)
                
                if self.rank == 0:
                    print(f"✓ 创建了包含 {len(self.states)} 个多样化初始状态的分布式数据集")
            
            def sample_batch(self, batch_size=1):
                """保持向后兼容的随机采样方法"""
                if batch_size > len(self.states):
                    indices = np.random.choice(len(self.states), batch_size, replace=True)
                else:
                    indices = np.random.choice(len(self.states), batch_size, replace=False)
                
                sampled_states = np.array([self.states[i] for i in indices])
                return sampled_states
            
            # 🔥 新增：环形索引采样方法
            def sample_batch_circular(self, batch_size: int) -> np.ndarray:
                """环形索引采样，确保不重复且有序"""
                if len(self.states) == 0:
                    raise ValueError("状态数据集为空，无法采样")
                
                selected_states = []
                
                for i in range(batch_size):
                    idx = (self.init_state_cursor + i) % len(self.states)
                    selected_states.append(self.states[idx])
                
                # 更新cursor
                self.init_state_cursor = (self.init_state_cursor + batch_size) % len(self.states)
                
                return np.array(selected_states)
            
            def get_states_for_envs(self, num_parallel_envs: int, rloo_batch_size: int) -> np.ndarray:
                """为并行环境分配不同的初始状态"""
                total_states_needed = num_parallel_envs * rloo_batch_size
                
                if self.rank == 0:
                    print(f"🔍 环形索引采样: 需要 {total_states_needed} 个状态 "
                          f"({num_parallel_envs} envs × {rloo_batch_size} batch)")
                
                states = self.sample_batch_circular(total_states_needed)
                
                if self.rank == 0:
                    print(f"✓ 已选择状态索引范围: "
                          f"{(self.init_state_cursor - total_states_needed) % len(self.states)} - "
                          f"{(self.init_state_cursor - 1) % len(self.states)}")
                
                return states
        
        return DistributedInitialStateDataset(rank=self.rank, world_size=self.world_size)
    
    def save_distributed_training_results(self, all_training_metrics):
        """保存分布式训练结果"""
        results_file = self.exp_output_dir / "distributed_training_results.json"
        
        results = {
            'config': self.config,
            'training_metrics': all_training_metrics,
            'distributed_info': {
                'world_size': self.world_size,
                'backend': self.config.get('distributed', {}).get('backend', 'nccl'),
                'exp_output_dir': str(self.exp_output_dir),
                'video_dir': str(self.video_dir),
                'timestamp': datetime.now().isoformat(),
            },
            'experiment_summary': {
                'total_iterations': len(all_training_metrics),
                'total_episodes': sum(m['total_episodes'] for m in all_training_metrics),
                'total_training_steps': sum(m.get('total_training_steps', 0) for m in all_training_metrics),
                'final_success_rate': all_training_metrics[-1]['success_rate'] if all_training_metrics else 0.0,
                'final_loss': all_training_metrics[-1].get('aggregated_mean_loss', 0.0) if all_training_metrics else 0.0
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n💾 分布式训练结果已保存到: {results_file}")
    
    def print_distributed_training_summary(self, all_training_metrics):
        """打印分布式训练摘要"""
        print(f"\\n" + "="*80)
        print("🎉 分布式训练循环结束！")
        print("="*80)
        
        print("\\n✅ 成功实现的分布式功能:")
        print("1. ✓ 标准PyTorch分布式训练 (DDP)")
        print("2. ✓ 任务分片和负载均衡")
        print("3. ✓ 分布式数据采样")
        print("4. ✓ 梯度自动同步和聚合")
        print("5. ✓ 分布式统计数据同步")
        print("6. ✓ 多GPU环境优化")
        
        if all_training_metrics:
            print(f"\\n📊 分布式训练轨迹总结:")
            print("Iter | Batches | Episodes | Reward | Success | Loss    | Steps | GPUs")
            print("-----|---------|----------|--------|---------|---------|-------|-----")
            for m in all_training_metrics:
                print(f"{m['iteration']:4d} | {m['collected_batches']:7d} | "
                      f"{m['total_episodes']:8d} | {m['mean_reward']:6.3f} | "
                      f"{m['success_rate']:7.2f} | {m.get('aggregated_mean_loss', 0.0):7.4f} | "
                      f"{m.get('total_training_steps', 0):5d} | {m['world_size']:4d}")
            
            first_loss = all_training_metrics[0].get('aggregated_mean_loss', 0.0)
            last_loss = all_training_metrics[-1].get('aggregated_mean_loss', 0.0)
            if first_loss > 0:
                loss_change = ((last_loss - first_loss) / first_loss) * 100
                print(f"\\n📈 分布式训练趋势:")
                print(f"  初始损失: {first_loss:.6f}")
                print(f"  最终损失: {last_loss:.6f}")
                print(f"  损失变化: {loss_change:+.2f}%")
                print(f"  总GPU数: {self.world_size}")
                print(f"  训练加速比: ~{self.world_size:.1f}x (理论)")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PI0 RIPT分布式训练 - 第10阶段")
    parser.add_argument("--config_path", type=str, required=True,
                        help="YAML配置文件路径")
    parser.add_argument("--override", action='append', dest='overrides',
                        help="覆盖配置参数，格式: key=value")
    
    # 保持与第9阶段的兼容性
    parser.add_argument("--training_iterations", type=int, default=None,
                        help="训练迭代数（覆盖配置文件）")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率（覆盖配置文件）")
    parser.add_argument("--debug_sampling", action='store_true',
                        help="启用采样调试")
    
    return parser.parse_args()

def main():
    print("=== 第10阶段：分布式训练系统 ===")
    print("从单机训练升级为多GPU分布式训练")
    print()
    
    # 解析参数
    args = parse_args()
    
    # 初始化分布式环境
    is_distributed, rank, world_size, local_rank, device = setup_distributed()
    
    if rank == 0:
        print(f"🌐 分布式环境初始化完成:")
        print(f"  Is Distributed: {is_distributed}")
        print(f"  World Size: {world_size}")
        print(f"  Rank: {rank}")
        print(f"  Local Rank: {local_rank}")
        print(f"  Device: {device}")
        print()
    
    # 构建配置覆盖列表
    overrides = args.overrides or []
    
    # 添加兼容性覆盖
    if args.training_iterations is not None:
        overrides.append(f"training.num_train_steps={args.training_iterations}")
    if args.lr is not None:
        overrides.append(f"algo.lr={args.lr}")
    if args.debug_sampling:
        overrides.append("enhanced_sampling.debug_sampling=true")
    
    # 加载和验证配置
    try:
        config = load_config_from_yaml(args.config_path, overrides)
        config = validate_distributed_config(config)
    except Exception as e:
        if rank == 0:
            print(f"❌ 配置加载失败: {e}")
        cleanup_distributed()
        return 1
    
    # 创建并运行分布式训练
    try:
        trainer = DistributedTrainingRunner(config)
        training_metrics = trainer.run_distributed_training()
        
        if rank == 0:
            print("\\n🎯 第10阶段完成！现在支持完整的分布式训练系统")
        
        cleanup_distributed()
        return 0
        
    except Exception as e:
        if rank == 0:
            print(f"❌ 分布式训练执行失败: {e}")
            import traceback
            traceback.print_exc()
        
        cleanup_distributed()
        return 1

if __name__ == "__main__":
    exit(main())