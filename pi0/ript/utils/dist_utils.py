"""
分布式训练工具模块
基于原版RIPT的分布式架构，提供完整的分布式协调功能
"""

import os
import json
import pickle
import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import time
import fcntl
import uuid
from datetime import datetime

def is_distributed():
    """检查是否在分布式环境中运行"""
    return dist.is_initialized()

def get_rank():
    """获取当前进程的rank"""
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size():
    """获取总进程数"""
    return dist.get_world_size() if dist.is_initialized() else 1

def barrier():
    """同步所有进程"""
    if dist.is_initialized():
        dist.barrier()

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    """在所有进程间聚合张量"""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor

def all_gather_tensors(tensor):
    """收集所有进程的张量"""
    if not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return gathered_tensors

def all_gather_objects(obj):
    """收集所有进程的对象"""
    if not dist.is_initialized():
        return [obj]
    
    world_size = dist.get_world_size()
    gathered_objects = [None] * world_size
    dist.all_gather_object(gathered_objects, obj)
    return gathered_objects

def broadcast_object(obj, src=0):
    """从源进程广播对象到所有进程"""
    if not dist.is_initialized():
        return obj
    
    objects = [obj]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]

class DistributedFileCounter:
    """
    分布式文件计数器，用于进程间协调
    基于RIPT的FileGlobalCounter实现
    """
    
    def __init__(self, filename: str):
        self.filename = filename
        self.setup_counter()
    
    def setup_counter(self):
        """设置计数器文件"""
        if get_rank() == 0:
            # 主进程创建文件
            if not os.path.exists(self.filename):
                with open(self.filename, "w") as f:
                    f.write("0")
        
        # 同步所有进程
        barrier()
    
    def reset(self, value=0):
        """重置计数器"""
        if get_rank() == 0:
            with open(self.filename, "w") as f:
                f.write(str(value))
        barrier()
    
    def update(self, increment=1):
        """原子性更新计数器"""
        with open(self.filename, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                content = f.read().strip()
                current = int(content) if content else 0
                current += increment
                f.seek(0)
                f.write(str(current))
                f.truncate()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
            return current
    
    def get(self):
        """获取当前计数值"""
        with open(self.filename, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                content = f.read().strip()
                current = int(content) if content else 0
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
            return current
    
    def cleanup(self):
        """清理计数器文件"""
        if get_rank() == 0 and os.path.exists(self.filename):
            try:
                os.remove(self.filename)
            except:
                pass

class DistributedStatsSynchronizer:
    """
    分布式统计数据同步器
    用于同步采样统计、训练指标等数据
    """
    
    def __init__(self, temp_dir: str = "/tmp/ript_sync"):
        self.temp_dir = Path(temp_dir)
        self.rank = get_rank()
        self.world_size = get_world_size()
        
        # 创建临时目录
        if self.rank == 0:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        barrier()
    
    def sync_stats_dict(self, local_stats: Dict[str, Any]) -> Dict[str, Any]:
        """同步统计字典"""
        if not is_distributed():
            return local_stats
        
        # 每个进程保存本地统计
        temp_file = self.temp_dir / f"stats_rank_{self.rank}.pkl"
        with open(temp_file, 'wb') as f:
            pickle.dump(local_stats, f)
        
        barrier()
        
        # 主进程收集和合并统计
        if self.rank == 0:
            merged_stats = {}
            
            for r in range(self.world_size):
                rank_file = self.temp_dir / f"stats_rank_{r}.pkl"
                if rank_file.exists():
                    with open(rank_file, 'rb') as f:
                        rank_stats = pickle.load(f)
                    
                    # 合并统计数据
                    for key, value in rank_stats.items():
                        if key not in merged_stats:
                            merged_stats[key] = []
                        
                        if isinstance(value, list):
                            merged_stats[key].extend(value)
                        else:
                            merged_stats[key].append(value)
            
            # 保存合并后的统计
            merged_file = self.temp_dir / "merged_stats.pkl"
            with open(merged_file, 'wb') as f:
                pickle.dump(merged_stats, f)
        
        barrier()
        
        # 所有进程加载合并后的统计
        merged_file = self.temp_dir / "merged_stats.pkl"
        with open(merged_file, 'rb') as f:
            merged_stats = pickle.load(f)
        
        barrier()
        return merged_stats
    
    def sync_rollout_stats(self, local_rollout_stats: Dict[str, Dict]) -> Dict[str, Dict]:
        """同步rollout统计数据"""
        if not is_distributed():
            return local_rollout_stats
        
        # 保存本地rollout统计
        temp_file = self.temp_dir / f"rollout_stats_rank_{self.rank}.pkl"
        with open(temp_file, 'wb') as f:
            pickle.dump(local_rollout_stats, f)
        
        barrier()
        
        # 主进程合并rollout统计
        if self.rank == 0:
            merged_rollout_stats = {}
            
            for r in range(self.world_size):
                rank_file = self.temp_dir / f"rollout_stats_rank_{r}.pkl"
                if rank_file.exists():
                    with open(rank_file, 'rb') as f:
                        rank_rollout_stats = pickle.load(f)
                    
                    # 合并rollout统计
                    for state_hash, state_info in rank_rollout_stats.items():
                        if state_hash not in merged_rollout_stats:
                            merged_rollout_stats[state_hash] = {
                                'attempts': 0,
                                'successes': [],
                                'last_success_rate': 0.0
                            }
                        
                        merged_rollout_stats[state_hash]['attempts'] += state_info.get('attempts', 0)
                        merged_rollout_stats[state_hash]['successes'].extend(state_info.get('successes', []))
                        
                        # 限制历史记录长度
                        if len(merged_rollout_stats[state_hash]['successes']) > 50:
                            merged_rollout_stats[state_hash]['successes'] = merged_rollout_stats[state_hash]['successes'][-50:]
                        
                        # 重新计算成功率
                        if merged_rollout_stats[state_hash]['successes']:
                            merged_rollout_stats[state_hash]['last_success_rate'] = (
                                sum(merged_rollout_stats[state_hash]['successes']) / 
                                len(merged_rollout_stats[state_hash]['successes'])
                            )
            
            # 保存合并后的rollout统计
            merged_file = self.temp_dir / "merged_rollout_stats.pkl"
            with open(merged_file, 'wb') as f:
                pickle.dump(merged_rollout_stats, f)
        
        barrier()
        
        # 所有进程加载合并后的rollout统计
        merged_file = self.temp_dir / "merged_rollout_stats.pkl"
        with open(merged_file, 'rb') as f:
            merged_rollout_stats = pickle.load(f)
        
        barrier()
        return merged_rollout_stats
    
    def cleanup(self):
        """清理临时文件"""
        if self.rank == 0:
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass

def sync_rollout_results_via_file(rollout_results, logger, step, is_train=False):
    """
    通过文件同步rollout结果
    基于RIPT的sync_rollout_results_via_file实现
    """
    if not is_distributed():
        # 非分布式环境直接记录
        if logger and get_rank() == 0:
            prefix = "rl_train" if is_train else "rl_eval"
            for key, value in rollout_results.items():
                logger.log_metric(f"{prefix}/{key}", value, step)
        return rollout_results
    
    rank = get_rank()
    world_size = get_world_size()
    
    # 创建临时文件目录
    temp_dir = Path(f"/tmp/rollout_sync_{step}")
    if rank == 0:
        temp_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    
    # 每个进程保存结果
    rank_file = temp_dir / f"rollout_results_rank_{rank}.json"
    with open(rank_file, 'w') as f:
        json.dump(rollout_results, f)
    
    barrier()
    
    # 主进程收集和聚合结果
    if rank == 0:
        all_results = {}
        task_results = {}
        
        for r in range(world_size):
            result_file = temp_dir / f"rollout_results_rank_{r}.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    rank_results = json.load(f)
                
                # 聚合结果
                for key, value in rank_results.items():
                    if key not in all_results:
                        all_results[key] = []
                    all_results[key].append(value)
        
        # 平均化结果
        aggregated_results = {}
        for key, values in all_results.items():
            if all(isinstance(v, (int, float)) for v in values):
                aggregated_results[key] = np.mean(values)
            else:
                aggregated_results[key] = values
        
        # 记录到logger
        if logger:
            prefix = "rl_train" if is_train else "rl_eval"
            for key, value in aggregated_results.items():
                if isinstance(value, (int, float)):
                    logger.log_metric(f"{prefix}/{key}", value, step)
        
        # 清理临时文件
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        return aggregated_results
    else:
        barrier()
        return rollout_results

def setup_distributed_counter(tmp_dir="/tmp", counter_name="global_counter"):
    """
    设置分布式计数器
    基于RIPT的setup_file_counter实现
    """
    if is_distributed():
        rank = get_rank()
        if rank == 0:
            counter_filename = f"{tmp_dir}/{counter_name}_{uuid.uuid4().hex}.txt"
        else:
            counter_filename = ""
        
        filename_list = [counter_filename]
        dist.broadcast_object_list(filename_list, src=0)
        counter_filename = filename_list[0]
        
        file_counter = DistributedFileCounter(counter_filename)
        if rank == 0:
            file_counter.reset(0)
        barrier()
    else:
        counter_filename = f"{tmp_dir}/{counter_name}_{uuid.uuid4().hex}.txt"
        file_counter = DistributedFileCounter(counter_filename)
        file_counter.reset(0)
    
    return file_counter, counter_filename

def log_metrics_distributed(metrics: Dict[str, float], step: int, logger=None):
    """分布式环境下记录指标"""
    if not is_distributed():
        if logger:
            for key, value in metrics.items():
                logger.log_metric(key, value, step)
        return metrics
    
    rank = get_rank()
    
    # 收集所有进程的指标
    all_metrics = all_gather_objects(metrics)
    
    if rank == 0 and logger:
        # 聚合指标
        aggregated_metrics = {}
        
        for metric_dict in all_metrics:
            for key, value in metric_dict.items():
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = []
                aggregated_metrics[key].append(value)
        
        # 计算平均值并记录
        for key, values in aggregated_metrics.items():
            if all(isinstance(v, (int, float)) for v in values):
                avg_value = np.mean(values)
                logger.log_metric(f"distributed/{key}", avg_value, step)
                logger.log_metric(f"distributed/{key}_std", np.std(values), step)
    
    return metrics

def reduce_gradients(model, world_size):
    """手动聚合梯度（用于非DDP情况）"""
    if not is_distributed():
        return
    
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size

def save_checkpoint_distributed(model, optimizer, step, save_path, keep_latest=3):
    """分布式环境下保存检查点"""
    if get_rank() != 0:
        return  # 只在主进程保存
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存检查点
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'timestamp': datetime.now().isoformat(),
    }
    
    checkpoint_path = save_path / f"checkpoint_step_{step}.pth"
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最新检查点链接
    latest_path = save_path / "checkpoint_latest.pth"
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(checkpoint_path.name)
    
    # 清理旧检查点
    if keep_latest > 0:
        checkpoints = sorted(save_path.glob("checkpoint_step_*.pth"), key=os.path.getmtime)
        if len(checkpoints) > keep_latest:
            for old_checkpoint in checkpoints[:-keep_latest]:
                try:
                    old_checkpoint.unlink()
                except:
                    pass
    
    print(f"✓ 检查点已保存: {checkpoint_path}")

def load_checkpoint_distributed(model, optimizer, checkpoint_path, device):
    """分布式环境下加载检查点"""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        if checkpoint_path.name == "checkpoint_latest.pth":
            # 尝试找到最新的检查点
            parent_dir = checkpoint_path.parent
            checkpoints = sorted(parent_dir.glob("checkpoint_step_*.pth"), key=os.path.getmtime)
            if checkpoints:
                checkpoint_path = checkpoints[-1]
            else:
                raise FileNotFoundError(f"没有找到检查点文件: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    print(f"📥 加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    step = checkpoint.get('step', 0)
    print(f"✓ 检查点加载完成，从步骤 {step} 继续")
    
    return step

class DistributedLogger:
    """分布式日志记录器"""
    
    def __init__(self, log_dir: str, rank: int = None):
        self.rank = rank if rank is not None else get_rank()
        self.world_size = get_world_size()
        self.log_dir = Path(log_dir)
        
        if self.rank == 0:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 每个进程有自己的日志文件
        self.log_file = self.log_dir / f"train_rank_{self.rank}.log"
        
        barrier()
    
    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [RANK {self.rank}] [{level}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        # 主进程也打印到控制台
        if self.rank == 0:
            print(f"[{level}] {message}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """记录指标"""
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.log(f"Step {step}: {metrics_str}")