"""
LIBERO Demo数据集加载器

实现与原版RIPT兼容的LIBERO数据集加载功能，为我们的训练系统提供demo初始状态。
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LIBERODemoDataset(Dataset):
    """
    LIBERO Demo数据集
    
    加载LIBERO数据集中的专家演示，提供初始状态和任务信息。
    """
    
    def __init__(self, 
                 data_prefix: str,
                 benchmark_name: str = "LIBERO_SPATIAL",
                 task_names_to_use: Optional[List[str]] = None,
                 load_full_trajectory: bool = False):
        """
        Args:
            data_prefix: LIBERO数据集根目录路径
            benchmark_name: 基准名称 (LIBERO_SPATIAL, LIBERO_GOAL等)
            task_names_to_use: 要使用的任务名称列表
            load_full_trajectory: 是否加载完整轨迹（否则只加载初始状态）
        """
        self.data_prefix = data_prefix
        self.benchmark_name = benchmark_name
        self.load_full_trajectory = load_full_trajectory
        
        # 获取LIBERO基准
        try:
            from libero.libero.benchmark import get_benchmark
            self.benchmark = get_benchmark(benchmark_name)()
            self.n_tasks = self.benchmark.n_tasks
        except ImportError:
            logger.error("无法导入LIBERO，请确保已正确安装")
            raise
        
        # 确定要使用的任务
        if task_names_to_use and len(task_names_to_use) > 0:
            task_names_to_idx = {self.benchmark.get_task(i).name: i for i in range(self.n_tasks)}
            self.task_indices = [task_names_to_idx[name] for name in task_names_to_use]
            logger.info(f"使用指定任务: {task_names_to_use}")
        else:
            self.task_indices = list(range(self.n_tasks))
            logger.info(f"使用所有任务: {self.n_tasks}个")
        
        # 加载数据
        self.demos = []
        self.task_descriptions = []
        self.task_names = []
        self._load_demos()
        
        logger.info(f"LIBERO数据集加载完成: {len(self.demos)}个demo样本")
    
    def _load_demos(self):
        """加载demo数据"""
        for task_idx in self.task_indices:
            task = self.benchmark.get_task(task_idx)
            task_name = task.name
            task_description = task.language
            
            # 🔥 修复：get_task_demonstration返回的是文件路径，不是目录路径
            demo_relative_path = self.benchmark.get_task_demonstration(task_idx)
            demo_path = os.path.join(self.data_prefix, "libero", demo_relative_path)
            
            if not os.path.exists(demo_path):
                logger.warning(f"任务数据路径不存在: {demo_path}")
                continue
            
            # 🔥 直接加载单个演示文件，而不是查找多个demo_i.hdf5文件
            try:
                demo_data = self._load_single_demo(demo_path, task_idx, task_name, task_description)
                if demo_data is not None:
                    self.demos.append(demo_data)
                    self.task_descriptions.append(task_description)
                    self.task_names.append(task_name)
                    logger.info(f"✓ 成功加载demo: {demo_path}")
            except Exception as e:
                logger.warning(f"加载demo失败 {demo_path}: {e}")
                continue
    
    def _load_single_demo(self, demo_path: str, task_idx: int, task_name: str, task_description: str) -> Optional[Dict]:
        """
        加载单个demo文件
        
        Args:
            demo_path: demo文件路径
            task_idx: 任务索引
            task_name: 任务名称
            task_description: 任务描述
            
        Returns:
            demo数据字典或None
        """
        try:
            with h5py.File(demo_path, 'r') as f:
                # 获取基本信息
                demo_data = {
                    'task_id': task_idx,
                    'task_name': task_name,
                    'task_description': task_description,
                    'demo_path': demo_path
                }
                
                # 获取轨迹长度
                if 'data' in f:
                    data_group = f['data']
                    
                    # 获取第一个episode的数据
                    episode_keys = [k for k in data_group.keys() if k.startswith('demo_')]
                    if not episode_keys:
                        return None
                    
                    episode_key = episode_keys[0]
                    episode_data = data_group[episode_key]
                    
                    # 提取初始状态
                    if 'obs' in episode_data:
                        obs_data = episode_data['obs']
                        
                        # 获取初始观测
                        initial_obs = {}
                        for obs_key in obs_data.keys():
                            obs_values = obs_data[obs_key][:]
                            if len(obs_values) > 0:
                                initial_obs[obs_key] = obs_values[0]  # 第一个时间步
                        
                        demo_data['initial_obs'] = initial_obs
                    
                    # 提取初始动作（如果需要）
                    if 'actions' in episode_data:
                        actions = episode_data['actions'][:]
                        if len(actions) > 0:
                            demo_data['initial_action'] = actions[0]
                    
                    # 如果需要完整轨迹
                    if self.load_full_trajectory:
                        demo_data['full_trajectory'] = {
                            'obs': {k: v[:] for k, v in episode_data['obs'].items()},
                            'actions': episode_data['actions'][:] if 'actions' in episode_data else None,
                            'rewards': episode_data.get('rewards', [])[:] if 'rewards' in episode_data else None
                        }
                
                return demo_data
                
        except Exception as e:
            logger.error(f"读取demo文件失败 {demo_path}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.demos)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取demo样本
        
        Returns:
            包含任务信息和初始状态的字典
        """
        demo = self.demos[idx]
        
        # 构建返回数据
        sample = {
            'task_id': torch.tensor(demo['task_id'], dtype=torch.long),
            'task_name': demo['task_name'],
            'task_description': demo['task_description'],
            'demo_path': demo['demo_path']
        }
        
        # 添加初始观测
        if 'initial_obs' in demo:
            sample['initial_obs'] = {}
            for key, value in demo['initial_obs'].items():
                if isinstance(value, np.ndarray):
                    sample['initial_obs'][key] = torch.from_numpy(value.copy())
                else:
                    sample['initial_obs'][key] = torch.tensor(value)
        
        # 添加初始动作
        if 'initial_action' in demo:
            sample['initial_action'] = torch.from_numpy(demo['initial_action'].copy())
        
        # 添加完整轨迹（如果加载了）
        if 'full_trajectory' in demo:
            sample['full_trajectory'] = demo['full_trajectory']
        
        return sample


def create_libero_demo_dataloader(
    data_prefix: str,
    benchmark_name: str = "LIBERO_SPATIAL",
    batch_size: int = 6,
    n_demos: int = 50,
    task_names_to_use: Optional[List[str]] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    load_full_trajectory: bool = False
) -> DataLoader:
    """
    创建LIBERO demo数据加载器
    
    Args:
        data_prefix: LIBERO数据集根目录
        benchmark_name: 基准名称
        batch_size: 批次大小
        n_demos: 每个任务的demo数量
        task_names_to_use: 要使用的任务名称
        shuffle: 是否打乱数据
        num_workers: 数据加载工作进程数
        load_full_trajectory: 是否加载完整轨迹
        
    Returns:
        DataLoader对象
    """
    dataset = LIBERODemoDataset(
        data_prefix=data_prefix,
        benchmark_name=benchmark_name,
        n_demos=n_demos,
        task_names_to_use=task_names_to_use,
        load_full_trajectory=load_full_trajectory
    )
    
    def collate_fn(batch):
        """自定义批次整理函数"""
        # 将batch中的数据整理成字典格式
        collated = {}
        
        # 处理标量字段
        for key in ['task_id', 'task_name', 'task_description', 'demo_path']:
            if key in batch[0]:
                if key == 'task_id':
                    collated[key] = torch.stack([item[key] for item in batch])
                else:
                    collated[key] = [item[key] for item in batch]
        
        # 处理初始观测
        if 'initial_obs' in batch[0]:
            collated['initial_obs'] = {}
            obs_keys = batch[0]['initial_obs'].keys()
            for obs_key in obs_keys:
                obs_values = [item['initial_obs'][obs_key] for item in batch]
                collated['initial_obs'][obs_key] = torch.stack(obs_values)
        
        # 处理初始动作
        if 'initial_action' in batch[0]:
            collated['initial_action'] = torch.stack([item['initial_action'] for item in batch])
        
        return collated
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )


# 测试函数
def test_libero_demo_loader():
    """测试LIBERO demo加载器"""
    # 假设数据路径
    data_prefix = "/path/to/libero/datasets"
    
    try:
        dataloader = create_libero_demo_dataloader(
            data_prefix=data_prefix,
            benchmark_name="LIBERO_SPATIAL",
            batch_size=3,
            n_demos=5,  # 测试时只加载5个demo
            shuffle=True
        )
        
        print(f"数据集大小: {len(dataloader.dataset)}")
        
        # 测试加载一个batch
        for batch in dataloader:
            print(f"Batch大小: {len(batch['task_id'])}")
            print(f"任务ID: {batch['task_id']}")
            print(f"任务名称: {batch['task_name']}")
            print(f"任务描述: {batch['task_description'][:2]}")  # 只显示前2个
            
            if 'initial_obs' in batch:
                print(f"初始观测键: {list(batch['initial_obs'].keys())}")
                for key, value in batch['initial_obs'].items():
                    print(f"  {key}: {value.shape}")
            
            break  # 只测试第一个batch
            
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    test_libero_demo_loader()
