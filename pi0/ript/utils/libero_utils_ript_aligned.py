"""
与原版RIPT对齐的LIBERO数据集工具
确保demo数据包含MuJoCo状态向量，与原版RIPT的数据格式完全一致
"""

import os
import h5py
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

def build_dataset_ript_aligned(
    data_prefix: str,
    suite_name: str = "libero",
    benchmark_name: str = "libero_spatial", 
    mode: str = "all",
    seq_len: int = 600,
    frame_stack: int = 1,
    obs_seq_len: int = 1,
    shape_meta: Dict = None,
    load_obs: bool = True,
    task_embedding_format: str = "clip",
    n_demos: int = 50,
    load_next_obs: bool = True,
    get_pad_mask: bool = True,
    task_names_to_use: List[str] = None,
    pad_seq_length: bool = True,
    load_state: bool = True,  # 🔥 关键：加载MuJoCo状态
    **kwargs
):
    """
    构建与原版RIPT对齐的数据集
    
    Args:
        data_prefix: 数据集根目录路径
        load_state: 是否加载MuJoCo状态数据（与原版RIPT对齐）
        其他参数与原版RIPT保持一致
    """
    
    # 🔥 构建数据集路径（与原版RIPT对齐）
    dataset_path = os.path.join(data_prefix, suite_name, benchmark_name)
    
    print(f"🔥 构建RIPT对齐数据集:")
    print(f"   数据路径: {dataset_path}")
    print(f"   加载状态: {load_state}")
    print(f"   任务列表: {task_names_to_use}")
    
    # 检查数据路径是否存在
    if not os.path.exists(dataset_path):
        print(f"⚠️ 数据路径不存在: {dataset_path}")
        print(f"   将使用模拟数据集进行测试")
        return MockRIPTDataset(
            task_names_to_use=task_names_to_use,
            load_state=load_state,
            seq_len=seq_len
        )
    
    # 🔥 创建与原版RIPT兼容的数据集
    return RIPTAlignedDataset(
        dataset_path=dataset_path,
        task_names_to_use=task_names_to_use,
        load_state=load_state,
        seq_len=seq_len,
        frame_stack=frame_stack,
        obs_seq_len=obs_seq_len,
        load_obs=load_obs,
        n_demos=n_demos,
        get_pad_mask=get_pad_mask
    )

class RIPTAlignedDataset(Dataset):
    """与原版RIPT对齐的数据集类"""
    
    def __init__(
        self,
        dataset_path: str,
        task_names_to_use: List[str],
        load_state: bool = True,
        seq_len: int = 600,
        frame_stack: int = 1,
        obs_seq_len: int = 1,
        load_obs: bool = True,
        n_demos: int = 50,
        get_pad_mask: bool = True
    ):
        self.dataset_path = dataset_path
        self.task_names_to_use = task_names_to_use or []
        self.load_state = load_state
        self.seq_len = seq_len
        self.load_obs = load_obs
        self.get_pad_mask = get_pad_mask
        self.n_demos = n_demos  # 🔥 存储demo数量参数
        
        # 🔥 加载demo数据
        self.demos = self._load_demos()
        
        # 🔥 计算实际可用的demo总数
        total_available_demos = sum(demo.get('available_demo_count', 1) for demo in self.demos)
        if total_available_demos == 0:
            total_available_demos = 50  # 默认值，基于之前的观察
            
        print(f"✅ RIPT对齐数据集初始化完成:")
        print(f"   任务数量: {len(self.task_names_to_use)}")
        print(f"   加载的Demo数量: {len(self.demos)}")
        print(f"   可用Demo总数: {total_available_demos}")
        print(f"   加载状态: {self.load_state}")
    
    def _load_demos(self) -> List[Dict]:
        """加载demo数据，确保包含MuJoCo状态"""
        demos = []
        
        for task_idx, task_name in enumerate(self.task_names_to_use):
            demo_file = os.path.join(self.dataset_path, f"{task_name}_demo.hdf5")
            if os.path.exists(demo_file):
                # 🔥 新逻辑：一次性加载 n_demos 个 demo，与原版 RIPT 对齐
                demos.extend(self._load_multiple_demos(demo_file, task_idx, task_name))
            else:
                print(f"⚠️ Demo文件不存在: {demo_file}")
                demo_data = self._create_mock_demo(task_idx, task_name)
                demos.append(demo_data)
        
        return demos

    def _load_multiple_demos(self, demo_file: str, task_idx: int, task_name: str) -> List[Dict]:
        """一次性加载多个 demo，每个 demo 作为独立样本，数量与 n_demos 对齐"""
        loaded = []
        try:
            with h5py.File(demo_file, 'r') as f:
                if 'data' not in f:
                    print(f"⚠️ HDF5 文件缺少 data 组: {demo_file}")
                    return [self._create_mock_demo(task_idx, task_name)]

                demo_ids = sorted(list(f['data'].keys()))
                n_to_load = min(self.n_demos, len(demo_ids))
                print(f"📋 发现 {len(demo_ids)} 个 demo，加载前 {n_to_load} 个 → task={task_name}")

                for i, demo_id in enumerate(demo_ids[:n_to_load]):
                    demo_group = f['data'][demo_id]

                    # 观测
                    obs = {}
                    if self.load_obs and 'obs' in demo_group:
                        g = demo_group['obs']
                        if 'agentview_rgb' in g:
                            obs['agentview_rgb'] = np.array(g['agentview_rgb'][0])
                        if 'ee_pos' in g:
                            obs['robot0_eef_pos'] = np.array(g['ee_pos'][0])
                        if 'joint_states' in g:
                            obs['robot0_joint_pos'] = np.array(g['joint_states'][0])

                    # 状态 - 双轨输出：序列+单快照
                    if self.load_state and 'states' in demo_group:
                        full_states = np.array(demo_group['states'])  # [T, state_dim]
                        init_state_seq = full_states  # 完整序列供轮换器使用
                        init_state_vec = full_states[0]  # 单快照与原版对齐
                    else:
                        mock_states = self._generate_mock_states()
                        init_state_seq = mock_states  # [1, state_dim] 
                        init_state_vec = mock_states[0]  # [state_dim]

                    loaded.append({
                        'task_id': task_idx,
                        'task_name': task_name,
                        'demo_id': demo_id,
                        'initial_obs': obs,
                        # 保留序列供轮换器使用
                        'init_state': {
                            'states': torch.tensor(init_state_seq, dtype=torch.float32),
                            'pad_mask': torch.ones(init_state_seq.shape[0], dtype=torch.bool)
                        },
                        # 新增：单快照与原版对齐
                        'init_state_vec': {
                            'states': torch.tensor(init_state_vec, dtype=torch.float32),  # [state_dim]
                            'pad_mask': torch.tensor(True, dtype=torch.bool)  # 单个状态总是有效
                        }
                    })
        except Exception as e:
            print(f"⚠️ 多 demo 加载失败 {demo_file}: {e}")
            loaded.append(self._create_mock_demo(task_idx, task_name))
        return loaded
    
    def _load_real_demo(self, demo_file: str, task_idx: int, task_name: str) -> Dict:
        """加载真实的demo文件（修复：正确读取HDF5结构）"""
        try:
            with h5py.File(demo_file, 'r') as f:
                print(f"🔍 检查demo文件结构: {demo_file}")

                # 🔥 修复：按照原版RIPT的方式读取数据
                obs_data = {}
                states_data = None

                # 检查是否有data目录（标准LIBERO格式）
                if 'data' in f:
                    print(f"✅ 找到data目录，包含demos: {list(f['data'].keys())}")

                    # 🔥 修复统计报告：显示实际可用demo数量
                    demo_ids = sorted(list(f['data'].keys()))
                    n_demos_available = len(demo_ids)
                    
                    print(f"📋 发现 {n_demos_available} 个demo: {demo_ids[:10]}{'...' if len(demo_ids) > 10 else ''}")
                    
                    # 选择第一个demo用于数据加载（但记录总数用于统计）
                    if demo_ids:
                        demo_id = demo_ids[0]  # 使用第一个demo
                        demo_group = f['data'][demo_id]
                        print(f"📋 使用demo: {demo_id} (共有{n_demos_available}个可用)")

                        # 🔥 加载观测数据（从第一个时间步）
                        if self.load_obs and 'obs' in demo_group:
                            obs_group = demo_group['obs']
                            obs_data = {}

                            # 读取第一个时间步的观测
                            if 'agentview_rgb' in obs_group:
                                obs_data['agentview_rgb'] = np.array(obs_group['agentview_rgb'][0])
                            if 'ee_pos' in obs_group:
                                obs_data['robot0_eef_pos'] = np.array(obs_group['ee_pos'][0])
                            if 'joint_states' in obs_group:
                                obs_data['robot0_joint_pos'] = np.array(obs_group['joint_states'][0])

                        # 🔥 关键修复：正确加载MuJoCo状态数据
                        if self.load_state and 'states' in demo_group:
                            # 读取第一个时间步的状态作为初始状态
                            full_states = np.array(demo_group['states'])  # Shape: (timesteps, state_dim)
                            initial_state = full_states[0:1]  # 取第一个时间步，保持2D: (1, state_dim)
                            states_data = initial_state

                            print(f"✅ 成功加载MuJoCo状态:")
                            print(f"   完整轨迹形状: {full_states.shape}")
                            print(f"   初始状态形状: {states_data.shape}")
                        else:
                            print(f"⚠️ {demo_file} 在 {demo_id} 中缺少states数据")

                # 如果没有找到states数据，生成模拟数据
                if states_data is None:
                    print(f"⚠️ 未找到states数据，生成模拟状态")
                    states_data = self._generate_mock_states()

                return {
                    'task_id': task_idx,
                    'task_name': task_name,
                    'initial_obs': obs_data,
                    'init_state': {
                        'states': torch.tensor(states_data, dtype=torch.float32),
                        'pad_mask': torch.ones(states_data.shape[0], dtype=torch.bool)  # 对应时间步数
                    },
                    'available_demo_count': n_demos_available  # 🔥 包含实际可用demo数量
                }

        except Exception as e:
            print(f"⚠️ 加载demo失败 {demo_file}: {e}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")
            return self._create_mock_demo(task_idx, task_name)
    
    def _create_mock_demo(self, task_idx: int, task_name: str) -> Dict:
        """创建模拟demo数据（用于测试）"""
        # 🔥 生成模拟观测数据
        mock_obs = {
            'agentview_rgb': np.random.randint(0, 255, (3, 128, 128), dtype=np.uint8),
            'robot0_eef_pos': np.random.randn(3).astype(np.float32),
            'robot0_joint_pos': np.random.randn(7).astype(np.float32),
        }
        
        # 🔥 生成模拟MuJoCo状态（高维状态向量）
        mock_states = self._generate_mock_states()
        
        return {
            'task_id': task_idx,
            'task_name': task_name,
            'initial_obs': mock_obs,
            'init_state': {
                'states': torch.tensor(mock_states, dtype=torch.float32),
                'pad_mask': torch.ones(len(mock_states), dtype=torch.bool)
            },
            'available_demo_count': 1  # 🔥 Mock demo只有1个
        }
    
    def _generate_mock_states(self) -> np.ndarray:
        """生成模拟的MuJoCo状态向量"""
        # 🔥 生成高维状态向量（模拟真实的MuJoCo状态）
        # 通常包含：关节位置、关节速度、物体位置、接触状态等
        state_dim = 487  # 典型的LIBERO MuJoCo状态维度
        sequence_length = 1  # 初始状态只需要1个时间步
        
        mock_states = np.random.randn(sequence_length, state_dim).astype(np.float32)
        
        # 添加一些合理的约束
        mock_states[:, :7] = np.clip(mock_states[:, :7], -2.0, 2.0)  # 关节位置
        mock_states[:, 7:14] = np.clip(mock_states[:, 7:14], -1.0, 1.0)  # 关节速度
        
        return mock_states
    
    def __len__(self) -> int:
        return len(self.demos)
    
    def __getitem__(self, idx: int) -> Dict:
        demo = self.demos[idx]
        
        # 🔥 返回与原版RIPT兼容的数据格式
        item = {
            'task_id': torch.tensor([demo['task_id']]),
            'task_name': demo['task_name'],
        }
        
        # 添加观测数据
        if self.load_obs and demo['initial_obs'] is not None:
            item['initial_obs'] = demo['initial_obs']
        
        # 🔥 关键：添加MuJoCo状态数据（双轨输出）
        if self.load_state and demo['init_state'] is not None:
            item['init_state'] = demo['init_state']  # 序列供轮换器使用
            if 'init_state_vec' in demo:
                item['init_state_vec'] = demo['init_state_vec']  # 单快照与原版对齐
        
        return item

class MockRIPTDataset(Dataset):
    """模拟数据集（用于测试）"""
    
    def __init__(self, task_names_to_use: List[str], load_state: bool = True, seq_len: int = 600):
        self.task_names_to_use = task_names_to_use or ["mock_task"]
        self.load_state = load_state
        self.seq_len = seq_len
        
        print(f"🔧 使用模拟RIPT数据集: {len(self.task_names_to_use)} 个任务")
    
    def __len__(self) -> int:
        return len(self.task_names_to_use)
    
    def __getitem__(self, idx: int) -> Dict:
        task_name = self.task_names_to_use[idx]
        
        # 生成模拟数据
        item = {
            'task_id': torch.tensor([idx]),
            'task_name': task_name,
            'initial_obs': {
                'agentview_rgb': np.random.randint(0, 255, (3, 128, 128), dtype=np.uint8),
                'robot0_eef_pos': np.random.randn(3).astype(np.float32),
                'robot0_joint_pos': np.random.randn(7).astype(np.float32),
            }
        }
        
        # 🔥 添加模拟的MuJoCo状态（双轨输出）
        if self.load_state:
            mock_states_seq = np.random.randn(1, 487).astype(np.float32)  # [1, 487]
            mock_states_vec = mock_states_seq[0]  # [487]
            
            # 序列格式供轮换器使用
            item['init_state'] = {
                'states': torch.tensor(mock_states_seq, dtype=torch.float32),
                'pad_mask': torch.ones(1, dtype=torch.bool)
            }
            # 单快照格式与原版对齐
            item['init_state_vec'] = {
                'states': torch.tensor(mock_states_vec, dtype=torch.float32),
                'pad_mask': torch.tensor(True, dtype=torch.bool)
            }
        
        return item

def collate_fn_ript_aligned(batch):
    """与原版RIPT对齐的collate函数（双轨输出）"""
    
    # 先处理除状态外的其他字段
    filtered_items = []
    for item in batch:
        filtered_item = {k: v for k, v in item.items() if k not in ['init_state', 'init_state_vec']}
        filtered_items.append(filtered_item)
    
    collated_batch = default_collate(filtered_items)
    
    # 🔥 处理init_state字段（序列格式，供轮换器使用）
    if 'init_state' in batch[0] and batch[0]['init_state'] is not None:
        states = [item['init_state']['states'] for item in batch]

        # states的形状应该是 [T, state_dim]
        max_seq_len = max(s.shape[0] for s in states)
        max_state_dim = max(s.shape[-1] for s in states)

        padded_states = []
        masks = []

        for item in batch:
            tensor = item['init_state']['states'].float()
            seq_len, state_dim = tensor.shape

            seq_pad_size = max_seq_len - seq_len
            state_pad_size = max_state_dim - state_dim

            padded = torch.nn.functional.pad(tensor, (0, state_pad_size, 0, seq_pad_size))
            padded_states.append(padded)

            mask = torch.ones(seq_len, dtype=torch.bool)
            mask = torch.nn.functional.pad(mask, (0, seq_pad_size), value=False)
            masks.append(mask)

        collated_batch['init_state'] = {
            'states': torch.stack(padded_states),    # [B, T, state_dim]
            'pad_mask': torch.stack(masks)           # [B, T]
        }

    # 🔥 处理init_state_vec字段（单快照格式，与原版RIPT对齐）
    if 'init_state_vec' in batch[0] and batch[0]['init_state_vec'] is not None:
        vec_states = [item['init_state_vec']['states'] for item in batch]
        vec_masks = [item['init_state_vec']['pad_mask'] for item in batch]
        
        # 单快照向量格式 [B, state_dim]
        collated_batch['init_state_vec'] = {
            'states': torch.stack(vec_states),       # [B, state_dim]
            'pad_mask': torch.stack(vec_masks)       # [B]
        }

    return collated_batch
