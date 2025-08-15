"""
OpenPI-RIPT兼容数据包装器
支持在OpenPI标准格式基础上添加可选的RIPT字段（advantages, init_hash）
确保向后兼容性，不破坏现有训练流程
"""

import copy
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Union
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torchvision.transforms.v2 import Resize
from utils.normalizers import Normalizer
from utils.state_dimension_adapter import StateDimensionAdapter, StateDimensionConfig, create_pi0_state_adapter


class OpenPIRiptDatasetWrapper(Dataset):
    """
    OpenPI和RIPT兼容的数据包装器
    
    Features:
    - 保持OpenPI标准数据格式
    - 支持可选RIPT字段（advantages, init_hash）
    - 向后兼容现有数据集
    - 支持状态维度自动对齐
    """
    
    def __init__(
        self,
        repo_id: str = "ZibinDong/so100_grab_screwdriver",
        image_size: int = 224,
        action_chunk_size: int = 50,
        fps: float = 30.0,
        target_state_dim: Optional[int] = None,
        enable_ript_fields: bool = False,
        default_advantages: Optional[torch.Tensor] = None,
        **dataset_kwargs
    ):
        """
        Args:
            repo_id: HuggingFace数据集ID
            image_size: 图像缩放尺寸
            action_chunk_size: 动作chunk长度
            fps: 帧率，用于计算时间戳
            target_state_dim: 目标状态维度，用于维度对齐
            enable_ript_fields: 是否启用RIPT字段
            default_advantages: 默认优势值
            **dataset_kwargs: 传递给LeRobotDataset的额外参数
        """
        
        self.repo_id = repo_id
        self.image_size = image_size
        self.action_chunk_size = action_chunk_size
        self.fps = fps
        self.target_state_dim = target_state_dim
        self.enable_ript_fields = enable_ript_fields
        
        # 初始化状态维度适配器
        self.state_adapter = None
        if target_state_dim is not None:
            self.state_adapter = create_pi0_state_adapter(
                target_state_dim=target_state_dim,
                padding_mode="zero",
                truncation_mode="first"
            )
        
        # 设置图像变换
        self.image_transforms = Resize((image_size, image_size))
        
        # 设置delta_timestamps (OpenPI标准)
        self.delta_timestamps = {
            "observation.images.base": [0],
            "observation.images.wrist": [0],
            "observation.state": [0],
            "action": [i / fps for i in range(action_chunk_size)],
        }
        
        # 初始化LeRobotDataset
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            image_transforms=self.image_transforms,
            delta_timestamps=self.delta_timestamps,
            **dataset_kwargs
        )
        
        # 初始化归一化器
        self.normalizer = Normalizer(
            norm_stats=self.dataset.meta.stats,
            norm_type={
                "observation.images.base": "identity",
                "observation.images.wrist": "identity", 
                "observation.state": "meanstd",
                "action": "std",
            },
        )
        
        # 设置默认优势值
        if default_advantages is None:
            self.default_advantages = torch.zeros(action_chunk_size)
        else:
            self.default_advantages = default_advantages
            
        print(f"✅ OpenPIRiptDatasetWrapper初始化完成")
        print(f"   - 数据集: {repo_id}")
        print(f"   - 图像尺寸: {image_size}x{image_size}")
        print(f"   - 动作chunk大小: {action_chunk_size}")
        print(f"   - RIPT字段: {'启用' if enable_ript_fields else '禁用'}")
        print(f"   - 数据集大小: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取数据样本，返回OpenPI标准格式 + 可选RIPT字段
        
        Returns:
            Dict containing:
            - image: {"base_0_rgb": tensor, "left_wrist_0_rgb": tensor}
            - state: 当前状态 (可能经过维度对齐)
            - action: 动作chunk
            - action_is_pad: padding掩码
            - prompt: 任务描述
            - advantages: (可选) RIPT优势值
            - init_hash: (可选) 初始状态哈希
        """
        try:
            # 1. 获取原始数据
            item = self.dataset[idx]
            
            # 2. 处理相对动作 (OpenPI标准)
            item["action"] = item["action"] - item["observation.state"]
            
            # 3. 归一化
            normalized_item = self.normalizer.normalize(item)
            
            # 4. 处理图像 (转换为uint8格式)
            base_image = (normalized_item["observation.images.base"] * 255).to(torch.uint8)
            wrist_image = (normalized_item["observation.images.wrist"] * 255).to(torch.uint8)
            
            # 5. 处理状态维度对齐
            current_state = normalized_item["observation.state"][0]  # 取当前时刻
            if self.state_adapter is not None:
                current_state = self.state_adapter.align_state_dimension(current_state, batch_idx=idx)
            
            # 6. 构造OpenPI标准格式
            sample = {
                "image": {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": wrist_image
                },
                "state": current_state,
                "action": normalized_item["action"],
                "action_is_pad": normalized_item["action_is_pad"],
                "prompt": item["task"],
            }
            
            # 7. 添加可选RIPT字段
            if self.enable_ript_fields:
                # 优势值：优先使用数据中的，否则使用默认值
                sample["advantages"] = item.get("advantages", self.default_advantages.clone())
                
                # 初始状态哈希：优先使用数据中的，否则计算
                sample["init_hash"] = item.get("init_hash", self._compute_init_hash(current_state))
            
            return sample
            
        except Exception as e:
            print(f"❌ 数据加载错误 (idx={idx}): {e}")
            # 返回一个安全的默认样本
            return self._create_fallback_sample()
    
    def get_state_adapter_stats(self) -> Dict[str, Any]:
        """
        获取状态适配器统计信息
        
        Returns:
            状态适配器的转换统计
        """
        if self.state_adapter is not None:
            return self.state_adapter.get_conversion_stats()
        else:
            return {"message": "状态适配器未启用"}
    
    def print_state_adapter_stats(self):
        """打印状态适配器统计信息"""
        if self.state_adapter is not None:
            self.state_adapter.print_stats()
        else:
            print("状态适配器未启用")
    
    def _compute_init_hash(self, state: torch.Tensor) -> str:
        """
        计算初始状态哈希
        
        Args:
            state: 状态张量
            
        Returns:
            状态哈希字符串
        """
        # 简单哈希计算，后续可以替换为更复杂的逻辑
        state_bytes = state.detach().cpu().numpy().tobytes()
        hash_value = str(hash(state_bytes))
        return hash_value
    
    def _create_fallback_sample(self) -> Dict[str, Any]:
        """
        创建安全的fallback样本，防止数据加载失败
        
        Returns:
            默认样本字典
        """
        # 创建默认的张量
        default_image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.uint8)
        
        # 创建默认状态并应用状态适配器
        fallback_state_dim = self.target_state_dim or 9
        default_state = torch.zeros(fallback_state_dim)
        if self.state_adapter is not None:
            # 状态适配器已启用，确保维度正确
            default_state = self.state_adapter.align_state_dimension(default_state)
        
        default_action = torch.zeros(self.action_chunk_size, 7)  # 假设7维动作
        default_padding = torch.zeros(self.action_chunk_size, dtype=torch.bool)
        
        sample = {
            "image": {
                "base_0_rgb": default_image,
                "left_wrist_0_rgb": default_image
            },
            "state": default_state,
            "action": default_action,
            "action_is_pad": default_padding,
            "prompt": "fallback_task",
        }
        
        if self.enable_ript_fields:
            sample["advantages"] = self.default_advantages.clone()
            sample["init_hash"] = "fallback_hash"
        
        print("⚠️  使用fallback样本")
        return sample
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        获取样本信息，用于调试
        
        Args:
            idx: 样本索引
            
        Returns:
            样本信息字典
        """
        sample = self[idx]
        
        info = {
            "index": idx,
            "image_shapes": {k: v.shape for k, v in sample["image"].items()},
            "state_shape": sample["state"].shape,
            "action_shape": sample["action"].shape,
            "action_is_pad_shape": sample["action_is_pad"].shape,
            "prompt": sample["prompt"],
        }
        
        if self.enable_ript_fields:
            info["advantages_shape"] = sample["advantages"].shape
            info["init_hash"] = sample["init_hash"]
        
        return info
    
    def validate_dataset(self, num_samples: int = 5) -> bool:
        """
        验证数据集的完整性
        
        Args:
            num_samples: 验证的样本数量
            
        Returns:
            是否验证通过
        """
        print(f"🔍 验证数据集 (检查{num_samples}个样本)...")
        
        try:
            for i in range(min(num_samples, len(self))):
                sample = self[i]
                
                # 验证必需字段
                required_fields = ["image", "state", "action", "action_is_pad", "prompt"]
                for field in required_fields:
                    if field not in sample:
                        print(f"❌ 缺少必需字段: {field}")
                        return False
                
                # 验证图像格式
                image_dict = sample["image"]
                for cam_name, img_tensor in image_dict.items():
                    if not isinstance(img_tensor, torch.Tensor):
                        print(f"❌ 图像不是张量: {cam_name}")
                        return False
                    if img_tensor.dim() != 3:  # (C, H, W)
                        print(f"❌ 图像维度错误: {cam_name}, shape={img_tensor.shape}")
                        return False
                
                # 验证动作维度一致性
                action_tensor = sample["action"]
                padding_tensor = sample["action_is_pad"]
                if action_tensor.shape[0] != padding_tensor.shape[0]:
                    print(f"❌ 动作与padding长度不匹配: {action_tensor.shape[0]} vs {padding_tensor.shape[0]}")
                    return False
                
                # 验证RIPT字段（如果启用）
                if self.enable_ript_fields:
                    if "advantages" not in sample or "init_hash" not in sample:
                        print(f"❌ 缺少RIPT字段")
                        return False
                
            print(f"✅ 数据集验证通过 ({num_samples}个样本)")
            return True
            
        except Exception as e:
            print(f"❌ 数据集验证失败: {e}")
            return False


def create_openpi_ript_dataset(
    repo_id: str,
    enable_ript: bool = False,
    **kwargs
) -> OpenPIRiptDatasetWrapper:
    """
    工厂函数：创建OpenPI-RIPT兼容数据集
    
    Args:
        repo_id: 数据集ID
        enable_ript: 是否启用RIPT功能
        **kwargs: 其他参数
        
    Returns:
        数据集实例
    """
    dataset = OpenPIRiptDatasetWrapper(
        repo_id=repo_id,
        enable_ript_fields=enable_ript,
        **kwargs
    )
    
    # 验证数据集
    if not dataset.validate_dataset():
        raise ValueError("数据集验证失败")
    
    return dataset


# 向后兼容的别名
PI0RiptDataset = OpenPIRiptDatasetWrapper
