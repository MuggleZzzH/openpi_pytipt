"""
状态维度适配器
处理不同数据源与PI0模型之间的状态维度对齐问题
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class StateDimensionConfig:
    """状态维度配置"""
    target_state_dim: int  # 目标状态维度（PI0模型期望的维度）
    source_state_dim: Optional[int] = None  # 源状态维度（数据集实际维度）
    padding_mode: str = "zero"  # 填充模式: "zero", "repeat_last", "mean"
    truncation_mode: str = "first"  # 截断模式: "first", "last", "middle"
    normalize_after_align: bool = False  # 对齐后是否归一化
    enable_dimension_check: bool = True  # 是否启用维度检查
    
    def __post_init__(self):
        assert self.padding_mode in ["zero", "repeat_last", "mean"], f"不支持的填充模式: {self.padding_mode}"
        assert self.truncation_mode in ["first", "last", "middle"], f"不支持的截断模式: {self.truncation_mode}"


class StateDimensionAdapter:
    """
    状态维度适配器
    
    功能：
    1. 自动检测源状态维度与目标维度的差异
    2. 智能填充或截断状态向量
    3. 提供多种对齐策略
    4. 记录维度转换统计信息
    """
    
    def __init__(self, config: StateDimensionConfig):
        self.config = config
        self.conversion_stats = {
            "total_conversions": 0,
            "expansions": 0,
            "truncations": 0,
            "no_change": 0,
            "dimension_history": {}
        }
        self._warned_dimensions = set()
        
        print(f"🔧 StateDimensionAdapter 初始化")
        print(f"   - 目标维度: {config.target_state_dim}")
        print(f"   - 填充模式: {config.padding_mode}")
        print(f"   - 截断模式: {config.truncation_mode}")
        
    def align_state_dimension(
        self, 
        state: torch.Tensor, 
        batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        对齐状态维度到目标维度
        
        Args:
            state: 输入状态张量 (..., state_dim)
            batch_idx: 批次索引（可选，用于调试）
            
        Returns:
            对齐后的状态张量 (..., target_state_dim)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            
        original_shape = state.shape
        current_dim = original_shape[-1]
        target_dim = self.config.target_state_dim
        
        # 更新统计信息
        self.conversion_stats["total_conversions"] += 1
        dim_key = f"{current_dim}->{target_dim}"
        self.conversion_stats["dimension_history"][dim_key] = \
            self.conversion_stats["dimension_history"].get(dim_key, 0) + 1
        
        # 如果维度已经匹配，直接返回
        if current_dim == target_dim:
            self.conversion_stats["no_change"] += 1
            return state
            
        # 维度检查和警告
        if self.config.enable_dimension_check and current_dim not in self._warned_dimensions:
            if current_dim < target_dim:
                print(f"🔧 状态维度扩展: {current_dim} -> {target_dim} (填充模式: {self.config.padding_mode})")
            else:
                print(f"⚠️ 状态维度截断: {current_dim} -> {target_dim} (截断模式: {self.config.truncation_mode})")
            self._warned_dimensions.add(current_dim)
        
        if current_dim < target_dim:
            # 需要扩展维度
            aligned_state = self._expand_state_dimension(state, current_dim, target_dim)
            self.conversion_stats["expansions"] += 1
        else:
            # 需要截断维度
            aligned_state = self._truncate_state_dimension(state, current_dim, target_dim)
            self.conversion_stats["truncations"] += 1
            
        # 可选的后处理归一化
        if self.config.normalize_after_align:
            aligned_state = self._normalize_state(aligned_state)
            
        # 验证输出维度
        assert aligned_state.shape[-1] == target_dim, \
            f"维度对齐失败: 期望{target_dim}, 实际{aligned_state.shape[-1]}"
            
        return aligned_state
    
    def _expand_state_dimension(
        self, 
        state: torch.Tensor, 
        current_dim: int, 
        target_dim: int
    ) -> torch.Tensor:
        """扩展状态维度"""
        expand_size = target_dim - current_dim
        
        if self.config.padding_mode == "zero":
            # 零填充
            padding_shape = list(state.shape)
            padding_shape[-1] = expand_size
            padding = torch.zeros(padding_shape, dtype=state.dtype, device=state.device)
            
        elif self.config.padding_mode == "repeat_last":
            # 重复最后一个元素
            last_element = state[..., -1:].expand(*state.shape[:-1], expand_size)
            padding = last_element
            
        elif self.config.padding_mode == "mean":
            # 使用均值填充
            state_mean = state.mean(dim=-1, keepdim=True)
            padding = state_mean.expand(*state.shape[:-1], expand_size)
            
        else:
            raise ValueError(f"不支持的填充模式: {self.config.padding_mode}")
            
        return torch.cat([state, padding], dim=-1)
    
    def _truncate_state_dimension(
        self, 
        state: torch.Tensor, 
        current_dim: int, 
        target_dim: int
    ) -> torch.Tensor:
        """截断状态维度"""
        if self.config.truncation_mode == "first":
            # 保留前target_dim个维度
            return state[..., :target_dim]
            
        elif self.config.truncation_mode == "last":
            # 保留后target_dim个维度
            return state[..., -target_dim:]
            
        elif self.config.truncation_mode == "middle":
            # 保留中间target_dim个维度
            start_idx = (current_dim - target_dim) // 2
            end_idx = start_idx + target_dim
            return state[..., start_idx:end_idx]
            
        else:
            raise ValueError(f"不支持的截断模式: {self.config.truncation_mode}")
    
    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """归一化状态"""
        # 简单的标准化：均值为0，标准差为1
        state_mean = state.mean(dim=-1, keepdim=True)
        state_std = state.std(dim=-1, keepdim=True)
        return (state - state_mean) / (state_std + 1e-8)
    
    def detect_state_dimension(self, dataset_or_sample: Union[torch.utils.data.Dataset, Dict[str, Any]]) -> int:
        """
        自动检测数据集或样本的状态维度
        
        Args:
            dataset_or_sample: 数据集对象或样本字典
            
        Returns:
            检测到的状态维度
        """
        if hasattr(dataset_or_sample, '__getitem__'):
            # 数据集对象
            sample = dataset_or_sample[0]
            state = sample["state"]
        else:
            # 样本字典
            state = dataset_or_sample["state"]
            
        if isinstance(state, (list, np.ndarray)):
            state = torch.tensor(state)
            
        detected_dim = state.shape[-1]
        print(f"🔍 检测到状态维度: {detected_dim}")
        return detected_dim
    
    def update_source_dimension(self, source_dim: int):
        """更新源状态维度配置"""
        self.config.source_state_dim = source_dim
        print(f"🔄 更新源状态维度: {source_dim}")
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """获取维度转换统计信息"""
        return {
            **self.conversion_stats,
            "config": {
                "target_state_dim": self.config.target_state_dim,
                "source_state_dim": self.config.source_state_dim,
                "padding_mode": self.config.padding_mode,
                "truncation_mode": self.config.truncation_mode,
            }
        }
    
    def print_stats(self):
        """打印转换统计信息"""
        stats = self.conversion_stats
        print("\n📊 状态维度转换统计:")
        print(f"   总转换次数: {stats['total_conversions']}")
        print(f"   扩展次数: {stats['expansions']}")
        print(f"   截断次数: {stats['truncations']}")
        print(f"   无变化次数: {stats['no_change']}")
        
        if stats['dimension_history']:
            print("   维度转换历史:")
            for conversion, count in stats['dimension_history'].items():
                print(f"     {conversion}: {count}次")


def create_pi0_state_adapter(
    target_state_dim: int,
    source_state_dim: Optional[int] = None,
    **kwargs
) -> StateDimensionAdapter:
    """
    创建PI0专用的状态维度适配器
    
    Args:
        target_state_dim: PI0模型期望的状态维度
        source_state_dim: 数据集的状态维度（可选）
        **kwargs: 其他配置参数
        
    Returns:
        配置好的状态维度适配器
    """
    config = StateDimensionConfig(
        target_state_dim=target_state_dim,
        source_state_dim=source_state_dim,
        **kwargs
    )
    
    adapter = StateDimensionAdapter(config)
    
    if source_state_dim is not None and source_state_dim != target_state_dim:
        print(f"⚠️ 检测到维度不匹配: 源({source_state_dim}) vs 目标({target_state_dim})")
        
    return adapter


def batch_align_states(
    batch: Dict[str, Any], 
    adapter: StateDimensionAdapter
) -> Dict[str, Any]:
    """
    批量对齐状态维度
    
    Args:
        batch: 包含state字段的批次数据
        adapter: 状态维度适配器
        
    Returns:
        对齐后的批次数据
    """
    if "state" in batch:
        batch["state"] = adapter.align_state_dimension(batch["state"])
    
    return batch


# 便捷函数
def auto_align_state_for_pi0(state: torch.Tensor, target_dim: int = 14) -> torch.Tensor:
    """
    为PI0模型自动对齐状态维度的便捷函数
    
    Args:
        state: 输入状态
        target_dim: 目标维度（默认14，可根据实际模型调整）
        
    Returns:
        对齐后的状态
    """
    config = StateDimensionConfig(
        target_state_dim=target_dim,
        padding_mode="zero",
        truncation_mode="first"
    )
    adapter = StateDimensionAdapter(config)
    return adapter.align_state_dimension(state)
