"""
优势值处理器
处理RIPT训练中的优势值归一化、截断和数值稳定性问题
"""

import torch
import numpy as np
import warnings
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum


class AdvantageNormalizationMode(Enum):
    """优势归一化模式"""
    NONE = "none"                    # 不归一化
    ZERO_MEAN = "zero_mean"          # 零均值归一化 (x - mean)
    STANDARD = "standard"            # 标准归一化 (x - mean) / std
    MIN_MAX = "min_max"              # 最小-最大归一化 (x - min) / (max - min)
    ROBUST = "robust"                # 鲁棒归一化，使用中位数和MAD


class AdvantageClippingMode(Enum):
    """优势截断模式"""
    NONE = "none"                    # 不截断
    SYMMETRIC = "symmetric"          # 对称截断 [-clip, +clip]
    QUANTILE = "quantile"            # 基于分位数截断
    SIGMA = "sigma"                  # 基于标准差截断 [mean ± n*std]


class NegativeHandlingMode(Enum):
    """负值处理模式"""
    KEEP = "keep"                    # 保持负值
    SOFTPLUS = "softplus"            # 使用softplus: log(1 + exp(x))
    RELU = "relu"                    # 使用ReLU: max(0, x)
    SHIFT_POSITIVE = "shift"         # 平移到正数区间
    EXP = "exp"                      # 指数变换: exp(x)


@dataclass
class AdvantageProcessingConfig:
    """优势处理配置"""
    # 归一化配置
    normalization_mode: AdvantageNormalizationMode = AdvantageNormalizationMode.STANDARD
    
    # 截断配置
    clipping_mode: AdvantageClippingMode = AdvantageClippingMode.SYMMETRIC
    clip_value: float = 3.0          # 截断阈值
    quantile_range: Tuple[float, float] = (0.05, 0.95)  # 分位数范围
    
    # 负值处理配置
    negative_handling: NegativeHandlingMode = NegativeHandlingMode.KEEP
    softplus_beta: float = 1.0       # softplus的beta参数
    shift_epsilon: float = 1e-3      # 平移的最小正值
    
    # 数值稳定性配置
    epsilon: float = 1e-8            # 避免除零的小数
    nan_replacement: float = 0.0     # NaN替换值
    inf_replacement: float = 1.0     # Inf替换值
    enable_validation: bool = True   # 启用输入验证
    
    # 调试配置
    verbose: bool = False            # 详细输出
    track_statistics: bool = True    # 跟踪统计信息


class AdvantageProcessor:
    """
    优势值处理器
    
    功能：
    1. 多种归一化策略
    2. 智能截断机制
    3. 负值处理策略
    4. 数值稳定性保障
    5. 统计信息跟踪
    """
    
    def __init__(self, config: AdvantageProcessingConfig):
        self.config = config
        
        # 统计信息跟踪
        self.processing_stats = {
            "total_processed": 0,
            "nan_count": 0,
            "inf_count": 0,
            "negative_count": 0,
            "clipped_count": 0,
            "original_stats": {"mean": [], "std": [], "min": [], "max": []},
            "processed_stats": {"mean": [], "std": [], "min": [], "max": []},
        }
        
        if config.verbose:
            print(f"🔧 AdvantageProcessor 初始化")
            print(f"   - 归一化模式: {config.normalization_mode.value}")
            print(f"   - 截断模式: {config.clipping_mode.value}")
            print(f"   - 负值处理: {config.negative_handling.value}")
    
    def process_advantages(
        self, 
        advantages: Union[torch.Tensor, np.ndarray],
        batch_info: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        处理优势值，应用归一化、截断和负值处理
        
        Args:
            advantages: 原始优势值
            batch_info: 批次信息（可选，用于调试）
            
        Returns:
            处理后的优势值
        """
        # 转换为torch.Tensor
        if isinstance(advantages, np.ndarray):
            advantages = torch.tensor(advantages, dtype=torch.float32)
        elif not isinstance(advantages, torch.Tensor):
            advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # 🔥 关键修复：确保数据类型为浮点型
        if not advantages.is_floating_point():
            warnings.warn(f"优势值数据类型为 {advantages.dtype}，自动转换为float32")
            advantages = advantages.float()
        
        # 输入验证
        if self.config.enable_validation:
            self._validate_input(advantages)
        
        # 记录原始统计信息
        original_advantages = advantages.clone()
        if self.config.track_statistics:
            self._record_original_stats(original_advantages)
        
        if self.config.verbose:
            print(f"🔄 处理优势值: shape={advantages.shape}")
            print(f"   原始统计: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
            print(f"   原始范围: [{advantages.min():.4f}, {advantages.max():.4f}]")
        
        # 步骤1: 处理NaN和Inf值
        advantages = self._handle_invalid_values(advantages)
        
        # 步骤2: 归一化
        advantages = self._apply_normalization(advantages)
        
        # 步骤3: 截断
        advantages = self._apply_clipping(advantages)
        
        # 步骤4: 负值处理
        advantages = self._handle_negative_values(advantages)
        
        # 步骤5: 最终验证
        advantages = self._final_validation(advantages)
        
        # 记录处理后统计信息
        if self.config.track_statistics:
            self._record_processed_stats(advantages)
            self.processing_stats["total_processed"] += 1
        
        if self.config.verbose:
            print(f"   处理后统计: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
            print(f"   处理后范围: [{advantages.min():.4f}, {advantages.max():.4f}]")
        
        return advantages
    
    def _validate_input(self, advantages: torch.Tensor):
        """验证输入优势值"""
        if advantages.numel() == 0:
            raise ValueError("优势值张量为空")
        
        if not advantages.dtype.is_floating_point:
            warnings.warn(f"优势值数据类型为 {advantages.dtype}，建议使用浮点类型")
    
    def _handle_invalid_values(self, advantages: torch.Tensor) -> torch.Tensor:
        """处理NaN和Inf值"""
        nan_mask = torch.isnan(advantages)
        inf_mask = torch.isinf(advantages)
        
        if nan_mask.any():
            nan_count = nan_mask.sum().item()
            self.processing_stats["nan_count"] += nan_count
            advantages = torch.where(nan_mask, self.config.nan_replacement, advantages)
            if self.config.verbose:
                print(f"   ⚠️ 处理了 {nan_count} 个NaN值")
        
        if inf_mask.any():
            inf_count = inf_mask.sum().item()
            self.processing_stats["inf_count"] += inf_count
            # 区分正无穷和负无穷
            pos_inf_mask = torch.isposinf(advantages)
            neg_inf_mask = torch.isneginf(advantages)
            advantages = torch.where(pos_inf_mask, self.config.inf_replacement, advantages)
            advantages = torch.where(neg_inf_mask, -self.config.inf_replacement, advantages)
            if self.config.verbose:
                print(f"   ⚠️ 处理了 {inf_count} 个Inf值")
        
        return advantages
    
    def _apply_normalization(self, advantages: torch.Tensor) -> torch.Tensor:
        """应用归一化"""
        mode = self.config.normalization_mode
        
        if mode == AdvantageNormalizationMode.NONE:
            return advantages
        
        elif mode == AdvantageNormalizationMode.ZERO_MEAN:
            # 零均值归一化
            mean_val = advantages.mean()
            normalized = advantages - mean_val
            if self.config.verbose:
                print(f"   📊 零均值归一化: mean={mean_val:.4f}")
        
        elif mode == AdvantageNormalizationMode.STANDARD:
            # 标准归一化
            mean_val = advantages.mean()
            std_val = advantages.std()
            if std_val < self.config.epsilon:
                std_val = torch.tensor(1.0)  # 避免除零
                if self.config.verbose:
                    print(f"   ⚠️ 标准差太小，使用1.0替代")
            normalized = (advantages - mean_val) / std_val
            if self.config.verbose:
                print(f"   📊 标准归一化: mean={mean_val:.4f}, std={std_val:.4f}")
        
        elif mode == AdvantageNormalizationMode.MIN_MAX:
            # 最小-最大归一化
            min_val = advantages.min()
            max_val = advantages.max()
            if (max_val - min_val) < self.config.epsilon:
                normalized = torch.zeros_like(advantages)  # 所有值相等
                if self.config.verbose:
                    print(f"   ⚠️ 值范围太小，归一化为零")
            else:
                normalized = (advantages - min_val) / (max_val - min_val)
                if self.config.verbose:
                    print(f"   📊 最小-最大归一化: range=[{min_val:.4f}, {max_val:.4f}]")
        
        elif mode == AdvantageNormalizationMode.ROBUST:
            # 鲁棒归一化（基于中位数和MAD）
            median_val = advantages.median()
            mad = torch.median(torch.abs(advantages - median_val))
            if mad < self.config.epsilon:
                mad = torch.tensor(1.0)
                if self.config.verbose:
                    print(f"   ⚠️ MAD太小，使用1.0替代")
            normalized = (advantages - median_val) / mad
            if self.config.verbose:
                print(f"   📊 鲁棒归一化: median={median_val:.4f}, MAD={mad:.4f}")
        
        else:
            raise ValueError(f"不支持的归一化模式: {mode}")
        
        return normalized
    
    def _apply_clipping(self, advantages: torch.Tensor) -> torch.Tensor:
        """应用截断"""
        mode = self.config.clipping_mode
        
        if mode == AdvantageClippingMode.NONE:
            return advantages
        
        original_range = (advantages.min().item(), advantages.max().item())
        
        if mode == AdvantageClippingMode.SYMMETRIC:
            # 对称截断
            clip_val = self.config.clip_value
            clipped = torch.clamp(advantages, min=-clip_val, max=clip_val)
            
        elif mode == AdvantageClippingMode.QUANTILE:
            # 基于分位数截断
            q_low, q_high = self.config.quantile_range
            low_val = advantages.quantile(q_low)
            high_val = advantages.quantile(q_high)
            clipped = torch.clamp(advantages, min=low_val, max=high_val)
            
        elif mode == AdvantageClippingMode.SIGMA:
            # 基于标准差截断
            mean_val = advantages.mean()
            std_val = advantages.std()
            clip_val = self.config.clip_value * std_val
            clipped = torch.clamp(advantages, min=mean_val - clip_val, max=mean_val + clip_val)
            
        else:
            raise ValueError(f"不支持的截断模式: {mode}")
        
        # 统计截断信息
        clipped_mask = (clipped != advantages)
        if clipped_mask.any():
            clipped_count = clipped_mask.sum().item()
            self.processing_stats["clipped_count"] += clipped_count
            if self.config.verbose:
                clipped_range = (clipped.min().item(), clipped.max().item())
                print(f"   ✂️ 截断了 {clipped_count} 个值")
                print(f"      原始范围: [{original_range[0]:.4f}, {original_range[1]:.4f}]")
                print(f"      截断范围: [{clipped_range[0]:.4f}, {clipped_range[1]:.4f}]")
        
        return clipped
    
    def _handle_negative_values(self, advantages: torch.Tensor) -> torch.Tensor:
        """处理负值"""
        mode = self.config.negative_handling
        
        negative_mask = advantages < 0
        if negative_mask.any():
            negative_count = negative_mask.sum().item()
            self.processing_stats["negative_count"] += negative_count
            
            if self.config.verbose:
                print(f"   ➖ 检测到 {negative_count} 个负值")
        
        if mode == NegativeHandlingMode.KEEP:
            return advantages
        
        elif mode == NegativeHandlingMode.SOFTPLUS:
            # 使用softplus变换
            beta = self.config.softplus_beta
            transformed = torch.nn.functional.softplus(advantages, beta=beta)
            if self.config.verbose:
                print(f"   🔄 应用softplus变换 (beta={beta})")
        
        elif mode == NegativeHandlingMode.RELU:
            # 使用ReLU
            transformed = torch.nn.functional.relu(advantages)
            if self.config.verbose:
                print(f"   🔄 应用ReLU变换")
        
        elif mode == NegativeHandlingMode.SHIFT_POSITIVE:
            # 平移到正数区间
            min_val = advantages.min()
            if min_val < 0:
                shift_amount = -min_val + self.config.shift_epsilon
                transformed = advantages + shift_amount
                if self.config.verbose:
                    print(f"   🔄 平移变换: +{shift_amount:.6f}")
            else:
                transformed = advantages
        
        elif mode == NegativeHandlingMode.EXP:
            # 指数变换
            # 为避免数值溢出，先进行截断
            safe_advantages = torch.clamp(advantages, max=10.0)  # exp(10) ≈ 22000
            transformed = torch.exp(safe_advantages)
            if self.config.verbose:
                print(f"   🔄 应用指数变换")
        
        else:
            raise ValueError(f"不支持的负值处理模式: {mode}")
        
        return transformed
    
    def _final_validation(self, advantages: torch.Tensor) -> torch.Tensor:
        """最终验证和清理"""
        # 最后一次NaN/Inf检查
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            advantages = torch.nan_to_num(
                advantages, 
                nan=self.config.nan_replacement,
                posinf=self.config.inf_replacement,
                neginf=-self.config.inf_replacement
            )
            if self.config.verbose:
                print(f"   🧹 最终清理：移除残留的NaN/Inf")
        
        return advantages
    
    def _record_original_stats(self, advantages: torch.Tensor):
        """记录原始统计信息"""
        stats = self.processing_stats["original_stats"]
        stats["mean"].append(advantages.mean().item())
        stats["std"].append(advantages.std().item())
        stats["min"].append(advantages.min().item())
        stats["max"].append(advantages.max().item())
        
        # 保持列表大小
        for key in stats:
            if len(stats[key]) > 100:
                stats[key] = stats[key][-50:]
    
    def _record_processed_stats(self, advantages: torch.Tensor):
        """记录处理后统计信息"""
        stats = self.processing_stats["processed_stats"]
        stats["mean"].append(advantages.mean().item())
        stats["std"].append(advantages.std().item())
        stats["min"].append(advantages.min().item())
        stats["max"].append(advantages.max().item())
        
        # 保持列表大小
        for key in stats:
            if len(stats[key]) > 100:
                stats[key] = stats[key][-50:]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.processing_stats.copy()
        
        # 计算平均统计
        for phase in ["original_stats", "processed_stats"]:
            phase_stats = stats[phase]
            for metric in ["mean", "std", "min", "max"]:
                if phase_stats[metric]:
                    stats[f"{phase}_{metric}_avg"] = np.mean(phase_stats[metric])
                else:
                    stats[f"{phase}_{metric}_avg"] = 0.0
        
        return stats
    
    def print_stats(self):
        """打印处理统计信息"""
        stats = self.get_processing_stats()
        
        print("\n📊 优势值处理统计:")
        print(f"   总处理次数: {stats['total_processed']}")
        print(f"   NaN处理次数: {stats['nan_count']}")
        print(f"   Inf处理次数: {stats['inf_count']}")
        print(f"   负值处理次数: {stats['negative_count']}")
        print(f"   截断次数: {stats['clipped_count']}")
        
        print(f"\n📈 原始优势值统计:")
        print(f"   平均mean: {stats['original_stats_mean_avg']:.4f}")
        print(f"   平均std: {stats['original_stats_std_avg']:.4f}")
        
        print(f"\n📈 处理后优势值统计:")
        print(f"   平均mean: {stats['processed_stats_mean_avg']:.4f}")
        print(f"   平均std: {stats['processed_stats_std_avg']:.4f}")
    
    def reset_stats(self):
        """重置统计信息"""
        self.processing_stats = {
            "total_processed": 0,
            "nan_count": 0,
            "inf_count": 0,
            "negative_count": 0,
            "clipped_count": 0,
            "original_stats": {"mean": [], "std": [], "min": [], "max": []},
            "processed_stats": {"mean": [], "std": [], "min": [], "max": []},
        }
        if self.config.verbose:
            print("🔄 优势处理统计已重置")


# 便捷函数
def create_advantage_processor(
    normalization: str = "standard",
    clipping: str = "symmetric",
    clip_value: float = 3.0,
    negative_handling: str = "keep",
    verbose: bool = False
) -> AdvantageProcessor:
    """
    创建优势处理器的便捷函数
    
    Args:
        normalization: 归一化模式
        clipping: 截断模式
        clip_value: 截断值
        negative_handling: 负值处理模式
        verbose: 是否详细输出
        
    Returns:
        配置好的优势处理器
    """
    config = AdvantageProcessingConfig(
        normalization_mode=AdvantageNormalizationMode(normalization),
        clipping_mode=AdvantageClippingMode(clipping),
        clip_value=clip_value,
        negative_handling=NegativeHandlingMode(negative_handling),
        verbose=verbose
    )
    
    return AdvantageProcessor(config)


def process_advantages_batch(
    advantages_list: list,
    processor: Optional[AdvantageProcessor] = None,
    **processor_kwargs
) -> list:
    """
    批量处理优势值列表
    
    Args:
        advantages_list: 优势值列表
        processor: 优势处理器（可选）
        **processor_kwargs: 处理器创建参数
        
    Returns:
        处理后的优势值列表
    """
    if processor is None:
        processor = create_advantage_processor(**processor_kwargs)
    
    processed_list = []
    for i, advantages in enumerate(advantages_list):
        processed = processor.process_advantages(advantages, batch_info={"batch_idx": i})
        processed_list.append(processed)
    
    return processed_list
