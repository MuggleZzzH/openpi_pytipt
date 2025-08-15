"""
安全批次拷贝工具
解决CFG训练中的浅拷贝问题，确保张量不共享内存
"""

import copy
import torch
import numpy as np
from typing import Dict, Any, Union, List, Tuple, Optional
import warnings


class SafeBatchCopier:
    """
    安全的批次拷贝器
    
    功能：
    1. 深拷贝批次数据，避免张量内存共享
    2. 提供多种拷贝策略
    3. 性能优化和内存管理
    4. 拷贝验证和调试支持
    """
    
    def __init__(
        self,
        copy_mode: str = "smart",  # "deep", "smart", "reconstruct"
        device_aware: bool = True,  # 是否考虑设备信息
        verify_independence: bool = False,  # 是否验证拷贝独立性
        track_performance: bool = False,  # 是否跟踪性能
        verbose: bool = False
    ):
        self.copy_mode = copy_mode
        self.device_aware = device_aware
        self.verify_independence = verify_independence
        self.track_performance = track_performance
        self.verbose = verbose
        
        # 性能统计
        self.performance_stats = {
            "total_copies": 0,
            "copy_times": [],
            "memory_usage": [],
            "copy_modes_used": {},
        }
        
        if verbose:
            print(f"🔧 SafeBatchCopier 初始化")
            print(f"   - 拷贝模式: {copy_mode}")
            print(f"   - 设备感知: {device_aware}")
            print(f"   - 验证独立性: {verify_independence}")
    
    def safe_copy_batch(
        self, 
        batch: Dict[str, Any], 
        copy_suffix: Optional[str] = None,
        preserve_requires_grad: bool = False
    ) -> Dict[str, Any]:
        """
        安全拷贝批次数据
        
        Args:
            batch: 原始批次数据
            copy_suffix: 拷贝后缀（用于调试）
            preserve_requires_grad: 是否保持梯度追踪
            
        Returns:
            安全拷贝的批次数据
        """
        import time
        start_time = time.time()
        
        if self.verbose:
            print(f"🔄 开始安全拷贝批次 (模式: {self.copy_mode})")
            if copy_suffix:
                print(f"   拷贝标识: {copy_suffix}")
        
        # 选择拷贝策略
        if self.copy_mode == "deep":
            copied_batch = self._deep_copy_batch(batch)
        elif self.copy_mode == "smart":
            copied_batch = self._smart_copy_batch(batch, preserve_requires_grad)
        elif self.copy_mode == "reconstruct":
            copied_batch = self._reconstruct_batch(batch)
        else:
            raise ValueError(f"不支持的拷贝模式: {self.copy_mode}")
        
        # 验证拷贝独立性
        if self.verify_independence:
            self._verify_copy_independence(batch, copied_batch, copy_suffix)
        
        # 性能统计
        if self.track_performance:
            copy_time = time.time() - start_time
            self._update_performance_stats(copy_time, len(batch))
        
        if self.verbose:
            print(f"✅ 批次拷贝完成")
        
        return copied_batch
    
    def _deep_copy_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """使用Python深拷贝（最安全但可能较慢）"""
        if self.verbose:
            print("   使用深拷贝策略")
        
        return copy.deepcopy(batch)
    
    def _smart_copy_batch(
        self, 
        batch: Dict[str, Any], 
        preserve_requires_grad: bool = False
    ) -> Dict[str, Any]:
        """
        智能拷贝策略（推荐）
        针对不同数据类型使用最合适的拷贝方法
        """
        if self.verbose:
            print("   使用智能拷贝策略")
        
        copied_batch = {}
        
        for key, value in batch.items():
            copied_batch[key] = self._smart_copy_value(
                value, 
                key, 
                preserve_requires_grad
            )
        
        return copied_batch
    
    def _smart_copy_value(
        self, 
        value: Any, 
        key: str, 
        preserve_requires_grad: bool = False
    ) -> Any:
        """智能拷贝单个值"""
        if isinstance(value, torch.Tensor):
            # 张量拷贝：使用clone()确保内存独立
            if preserve_requires_grad:
                copied_value = value.clone()
            else:
                copied_value = value.clone().detach()
            
            if self.verbose:
                print(f"     {key}: 张量拷贝 {value.shape} -> {copied_value.shape}")
                if self.device_aware:
                    print(f"       设备: {value.device} -> {copied_value.device}")
            
            return copied_value
        
        elif isinstance(value, np.ndarray):
            # NumPy数组拷贝
            copied_value = value.copy()
            if self.verbose:
                print(f"     {key}: NumPy数组拷贝 {value.shape} -> {copied_value.shape}")
            return copied_value
        
        elif isinstance(value, dict):
            # 嵌套字典递归拷贝
            copied_value = {}
            for sub_key, sub_value in value.items():
                copied_value[sub_key] = self._smart_copy_value(
                    sub_value, 
                    f"{key}.{sub_key}", 
                    preserve_requires_grad
                )
            return copied_value
        
        elif isinstance(value, (list, tuple)):
            # 列表/元组拷贝
            copied_items = []
            for i, item in enumerate(value):
                copied_item = self._smart_copy_value(
                    item, 
                    f"{key}[{i}]", 
                    preserve_requires_grad
                )
                copied_items.append(copied_item)
            
            # 保持原始类型
            copied_value = type(value)(copied_items)
            if self.verbose:
                print(f"     {key}: {type(value).__name__}拷贝 (长度: {len(value)})")
            return copied_value
        
        elif isinstance(value, (int, float, str, bool, type(None))):
            # 不可变类型直接返回
            if self.verbose:
                print(f"     {key}: 不可变类型 ({type(value).__name__})")
            return value
        
        else:
            # 其他类型使用深拷贝
            if self.verbose:
                print(f"     {key}: 其他类型深拷贝 ({type(value).__name__})")
            return copy.deepcopy(value)
    
    def _reconstruct_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        重构批次（完全重新构造字典）
        对于简单结构的批次，这可能最快
        """
        if self.verbose:
            print("   使用重构策略")
        
        reconstructed_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # 创建新张量
                reconstructed_batch[key] = torch.tensor(
                    value.detach().cpu().numpy(), 
                    dtype=value.dtype,
                    device=value.device
                )
            elif isinstance(value, np.ndarray):
                # 创建新数组
                reconstructed_batch[key] = np.array(value)
            elif isinstance(value, dict):
                # 递归重构嵌套字典
                reconstructed_batch[key] = self._reconstruct_batch(value)
            else:
                # 其他类型使用智能拷贝
                reconstructed_batch[key] = self._smart_copy_value(value, key)
        
        return reconstructed_batch
    
    def _verify_copy_independence(
        self, 
        original: Dict[str, Any], 
        copied: Dict[str, Any], 
        copy_suffix: Optional[str] = None
    ):
        """验证拷贝的独立性"""
        if self.verbose:
            print("🔍 验证拷贝独立性...")
        
        independence_issues = []
        
        def check_independence(orig_val, copy_val, path):
            if isinstance(orig_val, torch.Tensor) and isinstance(copy_val, torch.Tensor):
                # 检查张量是否共享存储
                if orig_val.data_ptr() == copy_val.data_ptr():
                    independence_issues.append(f"张量共享内存: {path}")
                
                # 检查设备是否一致
                if orig_val.device != copy_val.device:
                    independence_issues.append(f"设备不一致: {path} ({orig_val.device} vs {copy_val.device})")
                
                # 检查形状是否一致
                if orig_val.shape != copy_val.shape:
                    independence_issues.append(f"形状不一致: {path} ({orig_val.shape} vs {copy_val.shape})")
            
            elif isinstance(orig_val, np.ndarray) and isinstance(copy_val, np.ndarray):
                # 检查NumPy数组是否共享内存
                if np.shares_memory(orig_val, copy_val):
                    independence_issues.append(f"NumPy数组共享内存: {path}")
            
            elif isinstance(orig_val, dict) and isinstance(copy_val, dict):
                # 递归检查嵌套字典
                for key in orig_val:
                    if key in copy_val:
                        check_independence(orig_val[key], copy_val[key], f"{path}.{key}")
        
        # 执行独立性检查
        for key in original:
            if key in copied:
                check_independence(original[key], copied[key], key)
        
        # 报告问题
        if independence_issues:
            warning_msg = f"发现 {len(independence_issues)} 个独立性问题"
            if copy_suffix:
                warning_msg += f" (拷贝: {copy_suffix})"
            warning_msg += ":\n" + "\n".join(f"  - {issue}" for issue in independence_issues)
            warnings.warn(warning_msg)
            
            if self.verbose:
                print(f"⚠️ {warning_msg}")
        else:
            if self.verbose:
                print("✅ 拷贝独立性验证通过")
    
    def _update_performance_stats(self, copy_time: float, batch_size: int):
        """更新性能统计"""
        self.performance_stats["total_copies"] += 1
        self.performance_stats["copy_times"].append(copy_time)
        
        # 记录使用的拷贝模式
        mode = self.copy_mode
        if mode not in self.performance_stats["copy_modes_used"]:
            self.performance_stats["copy_modes_used"][mode] = 0
        self.performance_stats["copy_modes_used"][mode] += 1
        
        # 保持统计列表大小
        if len(self.performance_stats["copy_times"]) > 100:
            self.performance_stats["copy_times"] = self.performance_stats["copy_times"][-50:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        
        if stats["copy_times"]:
            import numpy as np
            stats["average_copy_time"] = np.mean(stats["copy_times"])
            stats["total_copy_time"] = sum(stats["copy_times"])
            stats["min_copy_time"] = min(stats["copy_times"])
            stats["max_copy_time"] = max(stats["copy_times"])
        else:
            stats["average_copy_time"] = 0.0
            stats["total_copy_time"] = 0.0
        
        return stats
    
    def print_performance_stats(self):
        """打印性能统计"""
        stats = self.get_performance_stats()
        
        print("\n📊 批次拷贝性能统计:")
        print(f"   总拷贝次数: {stats['total_copies']}")
        print(f"   平均拷贝时间: {stats['average_copy_time']:.6f}s")
        print(f"   总拷贝时间: {stats['total_copy_time']:.4f}s")
        
        if stats["copy_modes_used"]:
            print(f"   拷贝模式使用:")
            for mode, count in stats["copy_modes_used"].items():
                print(f"     {mode}: {count}次")


# CFG专用的安全拷贝函数
def safe_copy_cfg_batches(
    original_batch: Dict[str, Any],
    copier: Optional[SafeBatchCopier] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    安全拷贝CFG训练所需的正负分支批次
    
    Args:
        original_batch: 原始批次
        copier: 拷贝器实例（可选）
        
    Returns:
        (positive_batch, negative_batch) 两个独立的批次
    """
    if copier is None:
        copier = SafeBatchCopier(
            copy_mode="smart",
            verify_independence=True,
            verbose=False
        )
    
    # 拷贝正分支批次
    positive_batch = copier.safe_copy_batch(
        original_batch, 
        copy_suffix="positive"
    )
    
    # 拷贝负分支批次  
    negative_batch = copier.safe_copy_batch(
        original_batch,
        copy_suffix="negative"
    )
    
    return positive_batch, negative_batch


def safe_copy_with_modifications(
    original_batch: Dict[str, Any],
    modifications: Dict[str, Any],
    copier: Optional[SafeBatchCopier] = None
) -> Dict[str, Any]:
    """
    安全拷贝并应用修改
    
    Args:
        original_batch: 原始批次
        modifications: 要应用的修改
        copier: 拷贝器实例（可选）
        
    Returns:
        修改后的安全拷贝
    """
    if copier is None:
        copier = SafeBatchCopier(copy_mode="smart")
    
    # 安全拷贝
    copied_batch = copier.safe_copy_batch(original_batch)
    
    # 应用修改
    for key, value in modifications.items():
        copied_batch[key] = value
    
    return copied_batch


def create_cfg_safe_copier(
    verify_copies: bool = True,
    track_performance: bool = False,
    verbose: bool = False
) -> SafeBatchCopier:
    """
    创建CFG专用的安全拷贝器
    
    Args:
        verify_copies: 是否验证拷贝独立性
        track_performance: 是否跟踪性能
        verbose: 是否详细输出
        
    Returns:
        配置好的拷贝器
    """
    return SafeBatchCopier(
        copy_mode="smart",
        device_aware=True,
        verify_independence=verify_copies,
        track_performance=track_performance,
        verbose=verbose
    )


# 便捷装饰器
def with_safe_batch_copy(
    verify_independence: bool = True,
    verbose: bool = False
):
    """
    装饰器：为函数提供安全的批次拷贝功能
    
    Args:
        verify_independence: 是否验证独立性
        verbose: 是否详细输出
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 创建拷贝器
            copier = SafeBatchCopier(
                copy_mode="smart",
                verify_independence=verify_independence,
                verbose=verbose
            )
            
            # 将拷贝器注入到kwargs中
            kwargs['_safe_copier'] = copier
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# 兼容性函数（替换现有的.copy()调用）
def replace_shallow_copy(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    替换字典浅拷贝的安全函数
    这个函数可以直接替换现有代码中的 batch.copy() 调用
    
    Args:
        batch: 原始批次字典
        
    Returns:
        安全拷贝的批次
    """
    copier = SafeBatchCopier(copy_mode="smart", verbose=False)
    return copier.safe_copy_batch(batch)
