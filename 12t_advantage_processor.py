"""
测试优势值处理器
验证归一化、截断、负值处理等数值稳定性功能
"""

import torch
import numpy as np
from ript.utils.advantage_processor import (
    AdvantageProcessor,
    AdvantageProcessingConfig,
    AdvantageNormalizationMode,
    AdvantageClippingMode,
    NegativeHandlingMode,
    create_advantage_processor,
    process_advantages_batch
)


def test_normalization_modes():
    """测试不同的归一化模式"""
    print("🧪 测试归一化模式...")
    
    # 创建测试数据
    test_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # 测试各种归一化模式
    modes = [
        AdvantageNormalizationMode.NONE,
        AdvantageNormalizationMode.ZERO_MEAN,
        AdvantageNormalizationMode.STANDARD,
        AdvantageNormalizationMode.MIN_MAX,
        AdvantageNormalizationMode.ROBUST
    ]
    
    for mode in modes:
        config = AdvantageProcessingConfig(
            normalization_mode=mode,
            clipping_mode=AdvantageClippingMode.NONE,
            negative_handling=NegativeHandlingMode.KEEP,
            verbose=True
        )
        
        processor = AdvantageProcessor(config)
        result = processor.process_advantages(test_data.clone())
        
        print(f"模式 {mode.value}:")
        print(f"  原始: {test_data.tolist()}")
        print(f"  结果: {result.tolist()}")
        print(f"  均值: {result.mean():.4f}, 标准差: {result.std():.4f}")
        print()
        
        # 验证结果
        if mode == AdvantageNormalizationMode.NONE:
            assert torch.allclose(result, test_data)
        elif mode == AdvantageNormalizationMode.ZERO_MEAN:
            assert abs(result.mean()) < 1e-6  # 均值应该接近0
        elif mode == AdvantageNormalizationMode.STANDARD:
            assert abs(result.mean()) < 1e-6  # 均值应该接近0
            assert abs(result.std() - 1.0) < 1e-6  # 标准差应该接近1
        elif mode == AdvantageNormalizationMode.MIN_MAX:
            assert abs(result.min()) < 1e-6  # 最小值应该接近0
            assert abs(result.max() - 1.0) < 1e-6  # 最大值应该接近1
    
    print("✅ 归一化模式测试通过")


def test_clipping_modes():
    """测试不同的截断模式"""
    print("🧪 测试截断模式...")
    
    # 创建包含极值的测试数据
    test_data = torch.tensor([-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0])
    
    # 测试对称截断
    config = AdvantageProcessingConfig(
        normalization_mode=AdvantageNormalizationMode.NONE,
        clipping_mode=AdvantageClippingMode.SYMMETRIC,
        clip_value=2.0,
        verbose=True
    )
    
    processor = AdvantageProcessor(config)
    result = processor.process_advantages(test_data.clone())
    
    print(f"对称截断 (±2.0):")
    print(f"  原始: {test_data.tolist()}")
    print(f"  结果: {result.tolist()}")
    
    # 验证对称截断
    assert result.min() >= -2.0
    assert result.max() <= 2.0
    
    # 测试分位数截断
    config.clipping_mode = AdvantageClippingMode.QUANTILE
    config.quantile_range = (0.2, 0.8)
    
    processor = AdvantageProcessor(config)
    result = processor.process_advantages(test_data.clone())
    
    print(f"分位数截断 (20%-80%):")
    print(f"  结果: {result.tolist()}")
    
    # 测试sigma截断
    config.clipping_mode = AdvantageClippingMode.SIGMA
    config.clip_value = 1.0  # 1个标准差
    
    processor = AdvantageProcessor(config)
    result = processor.process_advantages(test_data.clone())
    
    print(f"Sigma截断 (±1σ):")
    print(f"  结果: {result.tolist()}")
    print()
    
    print("✅ 截断模式测试通过")


def test_negative_handling():
    """测试负值处理模式"""
    print("🧪 测试负值处理模式...")
    
    # 创建包含负值的测试数据
    test_data = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    
    # 测试各种负值处理模式
    modes = [
        NegativeHandlingMode.KEEP,
        NegativeHandlingMode.SOFTPLUS,
        NegativeHandlingMode.RELU,
        NegativeHandlingMode.SHIFT_POSITIVE,
        NegativeHandlingMode.EXP
    ]
    
    for mode in modes:
        config = AdvantageProcessingConfig(
            normalization_mode=AdvantageNormalizationMode.NONE,
            clipping_mode=AdvantageClippingMode.NONE,
            negative_handling=mode,
            verbose=True
        )
        
        processor = AdvantageProcessor(config)
        result = processor.process_advantages(test_data.clone())
        
        print(f"模式 {mode.value}:")
        print(f"  原始: {test_data.tolist()}")
        print(f"  结果: {[f'{x:.4f}' for x in result.tolist()]}")
        
        # 验证结果
        if mode == NegativeHandlingMode.KEEP:
            assert torch.allclose(result, test_data)
        elif mode in [NegativeHandlingMode.SOFTPLUS, NegativeHandlingMode.RELU, 
                     NegativeHandlingMode.SHIFT_POSITIVE, NegativeHandlingMode.EXP]:
            assert result.min() >= 0  # 应该没有负值
        
        print()
    
    print("✅ 负值处理测试通过")


def test_numerical_stability():
    """测试数值稳定性"""
    print("🧪 测试数值稳定性...")
    
    # 创建包含问题值的测试数据
    test_data = torch.tensor([
        float('nan'),     # NaN
        float('inf'),     # 正无穷
        float('-inf'),    # 负无穷
        1e-10,           # 很小的正数
        -1e-10,          # 很小的负数
        1e10,            # 很大的正数
        -1e10,           # 很大的负数
        0.0              # 零
    ])
    
    config = AdvantageProcessingConfig(
        normalization_mode=AdvantageNormalizationMode.STANDARD,
        clipping_mode=AdvantageClippingMode.SYMMETRIC,
        clip_value=5.0,
        negative_handling=NegativeHandlingMode.SOFTPLUS,
        verbose=True
    )
    
    processor = AdvantageProcessor(config)
    result = processor.process_advantages(test_data)
    
    print(f"原始数据: {test_data.tolist()}")
    print(f"处理结果: {[f'{x:.6f}' for x in result.tolist()]}")
    
    # 验证没有invalid值
    assert not torch.isnan(result).any(), "结果中不应该有NaN"
    assert not torch.isinf(result).any(), "结果中不应该有Inf"
    assert result.min() >= 0, "负值处理后应该没有负数"
    
    # 打印统计信息
    processor.print_stats()
    
    print("✅ 数值稳定性测试通过")


def test_edge_cases():
    """测试边界情况"""
    print("🧪 测试边界情况...")
    
    config = AdvantageProcessingConfig(verbose=True)
    processor = AdvantageProcessor(config)
    
    # 测试空张量
    try:
        empty_tensor = torch.tensor([])
        result = processor.process_advantages(empty_tensor)
        assert False, "应该抛出异常"
    except ValueError:
        print("✓ 空张量正确抛出异常")
    
    # 测试单个值
    single_value = torch.tensor([5.0])
    result = processor.process_advantages(single_value)
    assert result.shape == (1,), "单个值的形状应该保持"
    print(f"✓ 单个值处理: {single_value.item()} → {result.item():.4f}")
    
    # 测试相同值
    same_values = torch.tensor([3.0, 3.0, 3.0, 3.0])
    result = processor.process_advantages(same_values)
    print(f"✓ 相同值处理: {same_values.tolist()} → {result.tolist()}")
    
    # 测试不同数据类型
    int_data = torch.tensor([1, 2, 3, 4, 5])  # 整数类型
    result = processor.process_advantages(int_data)
    assert result.dtype.is_floating_point, "输出应该是浮点类型"
    print(f"✓ 整数输入处理: {int_data.tolist()} → {[f'{x:.4f}' for x in result.tolist()]}")
    
    print("✅ 边界情况测试通过")


def test_batch_processing():
    """测试批量处理"""
    print("🧪 测试批量处理...")
    
    # 创建多个优势值张量
    advantages_list = [
        torch.randn(10),      # 随机数据
        torch.tensor([1.0, 2.0, 3.0]),  # 小数据
        torch.randn(100) * 10,  # 大范围数据
        torch.tensor([-5.0, 0.0, 5.0])  # 包含负值
    ]
    
    # 使用便捷函数批量处理
    processed_list = process_advantages_batch(
        advantages_list,
        normalization="standard",
        clipping="symmetric",
        clip_value=2.0,
        negative_handling="softplus",
        verbose=False
    )
    
    assert len(processed_list) == len(advantages_list), "批量处理数量应该一致"
    
    for i, (original, processed) in enumerate(zip(advantages_list, processed_list)):
        print(f"批次 {i}: {original.shape} → {processed.shape}")
        assert original.shape == processed.shape, "形状应该保持一致"
        assert not torch.isnan(processed).any(), "不应该有NaN"
        assert not torch.isinf(processed).any(), "不应该有Inf"
    
    print("✅ 批量处理测试通过")


def test_convenience_functions():
    """测试便捷函数"""
    print("🧪 测试便捷函数...")
    
    # 测试便捷创建函数
    processor = create_advantage_processor(
        normalization="standard",
        clipping="quantile",
        clip_value=2.0,
        negative_handling="relu",
        verbose=False
    )
    
    test_data = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = processor.process_advantages(test_data)
    
    print(f"便捷函数创建的处理器:")
    print(f"  原始: {test_data.tolist()}")
    print(f"  结果: {[f'{x:.4f}' for x in result.tolist()]}")
    
    # 验证设置
    assert processor.config.normalization_mode == AdvantageNormalizationMode.STANDARD
    assert processor.config.clipping_mode == AdvantageClippingMode.QUANTILE
    assert processor.config.negative_handling == NegativeHandlingMode.RELU
    
    print("✅ 便捷函数测试通过")


def test_statistical_tracking():
    """测试统计跟踪"""
    print("🧪 测试统计跟踪...")
    
    config = AdvantageProcessingConfig(
        track_statistics=True,
        verbose=False
    )
    processor = AdvantageProcessor(config)
    
    # 处理多批数据
    for i in range(5):
        test_data = torch.randn(20) * (i + 1)  # 逐渐增大的数据
        if i == 2:  # 在第3批中加入一些问题值
            test_data[0] = float('nan')
            test_data[1] = float('inf')
            test_data[2] = -100.0  # 极值
        
        processor.process_advantages(test_data)
    
    # 获取统计信息
    stats = processor.get_processing_stats()
    
    print(f"统计跟踪结果:")
    print(f"  总处理次数: {stats['total_processed']}")
    print(f"  NaN处理次数: {stats['nan_count']}")
    print(f"  Inf处理次数: {stats['inf_count']}")
    print(f"  负值处理次数: {stats['negative_count']}")
    print(f"  截断次数: {stats['clipped_count']}")
    
    assert stats['total_processed'] == 5, "应该处理了5批数据"
    assert stats['nan_count'] > 0, "应该检测到NaN"
    assert stats['inf_count'] > 0, "应该检测到Inf"
    
    # 打印详细统计
    processor.print_stats()
    
    print("✅ 统计跟踪测试通过")


def test_real_world_scenario():
    """测试真实世界场景"""
    print("🧪 测试真实世界场景...")
    
    # 模拟RIPT训练中的真实优势值
    # 通常包含：不同大小的奖励、一些极值、可能的数值问题
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 模拟RLOO计算后的优势值
    batch_size = 32
    raw_advantages = []
    
    for i in range(batch_size):
        # 模拟不同成功率的rollouts
        if i < 10:  # 前10个：高奖励
            adv = torch.normal(2.0, 0.5, (1,))
        elif i < 20:  # 中间10个：中等奖励
            adv = torch.normal(0.0, 1.0, (1,))
        else:  # 后12个：低奖励或负奖励
            adv = torch.normal(-1.0, 0.8, (1,))
        
        raw_advantages.append(adv.item())
    
    # 添加一些数值问题
    raw_advantages[0] = float('nan')    # 模拟计算错误
    raw_advantages[1] = float('inf')    # 模拟溢出
    raw_advantages[2] = -1000.0         # 模拟极端失败
    raw_advantages[3] = 1000.0          # 模拟极端成功
    
    advantages_tensor = torch.tensor(raw_advantages, dtype=torch.float32)
    
    print(f"原始优势值统计:")
    print(f"  数量: {len(raw_advantages)}")
    # 兼容性修复：替换torch.nanmin/nanmax
    valid_values = advantages_tensor[~torch.isinf(advantages_tensor) & ~torch.isnan(advantages_tensor)]
    if len(valid_values) > 0:
        print(f"  范围: [{valid_values.min():.2f}, {valid_values.max():.2f}]")
    else:
        print(f"  范围: [无有效值]")
    print(f"  NaN数: {torch.isnan(advantages_tensor).sum()}")
    print(f"  Inf数: {torch.isinf(advantages_tensor).sum()}")
    
    # 使用推荐的生产环境配置
    processor = create_advantage_processor(
        normalization="standard",    # 标准归一化
        clipping="symmetric",       # 对称截断
        clip_value=3.0,            # 3倍标准差截断
        negative_handling="softplus", # 软性处理负值
        verbose=True
    )
    
    # 处理优势值
    processed_advantages = processor.process_advantages(advantages_tensor)
    
    print(f"\n处理后优势值统计:")
    print(f"  范围: [{processed_advantages.min():.4f}, {processed_advantages.max():.4f}]")
    print(f"  均值: {processed_advantages.mean():.4f}")
    print(f"  标准差: {processed_advantages.std():.4f}")
    print(f"  负值数: {(processed_advantages < 0).sum()}")
    
    # 验证结果适合用于训练
    assert not torch.isnan(processed_advantages).any()
    assert not torch.isinf(processed_advantages).any()
    assert processed_advantages.min() >= 0  # softplus确保非负
    assert processed_advantages.max() <= 50  # 合理的上界
    assert abs(processed_advantages.mean()) < 2  # 合理的均值
    
    processor.print_stats()
    
    print("✅ 真实世界场景测试通过")


def main():
    """运行所有测试"""
    print("🚀 开始测试优势值处理器")
    print("="*50)
    
    try:
        test_normalization_modes()
        test_clipping_modes()
        test_negative_handling()
        test_numerical_stability()
        test_edge_cases()
        test_batch_processing()
        test_convenience_functions()
        test_statistical_tracking()
        test_real_world_scenario()
        
        print("\n" + "="*50)
        print("🎉 所有测试通过！")
        print("\n💡 推荐的生产环境配置:")
        print("   - normalization: 'standard' (零均值单位方差)")
        print("   - clipping: 'symmetric' with clip_value=3.0")
        print("   - negative_handling: 'softplus' (平滑处理)")
        print("   - 启用统计跟踪以监控数值健康状况")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
