"""
测试OpenPI Rollout收集器
验证基本功能和数据格式转换
"""

import torch
import numpy as np
from typing import Dict, Any, List
from ript.collectors.openpi_rollout_collector import (
    OpenPIRolloutCollectorOpenPIStandard,
    OpenPIRolloutConfig,
    create_openpi_rollout_collector
)


def test_openpi_rollout_collector_basic():
    """测试基本的rollout收集功能"""
    print("🧪 测试OpenPI Rollout收集器基本功能...")
    
    # 创建测试配置
    config = OpenPIRolloutConfig(
        num_rollouts_per_collect=5,
        action_chunk_size=20,
        enable_dynamic_sampling=True,
        enable_state_skipping=False,  # 禁用跳过以便测试
        image_size=(128, 128),
        target_state_dim=10,
        action_dim=6,
    )
    
    # 创建收集器（无env_runner，会使用模拟数据）
    collector = OpenPIRolloutCollectorOpenPIStandard(
        config=config,
        env_runner=None,
        stats_tracker=None
    )
    
    # 收集rollouts
    samples = collector.collect_rollouts_openpi_format(
        task_name="test_task",
        num_rollouts=3
    )
    
    # 验证结果
    assert len(samples) > 0, "应该收集到至少一些样本"
    
    # 验证样本格式
    sample = samples[0]
    required_fields = ["image", "state", "action", "action_is_pad", "prompt"]
    
    for field in required_fields:
        assert field in sample, f"缺少必需字段: {field}"
    
    # 验证图像格式
    image_dict = sample["image"]
    assert "base_0_rgb" in image_dict, "缺少base_0_rgb图像"
    assert "left_wrist_0_rgb" in image_dict, "缺少left_wrist_0_rgb图像"
    
    base_img = image_dict["base_0_rgb"]
    assert isinstance(base_img, torch.Tensor), "图像应该是torch.Tensor"
    assert base_img.shape == (3, 128, 128), f"图像形状错误: {base_img.shape}"
    
    # 验证状态
    state = sample["state"]
    assert isinstance(state, torch.Tensor), "状态应该是torch.Tensor"
    assert state.shape == (10,), f"状态维度错误: {state.shape}"
    
    # 验证动作chunk
    action = sample["action"]
    action_padding = sample["action_is_pad"]
    assert isinstance(action, torch.Tensor), "动作应该是torch.Tensor"
    assert isinstance(action_padding, torch.Tensor), "padding应该是torch.Tensor"
    assert action.shape == (20, 6), f"动作chunk形状错误: {action.shape}"
    assert action_padding.shape == (20,), f"padding形状错误: {action_padding.shape}"
    
    # 验证RIPT字段
    assert "advantages" in sample, "缺少advantages字段"
    assert "init_hash" in sample, "缺少init_hash字段"
    
    print(f"✅ 基本功能测试通过 - 收集了 {len(samples)} 个样本")
    
    # 打印统计信息
    collector.print_stats()
    
    return collector, samples


def test_dynamic_sampling_filter():
    """测试动态采样过滤功能"""
    print("\n🧪 测试动态采样过滤...")
    
    config = OpenPIRolloutConfig(
        num_rollouts_per_collect=10,
        enable_dynamic_sampling=True,
        enable_state_skipping=False,
    )
    
    collector = OpenPIRolloutCollectorOpenPIStandard(
        config=config,
        env_runner=None,
        stats_tracker=None
    )
    
    # 创建全成功的rollouts（应该被过滤）
    uniform_success_rollouts = [
        {'success': True, 'total_reward': 1.0} for _ in range(5)
    ]
    
    # 创建全失败的rollouts（应该被过滤）
    uniform_failure_rollouts = [
        {'success': False, 'total_reward': 0.0} for _ in range(5)
    ]
    
    # 创建混合结果的rollouts（不应该被过滤）
    mixed_rollouts = [
        {'success': True, 'total_reward': 1.0},
        {'success': False, 'total_reward': 0.0},
        {'success': True, 'total_reward': 0.8},
    ]
    
    # 测试过滤
    filtered_uniform_success = collector._apply_dynamic_sampling_filter(uniform_success_rollouts)
    filtered_uniform_failure = collector._apply_dynamic_sampling_filter(uniform_failure_rollouts)
    filtered_mixed = collector._apply_dynamic_sampling_filter(mixed_rollouts)
    
    assert len(filtered_uniform_success) == 0, "全成功rollouts应该被过滤"
    assert len(filtered_uniform_failure) == 0, "全失败rollouts应该被过滤"
    assert len(filtered_mixed) == 3, "混合rollouts不应该被过滤"
    
    print("✅ 动态采样过滤测试通过")


def test_action_chunk_generation():
    """测试action chunk生成"""
    print("\n🧪 测试Action Chunk生成...")
    
    config = OpenPIRolloutConfig(
        action_chunk_size=15,
        action_dim=4
    )
    
    collector = OpenPIRolloutCollectorOpenPIStandard(
        config=config,
        env_runner=None,
        stats_tracker=None
    )
    
    # 测试长轨迹（需要截断）
    long_actions = torch.randn(25, 4)  # 25步，4维动作
    chunk, padding = collector._generate_action_chunk(long_actions)
    
    assert chunk.shape == (15, 4), f"长轨迹chunk形状错误: {chunk.shape}"
    assert padding.shape == (15,), f"长轨迹padding形状错误: {padding.shape}"
    assert not padding.any(), "长轨迹不应该有padding"
    
    # 测试短轨迹（需要padding）
    short_actions = torch.randn(8, 4)  # 8步，4维动作
    chunk, padding = collector._generate_action_chunk(short_actions)
    
    assert chunk.shape == (15, 4), f"短轨迹chunk形状错误: {chunk.shape}"
    assert padding.shape == (15,), f"短轨迹padding形状错误: {padding.shape}"
    assert padding[8:].all(), "短轨迹的padding部分应该为True"
    assert not padding[:8].any(), "短轨迹的有效部分应该为False"
    
    # 测试维度不匹配（需要维度对齐）
    wrong_dim_actions = torch.randn(10, 3)  # 3维动作，期望4维
    chunk, padding = collector._generate_action_chunk(wrong_dim_actions)
    
    assert chunk.shape == (15, 4), f"维度对齐chunk形状错误: {chunk.shape}"
    # 应该在最后一维padding了1维
    
    print("✅ Action Chunk生成测试通过")


def test_factory_function():
    """测试工厂函数"""
    print("\n🧪 测试工厂函数...")
    
    config_dict = {
        'rloo_batch_size': 6,
        'action_chunk_size': 30,
        'enable_dynamic_sampling': False,
        'enable_state_skipping': True,
        'image_size': [256, 256],
        'target_state_dim': 12,
        'action_dim': 8,
        'rollout_skip_threshold': 5,
        'rollout_stats_path': './test_stats.json',
        'task_id': 1,
    }
    
    collector = create_openpi_rollout_collector(
        config_dict=config_dict,
        env_runner=None,
        stats_tracker=None
    )
    
    # 验证配置传递
    assert collector.config.num_rollouts_per_collect == 6
    assert collector.config.action_chunk_size == 30
    assert collector.config.enable_dynamic_sampling == False
    assert collector.config.enable_state_skipping == True
    assert collector.config.image_size == (256, 256)
    assert collector.config.target_state_dim == 12
    assert collector.config.action_dim == 8
    
    print("✅ 工厂函数测试通过")


def test_image_formatting():
    """测试图像格式化"""
    print("\n🧪 测试图像格式化...")
    
    config = OpenPIRolloutConfig(image_size=(64, 64))
    collector = OpenPIRolloutCollectorOpenPIStandard(config, None, None)
    
    # 测试正常图像
    normal_images = {
        'base_camera': np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        'wrist_camera': np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    }
    
    formatted = collector._format_images_openpi(normal_images, 0)
    
    assert "base_0_rgb" in formatted
    assert "left_wrist_0_rgb" in formatted
    assert formatted["base_0_rgb"].shape == (3, 64, 64)
    assert formatted["left_wrist_0_rgb"].shape == (3, 64, 64)
    
    # 测试空图像字典
    empty_images = {}
    formatted_empty = collector._format_images_openpi(empty_images, 1)
    
    assert "base_0_rgb" in formatted_empty
    assert "left_wrist_0_rgb" in formatted_empty
    assert formatted_empty["base_0_rgb"].shape == (3, 64, 64)
    
    print("✅ 图像格式化测试通过")


def validate_openpi_compatibility(samples: List[Dict[str, Any]]):
    """验证与OpenPI格式的兼容性"""
    print("\n🔍 验证OpenPI格式兼容性...")
    
    for i, sample in enumerate(samples[:3]):  # 检查前3个样本
        print(f"样本 {i}:")
        
        # 验证必需字段
        required_fields = ["image", "state", "action", "action_is_pad", "prompt"]
        for field in required_fields:
            assert field in sample, f"样本{i}缺少字段: {field}"
            print(f"  ✓ {field}")
        
        # 验证数据类型和形状
        assert isinstance(sample["image"], dict), "image应该是字典"
        assert isinstance(sample["state"], torch.Tensor), "state应该是tensor"
        assert isinstance(sample["action"], torch.Tensor), "action应该是tensor"
        assert isinstance(sample["action_is_pad"], torch.Tensor), "action_is_pad应该是tensor"
        assert isinstance(sample["prompt"], str), "prompt应该是字符串"
        
        print(f"  ✓ 数据类型检查通过")
        print(f"  ✓ 状态形状: {sample['state'].shape}")
        print(f"  ✓ 动作形状: {sample['action'].shape}")
        print(f"  ✓ 图像: {list(sample['image'].keys())}")
    
    print("✅ OpenPI格式兼容性验证通过")


def main():
    """运行所有测试"""
    print("🚀 开始测试OpenPI Rollout收集器")
    print("="*50)
    
    try:
        # 基本功能测试
        collector, samples = test_openpi_rollout_collector_basic()
        
        # 动态采样测试
        test_dynamic_sampling_filter()
        
        # Action chunk测试
        test_action_chunk_generation()
        
        # 工厂函数测试
        test_factory_function()
        
        # 图像格式化测试
        test_image_formatting()
        
        # OpenPI兼容性验证
        validate_openpi_compatibility(samples)
        
        print("\n" + "="*50)
        print("🎉 所有测试通过！")
        
        # 显示最终统计
        print("\n📊 最终收集统计:")
        collector.print_stats()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
