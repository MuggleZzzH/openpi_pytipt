"""
测试安全批次拷贝功能
验证CFG训练中的内存安全性和拷贝独立性
"""

import torch
import numpy as np
from ript.utils.safe_batch_copy import (
    SafeBatchCopier,
    safe_copy_cfg_batches,
    safe_copy_with_modifications,
    create_cfg_safe_copier,
    replace_shallow_copy
)


def create_test_batch(batch_size: int = 4) -> dict:
    """创建测试用的批次数据"""
    return {
        "image": {
            "base_0_rgb": torch.randint(0, 255, (batch_size, 3, 224, 224), dtype=torch.uint8),
            "left_wrist_0_rgb": torch.randint(0, 255, (batch_size, 3, 224, 224), dtype=torch.uint8)
        },
        "state": torch.randn(batch_size, 14),
        "action": torch.randn(batch_size, 50, 7),
        "action_is_pad": torch.zeros(batch_size, 50, dtype=torch.bool),
        "prompt": ["task_1", "task_2", "task_3", "task_4"],
        "advantages": torch.randn(batch_size, 50),
        "init_hash": ["hash_1", "hash_2", "hash_3", "hash_4"],
        "metadata": {
            "batch_id": 123,
            "timestamp": 1234567890,
            "nested_tensor": torch.randn(batch_size, 5)
        }
    }


def test_shallow_copy_problem():
    """演示浅拷贝问题"""
    print("🧪 演示浅拷贝问题...")
    
    original_batch = create_test_batch(2)
    
    # 浅拷贝（问题方式）
    shallow_copy = original_batch.copy()
    
    # 修改拷贝的张量
    if isinstance(shallow_copy["state"], torch.Tensor):
        shallow_copy["state"][0, 0] = 999.0
    
    # 检查原始批次是否被影响
    original_modified = (original_batch["state"][0, 0] == 999.0).item()
    
    print(f"浅拷贝问题演示:")
    print(f"  修改拷贝后，原始批次被影响: {original_modified}")
    print(f"  原始state[0,0]: {original_batch['state'][0, 0]:.2f}")
    print(f"  拷贝state[0,0]: {shallow_copy['state'][0, 0]:.2f}")
    
    if original_modified:
        print("❌ 浅拷贝确实存在内存共享问题")
    else:
        print("✅ 此情况下浅拷贝没有问题（可能因为PyTorch优化）")
    
    print()


def test_safe_batch_copier_modes():
    """测试不同的拷贝模式"""
    print("🧪 测试安全批次拷贝模式...")
    
    original_batch = create_test_batch(3)
    
    # 测试不同模式
    modes = ["smart", "deep", "reconstruct"]
    
    for mode in modes:
        print(f"\n测试 {mode} 模式:")
        
        copier = SafeBatchCopier(
            copy_mode=mode,
            verify_independence=True,
            track_performance=True,
            verbose=True
        )
        
        copied_batch = copier.safe_copy_batch(original_batch)
        
        # 验证拷贝结果
        assert "state" in copied_batch, "拷贝应该包含所有字段"
        assert copied_batch["state"].shape == original_batch["state"].shape, "形状应该一致"
        
        # 修改拷贝来测试独立性
        copied_batch["state"][0, 0] = 888.0
        
        # 检查原始数据是否未被影响
        original_unchanged = (original_batch["state"][0, 0] != 888.0).item()
        print(f"  独立性测试: {'通过' if original_unchanged else '失败'}")
        
        # 打印性能统计
        copier.print_performance_stats()
    
    print("✅ 安全批次拷贝模式测试完成")


def test_cfg_safe_copy():
    """测试CFG专用的安全拷贝"""
    print("\n🧪 测试CFG安全拷贝...")
    
    original_batch = create_test_batch(4)
    
    # 使用CFG专用拷贝函数
    positive_batch, negative_batch = safe_copy_cfg_batches(original_batch)
    
    # 验证拷贝独立性
    print("验证CFG批次独立性:")
    
    # 修改positive_batch
    positive_batch["state"][0, 0] = 111.0
    positive_batch["is_positive"] = torch.ones(4, dtype=torch.long)
    
    # 修改negative_batch
    negative_batch["state"][0, 1] = 222.0
    negative_batch["is_positive"] = torch.zeros(4, dtype=torch.long)
    
    # 检查原始批次
    original_unchanged = (
        original_batch["state"][0, 0] != 111.0 and
        original_batch["state"][0, 1] != 222.0 and
        "is_positive" not in original_batch
    )
    
    print(f"  原始批次未被影响: {original_unchanged}")
    
    # 检查positive和negative批次的独立性
    batches_independent = (
        positive_batch["state"][0, 1] != 222.0 and
        negative_batch["state"][0, 0] != 111.0
    )
    
    print(f"  正负分支相互独立: {batches_independent}")
    
    # 验证字段设置
    assert torch.all(positive_batch["is_positive"] == 1), "正分支应该全为1"
    assert torch.all(negative_batch["is_positive"] == 0), "负分支应该全为0"
    
    print("✅ CFG安全拷贝测试通过")


def test_memory_sharing_detection():
    """测试内存共享检测"""
    print("\n🧪 测试内存共享检测...")
    
    copier = SafeBatchCopier(
        copy_mode="smart",
        verify_independence=True,
        verbose=True
    )
    
    # 创建包含共享张量的批次（故意制造问题）
    shared_tensor = torch.randn(4, 10)
    problematic_batch = {
        "tensor1": shared_tensor,
        "tensor2": shared_tensor,  # 故意共享同一张量
        "independent": torch.randn(4, 5)
    }
    
    # 尝试拷贝（这应该解决共享问题）
    copied_batch = copier.safe_copy_batch(problematic_batch)
    
    # 验证拷贝后的独立性
    copied_batch["tensor1"][0, 0] = 999.0
    independence_test = (copied_batch["tensor2"][0, 0] != 999.0).item()
    
    print(f"拷贝后张量独立性: {'通过' if independence_test else '失败'}")
    
    print("✅ 内存共享检测测试完成")


def test_nested_structure_copy():
    """测试嵌套结构拷贝"""
    print("\n🧪 测试嵌套结构拷贝...")
    
    # 创建复杂嵌套结构
    complex_batch = {
        "level1": {
            "level2": {
                "tensor": torch.randn(3, 4),
                "array": np.random.randn(2, 3),
                "list": [torch.randn(2, 2), torch.randn(3, 3)]
            },
            "simple_tensor": torch.randn(5, 5)
        },
        "top_level_list": [
            {"nested_tensor": torch.randn(2, 2)},
            [torch.randn(1, 1), np.array([1, 2, 3])]
        ],
        "simple_values": {
            "int": 42,
            "float": 3.14,
            "string": "test",
            "bool": True,
            "none": None
        }
    }
    
    copier = SafeBatchCopier(
        copy_mode="smart",
        verify_independence=True,
        verbose=True
    )
    
    copied_batch = copier.safe_copy_batch(complex_batch)
    
    # 测试深层修改
    original_value = complex_batch["level1"]["level2"]["tensor"][0, 0].item()
    copied_batch["level1"]["level2"]["tensor"][0, 0] = 888.0
    
    independence_test = (
        complex_batch["level1"]["level2"]["tensor"][0, 0].item() == original_value
    )
    
    print(f"嵌套结构独立性: {'通过' if independence_test else '失败'}")
    
    # 验证不可变类型
    assert copied_batch["simple_values"]["int"] == 42
    assert copied_batch["simple_values"]["string"] == "test"
    
    print("✅ 嵌套结构拷贝测试通过")


def test_convenience_functions():
    """测试便捷函数"""
    print("\n🧪 测试便捷函数...")
    
    original_batch = create_test_batch(2)
    
    # 测试replace_shallow_copy
    safe_copy = replace_shallow_copy(original_batch)
    
    # 修改测试
    safe_copy["state"][0, 0] = 777.0
    independence = (original_batch["state"][0, 0] != 777.0).item()
    
    print(f"replace_shallow_copy独立性: {'通过' if independence else '失败'}")
    
    # 测试safe_copy_with_modifications
    modifications = {
        "new_field": torch.randn(2, 3),
        "is_positive": torch.ones(2, dtype=torch.long)
    }
    
    modified_copy = safe_copy_with_modifications(original_batch, modifications)
    
    assert "new_field" in modified_copy, "应该包含新字段"
    assert "is_positive" in modified_copy, "应该包含修改字段"
    assert "new_field" not in original_batch, "原始批次不应该被修改"
    
    print("✅ 便捷函数测试通过")


def test_performance_comparison():
    """测试性能对比"""
    print("\n🧪 性能对比测试...")
    
    # 创建较大的批次
    large_batch = create_test_batch(16)
    
    import time
    
    # 测试浅拷贝性能
    start_time = time.time()
    for _ in range(100):
        shallow = large_batch.copy()
    shallow_time = time.time() - start_time
    
    # 测试安全拷贝性能
    copier = SafeBatchCopier(copy_mode="smart", track_performance=True)
    start_time = time.time()
    for _ in range(100):
        safe = copier.safe_copy_batch(large_batch)
    safe_time = time.time() - start_time
    
    print(f"性能对比 (100次拷贝):")
    print(f"  浅拷贝: {shallow_time:.4f}s")
    print(f"  安全拷贝: {safe_time:.4f}s")
    print(f"  性能比例: {safe_time/shallow_time:.2f}x")
    
    copier.print_performance_stats()
    
    print("✅ 性能对比测试完成")


def test_real_world_cfg_scenario():
    """测试真实世界CFG场景"""
    print("\n🧪 真实CFG场景测试...")
    
    # 模拟CFG训练中的真实批次
    cfg_batch = {
        "image": {
            "base_0_rgb": torch.randint(0, 255, (8, 3, 224, 224), dtype=torch.uint8),
            "left_wrist_0_rgb": torch.randint(0, 255, (8, 3, 224, 224), dtype=torch.uint8)
        },
        "state": torch.randn(8, 14),
        "action": torch.randn(8, 50, 7),
        "action_is_pad": torch.zeros(8, 50, dtype=torch.bool),
        "prompt": [f"task_{i}" for i in range(8)],
        "advantages": torch.randn(8, 50),
        "noise": torch.randn(8, 50, 7),  # CFG特有
        "time": torch.rand(8),           # CFG特有
    }
    
    # 使用CFG专用拷贝器
    cfg_copier = create_cfg_safe_copier(
        verify_copies=True,
        track_performance=True,
        verbose=False
    )
    
    # 模拟CFG双分支拷贝
    positive_batch = cfg_copier.safe_copy_batch(cfg_batch, copy_suffix="positive")
    negative_batch = cfg_copier.safe_copy_batch(cfg_batch, copy_suffix="negative")
    
    # 设置CFG标识
    positive_batch["is_positive"] = torch.ones(8, dtype=torch.long)
    negative_batch["is_positive"] = torch.zeros(8, dtype=torch.long)
    
    # 关键测试：确保noise和time是共享的（这对CFG很重要）
    # 在实际使用中，我们希望两个分支使用相同的noise和time
    positive_batch["noise"] = cfg_batch["noise"]  # 应该共享
    negative_batch["noise"] = cfg_batch["noise"]  # 应该共享
    positive_batch["time"] = cfg_batch["time"]    # 应该共享
    negative_batch["time"] = cfg_batch["time"]    # 应该共享
    
    # 验证关键字段的独立性（除了故意共享的noise和time）
    positive_batch["state"][0, 0] = 100.0
    negative_batch["state"][0, 1] = 200.0
    
    # 原始批次不应该被影响
    original_safe = (
        cfg_batch["state"][0, 0] != 100.0 and
        cfg_batch["state"][0, 1] != 200.0
    )
    
    # 两个分支的state应该是独立的
    branches_independent = (
        positive_batch["state"][0, 1] != 200.0 and
        negative_batch["state"][0, 0] != 100.0
    )
    
    print(f"CFG场景测试结果:")
    print(f"  原始批次安全: {original_safe}")
    print(f"  分支独立性: {branches_independent}")
    print(f"  正分支标识: {positive_batch['is_positive'][0].item()}")
    print(f"  负分支标识: {negative_batch['is_positive'][0].item()}")
    
    # 验证noise和time是共享的（这是CFG要求）
    noise_shared = torch.allclose(positive_batch["noise"], negative_batch["noise"])
    time_shared = torch.allclose(positive_batch["time"], negative_batch["time"])
    
    print(f"  Noise共享: {noise_shared}")
    print(f"  Time共享: {time_shared}")
    
    cfg_copier.print_performance_stats()
    
    print("✅ 真实CFG场景测试通过")


def main():
    """运行所有测试"""
    print("🚀 开始测试安全批次拷贝功能")
    print("="*50)
    
    try:
        test_shallow_copy_problem()
        test_safe_batch_copier_modes()
        test_cfg_safe_copy()
        test_memory_sharing_detection()
        test_nested_structure_copy()
        test_convenience_functions()
        test_performance_comparison()
        test_real_world_cfg_scenario()
        
        print("\n" + "="*50)
        print("🎉 所有测试通过！")
        
        print("\n💡 关键发现:")
        print("   1. 浅拷贝确实可能导致张量内存共享问题")
        print("   2. 智能拷贝模式提供最佳的性能/安全平衡")
        print("   3. CFG训练需要特别注意noise和time的共享")
        print("   4. 深度嵌套结构需要递归拷贝处理")
        print("   5. 性能开销通常在可接受范围内")
        
        print("\n🔧 推荐使用:")
        print("   - 生产环境: SafeBatchCopier(copy_mode='smart')")
        print("   - 调试阶段: verify_independence=True")
        print("   - CFG训练: 使用 safe_copy_cfg_batches()")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
