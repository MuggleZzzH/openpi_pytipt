"""
OpenPI-RIPT集成测试
验证所有组件是否正常整合工作
"""

import torch
import numpy as np
from train_openpi_ript_integrated import OpenPIRiptTrainer, TrainingConfig


def test_component_initialization():
    """测试组件初始化"""
    print("🧪 测试组件初始化...")
    
    # 创建测试配置（快速模式）
    config = TrainingConfig(
        experiment_name="integration_test",
        num_train_steps=5,  # 只测试5步
        rloo_batch_size=4,  # 小批次
        enable_wandb=False,
        verbose=True,
        checkpoint_path="/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch"  # 需要真实路径
    )
    
    try:
        # 创建训练器
        trainer = OpenPIRiptTrainer(config)
        
        # 测试组件设置
        trainer.setup_components()
        
        print("✅ 所有组件初始化成功")
        return trainer
        
    except Exception as e:
        print(f"❌ 组件初始化失败: {e}")
        return None


def test_data_flow():
    """测试数据流"""
    print("\n🧪 测试数据流...")
    
    config = TrainingConfig(
        rloo_batch_size=2,
        action_chunk_size=10,
        verbose=False
    )
    
    trainer = OpenPIRiptTrainer(config)
    trainer.setup_components()
    
    try:
        # 测试数据收集
        training_batch = trainer.collect_and_process_data()
        
        if training_batch is None:
            print("⚠️ 使用模拟数据模式")
            return True
        
        # 验证数据格式
        required_keys = ["image", "state", "action", "action_is_pad", "advantages"]
        for key in required_keys:
            assert key in training_batch, f"缺少字段: {key}"
        
        # 验证数据形状
        batch_size = training_batch["state"].shape[0]
        assert batch_size == config.rloo_batch_size, f"批次大小不匹配: {batch_size}"
        
        print(f"✅ 数据流测试成功")
        print(f"   批次大小: {batch_size}")
        print(f"   状态形状: {training_batch['state'].shape}")
        print(f"   动作形状: {training_batch['action'].shape}")
        print(f"   优势范围: [{training_batch['advantages'].min():.3f}, {training_batch['advantages'].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据流测试失败: {e}")
        return False


def test_training_step():
    """测试训练步骤"""
    print("\n🧪 测试训练步骤...")
    
    config = TrainingConfig(
        rloo_batch_size=2,
        action_chunk_size=10,
        gradient_accumulation_steps=1,
        verbose=False
    )
    
    trainer = OpenPIRiptTrainer(config)
    trainer.setup_components()
    
    try:
        # 生成测试数据
        mock_batch = {
            "image": {
                "base_0_rgb": torch.randint(0, 255, (2, 3, 224, 224), dtype=torch.uint8).to(trainer.device),
                "left_wrist_0_rgb": torch.randint(0, 255, (2, 3, 224, 224), dtype=torch.uint8).to(trainer.device)
            },
            "state": torch.randn(2, config.target_state_dim).to(trainer.device),
            "action": torch.randn(2, config.action_chunk_size, 7).to(trainer.device),
            "action_is_pad": torch.zeros(2, config.action_chunk_size, dtype=torch.bool).to(trainer.device),
            "prompt": ["test_task_1", "test_task_2"],
            "advantages": torch.randn(2).to(trainer.device)
        }
        
        # 执行训练步骤
        metrics = trainer.train_step(mock_batch)
        
        # 验证指标
        assert "loss" in metrics, "缺少损失指标"
        assert not np.isnan(metrics["loss"]), "损失为NaN"
        assert not np.isinf(metrics["loss"]), "损失为无穷"
        
        print(f"✅ 训练步骤测试成功")
        print(f"   损失: {metrics['loss']:.6f}")
        print(f"   梯度范数: {metrics.get('grad_norm', 0):.4f}")
        print(f"   优势均值: {metrics.get('advantages_mean', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advantage_processing():
    """测试优势值处理"""
    print("\n🧪 测试优势值处理...")
    
    config = TrainingConfig()
    trainer = OpenPIRiptTrainer(config)
    trainer._setup_advantage_processor()
    
    try:
        # 创建测试优势值（包含各种情况）
        test_advantages = torch.tensor([
            float('nan'),    # NaN值
            float('inf'),    # 无穷值
            -10.0,          # 极端负值
            10.0,           # 极端正值
            0.5, -0.5, 1.0, -1.0  # 正常值
        ])
        
        # 处理优势值
        processed = trainer.advantage_processor.process_advantages(test_advantages)
        
        # 验证结果
        assert not torch.isnan(processed).any(), "处理后仍有NaN值"
        assert not torch.isinf(processed).any(), "处理后仍有无穷值"
        assert processed.min() >= 0, "负值处理失败"  # softplus应该确保非负
        
        print(f"✅ 优势值处理测试成功")
        print(f"   原始范围: NaN/Inf/[-10, 10]")
        print(f"   处理后范围: [{processed.min():.4f}, {processed.max():.4f}]")
        print(f"   处理后均值: {processed.mean():.4f}")
        
        trainer.advantage_processor.print_stats()
        
        return True
        
    except Exception as e:
        print(f"❌ 优势值处理测试失败: {e}")
        return False


def test_safe_copy():
    """测试安全拷贝功能"""
    print("\n🧪 测试CFG安全拷贝...")
    
    config = TrainingConfig()
    trainer = OpenPIRiptTrainer(config)
    trainer.setup_components()
    
    try:
        # 创建测试批次
        original_batch = {
            "state": torch.randn(2, 14),
            "action": torch.randn(2, 10, 7),
            "nested": {"tensor": torch.randn(2, 5)}
        }
        
        # 测试安全拷贝
        copied_batch = trainer.cfg_adapter._safe_copy_batch(original_batch, "test")
        
        # 修改拷贝来测试独立性
        copied_batch["state"][0, 0] = 999.0
        
        # 验证原始数据未被影响
        independence_check = (original_batch["state"][0, 0] != 999.0).item()
        
        print(f"✅ 安全拷贝测试{'成功' if independence_check else '失败'}")
        print(f"   拷贝独立性: {'通过' if independence_check else '失败'}")
        
        return independence_check
        
    except Exception as e:
        print(f"❌ 安全拷贝测试失败: {e}")
        return False


def test_mock_training_loop():
    """测试完整训练循环（模拟模式）"""
    print("\n🧪 测试完整训练循环...")
    
    config = TrainingConfig(
        experiment_name="mock_integration_test",
        num_train_steps=3,  # 只训练3步
        rloo_batch_size=2,
        action_chunk_size=5,
        gradient_accumulation_steps=1,
        log_interval=1,
        verbose=True
    )
    
    trainer = OpenPIRiptTrainer(config)
    
    try:
        trainer.setup_components()
        trainer.run_training()
        
        # 验证训练完成
        assert len(trainer.training_metrics) == 3, f"训练步数不匹配: {len(trainer.training_metrics)}"
        
        # 验证所有步骤都有有效损失
        for i, metrics in enumerate(trainer.training_metrics):
            assert "loss" in metrics, f"步骤{i}缺少损失"
            assert not np.isnan(metrics["loss"]), f"步骤{i}损失为NaN"
        
        print(f"✅ 完整训练循环测试成功")
        print(f"   训练步数: {len(trainer.training_metrics)}")
        print(f"   最终损失: {trainer.training_metrics[-1]['loss']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练循环测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_integration_tests():
    """运行所有集成测试"""
    print("🚀 OpenPI-RIPT 集成测试")
    print("="*50)
    
    test_results = {}
    
    # 运行各项测试
    tests = [
        ("组件初始化", test_component_initialization),
        ("数据流", test_data_flow),
        ("训练步骤", test_training_step),
        ("优势值处理", test_advantage_processing),
        ("安全拷贝", test_safe_copy),
        ("完整训练循环", test_mock_training_loop),
    ]
    
    for test_name, test_func in tests:
        try:
            if test_name == "组件初始化":
                result = test_func() is not None
            else:
                result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            test_results[test_name] = False
    
    # 打印测试摘要
    print("\n" + "="*50)
    print("📊 测试结果摘要:")
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总结: {passed}个测试通过, {failed}个测试失败")
    
    if failed == 0:
        print("🎉 所有集成测试通过！系统集成成功！")
    else:
        print(f"⚠️ 有{failed}个测试失败，需要检查相应组件")
    
    return failed == 0


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
