"""
OpenPI-RIPT集成系统快速启动脚本
提供简单的界面来测试和运行集成训练
"""

import os
import sys
import argparse
from pathlib import Path


def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print(f"❌ Python版本过低: {sys.version_info}, 需要3.8+")
        return False
    
    # 检查PyTorch
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   ✅ CUDA: {torch.version.cuda}, GPU数量: {torch.cuda.device_count()}")
        else:
            print(f"   ⚠️ CUDA不可用，将使用CPU")
    except ImportError:
        print(f"   ❌ PyTorch未安装")
        return False
    
    # 检查关键依赖
    required_packages = ['numpy', 'tqdm']
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"   ✅ {pkg}")
        except ImportError:
            print(f"   ❌ {pkg}未安装")
            return False
    
    # 检查数据目录
    data_dirs = [
        "./output",
        "./checkpoints",
    ]
    
    for dir_path in data_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"   🔧 创建目录: {dir_path}")
    
    print("✅ 环境检查完成")
    return True


def run_integration_tests():
    """运行集成测试"""
    print("\n🧪 运行集成测试...")
    
    try:
        from test_integration import run_integration_tests
        return run_integration_tests()
    except ImportError as e:
        print(f"❌ 无法导入测试模块: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        return False


def run_quick_training():
    """运行快速训练测试"""
    print("\n🚀 运行快速训练测试...")
    
    try:
        from train_openpi_ript_integrated import OpenPIRiptTrainer, TrainingConfig
        
        # 创建快速测试配置
        config = TrainingConfig(
            experiment_name="quick_test",
            num_train_steps=10,
            rloo_batch_size=4,
            action_chunk_size=20,
            gradient_accumulation_steps=2,
            log_interval=2,
            save_interval=10,
            verbose=True,
            enable_wandb=False
        )
        
        print("配置:")
        print(f"   训练步数: {config.num_train_steps}")
        print(f"   批次大小: {config.rloo_batch_size}")
        print(f"   梯度累积: {config.gradient_accumulation_steps}")
        
        # 创建并运行训练器
        trainer = OpenPIRiptTrainer(config)
        trainer.setup_components()
        trainer.run_training()
        
        print("✅ 快速训练完成")
        return True
        
    except Exception as e:
        print(f"❌ 快速训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_component_demo():
    """运行组件演示"""
    print("\n🔧 运行组件演示...")
    
    try:
        print("1. 测试优势值处理器...")
        from ript.utils.advantage_processor import create_advantage_processor
        import torch
        
        processor = create_advantage_processor(
            normalization="standard",
            clipping="symmetric", 
            clip_value=2.0,
            negative_handling="softplus",
            verbose=True
        )
        
        test_advantages = torch.tensor([float('nan'), -5.0, 0.0, 5.0, float('inf')])
        processed = processor.process_advantages(test_advantages)
        print(f"   处理结果: {processed}")
        
        print("\n2. 测试安全批次拷贝...")
        from ript.utils.safe_batch_copy import create_cfg_safe_copier
        
        copier = create_cfg_safe_copier(verify_copies=True, verbose=True)
        test_batch = {
            "tensor": torch.randn(2, 3),
            "nested": {"inner": torch.randn(2, 2)}
        }
        
        copied = copier.safe_copy_batch(test_batch, copy_suffix="demo")
        copied["tensor"][0, 0] = 999.0  # 修改拷贝
        
        independence = (test_batch["tensor"][0, 0] != 999.0).item()
        print(f"   拷贝独立性: {'通过' if independence else '失败'}")
        
        print("\n3. 测试数据包装器...")
        from utils.openpi_ript_dataset_wrapper import OpenPIRiptDatasetWrapper
        
        # 注意：这里可能会因为数据集不存在而失败，但展示了使用方法
        print("   (数据集演示需要真实数据，跳过)")
        
        print("✅ 组件演示完成")
        return True
        
    except Exception as e:
        print(f"❌ 组件演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_system_info():
    """显示系统信息"""
    print("\n📊 OpenPI-RIPT集成系统信息")
    print("="*50)
    
    # 显示组件状态
    components = [
        ("数据格式兼容器", "utils/openpi_ript_dataset_wrapper.py"),
        ("状态维度适配器", "utils/state_dimension_adapter.py"), 
        ("Rollout收集器", "ript/collectors/openpi_rollout_collector.py"),
        ("优势值处理器", "ript/utils/advantage_processor.py"),
        ("CFG安全拷贝", "ript/utils/safe_batch_copy.py"),
        ("主训练循环", "train_openpi_ript_integrated.py"),
        ("集成测试", "test_integration.py"),
    ]
    
    print("核心组件状态:")
    for name, path in components:
        exists = os.path.exists(path)
        status = "✅ 存在" if exists else "❌ 缺失"
        print(f"   {name}: {status}")
    
    # 显示配置信息
    print("\n默认配置:")
    print("   模型: PI0 (PaliGemma-based)")
    print("   数据格式: OpenPI标准")
    print("   优势计算: RLOO")
    print("   训练方式: CFG-style")
    print("   安全拷贝: 启用")
    
    print("\n使用方法:")
    print("   python quick_start_integration.py --mode test    # 运行测试")
    print("   python quick_start_integration.py --mode train   # 快速训练")
    print("   python quick_start_integration.py --mode demo    # 组件演示")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="OpenPI-RIPT集成系统快速启动")
    parser.add_argument(
        "--mode", 
        choices=["info", "test", "train", "demo"], 
        default="info",
        help="运行模式"
    )
    parser.add_argument("--skip-env-check", action="store_true", help="跳过环境检查")
    
    args = parser.parse_args()
    
    print("🌟 OpenPI-RIPT 集成系统")
    print("="*50)
    
    # 环境检查
    if not args.skip_env_check:
        if not check_environment():
            print("\n❌ 环境检查失败，请修复后重试")
            return False
    
    # 执行对应模式
    success = True
    
    if args.mode == "info":
        show_system_info()
        
    elif args.mode == "test":
        success = run_integration_tests()
        
    elif args.mode == "train":
        success = run_quick_training()
        
    elif args.mode == "demo":
        success = run_component_demo()
    
    print("\n" + "="*50)
    if success:
        print("🎉 操作成功完成！")
    else:
        print("❌ 操作失败，请检查错误信息")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
