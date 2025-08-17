#!/usr/bin/env python3
"""
RIPT-VLA训练启动器（内存优化版本）

解决GPU显存碎片化和OOM问题的启动器
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_memory_optimization():
    """设置内存优化环境变量"""
    
    # PyTorch CUDA内存管理优化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 启用同步执行（便于调试）
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 启用cuDNN v8 API
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    # 禁用CUDA缓存（减少内存碎片）
    os.environ['CUDA_CACHE_DISABLE'] = '1'
    
    print("🔧 内存优化环境变量已设置:")
    print(f"  PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    print(f"  CUDA_LAUNCH_BLOCKING={os.environ['CUDA_LAUNCH_BLOCKING']}")
    print(f"  TORCH_CUDNN_V8_API_ENABLED={os.environ['TORCH_CUDNN_V8_API_ENABLED']}")

def check_gpu_memory():
    """检查GPU内存状态"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print("🔍 GPU内存状态:")
            for i, line in enumerate(lines):
                total, used, free = line.split(', ')
                print(f"  GPU {i}: {used}MB/{total}MB used, {free}MB free")
        else:
            print("⚠️ 无法获取GPU状态")
            
    except FileNotFoundError:
        print("⚠️ nvidia-smi未找到，跳过GPU状态检查")

def run_training_with_memory_optimization(config_path: str):
    """运行内存优化的训练"""
    
    print("🚀 启动RIPT-VLA训练（内存优化模式）")
    
    # 设置内存优化
    setup_memory_optimization()
    
    # 检查GPU状态
    check_gpu_memory()
    
    # 导入训练模块（在环境变量设置后）
    print("📦 导入训练模块...")
    
    # 添加项目根目录到路径
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        # 导入并运行训练
        from main_train_ript_vla_style import main
        
        print(f"🎯 开始训练，配置文件: {config_path}")
        
        # 模拟命令行参数
        sys.argv = ['train_with_memory_optimization.py', '--config', config_path]
        
        # 运行训练
        main()
        
        print("✅ 训练完成")
        
    except ImportError as e:
        print(f"❌ 导入训练模块失败: {e}")
        print("🔄 回退到subprocess方式...")
        
        # 回退到subprocess
        cmd = [sys.executable, '11_train_ript_vla_style.py', '--config', config_path]
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("✅ 训练完成")
        else:
            print(f"❌ 训练失败，退出码: {result.returncode}")
            
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RIPT-VLA内存优化训练启动器')
    parser.add_argument('--config', 
                       default='pi0/ript/config/stage11_unified_pool.yaml',
                       help='训练配置文件路径')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return 1
    
    # 运行训练
    run_training_with_memory_optimization(str(config_path))
    
    return 0

if __name__ == "__main__":
    exit(main())
