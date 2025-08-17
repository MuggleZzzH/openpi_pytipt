#!/bin/bash

# RIPT-VLA训练启动脚本（内存优化版本）
# 解决GPU显存碎片化问题

echo "🚀 启动RIPT-VLA训练（内存优化模式）"

# 设置PyTorch CUDA内存管理环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# 设置内存优化参数
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_CACHE_DISABLE=1

# 显示环境变量设置
echo "📊 内存优化环境变量:"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "  TORCH_CUDNN_V8_API_ENABLED=$TORCH_CUDNN_V8_API_ENABLED"

# 检查GPU状态
echo "🔍 GPU状态检查:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits

# 运行训练
echo "🎯 开始训练..."
python 11_train_ript_vla_style.py --config pi0/ript/config/stage11_unified_pool.yaml

echo "✅ 训练完成"
