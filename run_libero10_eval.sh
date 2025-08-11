#!/bin/bash

# LIBERO-10 Checkpoint 快速评估脚本
# 使用方法: ./run_libero10_eval.sh [GPU_ID] [ROLLOUTS_PER_TASK]

set -e

# 默认参数
GPU_ID=${1:-0}
ROLLOUTS_PER_TASK=${2:-20}  # 默认每个任务20次评估 (快速版)

echo "🚀 LIBERO-10 Checkpoint 快速评估"
echo "================================"
echo "GPU设备: $GPU_ID"
echo "每任务评估次数: $ROLLOUTS_PER_TASK"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID
export TOKENIZERS_PARALLELISM=false

# 检查项目目录
if [ ! -d "/zhaohan/ZJH/openpi_pytorch" ]; then
    echo "❌ 项目目录不存在，请检查路径"
    exit 1
fi

cd /zhaohan/ZJH/openpi_pytorch

# 检查配置文件
CONFIG_FILE="pi0/ript/config/libero10_eval.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查checkpoint
CHECKPOINT_PATH="/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint目录不存在: $CHECKPOINT_PATH"
    echo "请确保已正确转换并放置PI0 checkpoint"
    exit 1
fi

echo "✓ 环境检查通过"
echo ""

# 修改配置中的评估次数 (如果用户指定了非默认值)
if [ "$ROLLOUTS_PER_TASK" != "50" ]; then
    echo "📝 临时修改评估配置: rollouts_per_task = $ROLLOUTS_PER_TASK"
    
    # 创建临时配置文件
    TEMP_CONFIG="pi0/ript/config/libero10_eval_temp.yaml"
    cp "$CONFIG_FILE" "$TEMP_CONFIG"
    
    # 使用sed修改rollouts_per_task
    sed -i "s/rollouts_per_task: [0-9]*/rollouts_per_task: $ROLLOUTS_PER_TASK/" "$TEMP_CONFIG"
    CONFIG_FILE="$TEMP_CONFIG"
fi

# 运行评估
echo "🎯 开始LIBERO-10评估..."
python evaluate_libero10.py --config_path "$CONFIG_FILE"

EVAL_EXIT_CODE=$?

# 清理临时文件
if [ -f "pi0/ript/config/libero10_eval_temp.yaml" ]; then
    rm "pi0/ript/config/libero10_eval_temp.yaml"
fi

# 检查评估结果
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 LIBERO-10评估完成!"
    echo ""
    echo "📁 结果文件位置:"
    find ./output/libero10_eval -name "evaluation_summary.txt" -exec echo "  汇总报告: {}" \;
    find ./output/libero10_eval -name "evaluation_results.json" -exec echo "  详细结果: {}" \;
    
    echo ""
    echo "📊 查看结果:"
    echo "  cat \$(find ./output/libero10_eval -name 'evaluation_summary.txt' | head -1)"
    
else
    echo "❌ 评估过程中出现错误"
    exit $EVAL_EXIT_CODE
fi