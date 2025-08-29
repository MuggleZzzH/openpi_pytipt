#!/bin/bash

# ==============================================================================
# Bash 脚本: LIBERO 10 评估实验 (本地环境版本)
#
# 功能:
#   - 仿照 libero_spatial.sh 案例，实现健壮的脚本执行
#   - 双重输出: 使用 'tee' 命令将所有输出同时打印到终端并写入日志文件。
#   - 日志管理: 自动在 'eval_logs/libero_10' 目录下创建带时间戳的日志文件。
#   - 健壮性设计: 任何命令失败时立即退出 (set -eo pipefail)。
#   - 动态路径: 自动定位项目根目录，确保脚本可在任何位置执行。
#   - 环境激活: 自动激活指定的 Conda 环境 (mix)。
# ==============================================================================

# --- 脚本设置 ---
set -eo pipefail

# --- 步骤 1: 设置日志文件 ---
LOG_DIR="eval_logs/libero_10"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/eval_run_${TIMESTAMP}.log"

# --- 脚本主体通过 tee 同时输出到终端和日志文件 ---
{
    # --- 步骤 2: 定位项目根目录并切换 ---
    echo "--- 步骤 2: 定位项目根目录并切换工作目录 ---"
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
    PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
    cd "$PROJECT_ROOT"
    echo "✅ 已成功切换到项目目录: $(pwd)"
    echo "--- 所有日志将同时打印并在以下位置存档: $(pwd)/$LOG_FILE ---"
    echo

    # --- 步骤 3: 初始化并激活 Conda 环境 ---
    echo "--- 步骤 3: 初始化并激活 Conda 环境 (本地环境) ---"
    # 本地环境的 conda 初始化
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        echo "✅ Conda 初始化成功。"
    else
        echo "❌ 错误: 未找到 conda 命令。请确保已安装 conda。"
        exit 1
    fi
    conda activate mix
    echo "✅ Conda 环境 'mix' 已激活。"
    echo "   当前使用的 Python: $(which python)"
    echo

    # --- 步骤 4: 设置环境变量 ---
    echo "--- 步骤 4: 设置必要的环境变量 ---"
    export HYDRA_FULL_ERROR=1
    export PYTHONPATH=$PYTHONPATH:"$PROJECT_ROOT/LIBERO":"$PROJECT_ROOT"
    export HF_ENDPOINT=https://hf-mirror.com
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "✅ HYDRA_FULL_ERROR, PYTHONPATH, HF_ENDPOINT, PYTORCH_CUDA_ALLOC_CONF 已设置。"
    echo

    # --- 步骤 5: 设置实验参数 ---
    echo "--- 步骤 5: 设置实验参数 ---"
    PROJECT_CATEGORY="ript-eval"           # 大类：评估实验
    BENCHMARK_TYPE="libero_10"             # 小类：10任务基准
    EXPERIMENT_BASE="eval_10"              # 实验基础名
    EXPERIMENT_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    EXPERIMENT_NAME="${EXPERIMENT_BASE}_${EXPERIMENT_TIMESTAMP}"

    # 分层名称
    SWANLAB_PROJECT="${PROJECT_CATEGORY}"  
    SWANLAB_RUN_NAME="${BENCHMARK_TYPE}_${EXPERIMENT_NAME}"

    echo "🎯 LIBERO 10 评估实验"
    echo "📊 SwanLab项目: ${SWANLAB_PROJECT}"
    echo "🏷️  SwanLab运行: ${SWANLAB_RUN_NAME}"
    echo "----------------------------------------"
    echo

    # --- 步骤 6: 执行评估命令 ---
    echo "--- 步骤 6: 执行 Python 评估命令 ---"
    python 11_train_ript_vla_style.py \
      --config_path pi0/ript/config/stage11_unified_pool.yaml \
      global_seed=42 \
      exp_name="${EXPERIMENT_NAME}" \
      logging.swanlab_project="${SWANLAB_PROJECT}" \
      logging.swanlab_run_name="${SWANLAB_RUN_NAME}" \
      logging.use_swanlab=true \
      task.benchmark_name="${BENCHMARK_TYPE}" \
      features.eval_only=true \
      rollout.enabled=true \
      rollout.rollouts_per_env=50 \
      rollout.num_parallel_envs=50 \
      rollout.n_video_final=0

    echo
    echo "========================================================"
    echo "🎉 评估脚本执行完毕 ---"
    echo "--- 脚本结束: $(date) ---"
    echo "--- 完整的执行日志已保存在: $(pwd)/$LOG_FILE ---"
    echo "========================================================"

} 2>&1 | tee "$LOG_FILE"



