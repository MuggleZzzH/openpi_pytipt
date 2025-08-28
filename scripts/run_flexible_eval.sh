#!/bin/bash

# 🎯 灵活任务选择评估脚本
# 演示不同的任务选择模式

# 参数配置（可修改）
BENCHMARK_TYPE="${1:-libero_spatial}"        # 基准类型：spatial/goal/object
TASK_MODE="${2:-auto_all}"                   # 任务选择模式
TASK_COUNT="${3:-3}"                         # 子集模式的任务数量

PROJECT_CATEGORY="ript-eval"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

case "$TASK_MODE" in
    "auto_all")
        EXPERIMENT_NAME="all_${BENCHMARK_TYPE}_${TIMESTAMP}"
        MODE_DESC="所有任务"
        ;;
    "auto_subset")
        EXPERIMENT_NAME="subset${TASK_COUNT}_${BENCHMARK_TYPE}_${TIMESTAMP}"  
        MODE_DESC="前${TASK_COUNT}个任务"
        ;;
    "manual")
        EXPERIMENT_NAME="manual_${BENCHMARK_TYPE}_${TIMESTAMP}"
        MODE_DESC="手动指定任务"
        ;;
    *)
        echo "❌ 无效的任务模式: $TASK_MODE"
        echo "💡 可用模式: auto_all, auto_subset, manual"
        exit 1
        ;;
esac

SWANLAB_PROJECT="${PROJECT_CATEGORY}"
SWANLAB_RUN_NAME="${BENCHMARK_TYPE}_${EXPERIMENT_NAME}"

echo "🎯 灵活任务选择评估"
echo "🎪 基准类型: ${BENCHMARK_TYPE}"
echo "⚙️  任务模式: ${TASK_MODE} (${MODE_DESC})"
echo "📊 SwanLab项目: ${SWANLAB_PROJECT}"
echo "🏷️  SwanLab运行: ${SWANLAB_RUN_NAME}"
echo "----------------------------------------"

# 根据任务模式构建不同的命令
if [ "$TASK_MODE" = "manual" ]; then
    echo "💡 手动模式：需要在配置文件中指定 task_names_to_use"
    EXTRA_ARGS="task.task_selection_mode=manual"
else
    EXTRA_ARGS="task.task_selection_mode=${TASK_MODE}"
    if [ "$TASK_MODE" = "auto_subset" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} task.auto_task_count=${TASK_COUNT}"
    fi
fi

python 11_train_ript_vla_style.py \
  --config_path pi0/ript/config/stage11_unified_pool.yaml \
  global_seed=42 \
  exp_name="${EXPERIMENT_NAME}" \
  logging.swanlab_project="${SWANLAB_PROJECT}" \
  logging.swanlab_run_name="${SWANLAB_RUN_NAME}" \
  logging.use_swanlab=true \
  task.benchmark_name="${BENCHMARK_TYPE}" \
  ${EXTRA_ARGS} \
  features.eval_only=true \
  rollout.enabled=true \
  rollout.rollouts_per_env=5 \
  rollout.n_video_final=2 \
  features.dynamic_sampling.enabled=false

echo "----------------------------------------"
echo "✅ 评估完成"
echo "📊 结果查看: SwanLab > ${SWANLAB_PROJECT} > ${SWANLAB_RUN_NAME}"
