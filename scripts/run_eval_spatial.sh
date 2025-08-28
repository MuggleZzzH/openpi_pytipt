#!/bin/bash

# 🎯 LIBERO Spatial 评估实验
# SwanLab项目: ript-eval / 运行: spatial_xxx

PROJECT_CATEGORY="ript-eval"           # 大类：评估实验  
BENCHMARK_TYPE="libero_spatial"        # 小类：spatial基准
EXPERIMENT_BASE="eval_spatial"         # 实验基础名
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="${EXPERIMENT_BASE}_${TIMESTAMP}"

# 分层名称
SWANLAB_PROJECT="${PROJECT_CATEGORY}"  
SWANLAB_RUN_NAME="${BENCHMARK_TYPE}_${EXPERIMENT_NAME}"

echo "🎯 LIBERO Spatial 评估实验"
echo "📊 SwanLab项目: ${SWANLAB_PROJECT}"
echo "🏷️  SwanLab运行: ${SWANLAB_RUN_NAME}"
echo "----------------------------------------"

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
  rollout.rollouts_per_env=10 \
  rollout.n_video_final=3 \
  features.dynamic_sampling.enabled=false
