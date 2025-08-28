#!/bin/bash

# 🚀 RIPT-VLA分层实验训练脚本
# 用法: ./run_ript_training_fixed.sh [可修改下面的参数]

# 🎯 分层实验配置
PROJECT_CATEGORY="ript-eval"          # 大类：eval/train/debug等
BENCHMARK_TYPE="libero_spatial"       # 基准测试：spatial/goal/object等  
EXPERIMENT_BASE="stage11_unified_pool_test"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="${EXPERIMENT_BASE}_${TIMESTAMP}"

# 构建分层名称
SWANLAB_PROJECT="${PROJECT_CATEGORY}"                    # 大类项目名
SWANLAB_RUN_NAME="${BENCHMARK_TYPE}_${EXPERIMENT_NAME}"  # 小类运行名

echo "🚀 启动RIPT-VLA分层实验"
echo "📊 SwanLab项目（大类）: ${SWANLAB_PROJECT}"
echo "🏷️  SwanLab运行（小类）: ${SWANLAB_RUN_NAME}"
echo "🎯 基准测试类型: ${BENCHMARK_TYPE}"
echo "📝 实验基础名: ${EXPERIMENT_NAME}"
echo "----------------------------------------"

python 11_train_ript_vla_style.py \
  --config_path pi0/ript/config/stage11_unified_pool.yaml \
  global_seed=42 \
  exp_name="${EXPERIMENT_NAME}" \
  logging.swanlab_project="${SWANLAB_PROJECT}" \
  logging.swanlab_run_name="${SWANLAB_RUN_NAME}" \
  logging.use_swanlab=true \
  logging.swanlab_mode="online" \
  task.benchmark_name="${BENCHMARK_TYPE}" \
  task.num_parallel_envs=2 \
  algo.demo_batch_size=1 \
  algo.rloo_batch_size=2 \
  algo.gradient_accumulation_steps=2 \
  algo.collection_cfg_scale=1.0 \
  policy.cfg_enabled=false \
  training.num_train_steps=1 \
  rollout.enabled=true \
  rollout.steps=1 \
  rollout.rollouts_per_env=2 \
  rollout.num_parallel_envs=2 \
  rollout.n_video_final=0 \
  features.dynamic_sampling.enabled=false \
  unified_pool_batch_size=8

echo "----------------------------------------"
echo "✅ 训练脚本执行完成"
echo "📊 SwanLab查看路径:"
echo "   项目: ${SWANLAB_PROJECT}"
echo "   运行: ${SWANLAB_RUN_NAME}"
