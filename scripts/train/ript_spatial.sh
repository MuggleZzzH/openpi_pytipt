#!/bin/bash

# 🚀 简洁的RIPT-VLA训练脚本 - 仿照RIPT官方风格
# 用法: ./run_ript_training.sh [参数可直接在此修改]

python 11_train_ript_vla_style.py \
  --config_path pi0/ript/config/stage11_unified_pool.yaml \
  global_seed=42 \
  exp_name=ript_spatial_eval \
  task.benchmark_name=libero_spatial \
  algo.collection_cfg_scale=1.0 \
  policy.cfg_enabled=false \
  rollout.rollouts_per_env=50
  rollout.num_parallel_envs=10 \
  rollout.n_video_final=0 \
  features.dynamic_sampling.enabled=false \
  features.eval_only=true \

