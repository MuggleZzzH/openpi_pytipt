#!/bin/bash

# 🚀 简洁的RIPT-VLA训练脚本 - 仿照RIPT官方风格
# 用法: ./run_ript_training.sh [参数可直接在此修改]

python 11_train_ript_vla_style.py \
  --config_path pi0/ript/config/stage11_unified_pool.yaml \
  global_seed=42 \
  exp_name=stage11_unified_pool_test \
  task.benchmark_name=libero_spatial \
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
