# 🚀 RIPT-VLA 简洁训练指南

## 快速启动

```bash
chmod +x run_ript_training.sh
./run_ript_training.sh
```

## 自定义配置

直接编辑 `run_ript_training.sh` 中的参数：

### 🔥 核心参数说明

```bash
# 统一种子 - 控制所有随机性
global_seed=42

# 实验配置
exp_name=stage11_unified_pool_test      # 实验名称
task.benchmark_name=libero_spatial       # 基准测试
task.num_parallel_envs=2                # 并行环境数

# 批次配置  
algo.demo_batch_size=1                  # Demo批次大小
algo.rloo_batch_size=2                  # RLOO批次大小
algo.gradient_accumulation_steps=2      # 梯度累积步数
unified_pool_batch_size=8               # 统一池批次大小

# CFG配置
algo.collection_cfg_scale=1.0           # CFG缩放因子
policy.cfg_enabled=false                # CFG启用/禁用

# 训练配置
training.num_train_steps=1              # 训练步数

# 评估配置
rollout.enabled=true                    # 评估启用
rollout.steps=1                         # 评估频率(每N步)
rollout.rollouts_per_env=2              # 每环境rollout数
rollout.num_parallel_envs=2             # 评估并行环境数
rollout.n_video_final=0                 # 最终评估视频数

# 动态采样
features.dynamic_sampling.enabled=false # 动态采样启用/禁用
```

## 配置示例

### 快速测试
```bash
global_seed=42
training.num_train_steps=1
rollout.enabled=false
```

### 完整训练
```bash
global_seed=123
exp_name=my_experiment
training.num_train_steps=50
algo.demo_batch_size=3
algo.rloo_batch_size=4
rollout.enabled=true
rollout.steps=10
```

### 大批次训练
```bash
algo.demo_batch_size=4
algo.rloo_batch_size=6
algo.gradient_accumulation_steps=4
unified_pool_batch_size=16
task.num_parallel_envs=4
```

### CFG启用训练
```bash
policy.cfg_enabled=true
algo.collection_cfg_scale=1.5
```

## 🎯 参数对应关系

| Shell参数 | YAML路径 | 说明 |
|-----------|----------|------|
| `global_seed=42` | `global_seed` | 统一种子 |
| `exp_name=xxx` | `exp_name` | 实验名 |
| `task.benchmark_name=libero_spatial` | `task.benchmark_name` | 基准 |
| `task.num_parallel_envs=2` | `task.num_parallel_envs` | 并行环境 |
| `algo.demo_batch_size=1` | `algo.demo_batch_size` | Demo批次 |
| `algo.rloo_batch_size=2` | `algo.rloo_batch_size` | RLOO批次 |
| `unified_pool_batch_size=8` | `unified_pool_batch_size` | 统一池批次 |

## SwanLab记录

- 所有指标自动记录到SwanLab
- 项目名：`openpi-ript-vla` 
- 运行名：使用`exp_name`参数

## 注意事项

1. **种子统一**：所有随机性都由`global_seed`控制
2. **参数格式**：使用点记法，如`algo.lr=1e-4`
3. **布尔值**：使用`true/false`（小写）
4. **字符串**：可以直接写，无需引号

## 故障排除

如果遇到问题：
1. 检查YAML语法是否正确
2. 确认所有路径存在
3. 查看控制台输出的参数override信息
