# 🎯 LIBERO Demo集成使用指南

## 📋 概述

我们已经成功集成了与原版RIPT兼容的LIBERO demo数据加载功能，现在可以像原版RIPT一样从LIBERO数据集中加载专家演示的初始状态进行训练。

## 🔍 实现原理

### **原版RIPT的Demo机制**
```python
# 原版RIPT的工作流程
demo_batch = dataloader.next()  # 从LIBERO数据集加载6个demo
for demo in demo_batch:
    # 每个demo提供：
    # - 任务描述
    # - 初始状态
    # - 任务ID
    rollouts = generate_rollouts(demo, rloo_batch_size=8)  # 基于demo生成8个rollout
```

### **我们的实现**
```python
# 我们的实现
demo_dataloader = create_libero_demo_dataloader(...)  # 创建LIBERO数据加载器
demo_batch = next(demo_dataloader)  # 获取demo初始状态
group_episodes = collect_rollouts_ript_vla_style(..., demo_initial_state=demo_batch)
```

## 🛠️ 使用方法

### **1. 准备LIBERO数据集**

首先需要下载LIBERO数据集：

```bash
# 下载LIBERO数据集
cd /path/to/your/data/directory
python -m libero.benchmark_scripts.download_libero_datasets

# 数据集结构应该如下：
# /path/to/libero/datasets/
# ├── libero/
# │   ├── LIBERO_SPATIAL/
# │   │   ├── LIBERO_SPATIAL_0/
# │   │   │   ├── demo_0.hdf5
# │   │   │   ├── demo_1.hdf5
# │   │   │   └── ...
# │   │   └── ...
# │   └── ...
```

### **2. 修改配置文件**

编辑 `pi0/ript/config/stage11_unified_pool.yaml`：

```yaml
# 任务配置
task:
  benchmark_name: "LIBERO_SPATIAL"
  task_names_to_use:
    - "LIBERO_SPATIAL_0"

# LIBERO数据集配置
use_libero_demos: true
libero_data_prefix: "/path/to/your/libero/datasets"  # 🔥 修改为实际路径
benchmark_name: "LIBERO_SPATIAL"

# 算法配置
algo:
  demo_batch_size: 6  # 每步从数据集中采样6个demo（与原版RIPT一致）
  rloo_batch_size: 8  # 每个demo生成8个rollout
```

### **3. 运行训练**

```bash
python 11_train_ript_vla_style.py --config_path pi0/ript/config/stage11_unified_pool.yaml
```

## 📊 预期效果

### **训练日志示例**
```
✅ LIBERO demo数据加载器创建成功
  数据路径: /path/to/libero/datasets
  基准: LIBERO_SPATIAL
  数据集大小: 500

🔧 批次配置:
  demo_batch_size: 6 (每步收集的组数)
  rloo_batch_size: 8 (每组内样本数)
  有效批次大小: 48
  使用LIBERO demos: 是

=== 训练步骤 1/10 ===
🔄 收集第 1/6 组...
  📋 使用LIBERO demo: 任务0
  📋 使用demo初始状态: 任务 LIBERO_SPATIAL_0
正在收集 8 个rollouts...
```

### **显存使用对比**
```
# 原来的方式（无demo，固定初始状态）
demo_batch_size: 1, rloo_batch_size: 2
总rollout数: 1 × 2 = 2个

# 新的方式（使用LIBERO demos）
demo_batch_size: 6, rloo_batch_size: 8  
总rollout数: 6 × 8 = 48个
```

## 🔧 配置选项详解

### **核心配置参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_libero_demos` | `true` | 是否启用LIBERO demo加载 |
| `libero_data_prefix` | 需要设置 | LIBERO数据集根目录路径 |
| `benchmark_name` | `"LIBERO_SPATIAL"` | LIBERO基准名称 |
| `demo_batch_size` | `6` | 每训练步采样的demo数量 |
| `task_names_to_use` | `["LIBERO_SPATIAL_0"]` | 要使用的具体任务 |

### **支持的LIBERO基准**
- `LIBERO_SPATIAL` (10个任务)
- `LIBERO_OBJECT` (10个任务)  
- `LIBERO_GOAL` (10个任务)
- `LIBERO_LONG` (10个任务)

### **数据集统计**
- 每个基准：10个任务
- 每个任务：50个专家演示
- 总demo数量：10 × 50 = 500个不同的初始状态

## 🚨 注意事项

### **1. 显存管理**
```yaml
# 推荐的显存友好配置
algo:
  demo_batch_size: 3      # 减少demo数量
  rloo_batch_size: 4      # 减少每个demo的rollout数
  # 总rollout数: 3 × 4 = 12个
```

### **2. 数据路径配置**
确保 `libero_data_prefix` 指向正确的数据集路径：
```bash
# 检查数据集是否存在
ls /path/to/libero/datasets/libero/LIBERO_SPATIAL/
# 应该看到 LIBERO_SPATIAL_0, LIBERO_SPATIAL_1, ... 等目录
```

### **3. 任务名称匹配**
确保 `task.task_names_to_use` 与 `benchmark_name` 匹配：
```yaml
# 正确的匹配
task:
  benchmark_name: "LIBERO_SPATIAL"
  task_names_to_use: ["LIBERO_SPATIAL_0"]

benchmark_name: "LIBERO_SPATIAL"  # 保持一致
```

## 🔄 回退方案

如果LIBERO demo加载失败，系统会自动回退到传统模式：

```yaml
# 禁用LIBERO demos
use_libero_demos: false

# 或者在运行时会自动回退
⚠️ LIBERO demo加载器创建失败: [错误信息]
  将使用传统的环境重置方式
```

## 🎯 与原版RIPT的对比

| 特性 | 原版RIPT | 我们的实现 | 状态 |
|------|----------|------------|------|
| **Demo数据加载** | ✅ 从HDF5文件加载 | ✅ 兼容的HDF5加载器 | 完成 |
| **初始状态多样性** | ✅ 500个不同初始状态 | ✅ 相同的500个初始状态 | 完成 |
| **批次处理** | ✅ DataLoader + batch | ✅ 兼容的DataLoader | 完成 |
| **任务描述** | ✅ 自动提取 | ✅ 自动提取 | 完成 |
| **动态batch大小** | ✅ 支持 | ✅ 支持 | 完成 |
| **分布式训练** | ✅ 支持 | 🔄 待测试 | 进行中 |

## 🚀 下一步计划

1. **性能测试**：对比使用/不使用LIBERO demos的训练效果
2. **分布式支持**：确保在多GPU环境下正常工作
3. **更多基准**：支持LIBERO_OBJECT、LIBERO_GOAL等其他基准
4. **优化加载**：实现更高效的数据预加载和缓存机制

这个集成让我们的训练系统与原版RIPT在数据使用方式上完全对等，为后续的性能对比和优化提供了坚实的基础。
