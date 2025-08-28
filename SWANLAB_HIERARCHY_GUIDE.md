# 🎯 SwanLab分层实验管理指南

## 🏗️ **分层命名结构**

为了更好地组织实验，我们采用**两层命名结构**：

### **第一层：项目分类（大类）**
```
ript-eval     # 评估实验项目  
ript-train    # 训练实验项目
ript-debug    # 调试实验项目
ript-ablation # 消融实验项目
```

### **第二层：运行名称（小类）** 
格式：`{基准类型}_{具体实验名}_{时间戳}`
```
spatial_eval_spatial_20241201_143022
goal_train_goal_20241201_143155  
object_debug_object_20241201_143300
```

---

## 🚀 **使用方式**

### **快速评估实验**
```bash
# LIBERO Spatial 评估
./scripts/run_eval_spatial.sh
# → SwanLab项目: ript-eval
# → SwanLab运行: spatial_eval_spatial_20241201_143022

# LIBERO Goal 评估  
./scripts/run_eval_goal.sh
# → SwanLab项目: ript-eval
# → SwanLab运行: goal_eval_goal_20241201_143155
```

### **训练实验**
```bash
# LIBERO Spatial 训练
./scripts/run_train_spatial.sh
# → SwanLab项目: ript-train
# → SwanLab运行: spatial_train_spatial_20241201_143300
```

### **自定义分层实验**
```bash
# 修复版脚本（支持完全自定义）
./scripts/run_ript_training_fixed.sh
# → 可在脚本内修改PROJECT_CATEGORY和BENCHMARK_TYPE
```

---

## 📊 **SwanLab界面效果**

在SwanLab界面中，您将看到：

### **项目列表**
```
📁 ript-eval     (所有评估实验)
   └── spatial_eval_spatial_xxx
   └── goal_eval_goal_xxx
   └── object_eval_object_xxx

📁 ript-train    (所有训练实验)  
   └── spatial_train_spatial_xxx
   └── goal_train_goal_xxx

📁 ript-debug    (所有调试实验)
   └── spatial_debug_xxx
```

### **运行对比**
- **同项目内对比**：比较不同基准测试的性能
- **跨项目对比**：比较训练 vs 评估结果
- **时间序列追踪**：通过时间戳追踪实验演进

---

## ⚙️ **配置文件说明**

### **YAML配置** (`pi0/ript/config/stage11_unified_pool.yaml`)
```yaml
logging:
  use_swanlab: true
  swanlab_project: "ript-eval"                           # 大类项目名
  swanlab_run_name: "${task.benchmark_name}_${exp_name}" # 小类运行名
  swanlab_tags: ["${task.benchmark_name}", "stage11"]    # 动态标签
```

### **脚本变量**
```bash
PROJECT_CATEGORY="ript-eval"          # 修改这里改变大类
BENCHMARK_TYPE="libero_spatial"       # 修改这里改变基准类型
EXPERIMENT_BASE="eval_spatial"        # 修改这里改变实验名
```

---

## 🎯 **最佳实践**

1. **评估实验**：使用 `ript-eval` 项目
2. **训练实验**：使用 `ript-train` 项目  
3. **调试实验**：使用 `ript-debug` 项目
4. **消融实验**：使用 `ript-ablation` 项目

### **命名建议**
- **大类名称**：简洁明了，描述实验目的
- **小类名称**：包含基准类型和具体内容
- **时间戳**：自动添加，确保唯一性

### **标签使用**
- 基准类型标签：`spatial`, `goal`, `object` 
- 阶段标签：`stage11`, `stage12`
- 功能标签：`unified-pool`, `cfg-enabled`

---

## 🔍 **故障排除**

### **SwanLab不同步**
1. 确认已激活 `mix` 环境
2. 检查 `logging.use_swanlab: true`
3. 验证网络连接

### **名称不匹配**
1. 检查YAML中的变量引用
2. 确认脚本中的覆盖参数
3. 查看控制台输出的实际名称

### **项目找不到**
1. 确认 `swanlab_project` 设置正确
2. 检查SwanLab账号权限
3. 尝试手动创建项目

---

## 📝 **示例输出**

运行脚本时，您将看到：
```
🚀 启动RIPT-VLA分层实验
📊 SwanLab项目（大类）: ript-eval
🏷️  SwanLab运行（小类）: spatial_eval_spatial_20241201_143022
🎯 基准测试类型: libero_spatial
📝 实验基础名: eval_spatial_20241201_143022
----------------------------------------
... 训练/评估过程 ...
----------------------------------------
✅ 训练脚本执行完成
📊 SwanLab查看路径:
   项目: ript-eval
   运行: spatial_eval_spatial_20241201_143022
```

这样您就可以在SwanLab中轻松找到和管理所有实验了！ 🎉


