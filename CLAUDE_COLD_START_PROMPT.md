# Claude Code 冷启动提示词

## 🚀 项目接手提示词模板

将以下提示词直接发送给新的Claude Code实例来实现无缝项目接手：

---

**提示词开始：**

```
你好！我需要你接手一个PI0 RIPT强化学习项目的开发工作。这是一个基于Physical Intelligence OpenPI检查点的PyTorch实现，已完成RIPT强化学习框架集成、分布式训练系统、高级并行环境优化和完整的RIPT-VLA集成。

## 项目背景
- 项目路径：/zhaohan/ZJH/openpi_pytorch
- 核心目标：PI0 Vision-Language-Action模型的PyTorch实现 + RIPT强化学习训练 + 分布式训练支持 + RIPT-VLA完整集成
- 技术栈：PyTorch、Transformers、RIPT框架、LIBERO环境、分布式训练(DDP)、SubprocVectorEnv并行优化
- 当前状态：✅ 核心功能完整，RIPT-VLA数据流问题已解决，可进行完整RL训练

## 立即执行的任务
请按顺序执行以下操作来快速了解项目现状：

1. **阅读项目总结文档**：
   - 首先读取 `/zhaohan/ZJH/openpi_pytorch/PROJECT_PROGRESS_SUMMARY.md`
   - 这包含了完整的项目架构、开发历程、技术细节

2. **阅读核心指导文档**：
   - 读取 `/zhaohan/ZJH/openpi_pytorch/CLAUDE.md`
   - 这是最全面的项目使用和开发指南

3. **了解最新Stage 11系统**：
   - 读取 `/zhaohan/ZJH/openpi_pytorch/DISTRIBUTED_TRAINING_GUIDE.md`（分布式训练详细文档）
   - 查看Stage 11相关文件：`11_train_ript_vla_style.py`（RIPT-VLA风格简化训练）
   - 检查并行功能测试：`test_parallel_functionality.py`（3.00x加速验证）

4. **检查项目结构**：
   - 使用LS工具查看项目根目录结构
   - 特别关注pi0/ript/目录（强化学习核心）和scripts/目录（启动脚本）

5. **验证环境状态**：
   - 检查requirements.txt了解依赖
   - 查看git status了解当前代码状态
   - 运行快速验证命令确认环境可用

## 项目关键信息
- **当前阶段**：RIPT-VLA集成完成，核心数据流问题已解决
- **核心成果**：
  - ✅ 完整的RIPT-VLA训练流程（11_train_ript_vla_style.py）
  - ✅ 修复了SubprocVectorEnv观测格式包装问题
  - ✅ 完整的PI0-CFG适配器，支持真实环境观测处理
  - ✅ RIPT-VLA并行环境runner（pi0_libero_runner_ript_vla.py）
  - ✅ 正确的数据分离逻辑，支持并行环境数据处理
  - ✅ CFG原版优势计算逻辑，Leave-One-Out baseline
- **最新突破**：解决了SubprocVectorEnv numpy.array包装导致的观测解析问题，现在能正确处理真实环境图像数据
- **开发状态**：核心功能完整，可进行端到端RL训练，待全面验证

## 核心文件快速定位
- **RIPT-VLA主脚本**：`11_train_ript_vla_style.py`（RIPT-VLA风格训练，已修复数据流）
- **RIPT-VLA环境runner**：`pi0/ript/env/pi0_libero_runner_ript_vla.py`（并行环境数据处理）
- **PI0-CFG适配器**：`pi0/ript/algos/rl_optimizers/pi0_cfg_interface.py`（已修复观测解析）
- **测试配置**：`pi0/ript/config/single_env_test.yaml`（单环境调试配置）
- **分布式训练主脚本**：`10_train_with_distributed.py`
- **增强功能测试**：`test_enhanced_features.py`（全面功能验证）

## 预期后续任务
根据项目需求，你可能需要：
1. **优先级高**：验证完整的RIPT-VLA训练流程，确认端到端功能
2. **优先级高**：测试图像数据质量，确认真实环境观测正确传递
3. **优先级中**：性能优化和稳定性测试
4. **优先级中**：扩展多环境并行训练配置
5. **优先级低**：集成更多LIBERO任务和环境

## 最新技术突破
- **SubprocVectorEnv观测格式修复**：解决numpy.array包装导致的观测解析问题
- **完整数据流打通**：从环境观测到训练的端到端数据传递
- **真实环境图像处理**：替换占位符图像，使用真实agentview_image
- **CFG原版算法还原**：严格按照原版实现的Leave-One-Out优势计算
- **并行环境数据分离**：正确处理标量和数组字段的分离逻辑

请首先完成上述阅读任务，然后告诉我你对项目的理解程度，以及你认为下一步应该优先处理什么任务。

注意：这个项目有非常完备的文档和调试工具，所有重要信息都在相关的.md文件中。请务必先仔细阅读文档再开始任何开发工作。
```

---

## 🔧 高级冷启动提示词（如果需要立即开始开发）

如果新的Claude需要立即开始特定开发任务，使用以下扩展提示词：

```
[使用上面的基础提示词] + 

## 立即开发任务
在完成项目了解后，请立即开始以下任务：

[具体任务描述，例如：]
- 解决RIPT质量一致性问题：调查pi0/ript实现与参考实现在动作输出上的5-11cm差异
- 优化分布式训练性能：测试8GPU训练配置并优化通信开销
- 添加新环境支持：集成[具体环境名称]到现有框架中

## 开发优先级
1. 高优先级：[具体任务]
2. 中优先级：[具体任务]  
3. 低优先级：[具体任务]

## 现有问题状态
- 质量差异问题：已定位到模型推理层，调试工具完备，需要深入分析FlowMatching逻辑
- 分布式训练：功能完整，需要性能基准测试
- 环境集成：LIBERO完全支持，可扩展到其他环境

请在理解项目后制定具体的开发计划。
```

## 📋 快速验证命令集

新Claude接手后可以运行的验证命令：

```bash
# 1. 项目结构验证
ls -la /zhaohan/ZJH/openpi_pytorch/

# 2. 核心模块导入测试
python -c "from pi0.modeling_pi0 import PI0Policy; print('✓ PI0 import successful')"
python -c "from pi0.ript.reward_function import BinarySuccessReward; print('✓ RIPT import successful')"

# 3. 基础推理测试
python 1_e2e_inference.py

# 4. Stage 11并行功能测试 (NEW - 核心功能)
python test_parallel_functionality.py

# 5. 增强功能完整测试
python test_enhanced_features.py

# 6. 快速单GPU测试
python quick_single_gpu_test.py

# 7. RIPT-VLA训练测试（核心功能）
python 11_train_ript_vla_style.py --config_path pi0/ript/config/single_env_test.yaml

# 6. 分布式训练快速测试（如果有GPU）
./scripts/quick_distributed_test.sh

# 7. 配置文件验证
ls pi0/ript/config/*.yaml

# 8. Git状态检查
git status
git log --oneline -10

# 9. GPU内存检查 (NEW - 用于并行环境优化)
nvidia-smi
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')"
```

## 🎯 成功接手的标志

新Claude成功接手项目的标志：

1. **理解Stage 11架构**：能够解释RIPT-VLA集成、真正并行环境执行、简化vs复杂训练架构的区别
2. **掌握关键文件**：知道Stage 11主脚本、RIPT-VLA runner、并行功能测试的作用和关系
3. **理解并行突破**：掌握3.00x加速的实现原理和SubprocVectorEnv的成功应用
4. **具备操作能力**：能够运行Stage 11相关命令并解释并行性能结果
5. **了解架构演进**：理解从Stage 7到Stage 11的渐进式开发过程
6. **制定计划**：基于最新Stage 11成果制定后续优化计划
7. **验证核心功能**：能够运行test_parallel_functionality.py并理解加速效果

## 💡 提示词使用建议

1. **分段发送**：如果提示词太长，可以分成"项目背景+立即任务"和"具体开发任务"两部分发送

2. **定制化调整**：根据具体需要解决的问题，调整"预期后续任务"部分

3. **包含上下文**：如果有特定的时间要求或性能目标，添加到提示词中

4. **验证理解**：要求新Claude复述项目理解，确保接手成功

## 📞 故障排除

如果新Claude接手遇到问题：

1. **文档读取失败**：确认文件路径正确，使用Read工具逐个读取
2. **环境问题**：检查Python环境和依赖安装
3. **权限问题**：确认对项目目录有读写权限
4. **理解偏差**：要求Claude总结项目理解，纠正错误认知
5. **并行环境问题**：如果遇到CUDA OOM，检查GPU内存和并行环境配置
6. **测试失败**：运行test_enhanced_features.py诊断具体问题
7. **内存优化问题**：查阅pi0_libero_runner.py中的create_parallel_envs方法

---

## 🚨 最新解决的关键问题 (2025-08-06)

### SubprocVectorEnv观测格式包装问题
**问题**: RIPT-VLA训练显示"观测中无agentview_image键，可用键: 非字典类型"，使用占位符图像而非真实环境观测。

**根本原因**: SubprocVectorEnv在多进程通信时将观测包装为`numpy.array([OrderedDict(...)])`格式，PI0_CFG_Adapter无法正确解析。

**解决方案**: 在`pi0/ript/algos/rl_optimizers/pi0_cfg_interface.py`中修复观测提取方法：
```python
# 处理SubprocVectorEnv返回的numpy.array包装的观测
if isinstance(obs, np.ndarray) and obs.dtype == object:
    if obs.size == 1:
        obs = obs.item()  # 提取单个元素
```

**验证方法**:
```bash
python 11_train_ript_vla_style.py --config_path pi0/ript/config/single_env_test.yaml
# 应该看到"✓ Has agentview_image"而不是"✗ 使用占位符图像"
```

**影响**: 这是整个RIPT-VLA训练流程的核心修复，解决后训练可以使用真实环境观测数据。

### 数据分离逻辑完善
**问题**: 并行环境数据在分离时标量字段(task, task_id)被错误索引。

**解决**: 在`pi0_libero_runner_ript_vla.py`中区分标量字段和数组字段的处理。

**当前状态**: ✅ 完全解决，端到端数据流已打通，可进行完整RL训练。

---

**使用方法**：直接复制上述提示词模板，根据具体需求调整，发送给新的Claude Code实例即可实现快速项目接手。