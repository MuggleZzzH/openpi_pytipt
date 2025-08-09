### Stage 12 增量完善清单（可验证版）

本文档列出从当前单任务“小人物”方案，逐步演进到可跑“libero-10”多任务的最小可用迭代步骤。每一步都是“可独立验证”的优化点：完成后即可立即运行脚本观察结果与产物。

请仅按顺序实施，确保每步通过后再进行下一步。

---

## 0. 概述与目标

- **目标大任务**: `libero_10`（多任务）
- **当前瓶颈**:
  - 批次级动态采样过于激进，导致训练步数为 0
  - 未按样本级（rloo 组）过滤且无历史统计/跳过机制
  - 未采用“demo 批 × rloo 组”双层批次组织
  - 缺乏任务感知的初始状态数据源与多任务 env 复用
- **原则**: 先“能跑通”，再“对齐 ript 的结构”。优先打通核心数据流与指标产出。

---

## 1) 禁用动态采样（立即产出训练步与指标）

- **目标**: 规避“整批全1/全0被丢弃 → total_steps=0”，先让指标产出。
- **变更范围**: `openpi_pytorch/pi0/ript/config/stage11_ript_vla.yaml`
- **配置改动**:
  - `algo.enable_dynamic_sampling: false`
  - 可选提速：`features.save_video: false`
- **验证**:
  - 命令: `python 11_train_ript_vla_style.py --config_path pi0/ript/config/stage11_ript_vla.yaml`
  - 观察终端“✓ 步骤 1 完成 …”、`training_metrics` 非空；产物 `final_training_results.json` 中 `total_steps > 0`
- **预期**: 至少 1 个训练步、指标非空

---

## 2) 接入本仓库 RolloutGenerator（样本级收集、可统计）

- **目标**: 用 `pi0/ript/algos/rl_optimizers/rollout_generator.py` 替换当前“批次级过滤”，为后续动态采样/统计/早停打基础。
- **变更范围**: `openpi_pytorch/11_train_ript_vla_style.py`
- **实现要点**:
  - 创建“初始状态数据集 + DataLoader”（shape ~ `[N, 8]`，`batch_size=task.num_parallel_envs`）
  - 实例化 RolloutGenerator（关键参数）：
    - `env_runner`、`rollouts_per_env=algo.rloo_batch_size`、`num_envs=task.num_parallel_envs`、`max_steps=task.max_episode_length`
    - `agent_gpus=[0]`（或当前设备）
    - `init_state_dataloader`、`init_state_dataset`
    - 透传：`enable_dynamic_sampling`、`rollout_skip_threshold`、`enable_rollout_stats_tracking`、`rollout_stats_path`、`use_val_init`
  - 用 `episodes = rollout_generator.generate_rollouts()` 替换原 `collect_rollouts... + _dynamic_filter_rollouts(...)`
  - 训练结束前 `rollout_generator.cleanup()`
- **验证**: 运行同上；日志出现 RolloutGenerator 收集与统计输出；动态采样关闭时稳定产出 episodes 与训练步
- **预期**: 收集与训练流程稳定，`training_metrics` 增长

---

## 3) 打开统计追踪（不启用动态采样）

- **目标**: 开始记录“每个初始状态”的成功历史，为后续启用动态采样做准备。
- **配置改动**:
  - `algo.enable_rollout_stats_tracking: true`
  - `algo.rollout_stats_path: ./output/stage11_ript_vla/rollout_stats.json`
  - 保持：`algo.enable_dynamic_sampling: false`
- **验证**: 连续跑 2-3 次，`rollout_stats.json` 持续增长，按 init_state hash 累积成功历史
- **预期**: 统计文件持续更新

---

## 4) 启用“样本级”动态采样（避免“整批丢弃”）

- **目标**: 按“样本内 rloo 组”粒度过滤；结合历史统计跳过“已熟练”初始状态。
- **配置改动**:
  - `algo.enable_dynamic_sampling: true`
  - `algo.rollout_skip_threshold: 3`（或更大）
- **验证**:
  - 日志出现“由于持续成功而跳过初始状态 …”“由于成功率一致 … 而丢弃来自 … 的轨迹”
  - 其他样本仍能产生 episodes 与训练步；`training_metrics` 继续增长
- **预期**: 不再出现“整批被丢弃 → 0 步”

---

## 5) 切换到 `libero_10`（多任务）

- **目标**: 改为大任务，初步验证多任务配置下的跑通性。
- **配置改动**:
  - `task.benchmark_name: libero_10`
  - 删除/清空 `task.task_names_to_use`（使用 benchmark 全任务）
  - 建议先降负载：`task.num_parallel_envs: 2`，`algo.rloo_batch_size: 4`，`features.save_video: false`
- **验证**: 日志可见不同任务名的运行记录（若 Runner 输出）；指标文件训练步与奖励/成功率仍增长
- **预期**: 多任务下训练稳定

---

## 6) 任务感知的初始状态（基础版）

- **目标**: 让 RolloutGenerator 能按样本的 `task_id` 选择对应任务及其初始状态。
- **变更范围**: `11_train_ript_vla_style.py`
- **实现要点**:
  - 初始状态 DataLoader 返回结构：`{"task_id": int, "init_state": np.ndarray(8,)}`（或 batch 化）
  - 生成轨迹时：逐样本取 `task_id→task_name`，传入对应 `created_env`（见下一步）
- **验证**: `libero_10` 下运行，观察不同 `task_id` 被轮询采样，episodes 中 `task/描述` 对应一致
- **预期**: 样本与任务一一对应，训练正常

---

## 7) 多任务 env 复用（性能优化，可开关）

- **目标**: 跨训练步复用每个任务的并行 env，降低重复创建开销。
- **实现要点**:
  - RolloutGenerator 内维护 `created_envs: Dict[int, Tuple]` 缓存
  - 首次 `task_id`：`env_runner.create_parallel_envs(task_name)` 获得 `(env, env_id, env_num)` 并缓存
  - 后续 `run_policy_in_env(..., created_env=created_envs[task_id])`
  - `cleanup()` 里统一关闭缓存 env
- **验证**: 跑 2-3 步，日志不再每步“创建 4 个独立并行环境…”，总时延下降
- **预期**: 收集/训练速度提升

---

## 8) RLOO 优势按“样本内组”计算（替代全批均值）

- **目标**: 用真正的 RLOO（Leave-One-Out）基线，按“每个样本的 rloo 组”计算优势。
- **实现要点**:
  - 训练循环中，按“每个样本的 rloo 组”聚合奖励，计算 `adv_i = r_i - mean(r_others)`，再拼接成一维优势张量传给 `PI0_CFG_Adapter.compute_weighted_loss`
- **验证**: 与“全批均值基线”对比损失与收敛曲线（同样配置）
- **预期**: 优势更稳定，损失收敛更平滑

---

## 9) 周期性评估与指标落盘（多任务聚合）

- **目标**: 每 `logging.log_freq` 步对当前策略进行 rollout 评估，聚合多任务成功率/奖励并写入 JSON。
- **实现要点**:
  - 训练循环中周期调用 `env_runner.run(...)`（不开视频），聚合指标，追加到 `output_dir/metrics.json`
- **验证**: `metrics.json` 持续追加记录，含 step、overall_success_rate、per_task 成功率
- **预期**: 观察多任务上的进展与波动

---

## 10) 配置统一与“libero-10”长跑预设

- **目标**: 避免同名 YAML 混淆，准备“libero-10 可长跑”的稳健配置。
- **操作**:
  - 仅保留并使用：`openpi_pytorch/pi0/ript/config/stage11_ript_vla.yaml`
  - 新增：`openpi_pytorch/pi0/ript/config/stage12_libero10.yaml`（推荐值）：
    - `task.benchmark_name: libero_10`
    - `task.num_parallel_envs: 2`
    - `algo.rloo_batch_size: 4`
    - `algo.enable_dynamic_sampling: true`
    - `algo.enable_rollout_stats_tracking: true`
    - `algo.rollout_stats_path: ./output/stage12/rollout_stats.json`
    - `features.save_video: false`
    - `training.num_train_steps: 1000`
- **验证**: 使用新配置长跑，`metrics.json` 曲线与 `rollout_stats.json` 体量持续增长

---

## 附：初始状态来源说明

- **固定集合**: `benchmark.get_task_init_states(task_id, split='train'|'test')`
- **数据集侧**: DataLoader 提供 `init_state`（可自定义多样性/采样策略）
- **随机初始化**: Runner 可选 `random_init=True`，不建议作为基准对齐默认路径

---

## 常见建议

- 先关视频：`features.save_video: false`
- 逐步提升 `num_parallel_envs` 与 `rloo_batch_size`
- 关注动态采样/统计日志：“跳过初始状态/成功率一致丢弃”等

---

## 里程碑

- 完成 1–5 步：可稳定跑通“libero-10”基础版
- 完成 6–9 步：多任务感知 + 环境复用 + 组内 RLOO + 周期评估
- 完成 10 步：配置统一与长跑稳态 