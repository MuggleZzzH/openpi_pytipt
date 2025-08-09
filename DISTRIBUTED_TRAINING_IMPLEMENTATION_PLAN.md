# 分布式训练核心逻辑实现详细规划

## 📋 目标逻辑描述

### 理想分布式训练流程
1. **启动时**：将9个任务平均分到各GPU
   - 2 GPU时：rank-0（GPU0）→ 任务[0,2,4,6,8]，rank-1（GPU1）→ 任务[1,3,5,7]
   - 当前调试YAML只有1个task(9)：rank-0→[9]，rank-1→[]

2. **训练循环**：进行`num_train_steps`次迭代（调试YAML=3次）
   - 每次迭代：当前进程只处理自己任务列表中的**一个任务**
   - 任务轮询：通过`task_cursor`循环切换任务

3. **数据采样**：在当前任务的全部init-state列表中
   - 用环形索引取`num_parallel_envs × rloo_batch_size`条状态
   - 确保不同环境获得不同初始状态

4. **并行执行**：启动`num_parallel_envs`个环境并行跑rollout
   - 收集到满足`data_batch_size`条轨迹后停止采样

5. **训练更新**：用收集的数据做反向传播 + DDP梯度同步

## 📊 当前实现状态深度分析

### ✅ 已正确实现的部分

#### 1. 任务分片机制 (`get_distributed_tasks()`)
**位置**: `10_train_with_distributed.py:381-406`

**实现状态**: ✅ **核心逻辑正确**
```python
# 正确的轮询分配算法
for task_i, task_name in enumerate(all_tasks):
    rank_to_tasks[task_i % self.world_size].append(task_name)
```

**验证结果**:
- 9个任务 → rank0: [0,2,4,6,8], rank1: [1,3,5,7]
- 1个任务 → rank0: [9], rank1: [] ✅ 符合预期

**问题**: 当前YAML`task_name: 9`只有1个任务，导致rank1为空

#### 2. 分布式训练主循环
**位置**: `10_train_with_distributed.py:804-876`

**实现状态**: ✅ **架构正确**
```python
for iteration in range(self.config['training']['num_train_steps']):
    # 数据收集 → 优势计算 → 梯度更新
```

**优点**: DDP梯度同步、分布式指标聚合已实现

#### 3. 并行环境支持
**位置**: `pi0/ript/env/pi0_libero_runner.py:181-220`

**实现状态**: ✅ **功能完整**
```python
def create_parallel_envs(self, env_name: str):
    parallel_env = SubprocVectorEnv([env_factory for _ in range(env_num)])
```

**优点**: 真正的进程级并行，支持串行回退

### ❌ 核心逻辑缺失分析

#### 1. **任务轮询机制** - 🚨 **完全缺失**

**期望逻辑**:
```python
# 每次迭代只处理一个任务
current_task = local_tasks[task_cursor % len(local_tasks)]
task_cursor += 1  # 下次迭代切换到下一个任务
```

**当前实现**: 
```python
# 错误：每次都传递整个任务列表
env_runner = LIBEROEnvRunner(task_names_to_use=local_tasks)  # 所有任务
```

**问题**:
- `LIBEROEnvRunner`没有`task_cursor`概念
- 每次迭代处理所有任务，而非单个任务轮询

#### 2. **环形索引状态采样** - 🚨 **完全缺失**

**期望逻辑**:
```python
# 环形索引选择初始状态
start_idx = (init_state_cursor) % len(all_init_states)
required_states = num_parallel_envs * rloo_batch_size
selected_states = []
for i in range(required_states):
    idx = (start_idx + i) % len(all_init_states)
    selected_states.append(all_init_states[idx])
init_state_cursor += required_states
```

**当前实现**:
```python
# 错误：随机采样
def sample_batch(self, batch_size=1):
    indices = np.random.choice(len(self.states), batch_size, replace=True)
```

**问题**:
- 完全随机采样，没有环形索引
- 可能重复选择相同状态
- 不同环境可能获得相同初始状态

#### 3. **精确数据收集控制** - 🚨 **逻辑错误**

**期望逻辑**:
```python
# 收集到data_batch_size条轨迹就停止
collected_episodes = []
while len(collected_episodes) < data_batch_size:
    # 采样轨迹...
    collected_episodes.extend(new_episodes)
```

**当前实现**:
```python
# 错误：按批次数收集，不精确
for i in range(config.target_batches_per_iteration):  # 固定批次数
    # 每批固定收集数据
```

**问题**:
- `target_batches_per_iteration`与`data_batch_size`没有精确对应
- 可能收集过多或过少数据

#### 4. **单进程内任务处理** - 🚨 **架构错误**

**期望逻辑**:
```python
# 每次迭代只处理当前任务
current_task = get_current_task()
rollouts = env_runner.run_single_task(current_task, init_states)
```

**当前实现**:
```python
# 错误：env_runner固定处理所有任务
env_runner = LIBEROEnvRunner(task_names_to_use=local_tasks)
# 无法动态切换单个任务
```

## 🎯 详细实现计划

### 阶段1：任务轮询机制重构

#### 1.1 修改`LIBEROEnvRunner`类
**文件**: `pi0/ript/env/pi0_libero_runner.py`

**需要添加**:
```python
class LIBEROEnvRunner:
    def __init__(self, ...):
        # 新增：任务轮询相关属性
        self.assigned_tasks = task_names_to_use or []  # 分配的任务列表
        self.task_cursor = 0                          # 任务轮询索引
        self.current_task = None                      # 当前处理的任务
        
    def set_current_task_by_cursor(self):
        """根据cursor设置当前任务"""
        if self.assigned_tasks:
            self.current_task = self.assigned_tasks[self.task_cursor % len(self.assigned_tasks)]
            return self.current_task
        return None
    
    def advance_task_cursor(self):
        """推进任务cursor到下一个任务"""
        self.task_cursor += 1
        return self.set_current_task_by_cursor()
```

#### 1.2 修改分布式训练循环
**文件**: `10_train_with_distributed.py`

**修改位置**: 第804-820行的训练循环

**新逻辑**:
```python
for iteration in range(self.config['training']['num_train_steps']):
    # 🔥 新增：设置当前迭代的任务
    current_task = env_runner.set_current_task_by_cursor()
    
    if current_task is None:  # 该rank没有分配任务
        if self.rank == 0:
            print(f"Rank {self.rank} 没有分配任务，跳过迭代 {iteration + 1}")
        continue
    
    if self.rank == 0:
        print(f"🎯 迭代 {iteration + 1}: 处理任务 {current_task}")
    
    # 数据收集（只处理当前任务）
    collected_batches = self.distributed_collect_batches(...)
    
    # 🔥 新增：迭代结束后推进cursor
    env_runner.advance_task_cursor()
```

### 阶段2：环形索引初始状态采样

#### 2.1 重构初始状态数据集
**文件**: `10_train_with_distributed.py:878-914`

**新实现**:
```python
class DistributedInitialStateDataset:
    def __init__(self, num_states=50, state_dim=8, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.init_state_cursor = 0  # 🔥 新增：环形索引
        self.states = self._generate_states(num_states, state_dim, rank)
    
    def sample_batch_circular(self, batch_size: int) -> np.ndarray:
        """环形索引采样，确保不重复且有序"""
        selected_states = []
        
        for i in range(batch_size):
            idx = (self.init_state_cursor + i) % len(self.states)
            selected_states.append(self.states[idx])
        
        # 更新cursor
        self.init_state_cursor = (self.init_state_cursor + batch_size) % len(self.states)
        
        return np.array(selected_states)
    
    def get_states_for_envs(self, num_parallel_envs: int, rloo_batch_size: int) -> np.ndarray:
        """为并行环境分配不同的初始状态"""
        total_states_needed = num_parallel_envs * rloo_batch_size
        return self.sample_batch_circular(total_states_needed)
```

#### 2.2 修改采样逻辑
**文件**: `pi0/ript/utils/enhanced_smart_sampling.py`

**修改方法**: `smart_sample_init_state()`

**新逻辑**:
```python
def smart_sample_init_state(self, init_dataset):
    # 🔥 使用环形索引而非随机采样
    init_states = init_dataset.get_states_for_envs(
        self.config.num_parallel_envs, 
        self.config.rloo_batch_size
    )
    state_hash = self.compute_state_hash(init_states)
    return init_states, state_hash
```

### 阶段3：精确数据收集控制

#### 3.1 修改数据收集逻辑
**文件**: `10_train_with_distributed.py:516-531`

**新实现**:
```python
def distributed_collect_batches(self, env_runner, reward_function, init_dataset, sampler, config, iteration):
    """精确控制数据收集数量"""
    if self.rank == 0:
        print(f"🎯 目标收集轨迹数: {config.data_batch_size}")
    
    collected_episodes = []
    attempt = 0
    max_attempts = config.max_sampling_attempts
    
    # 🔥 精确控制：收集到data_batch_size条就停止
    while len(collected_episodes) < config.data_batch_size and attempt < max_attempts:
        # 计算还需要多少轨迹
        remaining_needed = config.data_batch_size - len(collected_episodes)
        
        # 环形索引选择初始状态
        init_states, _ = sampler.smart_sample_init_state(init_dataset)
        
        try:
            rollout_generator = env_runner.run_policy_in_env(
                env_name=env_runner.current_task,  # 🔥 使用当前任务
                all_init_states=init_states
            )
            
            batch_episodes = []
            for success, total_reward, episode_data in rollout_generator:
                episode = {
                    'success': success,
                    'total_reward': total_reward,
                    **episode_data
                }
                batch_episodes.append(episode)
                
                # 🔥 精确控制：达到目标数量就停止
                if len(collected_episodes) + len(batch_episodes) >= config.data_batch_size:
                    break
            
            collected_episodes.extend(batch_episodes)
            
        except Exception as e:
            print(f"采样失败 (尝试 {attempt + 1}): {e}")
        
        attempt += 1
    
    # 🔥 裁剪到精确数量
    if len(collected_episodes) > config.data_batch_size:
        collected_episodes = collected_episodes[:config.data_batch_size]
    
    if self.rank == 0:
        print(f"✅ 实际收集轨迹数: {len(collected_episodes)}")
    
    return [collected_episodes] if collected_episodes else []
```

### 阶段4：配置参数修复

#### 4.1 创建多任务测试配置
**文件**: `pi0/ript/config/multi_task_distributed.yaml`

**关键配置**:
```yaml
task:
  benchmark_name: "libero_spatial"
  task_names_to_use: [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 🔥 9个任务
  num_parallel_envs: 2
  rollouts_per_env: 2
  max_episode_length: 50

algo:
  rloo_batch_size: 2
  data_batch_size: 4          # 🔥 精确控制目标
  num_epochs: 1
  gradient_accumulation_steps: 1

training:
  num_train_steps: 6          # 🔥 6步测试任务轮询
```

## 🧪 验证计划

### 测试场景1：双GPU多任务轮询
- **任务分配**: GPU0→[0,2,4,6,8], GPU1→[1,3,5,7]
- **迭代验证**: 6次迭代，每GPU轮询5个任务
- **预期输出**:
  ```
  迭代1: GPU0处理任务0, GPU1处理任务1
  迭代2: GPU0处理任务2, GPU1处理任务3
  ...
  迭代6: GPU0处理任务8, GPU1处理任务7
  ```

### 测试场景2：环形索引验证
- **初始状态**: 10个状态，需要4个状态(2envs×2batch)
- **预期选择**: 第1次[0,1,2,3]，第2次[4,5,6,7]，第3次[8,9,0,1]

### 测试场景3：精确数据收集
- **目标**: data_batch_size=4
- **验证**: 每次迭代精确收集4条轨迹，不多不少

## 📈 实现优先级

1. **HIGH**: 任务轮询机制 (阶段1) - 解决rank1空任务问题
2. **HIGH**: 环形索引采样 (阶段2) - 确保状态分配合理
3. **MEDIUM**: 精确数据收集 (阶段3) - 优化训练效率
4. **LOW**: 多任务配置 (阶段4) - 全面测试验证

## 📝 实现检查清单

- [ ] `LIBEROEnvRunner`添加`task_cursor`属性
- [ ] 实现`set_current_task_by_cursor()`方法
- [ ] 修改训练循环支持任务轮询
- [ ] 重构`DistributedInitialStateDataset`支持环形索引
- [ ] 修改`smart_sample_init_state()`使用环形采样
- [ ] 实现精确的`data_batch_size`控制逻辑
- [ ] 创建多任务测试配置文件
- [ ] 编写验证测试脚本

## 🎯 成功标准

实现完成后，分布式训练应该能够：
1. 正确将9个任务分配到2个GPU
2. 每次迭代只处理一个任务，按cursor轮询
3. 使用环形索引避免状态重复
4. 精确收集data_batch_size数量的轨迹
5. 支持任意数量的任务和GPU组合

这样就能实现你描述的完整分布式训练逻辑。