# 优先级修复清单

## 🚨 高优先级（影响训练正确性）

### 1. **PPO算法实现缺失** 
**影响**：当前使用的CFG加权损失与原版RIPT的PPO算法完全不同，会导致训练效果差异巨大
**修复位置**：`pi0/ript/algos/rl_optimizers/pi0_cfg_interface.py`
**需要添加**：
- `compute_act_logits()` 方法
- PPO损失计算（policy ratio, clip loss）
- 多个PPO epochs支持

### 2. **环境交互数据缺失**
**影响**：缺少context tokens和action indices会导致模型无法正确学习
**修复位置**：`pi0/ript/env/pi0_libero_runner.py`
**需要添加**：
- context token收集
- action indices记录
- policy inference steps跟踪

### 3. **RLOO优势计算验证**
**影响**：优势计算直接影响策略梯度方向
**当前状态**：已实现但需要验证与原版一致性
**检查点**：
- baseline计算是否正确（leave-one-out平均）
- 优势归一化处理

## ⚠️ 中优先级（影响性能和稳定性）

### 4. **任务嵌入集成**
**影响**：任务条件化学习效果
**需要添加**：CLIP/BERT任务嵌入支持
**参考**：原版 `get_task_embs()` 函数

### 5. **梯度累积实现验证**
**当前状态**：已实现但与原版有差异
**需要确认**：
- AMP scaler使用正确性
- 梯度同步时机

### 6. **数据加载器对齐**
**当前状态**：基本对齐但细节需验证
**检查点**：
- MuJoCo状态加载正确性
- Demo采样策略一致性

## ✅ 低优先级（已基本对齐）

### 7. **SO100处理器**
**状态**：已实现且逻辑正确

### 8. **Reward函数**
**状态**：已正确实现二值奖励

### 9. **统计跟踪器**
**状态**：已实现基本功能

## 🔧 快速修复指南

### 最小可行修复（2小时内完成）：
1. 在`PI0_CFG_Adapter`中添加简化的PPO支持
2. 修改`collect_rollouts_ript_vla_style`确保返回必要字段
3. 调整配置文件添加PPO参数

### 代码片段示例：
```python
# 1. 添加compute_act_logits（简化版）
def compute_act_logits(self, model, episodes, device, max_seq_len=None):
    # 使用当前的forward计算，提取action的log概率
    log_probs = []
    for ep in episodes:
        # 处理episode获取log prob
        log_prob = self._get_action_log_prob(model, ep)
        log_probs.append(log_prob)
    return torch.stack(log_probs), max_seq_len

# 2. 修改update函数支持PPO
def update_policy_ppo_style(policy, optimizer, episodes, advantages, old_logprobs):
    # 计算新log probs
    new_logprobs = compute_act_logits(policy, episodes)
    
    # PPO loss
    ratio = torch.exp(new_logprobs - old_logprobs)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 0.8, 1.2)
    loss = torch.max(pg_loss1, pg_loss2).mean()
    
    # 优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 📊 验证检查清单

运行训练后检查：
- [ ] Policy ratio是否在合理范围（0.8-1.2）
- [ ] Advantage mean是否接近0
- [ ] 成功率是否稳定上升
- [ ] Loss是否稳定下降
- [ ] Episode数据是否包含所需字段






































