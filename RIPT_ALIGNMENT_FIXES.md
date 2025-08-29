# RIPT对齐修复方案

## 一、紧急修复项（影响训练正确性）

### 1. PPO实现修复
**问题**：当前使用CFG风格的加权L2损失，而非标准PPO损失
**修复方案**：
```python
# 在pi0_cfg_interface.py中添加PPO支持
def compute_ppo_loss(self, episodes, advantages, old_logprobs):
    # 计算新的action log probabilities
    new_logprobs = self.compute_act_logits(episodes)
    
    # 计算policy ratio
    ratio = torch.exp(new_logprobs - old_logprobs)
    
    # PPO clip loss
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1-clip_range, 1+clip_range)
    pg_loss = torch.max(pg_losses1, pg_losses2).mean()
    
    return pg_loss
```

### 2. 环境Runner修复
**问题**：缺少context token缓存和policy inference步数跟踪
**修复方案**：
```python
# 创建新的pi0_libero_runner_ript_aligned.py
class PI0LiberoRunnerRIPTAligned(LIBEROEnvRunner):
    def run_episode(self, env, env_name, policy, init_states_, env_num, render=False):
        # 添加context缓存
        all_context_tokens = []
        all_action_indices = []
        policy_inference_steps = 0
        
        # 在每步中收集context
        action, context_tokens, action_indices = policy(obs, task_id, task_emb)
        if context_tokens is not None:
            all_context_tokens.append(context_tokens)
            all_action_indices.append(action_indices)
            policy_inference_steps += 1
        
        # 返回时包含这些信息
        episode['context_tokens'] = np.stack(all_context_tokens)
        episode['action_indices'] = np.stack(all_action_indices)
        episode['policy_inference_steps'] = policy_inference_steps
```

### 3. 模型接口完善
**问题**：缺少compute_act_logits等关键方法
**修复方案**：
```python
# 在PI0_CFG_Adapter中添加
def compute_act_logits(self, model, episodes, device, max_seq_len=None):
    """计算动作的log probabilities（RIPT兼容）"""
    batch_size = len(episodes)
    all_logprobs = []
    
    for episode in episodes:
        # 处理episode数据
        obs = self._prepare_observation(episode)
        actions = episode['actions']
        
        # 通过模型前向传播获取action分布
        with torch.no_grad():
            outputs = model(obs)
            action_dist = outputs['action_distribution']
            
        # 计算log prob
        log_prob = action_dist.log_prob(actions)
        all_logprobs.append(log_prob)
    
    return torch.stack(all_logprobs), max_seq_len
```

## 二、功能增强项（提升性能）

### 1. 任务嵌入集成
```python
# 添加任务嵌入支持
from transformers import AutoModel, AutoTokenizer

def get_task_embs(task_embedding_format, descriptions):
    if task_embedding_format == "clip":
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        
        tokens = tokenizer(descriptions, padding=True, return_tensors="pt")
        with torch.no_grad():
            task_embs = model.get_text_features(**tokens)
            
    return task_embs
```

### 2. 分布式支持（可选，单GPU可跳过）
```python
# 如需要分布式，添加以下初始化
import torch.distributed as dist

def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        return rank, world_size
    return 0, 1
```

## 三、配置文件修正

### stage11_unified_pool.yaml需要添加：
```yaml
# PPO相关参数
algo:
  # 添加PPO参数
  num_ppo_epochs: 5  # 原版默认5-20
  ppo_clip_range: 0.2
  ppo_batch_size: 4
  use_token_level_loss_avg: true
  
  # 修正RLOO配置
  rloo_batch_size: 8  # 原版默认8
```

## 四、测试验证步骤

1. **验证PPO损失计算**：
   - 打印policy ratio的分布，应在0.8-1.2范围内
   - 监控clip fraction，应在0.1-0.3之间

2. **验证数据流**：
   - 确保episode包含context_tokens和action_indices
   - 检查advantage计算是否正确（mean应接近0）

3. **验证收敛性**：
   - 监控成功率是否稳定上升
   - 检查loss是否稳定下降

## 五、最小可行修复

如果时间紧迫，至少需要：
1. 修复PPO损失计算（最关键）
2. 确保episode数据包含所需字段
3. 调整超参数匹配原版






































