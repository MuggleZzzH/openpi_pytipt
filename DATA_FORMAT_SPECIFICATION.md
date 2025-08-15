# OpenPI-RIPT 数据格式规范

## 🎯 **完整数据格式说明**

### **第一层：OpenPI标准格式**

```python
# 标准OpenPI样本格式（基于so100_train.py）
openpi_sample = {
    "image": {
        "base_0_rgb": torch.Tensor,        # [3, 224, 224] uint8, 主视角RGB图像
        "left_wrist_0_rgb": torch.Tensor,  # [3, 224, 224] uint8, 手腕视角RGB图像
    },
    "state": torch.Tensor,                 # [state_dim] float32, 机器人状态
    "action": torch.Tensor,                # [50, 7] float32, 50步动作序列chunk
    "action_is_pad": torch.Tensor,         # [50] bool, 动作填充掩码
    "prompt": str,                         # 任务描述字符串
}
```

**关键特征:**
- **Action Chunking**: 50步未来动作序列 (1.67秒 @ 30fps)
- **多视角图像**: 主视角 + 手腕视角
- **归一化状态**: meanstd归一化的机器人状态
- **相对动作**: `action = target_action - current_state`
- **填充掩码**: 标记有效动作vs填充动作

### **第二层：RIPT扩展格式**

```python
# RIPT扩展样本格式
ript_extended_sample = {
    # === OpenPI标准字段 ===
    "image": {
        "base_0_rgb": torch.Tensor,        # [3, 224, 224] uint8
        "left_wrist_0_rgb": torch.Tensor,  # [3, 224, 224] uint8
    },
    "state": torch.Tensor,                 # [target_state_dim] float32, 维度对齐后
    "action": torch.Tensor,                # [50, 7] float32
    "action_is_pad": torch.Tensor,         # [50] bool
    "prompt": str,                         # 任务描述
    
    # === RIPT扩展字段 ===  
    "advantages": torch.Tensor,            # [50] float32, RLOO优势值
    "init_hash": str,                      # 初始状态哈希（用于状态跳过）
    
    # === 可选统计字段 ===
    "rollout_success": bool,               # Rollout是否成功
    "rollout_reward": float,               # Rollout总奖励
    "episode_length": int,                 # Episode长度
}
```

### **第三层：训练批次格式**

```python
# GPU训练批次格式
training_batch = {
    "image": {
        "base_0_rgb": torch.Tensor,        # [batch_size, 3, 224, 224] uint8, device=cuda
        "left_wrist_0_rgb": torch.Tensor,  # [batch_size, 3, 224, 224] uint8, device=cuda  
    },
    "state": torch.Tensor,                 # [batch_size, target_state_dim] float32, device=cuda
    "action": torch.Tensor,                # [batch_size, 50, 7] float32, device=cuda
    "action_is_pad": torch.Tensor,         # [batch_size, 50] bool, device=cuda
    "prompt": List[str],                   # [batch_size] 任务描述列表
    "advantages": torch.Tensor,            # [batch_size] float32, 处理后的优势值, device=cuda
}
```

### **第四层：CFG模型输入格式**

```python
# CFG适配器期望的Episode格式
episode_format = {
    'observations': [{
        'agentview_image': np.ndarray,           # [224, 224, 3] uint8, HWC格式  
        'robot0_eye_in_hand_image': np.ndarray,  # [224, 224, 3] uint8, HWC格式
        'robot0_eef_pos': np.ndarray,            # [3] float32, 末端位置
        'robot0_eef_quat': List[float],          # [4] 四元数 [0,0,0,1]
        'robot0_gripper_qpos': np.ndarray,       # [2] float32, 夹爪位置
    }],
    'actions': [np.ndarray],                     # [50, 7] float32, 动作序列
    'task': str,                                 # 任务描述
    'success': bool,                             # Episode成功标志  
    'total_reward': float,                       # 总奖励（来自优势值）
}
```

## 🔄 **数据转换流程**

### **1. 原始数据加载** (`LeRobotDataset`)

```python
# delta_timestamps 定义动作chunk
delta_timestamps = {
    "observation.images.base": [0],          # 当前时刻图像
    "observation.images.wrist": [0],         # 当前时刻图像  
    "observation.state": [0],                # 当前时刻状态
    "action": [i / 30 for i in range(50)],  # 未来50步动作 (1.67秒)
}

raw_item = {
    "observation.images.base": tensor([3, 224, 224]),    # 0-1归一化
    "observation.images.wrist": tensor([3, 224, 224]),   # 0-1归一化
    "observation.state": tensor([state_dim]),            # 原始状态
    "action": tensor([50, action_dim]),                  # 绝对动作
    "action_is_pad": tensor([50], dtype=bool),           # 填充掩码
    "task": "grab screwdriver",                          # 任务描述
}
```

### **2. OpenPI标准化处理**

```python
# 图像处理：0-1 → 0-255
base_image = (normalized_item["observation.images.base"] * 255).to(torch.uint8)
wrist_image = (normalized_item["observation.images.wrist"] * 255).to(torch.uint8)

# 相对动作计算  
relative_action = item["action"] - item["observation.state"]  # 重要！

# 状态归一化
normalized_state = normalizer.normalize(item["observation.state"])

# 标准OpenPI格式
openpi_sample = {
    "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": wrist_image},
    "state": normalized_state[0],          # 取第一个时刻
    "action": relative_action,             # 相对动作
    "action_is_pad": normalized_item["action_is_pad"],
    "prompt": item["task"],
}
```

### **3. RIPT字段扩展**

```python
# 优势值计算（RLOO）
rewards = [sample.rollout_reward for sample in samples]
advantages = compute_rloo_advantages(rewards)  # [batch_size] 

# 优势值处理（数值稳定）
processed_advantages = advantage_processor.process_advantages(
    advantages,
    normalization="standard",     # (adv - mean) / std
    clipping="symmetric",         # [-clip_value, clip_value] 
    clip_value=3.0,
    negative_handling="softplus"  # log(1 + exp(x))
)

# 扩展RIPT字段
for i, sample in enumerate(openpi_samples):
    sample["advantages"] = torch.full_like(
        sample["action"][:, 0], 
        processed_advantages[i].item()
    )  # [50] 每个动作都有相同优势值
    sample["init_hash"] = compute_hash(initial_state)
```

### **4. 批次化和设备转移**

```python
# 批量组装
training_batch = {
    "image": {
        "base_0_rgb": torch.stack([s["image"]["base_0_rgb"] for s in samples]).to(device),
        "left_wrist_0_rgb": torch.stack([s["image"]["left_wrist_0_rgb"] for s in samples]).to(device)
    },
    "state": torch.stack([s["state"] for s in samples]).to(device),
    "action": torch.stack([s["action"] for s in samples]).to(device), 
    "action_is_pad": torch.stack([s["action_is_pad"] for s in samples]).to(device),
    "prompt": [s["prompt"] for s in samples],
    "advantages": processed_advantages.to(device)  # [batch_size]
}
```

### **5. CFG双分支处理**

```python
# CFG适配器中的安全拷贝和双分支处理
batch_positive = safe_copy_batch(batch, "positive")  # 深拷贝
batch_positive["is_positive"] = torch.ones(batch_size, device=device, dtype=torch.long)

batch_uncond = safe_copy_batch(batch, "negative")    # 独立深拷贝  
batch_uncond["is_positive"] = torch.zeros(batch_size, device=device, dtype=torch.long)

# 转换为Episode格式
episodes_positive = batch_to_episodes(batch_positive)
episodes_uncond = batch_to_episodes(batch_uncond)

# 模型前向传播
out_positive = policy.forward(episodes_positive)  # 条件分支
out_uncond = policy.forward(episodes_uncond)      # 无条件分支

# CFG损失计算
cfg_loss = compute_cfg_loss(out_positive, out_uncond, advantages)
```

## 📊 **数据维度总结**

| 数据类型 | 维度 | 数据类型 | 设备 | 说明 |
|----------|------|----------|------|------|
| **base_0_rgb** | `[B, 3, 224, 224]` | `uint8` | GPU | 主视角RGB图像 |
| **left_wrist_0_rgb** | `[B, 3, 224, 224]` | `uint8` | GPU | 手腕视角RGB图像 |  
| **state** | `[B, target_state_dim]` | `float32` | GPU | 归一化机器人状态 |
| **action** | `[B, 50, 7]` | `float32` | GPU | 50步相对动作chunk |
| **action_is_pad** | `[B, 50]` | `bool` | GPU | 动作填充掩码 |
| **prompt** | `[B]` | `str` | CPU | 任务描述字符串 |
| **advantages** | `[B]` | `float32` | GPU | 处理后的优势值 |

**B = batch_size** (通常是4-16)

## 🔧 **关键设计决策**

### **1. Action Chunking (50步)**
- **原因**: 减少重规划频率，提升执行稳定性
- **时长**: 1.67秒 @ 30fps
- **优势**: 开环执行，避免累积误差

### **2. 相对动作**  
```python
action = target_action - current_state  # 相对动作，更稳定
```

### **3. 状态维度适配**
- **Panda**: 14维 → 保持不变
- **UR5e**: 6维 → 零填充到14维  
- **其他**: 任意维 → 截断/填充到目标维度

### **4. 优势值广播**
```python
# 每个时刻动作都使用相同的episode级优势值
sample["advantages"] = torch.full_like(sample["action"][:, 0], advantage_value)  # [50]
```

### **5. CFG安全拷贝**
- **问题**: 浅拷贝导致正/负分支内存共享
- **解决**: 深度拷贝确保分支独立性

## 🚨 **注意事项**

### **数据预处理**
1. **图像**: 必须是`uint8`格式，范围[0,255]
2. **状态**: 必须经过meanstd归一化
3. **动作**: 必须是相对动作，不是绝对动作
4. **掩码**: `action_is_pad`必须正确标记有效长度

### **内存管理**  
1. **张量拷贝**: CFG训练需要独立的batch拷贝
2. **设备一致**: 所有张量必须在相同设备上
3. **数据类型**: 严格匹配模型期望的数据类型

### **数值稳定性**
1. **优势值**: 必须处理NaN、Inf、极端值
2. **梯度**: 需要梯度裁剪防止爆炸
3. **损失**: CFG损失需要数值稳定的计算

---

*此格式规范确保了OpenPI标准兼容性和RIPT训练需求的完美结合* 🎯
