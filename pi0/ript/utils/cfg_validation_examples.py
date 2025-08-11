#!/usr/bin/env python3
"""
CFG验证使用示例
===============

演示如何在训练和推理脚本中使用CFG验证功能
"""

# 示例1: 在训练脚本中验证CFG训练逻辑
def example_training_validation():
    """在RIPT训练脚本中的使用示例"""
    
    # 假设您已经有了这些对象（来自实际训练脚本）
    # policy = PI0Policy.from_pretrained(checkpoint_path)
    # episodes = [...] # 从环境收集的episodes
    # advantages = torch.tensor([...]) # 计算的优势值
    
    # 导入验证函数
    from pi0.ript.utils.cfg_validation import (
        validate_cfg_training, 
        validate_cfg_inference, 
        validate_noise_time_consistency
    )
    
    print("🔍 开始CFG实现验证...")
    
    # 1. 验证训练逻辑
    # training_ok = validate_cfg_training(policy, episodes, advantages, device="cuda")
    
    # 2. 验证推理逻辑  
    # observation = {...} # 构造的观测
    # inference_ok = validate_cfg_inference(policy, observation, device="cuda")
    
    # 3. 验证noise/time一致性
    # consistency_ok = validate_noise_time_consistency(policy, episodes, advantages, device="cuda")
    
    # if all([training_ok, inference_ok, consistency_ok]):
    #     print("✅ CFG实现验证全部通过！")
    #     return True
    # else:
    #     print("❌ CFG实现存在问题，请检查！")
    #     return False
    
    print("⚠️  请替换上面注释的代码为实际对象")

# 示例2: 在现有训练脚本中添加验证检查点
def add_validation_to_existing_script():
    """演示如何在现有脚本中添加验证"""
    
    example_code = '''
# 在您的训练脚本中，例如 train_ript_pi0.py 的适当位置添加：

# === 在训练循环开始前 ===
if step == 0:  # 只在第一步验证一次
    from pi0.ript.utils.cfg_validation import validate_cfg_training, validate_noise_time_consistency
    
    print("🔍 验证CFG训练实现...")
    training_ok = validate_cfg_training(adapter.policy, batch_episodes, advantages)
    consistency_ok = validate_noise_time_consistency(adapter.policy, batch_episodes, advantages)
    
    if not (training_ok and consistency_ok):
        print("❌ CFG验证失败，停止训练")
        return
    print("✅ CFG验证通过，开始训练")

# === 在推理/评估时 ===
if eval_step % 1000 == 0:  # 定期验证推理
    from pi0.ript.utils.cfg_validation import validate_cfg_inference
    
    # 使用当前observation进行验证
    inference_ok = validate_cfg_inference(policy, current_observation)
    if not inference_ok:
        print("⚠️ CFG推理验证失败，请检查cfg_scale参数")
'''
    
    print("💡 集成示例代码：")
    print(example_code)

# 示例3: 独立运行验证脚本
def standalone_validation_example():
    """独立运行CFG验证的完整示例"""
    
    example_code = '''
#!/usr/bin/env python3
"""
独立CFG验证脚本
"""
import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append('/zhaohan/ZJH/openpi_pytorch')

def main():
    # 1. 加载模型
    from pi0.modeling_pi0 import PI0Policy
    checkpoint_path = "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch"
    policy = PI0Policy.from_pretrained(checkpoint_path)
    policy = policy.cuda().eval()
    
    # 2. 构造测试数据
    # 模拟一个简单的observation
    observation = {
        "image": {
            "base_0_rgb": torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.uint8).cuda(),
        },
        "state": torch.randn(1, 8).cuda(),
        "prompt": ["test task"]
    }
    
    # 模拟简单的episodes和advantages（用于训练验证）
    episodes = [{"observations": [{}], "actions": [torch.randn(7)], "task": "test"}]
    advantages = torch.tensor([1.0]).cuda()
    
    # 3. 运行验证
    from pi0.ript.utils.cfg_validation import (
        validate_cfg_training, 
        validate_cfg_inference, 
        validate_noise_time_consistency
    )
    
    print("🚀 开始独立CFG验证...")
    
    # 验证推理（最简单的验证）
    inference_ok = validate_cfg_inference(policy, observation, device="cuda")
    
    if inference_ok:
        print("🎉 CFG推理验证成功！")
    else:
        print("❌ CFG推理验证失败！")
    
    # 如果需要验证训练，需要更完整的数据
    # training_ok = validate_cfg_training(policy, episodes, advantages, device="cuda")

if __name__ == "__main__":
    main()
'''
    
    print("📝 独立验证脚本示例：")
    print(example_code)
    print("\n💾 您可以将上述代码保存为 test_cfg_validation.py 并运行")

if __name__ == "__main__":
    print("CFG验证工具使用指南")
    print("=" * 50)
    
    print("\n1️⃣ 训练脚本集成示例:")
    example_training_validation()
    
    print("\n2️⃣ 现有脚本添加验证:")
    add_validation_to_existing_script()
    
    print("\n3️⃣ 独立验证脚本:")
    standalone_validation_example()
    
    print("\n🎯 推荐使用方式:")
    print("- 开发阶段: 使用独立验证脚本快速检查")
    print("- 训练阶段: 在训练脚本开始时添加一次性验证")  
    print("- 推理阶段: 在评估脚本中定期验证")
    print("- 调试阶段: 使用完整验证功能定位问题")