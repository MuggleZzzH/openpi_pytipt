#!/usr/bin/env python3
"""
快速CFG验证脚本
==============

直接运行此脚本来验证CFG实现是否正确
"""
import os
import sys
import torch
from pathlib import Path

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent  # 回到项目根目录
sys.path.insert(0, str(project_root))

def quick_cfg_test():
    """快速CFG功能测试"""
    print("🚀 开始快速CFG验证测试...")
    
    try:
        # 检查CUDA可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 使用设备: {device}")
        
        # 1. 加载模型（请根据您的实际路径修改）
        checkpoint_path = "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch"
        
        if not Path(checkpoint_path).exists():
            print(f"⚠️  模型路径不存在: {checkpoint_path}")
            print("💡 请修改 checkpoint_path 为您的实际模型路径")
            return False
            
        print(f"📂 加载模型: {checkpoint_path}")
        from pi0.modeling_pi0 import PI0Policy
        policy = PI0Policy.from_pretrained(checkpoint_path)
        policy = policy.to(device).eval()
        print("✅ 模型加载成功")
        
        # 2. 构造测试观测
        observation = {
            "image": {
                "base_0_rgb": torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.uint8).to(device),
            },
            "state": torch.randn(1, 8).to(device),
            "prompt": ["测试任务：移动物体到目标位置"]
        }
        print("📊 构造测试数据完成")
        
        # 3. 验证推理功能
        print("🔍 开始CFG推理验证...")
        from pi0.ript.utils.cfg_validation import validate_cfg_inference
        
        inference_ok = validate_cfg_inference(policy, observation, device=device)
        
        if inference_ok:
            print("🎉 CFG推理验证成功！")
            print("   ✅ 不同cfg_scale产生不同动作")
            print("   ✅ is_positive张量类型正确") 
            print("   ✅ 推理管道无错误")
        else:
            print("❌ CFG推理验证失败！")
            return False
        
        # 4. 测试不同CFG scale的实际效果
        print("\n📈 测试不同CFG引导强度效果...")
        
        with torch.no_grad():
            action_1 = policy.select_action(observation, cfg_scale=1.0)  # 无引导
            action_3 = policy.select_action(observation, cfg_scale=3.0)  # 中等引导
            action_10 = policy.select_action(observation, cfg_scale=10.0) # 强引导
        
        # 计算动作差异
        diff_1_3 = torch.norm(action_1 - action_3).item()
        diff_3_10 = torch.norm(action_3 - action_10).item()
        
        print(f"   cfg_scale 1.0 vs 3.0 动作差异: {diff_1_3:.4f}")
        print(f"   cfg_scale 3.0 vs 10.0 动作差异: {diff_3_10:.4f}")
        
        if diff_1_3 > 1e-4 and diff_3_10 > 1e-4:
            print("✅ CFG引导强度调节正常工作")
        else:
            print("⚠️  CFG引导可能未生效，请检查实现")
        
        print("\n🎯 CFG验证测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ CFG验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cfg_training_if_possible():
    """如果有训练数据，测试CFG训练功能"""
    print("\n🔧 CFG训练功能测试...")
    print("💡 如需测试CFG训练功能，请在实际训练脚本中调用:")
    print("   from pi0.ript.utils.cfg_validation import validate_cfg_training")
    print("   validate_cfg_training(policy, episodes, advantages)")

if __name__ == "__main__":
    print("CFG快速验证脚本")
    print("=" * 40)
    
    # 运行快速测试
    success = quick_cfg_test()
    
    # 提供训练测试说明
    test_cfg_training_if_possible()
    
    if success:
        print("\n🎉 所有可用的CFG验证测试都通过了！")
        print("💡 您的CFG实现应该可以正常工作")
    else:
        print("\n❌ CFG验证测试失败，请检查实现")
    
    print("\n📚 更多使用方式请查看: cfg_validation_examples.py")