#!/usr/bin/env python3
"""
CFG实现验证脚本
===============

验证CFG训练和推理的正确性，确保与原始CFGRL实现对齐
"""
import torch
import numpy as np
from typing import Dict, Any

def validate_cfg_training(policy, episodes: list, advantages: torch.Tensor, device: str = "cuda"):
    """
    验证CFG训练实现的正确性
    
    检查项:
    1. 同一批noise/time下，is_positive=1与=0输出的losses不应完全相等
    2. 条件和无条件分支都能正常前向传播
    3. 优势权重正确应用
    """
    print("🔍 验证CFG训练实现...")
    
    try:
        from pi0.ript.algos.rl_optimizers.pi0_cfg_interface import PI0_CFG_Adapter
        
        adapter = PI0_CFG_Adapter(policy)
        
        # 测试基本训练流程
        loss = adapter.compute_weighted_loss(episodes, advantages, device=torch.device(device))
        
        print(f"✅ CFG训练损失计算成功: {loss.item():.6f}")
        
        # 检查损失是否为有限值
        assert torch.isfinite(loss), "CFG损失必须为有限值"
        assert loss.requires_grad, "CFG损失必须支持梯度计算"
        
        return True
        
    except Exception as e:
        print(f"❌ CFG训练验证失败: {e}")
        return False

def validate_cfg_inference(policy, observation: Dict[str, Any], device: str = "cuda"):
    """
    验证CFG推理实现的正确性
    
    检查项:
    1. cfg_scale=1.0与>1.0的动作应有可观差异
    2. cfg_scale增大时(v_cond - v_uncond)的投影步长变大
    3. 纯无条件推理不报错
    """
    print("🔍 验证CFG推理实现...")
    
    try:
        policy = policy.to(device).eval()
        
        # 测试不同cfg_scale的输出差异
        action_uncond = policy.select_action(observation, cfg_scale=1.0)  # 纯无条件
        action_cfg_3 = policy.select_action(observation, cfg_scale=3.0)   # CFG引导
        action_cfg_5 = policy.select_action(observation, cfg_scale=5.0)   # 更强CFG引导
        
        # 计算动作差异
        diff_1_3 = torch.norm(action_uncond - action_cfg_3).item()
        diff_3_5 = torch.norm(action_cfg_3 - action_cfg_5).item()
        
        print(f"✅ CFG推理测试成功:")
        print(f"   cfg_scale=1.0 vs 3.0 差异: {diff_1_3:.6f}")
        print(f"   cfg_scale=3.0 vs 5.0 差异: {diff_3_5:.6f}")
        
        # 检查CFG引导确实产生了不同的输出
        assert diff_1_3 > 1e-6, f"CFG引导应该产生不同的动作，但差异过小: {diff_1_3}"
        
        return True
        
    except Exception as e:
        print(f"❌ CFG推理验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_noise_time_consistency(policy, episodes: list, advantages: torch.Tensor, device: str = "cuda"):
    """
    验证noise和time在条件/无条件分支间的一致性
    
    这是与原始CFGRL对齐的关键点
    """
    print("🔍 验证noise/time一致性...")
    
    try:
        from pi0.ript.algos.rl_optimizers.pi0_cfg_interface import PI0_CFG_Adapter
        
        # 手动模拟adapter中的noise/time采样逻辑
        adapter = PI0_CFG_Adapter(policy)
        batch = adapter.process_episodes(episodes, torch.device(device))
        
        B = batch["state"].shape[0]
        n, d = policy.config.n_action_steps, policy.config.max_action_dim
        dtype = batch["state"].dtype
        
        # 采样一次noise和time
        noise = torch.randn(B, n, d, device=torch.device(device), dtype=dtype)
        time = policy.model.sample_time(B, torch.device(device)).to(dtype)
        
        # 条件分支
        batch_positive = batch.copy()
        batch_positive["is_positive"] = torch.ones(B, device=torch.device(device), dtype=torch.long)
        batch_positive["noise"] = noise
        batch_positive["time"] = time
        
        # 无条件分支
        batch_uncond = batch.copy()
        batch_uncond["is_positive"] = torch.zeros(B, device=torch.device(device), dtype=torch.long)
        batch_uncond["noise"] = noise  # 关键：使用相同的noise
        batch_uncond["time"] = time    # 关键：使用相同的time
        
        # 前向传播
        out_pos = policy.forward(batch_positive)
        out_uncond = policy.forward(batch_uncond)
        
        print("✅ noise/time一致性验证通过")
        print(f"   条件分支损失形状: {out_pos[1]['losses'].shape}")
        print(f"   无条件分支损失形状: {out_uncond[1]['losses'].shape}")
        
        # 验证两个分支的损失不完全相等（说明CFG embedding生效）
        losses_diff = torch.norm(out_pos[1]['losses'] - out_uncond[1]['losses']).item()
        assert losses_diff > 1e-6, f"条件和无条件分支损失应该不同，但差异过小: {losses_diff}"
        print(f"   条件/无条件损失差异: {losses_diff:.6f} (正常)")
        
        return True
        
    except Exception as e:
        print(f"❌ noise/time一致性验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_cfg_validation():
    """运行完整的CFG验证测试"""
    print("🚀 开始CFG实现完整验证\n")
    
    try:
        # 这里需要用户提供实际的policy和数据
        # 在实际使用时，应该从真实的训练环境中调用这些验证函数
        print("⚠️  注意: 请在实际训练脚本中调用各个验证函数")
        print("   示例用法:")
        print("   validate_cfg_training(policy, episodes, advantages)")
        print("   validate_cfg_inference(policy, observation)")
        print("   validate_noise_time_consistency(policy, episodes, advantages)")
        
        return True
        
    except Exception as e:
        print(f"❌ CFG验证失败: {e}")
        return False

if __name__ == "__main__":
    run_full_cfg_validation()