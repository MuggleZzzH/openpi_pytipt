#!/usr/bin/env python3
"""
设备不匹配问题修复测试

测试修复后的统一样本池是否能正确处理设备转移
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pi0.ript.data.so100_style_processor import SO100StyleProcessor
from pi0.ript.data.sample_generator import TrajectoryToSampleGenerator


def create_test_episode(episode_length: int = 60) -> Dict[str, Any]:
    """创建测试episode"""
    
    observations = []
    actions = []
    
    for t in range(episode_length):
        # 创建观测
        obs = {
            'image': {
                'base_0_rgb': torch.randn(224, 224, 3).cpu(),  # 确保在CPU上
                'left_wrist_0_rgb': torch.randn(224, 224, 3).cpu()
            },
            'state': torch.randn(8).cpu(),  # 确保在CPU上
            'prompt': ['test task description']
        }
        observations.append(obs)
        
        # 创建动作
        action = np.random.randn(7).astype(np.float32)
        actions.append(action)
    
    episode = {
        'observations': observations,
        'actions': actions,
        'total_reward': 0.5,
        'success': True,
        'task_description': 'test task'
    }
    
    return episode


def test_device_consistency():
    """测试设备一致性"""
    print("🧪 Testing device consistency fix...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target device: {device}")
    
    try:
        # 创建处理器
        processor = SO100StyleProcessor(
            action_chunk_size=50,
            state_mean=np.zeros(8),
            state_std=np.ones(8),
            action_mean=np.zeros(7),
            action_std=np.ones(7)
        )
        
        sample_generator = TrajectoryToSampleGenerator(processor)
        
        # 创建测试episode
        episode = create_test_episode(episode_length=60)
        
        print(f"📊 Test episode created:")
        print(f"  - Episode length: {len(episode['observations'])}")
        print(f"  - Action length: {len(episode['actions'])}")
        
        # 处理episode为样本
        print("🔄 Processing episode to samples...")
        samples, episode_to_samples_map = sample_generator.process_episodes_to_samples(
            [episode], device
        )
        
        print(f"✅ Sample generation successful:")
        print(f"  - Generated samples: {len(samples)}")
        print(f"  - Episode mapping: {episode_to_samples_map}")
        
        # 检查样本设备
        print("🔍 Checking sample devices...")
        for i, sample in enumerate(samples[:3]):  # 检查前3个样本
            print(f"  Sample {i}:")
            print(f"    base_image device: {sample['image']['base_0_rgb'].device}")
            print(f"    wrist_image device: {sample['image']['left_wrist_0_rgb'].device}")
            print(f"    state device: {sample['state'].device}")
            print(f"    action device: {sample['action'].device}")
            print(f"    action_is_pad device: {sample['action_is_pad'].device}")
        
        # 测试collate
        print("🔄 Testing batch collation...")
        batch = sample_generator.collate_samples_to_batch(samples[:4], device)
        
        print(f"✅ Batch collation successful:")
        print(f"  - Batch size: {batch['batch_size']}")
        print(f"  - Image device: {batch['image']['base_0_rgb'].device}")
        print(f"  - State device: {batch['state'].device}")
        print(f"  - Action device: {batch['action'].device}")
        print(f"  - Prompts: {batch['prompt']}")
        
        # 验证所有tensor都在同一设备
        expected_device = device
        tensors_to_check = [
            ('base_image', batch['image']['base_0_rgb']),
            ('wrist_image', batch['image']['left_wrist_0_rgb']),
            ('state', batch['state']),
            ('action', batch['action']),
            ('action_is_pad', batch['action_is_pad'])
        ]
        
        all_on_correct_device = True
        for name, tensor in tensors_to_check:
            if tensor.device != expected_device:
                print(f"❌ {name} on wrong device: {tensor.device}, expected: {expected_device}")
                all_on_correct_device = False
        
        if all_on_correct_device:
            print("✅ All tensors on correct device!")
            return True
        else:
            print("❌ Device mismatch detected!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advantages_device():
    """测试优势tensor设备处理"""
    print("\n🧪 Testing advantages device handling...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 创建CPU上的优势tensor
        advantages = torch.tensor([0.5, -0.3], dtype=torch.float32)
        print(f"Original advantages device: {advantages.device}")
        
        # 模拟CFG适配器中的设备转移
        advantages = advantages.to(device)
        w_pos = (advantages > 0).float().to(device)
        
        print(f"Transferred advantages device: {advantages.device}")
        print(f"w_pos device: {w_pos.device}")
        print(f"w_pos values: {w_pos}")
        
        if advantages.device == device and w_pos.device == device:
            print("✅ Advantages device handling correct!")
            return True
        else:
            print("❌ Advantages device handling failed!")
            return False
            
    except Exception as e:
        print(f"❌ Advantages test failed: {e}")
        return False


def main():
    """运行所有测试"""
    print("🚀 Device Fix Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: 设备一致性
    if test_device_consistency():
        tests_passed += 1
    
    # Test 2: 优势tensor设备
    if test_advantages_device():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All device fix tests passed!")
        print("✅ Device mismatch issue should be resolved")
    else:
        print("❌ Some tests failed")
        print("⚠️  Device mismatch issue may still exist")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
