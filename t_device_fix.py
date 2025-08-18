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
        'processed_observations': observations,
        'actions': actions,
        'states': [obs['state'] for obs in observations],  # 添加states序列
        'total_reward': 0.5,
        'success': True,
        'task_description': 'test task',
        'id': 'test_episode_0'
    }
    
    return episode


def test_device_consistency():
    """测试设备一致性"""
    print("🧪 Testing device consistency fix...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target device: {device}")
    
    try:
        # 创建处理器配置
        config = {
            'action_chunk_size': 50,
            'norm_stats_path': '/zhaohan/ZJH/openpi/physical-intelligence/libero/norm_stats.json'
        }
        
        # 创建处理器
        processor = SO100StyleProcessor(config)
        
        sample_generator = TrajectoryToSampleGenerator(processor)
        
        # 创建测试episode
        episode = create_test_episode(episode_length=60)
        
        print(f"📊 Test episode created:")
        print(f"  - Episode length: {len(episode['processed_observations'])}")
        print(f"  - Action length: {len(episode['actions'])}")
        
        # 处理episode为样本
        print("🔄 Processing episode to samples...")
        samples, episode_to_samples_map = sample_generator.generate_samples_from_episodes(
            [episode]
        )
        
        # 转换为OpenPI格式
        openpi_samples = []
        for sample in samples:
            openpi_sample = sample_generator.processor.convert_to_openpi_format(sample)
            openpi_samples.append(openpi_sample)
        
        print(f"✅ Sample generation successful:")
        print(f"  - Generated samples: {len(samples)}")
        print(f"  - OpenPI samples: {len(openpi_samples)}")
        print(f"  - Episode mapping: {episode_to_samples_map}")
        
        # 检查样本设备
        print("🔍 Checking sample devices...")
        for i, sample in enumerate(openpi_samples[:3]):  # 检查前3个样本
            print(f"  Sample {i}:")
            print(f"    base_image device: {sample['image']['base_0_rgb'].device}")
            print(f"    wrist_image device: {sample['image']['left_wrist_0_rgb'].device}")
            print(f"    state device: {sample['state'].device}")
            print(f"    action device: {sample['action'].device}")
            print(f"    action_is_pad device: {sample['action_is_pad'].device}")
        
        # 测试collate
        print("🔄 Testing batch collation...")
        batch = sample_generator.collate_samples_to_batch(openpi_samples[:4], device)
        
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
        
        # 使用相同的设备匹配函数
        def devices_match(device1, device2):
            """检查两个设备是否匹配，考虑cuda:0和cuda的等价性"""
            device1_str = str(device1)
            device2_str = str(device2)
            
            # 直接相等
            if device1_str == device2_str:
                return True
            
            # cuda和cuda:0等价
            if (device1_str == 'cuda' and device2_str.startswith('cuda:')) or \
               (device2_str == 'cuda' and device1_str.startswith('cuda:')):
                return True
                
            return False
        
        all_on_correct_device = True
        for name, tensor in tensors_to_check:
            if not devices_match(tensor.device, expected_device):
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
        
        # 检查设备是否正确（考虑cuda:0和cuda的等价性）
        def devices_match(device1, device2):
            """检查两个设备是否匹配，考虑cuda:0和cuda的等价性"""
            device1_str = str(device1)
            device2_str = str(device2)
            
            # 直接相等
            if device1_str == device2_str:
                return True
            
            # cuda和cuda:0等价
            if (device1_str == 'cuda' and device2_str.startswith('cuda:')) or \
               (device2_str == 'cuda' and device1_str.startswith('cuda:')):
                return True
                
            return False
        
        advantages_correct = devices_match(advantages.device, device)
        w_pos_correct = devices_match(w_pos.device, device)
        
        print(f"Advantages device check: {advantages.device} matches {device} -> {advantages_correct}")
        print(f"w_pos device check: {w_pos.device} matches {device} -> {w_pos_correct}")
        
        if advantages_correct and w_pos_correct:
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
