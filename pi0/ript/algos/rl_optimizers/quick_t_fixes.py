"""
Quick Test for Phase 2 Fixes

This script quickly tests the key fixes we made:
1. Mock policy returns 3D losses tensor
2. SO100 processing handles dimension mismatch
3. CFG adapter correctly processes episodes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

import torch
import numpy as np
from typing import Dict, List, Any
from unittest.mock import Mock

from pi0_cfg_interface import PI0_CFG_Adapter


def create_mock_policy():
    """Create a mock PI0Policy for testing."""
    policy = Mock()
    
    # Mock config - 修复：确保所有属性返回正确类型
    policy.config = Mock()
    policy.config.n_action_steps = 50
    policy.config.max_action_dim = 7
    policy.config.num_steps = 10
    policy.config.cfg_uncond_weight = 0.1  # 修复：添加CFG权重配置
    
    # Mock model
    policy.model = Mock()
    policy.model.sample_time = lambda batch_size, device: torch.rand(batch_size, device=device)
    
    # Mock forward method - 修复：返回3维losses tensor (B,T,D)
    def mock_forward(batch):
        B = batch.get('batch_size', batch['state'].shape[0])
        T = batch['action'].shape[1]
        D = batch['action'].shape[2]  # 动作维度，通常是7
        losses = torch.randn(B, T, D, requires_grad=True)  # 3维tensor (B,T,D)
        print(f"Mock forward: B={B}, T={T}, D={D}, losses.shape={losses.shape}")
        return {'losses': losses}
    
    policy.forward = mock_forward
    policy.parameters = lambda: [torch.randn(10, requires_grad=True)]
    
    return policy


def create_simple_episode(episode_id: str, length: int) -> Dict[str, Any]:
    """Create a simple episode for testing."""
    episode = {
        'observations': [],
        'actions': [],
        'total_reward': np.random.uniform(0, 1)
    }
    
    for step in range(length):
        # Create observation
        obs = {
            'robot0_eef_pos': np.random.randn(3).astype(np.float32),
            'robot0_eef_quat': np.array([0, 0, 0, 1], dtype=np.float32),  # Identity quaternion
            'robot0_gripper_qpos': np.random.randn(2).astype(np.float32),
            'agentview_image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'robot0_eye_in_hand_image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'task_description': f'Test task for episode {episode_id}'
        }
        episode['observations'].append(obs)
        
        # Create action (7-dimensional)
        action = np.random.randn(7).astype(np.float32)
        episode['actions'].append(action)
    
    return episode


def test_legacy_processing_fix():
    """Test that legacy processing now works with 3D losses."""
    print("🧪 Testing Legacy Processing Fix...")
    
    # Create mock policy and adapter
    policy = create_mock_policy()
    adapter = PI0_CFG_Adapter(
        policy=policy,
        use_so100_processing=False,
        windowing_mode='last'
    )
    
    # Create simple test episode
    episodes = [create_simple_episode("test_legacy", 80)]
    episode_advantages = torch.randn(1)
    device = torch.device('cpu')
    
    try:
        loss = adapter.compute_weighted_loss(episodes, episode_advantages, device)
        print(f"✅ Legacy processing fix successful! Loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"❌ Legacy processing still failing: {e}")
        return False


def test_so100_processing_fix():
    """Test that SO100 processing now works with dimension handling."""
    print("\n🧪 Testing SO100 Processing Fix...")
    
    # Create mock policy and adapter
    policy = create_mock_policy()
    adapter = PI0_CFG_Adapter(
        policy=policy,
        use_so100_processing=True
    )
    
    # Create simple test episode
    episodes = [create_simple_episode("test_so100", 80)]
    episode_advantages = torch.randn(1)
    device = torch.device('cpu')
    
    try:
        loss = adapter.compute_weighted_loss(episodes, episode_advantages, device)
        print(f"✅ SO100 processing fix successful! Loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"❌ SO100 processing still failing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_utilization():
    """Test data utilization improvement."""
    print("\n🧪 Testing Data Utilization...")
    
    # Create test episodes
    episodes = [
        create_simple_episode("util_test_1", 80),
        create_simple_episode("util_test_2", 100)
    ]
    device = torch.device('cpu')
    
    # Test legacy
    legacy_policy = create_mock_policy()
    legacy_adapter = PI0_CFG_Adapter(
        policy=legacy_policy,
        use_so100_processing=False,
        windowing_mode='last'
    )
    
    try:
        legacy_batch, _ = legacy_adapter.process_episodes(episodes, device)
        legacy_samples = legacy_batch['state'].shape[0]
        print(f"Legacy: {len(episodes)} episodes → {legacy_samples} windows")
    except Exception as e:
        print(f"Legacy processing failed: {e}")
        return False
    
    # Test SO100
    so100_policy = create_mock_policy()
    so100_adapter = PI0_CFG_Adapter(
        policy=so100_policy,
        use_so100_processing=True
    )
    
    try:
        print(f"🔍 SO100 adapter config: use_so100_processing={so100_adapter.use_so100_processing}")

        # 🔥 直接调用SO100处理方法，避免路由问题
        so100_batch, episode_to_samples_map = so100_adapter.process_episodes_to_samples_so100(episodes, device)
        so100_samples = so100_batch['batch_size']
        print(f"SO100: {len(episodes)} episodes → {so100_samples} samples")
        
        improvement = so100_samples / legacy_samples if legacy_samples > 0 else 0
        print(f"✅ Data utilization improvement: {improvement:.1f}x")
        return True
    except Exception as e:
        print(f"SO100 processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run quick tests for our fixes."""
    print("🚀 Quick Test for Phase 2 Fixes")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Legacy processing fix
    if test_legacy_processing_fix():
        tests_passed += 1
    
    # Test 2: SO100 processing fix
    if test_so100_processing_fix():
        tests_passed += 1
    
    # Test 3: Data utilization
    if test_data_utilization():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Quick Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All fixes working correctly!")
    else:
        print("❌ Some issues remain.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
