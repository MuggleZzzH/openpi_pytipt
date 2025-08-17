"""
Test Script for Phase 2 CFG Adapter Integration

This script validates that the modified PI0_CFG_Adapter works correctly with:
1. Legacy windowing processing (backward compatibility)
2. New SO100-style sample processing (Phase 2 feature)
3. Episode-to-sample advantage mapping
4. CFG loss computation with both methods

Run this script to verify Phase 2 integration before using in main training.
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
    
    # Mock config
    policy.config = Mock()
    policy.config.n_action_steps = 50
    policy.config.max_action_dim = 7
    policy.config.num_steps = 10
    
    # Mock model
    policy.model = Mock()
    policy.model.sample_time = lambda batch_size, device: torch.rand(batch_size, device=device)
    
    # Mock forward method
    def mock_forward(batch):
        B = batch.get('batch_size', batch['state'].shape[0])
        T = batch['action'].shape[1]
        losses = torch.randn(B, T, requires_grad=True)
        return {'losses': losses}
    
    policy.forward = mock_forward
    policy.parameters = lambda: [torch.randn(10, requires_grad=True)]
    
    return policy


def create_mock_episode(episode_id: str, length: int) -> Dict[str, Any]:
    """Create a mock episode for testing."""
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
        
        # Create action
        action = np.random.randn(7).astype(np.float32)
        episode['actions'].append(action)
    
    return episode


def test_legacy_windowing_processing():
    """Test legacy windowing processing (backward compatibility)."""
    print("=" * 60)
    print("Testing Legacy Windowing Processing")
    print("=" * 60)
    
    # Create mock policy and adapter
    policy = create_mock_policy()
    adapter = PI0_CFG_Adapter(
        policy=policy,
        use_so100_processing=False,  # Use legacy processing
        windowing_mode='last'
    )
    
    # Create test episodes
    episodes = []
    episode_lengths = [80, 120, 60, 100]
    
    for i, length in enumerate(episode_lengths):
        episode = create_mock_episode(f"legacy_episode_{i}", length)
        episodes.append(episode)
    
    # Create episode advantages
    episode_advantages = torch.randn(len(episodes))
    
    # Test processing
    device = torch.device('cpu')
    
    try:
        # Test compute_weighted_loss
        loss = adapter.compute_weighted_loss(episodes, episode_advantages, device)
        
        print(f"✅ Legacy processing successful:")
        print(f"  - Episodes: {len(episodes)}")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Loss requires grad: {loss.requires_grad}")
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_so100_sample_processing():
    """Test SO100-style sample processing (Phase 2 feature)."""
    print("\n" + "=" * 60)
    print("Testing SO100-Style Sample Processing")
    print("=" * 60)
    
    # Create mock policy and adapter
    policy = create_mock_policy()
    adapter = PI0_CFG_Adapter(
        policy=policy,
        use_so100_processing=True,  # Use SO100 processing
    )
    
    # Create test episodes
    episodes = []
    episode_lengths = [80, 120, 60, 100, 90]
    
    for i, length in enumerate(episode_lengths):
        episode = create_mock_episode(f"so100_episode_{i}", length)
        episodes.append(episode)
    
    # Create episode advantages
    episode_advantages = torch.randn(len(episodes))
    
    # Test processing
    device = torch.device('cpu')
    
    try:
        # Test compute_weighted_loss
        loss = adapter.compute_weighted_loss(episodes, episode_advantages, device)
        
        print(f"✅ SO100 processing successful:")
        print(f"  - Episodes: {len(episodes)}")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Loss requires grad: {loss.requires_grad}")
        
        return True
        
    except Exception as e:
        print(f"❌ SO100 processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_utilization_comparison():
    """Compare data utilization between legacy and SO100 processing."""
    print("\n" + "=" * 60)
    print("Testing Data Utilization Comparison")
    print("=" * 60)
    
    # Create test episodes
    episodes = []
    episode_lengths = [80, 120, 60, 100, 90, 110, 75, 130]
    
    for i, length in enumerate(episode_lengths):
        episode = create_mock_episode(f"comparison_episode_{i}", length)
        episodes.append(episode)
    
    episode_advantages = torch.randn(len(episodes))
    device = torch.device('cpu')
    
    # Test legacy processing
    policy = create_mock_policy()
    legacy_adapter = PI0_CFG_Adapter(
        policy=policy,
        use_so100_processing=False,
        windowing_mode='last'
    )
    
    try:
        legacy_batch, legacy_owner_indices = legacy_adapter.process_episodes(episodes, device)
        legacy_samples = legacy_batch['state'].shape[0]
        print(f"Legacy processing: {len(episodes)} episodes → {legacy_samples} windows")
    except Exception as e:
        print(f"Legacy processing failed: {e}")
        return False
    
    # Test SO100 processing
    so100_adapter = PI0_CFG_Adapter(
        policy=policy,
        use_so100_processing=True
    )
    
    try:
        so100_batch, so100_owner_indices = so100_adapter.process_episodes(episodes, device)
        so100_samples = so100_batch['batch_size']
        print(f"SO100 processing: {len(episodes)} episodes → {so100_samples} samples")
        
        # Calculate improvement
        improvement = so100_samples / legacy_samples if legacy_samples > 0 else 0
        print(f"Data utilization improvement: {improvement:.1f}x")
        
        # Verify improvement is significant
        assert improvement > 10, f"Expected >10x improvement, got {improvement:.1f}x"
        
        print("✅ Data utilization comparison successful!")
        return True
        
    except Exception as e:
        print(f"SO100 processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 2 integration tests."""
    print("🚀 Starting Phase 2 CFG Adapter Integration Tests")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Legacy windowing processing
    if test_legacy_windowing_processing():
        tests_passed += 1
    
    # Test 2: SO100 sample processing
    if test_so100_sample_processing():
        tests_passed += 1
    
    # Test 3: Data utilization comparison
    if test_data_utilization_comparison():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print(f"Phase 2 Integration Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Phase 2 CFG adapter integration is ready.")
        print("✅ Both legacy windowing and SO100 sample processing work correctly.")
        print("✅ Data utilization improvement verified.")
    else:
        print("❌ Some tests failed. Please review the errors above.")
    
    print("=" * 80)
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
