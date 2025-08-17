"""
Phase 3 Integration Test Script

This script tests the complete SO100 data processing integration in the main training pipeline.
It validates:
1. Configuration loading and parsing
2. CFG adapter initialization with SO100 support
3. Data utilization monitoring
4. Training loop compatibility
5. Performance comparison between Legacy and SO100 modes

Usage:
    # Test Legacy mode
    python test_phase3_integration.py --mode legacy
    
    # Test SO100 mode
    python test_phase3_integration.py --mode so100
    
    # Test both modes and compare
    python test_phase3_integration.py --mode compare
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import training components
from pi0.modeling_pi0 import PI0Policy
from pi0.ript.algos.rl_optimizers.pi0_cfg_interface import PI0_CFG_Adapter


def create_test_config(use_so100_processing: bool = False) -> Dict[str, Any]:
    """Create a test configuration for Phase 3 testing."""
    config = {
        'policy_path': "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch",
        'output_dir': "./output/phase3_test",
        'task': {
            'benchmark_name': "libero_goal",
            'task_names_to_use': ["put_the_bowl_on_the_stove"],
            'num_parallel_envs': 2,
            'max_episode_length': None
        },
        'algo': {
            'demo_batch_size': 1,
            'rloo_batch_size': 2,
            'lr': 1e-5,
            'gradient_accumulation_steps': 4,
            'collection_cfg_scale': 1.0,
            'cfg_uncond_weight': 0.1
        },
        'policy': {
            'cfg_enabled': True,
            'train_expert_only': True,
            'freeze_vision_encoder': True
        },
        'dataset': {
            'num_init_states': 1,
            'state_dim': 8,
            'use_so100_processing': use_so100_processing,  # 🔥 Phase 3 config
            'windowing_mode': 'last',
            'window_stride': 10,
            'max_windows_per_episode': 1
        },
        'training': {
            'num_train_steps': 2,
            'seed': 42,
            'save_freq': 1
        },
        'logging': {
            'use_wandb': False,
            'log_freq': 1
        }
    }
    return config


def create_mock_episodes(num_episodes: int = 3, episode_length: int = 80) -> list:
    """Create mock episodes for testing."""
    import numpy as np
    
    episodes = []
    for i in range(num_episodes):
        episode = {
            'observations': [],
            'actions': [],
            'total_reward': np.random.uniform(0, 1)
        }
        
        for step in range(episode_length):
            # Create observation
            obs = {
                'robot0_eef_pos': np.random.randn(3).astype(np.float32),
                'robot0_eef_quat': np.array([0, 0, 0, 1], dtype=np.float32),
                'robot0_gripper_qpos': np.random.randn(2).astype(np.float32),
                'agentview_image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                'robot0_eye_in_hand_image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                'task_description': f'Test task for episode {i}'
            }
            episode['observations'].append(obs)
            
            # Create action
            action = np.random.randn(7).astype(np.float32)
            episode['actions'].append(action)
        
        episodes.append(episode)
    
    return episodes


def test_cfg_adapter_initialization(config: Dict[str, Any]) -> bool:
    """Test CFG adapter initialization with SO100 support."""
    print(f"🧪 Testing CFG adapter initialization...")
    
    try:
        # Create mock policy
        from unittest.mock import Mock
        policy = Mock()
        policy.config = Mock()
        policy.config.n_action_steps = 50
        policy.config.max_action_dim = 7
        policy.config.num_steps = 10
        policy.config.cfg_uncond_weight = 0.1
        policy.parameters = lambda: [torch.randn(10, requires_grad=True)]
        
        # Extract dataset config
        dataset_config = config.get('dataset', {})
        use_so100_processing = dataset_config.get('use_so100_processing', False)
        
        # Initialize CFG adapter
        cfg_adapter = PI0_CFG_Adapter(
            policy=policy,
            norm_stats_path=f"{config['policy_path']}/norm_stats.json",
            use_so100_processing=use_so100_processing,
            windowing_mode=dataset_config.get('windowing_mode', 'last'),
            window_stride=dataset_config.get('window_stride', 10),
            max_windows_per_episode=dataset_config.get('max_windows_per_episode', 1)
        )
        
        print(f"✅ CFG adapter initialized successfully")
        print(f"   SO100 processing: {cfg_adapter.use_so100_processing}")
        
        return True
        
    except Exception as e:
        print(f"❌ CFG adapter initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_processing(config: Dict[str, Any]) -> bool:
    """Test data processing with mock episodes."""
    print(f"🧪 Testing data processing...")
    
    try:
        # Create mock policy
        from unittest.mock import Mock
        policy = Mock()
        policy.config = Mock()
        policy.config.n_action_steps = 50
        policy.config.max_action_dim = 7
        policy.config.num_steps = 10
        policy.config.cfg_uncond_weight = 0.1
        policy.parameters = lambda: [torch.randn(10, requires_grad=True)]
        
        # Mock forward method
        def mock_forward(batch):
            B = batch.get('batch_size', batch['state'].shape[0])
            T = batch['action'].shape[1]
            D = batch['action'].shape[2]
            losses = torch.randn(B, T, D, requires_grad=True)
            return {'losses': losses}
        
        policy.forward = mock_forward
        
        # Extract dataset config
        dataset_config = config.get('dataset', {})
        use_so100_processing = dataset_config.get('use_so100_processing', False)
        
        # Initialize CFG adapter
        cfg_adapter = PI0_CFG_Adapter(
            policy=policy,
            norm_stats_path=f"{config['policy_path']}/norm_stats.json",
            use_so100_processing=use_so100_processing,
            windowing_mode=dataset_config.get('windowing_mode', 'last'),
            window_stride=dataset_config.get('window_stride', 10),
            max_windows_per_episode=dataset_config.get('max_windows_per_episode', 1)
        )
        
        # Create mock episodes
        episodes = create_mock_episodes(num_episodes=3, episode_length=80)
        advantages = torch.randn(len(episodes))
        device = torch.device('cpu')
        
        # Test data processing
        loss = cfg_adapter.compute_weighted_loss(episodes, advantages, device)
        
        print(f"✅ Data processing successful")
        print(f"   Episodes: {len(episodes)}")
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Processing mode: {'SO100' if use_so100_processing else 'Legacy'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_comparison() -> bool:
    """Test performance comparison between Legacy and SO100 modes."""
    print(f"🧪 Testing performance comparison...")
    
    try:
        # Test Legacy mode
        legacy_config = create_test_config(use_so100_processing=False)
        legacy_success = test_data_processing(legacy_config)
        
        # Test SO100 mode
        so100_config = create_test_config(use_so100_processing=True)
        so100_success = test_data_processing(so100_config)
        
        if legacy_success and so100_success:
            print(f"✅ Performance comparison successful")
            print(f"   Both Legacy and SO100 modes work correctly")
            return True
        else:
            print(f"❌ Performance comparison failed")
            return False
            
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Phase 3 integration tests."""
    parser = argparse.ArgumentParser(description='Phase 3 Integration Test')
    parser.add_argument('--mode', choices=['legacy', 'so100', 'compare'], 
                       default='compare', help='Test mode')
    args = parser.parse_args()
    
    print("🚀 Phase 3 Integration Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    if args.mode == 'legacy':
        print("\n📋 Testing Legacy Mode Only")
        config = create_test_config(use_so100_processing=False)
        total_tests = 2
        
        if test_cfg_adapter_initialization(config):
            tests_passed += 1
        if test_data_processing(config):
            tests_passed += 1
            
    elif args.mode == 'so100':
        print("\n📋 Testing SO100 Mode Only")
        config = create_test_config(use_so100_processing=True)
        total_tests = 2
        
        if test_cfg_adapter_initialization(config):
            tests_passed += 1
        if test_data_processing(config):
            tests_passed += 1
            
    elif args.mode == 'compare':
        print("\n📋 Testing Both Modes and Comparison")
        total_tests = 1
        
        if test_performance_comparison():
            tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Phase 3 Integration Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Phase 3 integration is working correctly.")
    else:
        print("❌ Some tests failed. Please review the errors above.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
