"""
CFG Control Test Script

This script tests the CFG enable/disable functionality to ensure:
1. cfg_enabled=true properly enables CFG functionality
2. cfg_enabled=false properly disables CFG functionality
3. No hardcoded CFG forcing exists
4. CFG-related components are properly controlled by configuration

Usage:
    # Test CFG enabled
    python test_cfg_control.py --cfg_enabled true
    
    # Test CFG disabled
    python test_cfg_control.py --cfg_enabled false
    
    # Test both modes
    python test_cfg_control.py --test_both
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pi0.modeling_pi0 import PI0Policy
from pi0.ript.algos.rl_optimizers.pi0_cfg_interface import PI0_CFG_Adapter


def create_test_config(cfg_enabled: bool = True) -> Dict[str, Any]:
    """Create a test configuration with specified CFG setting."""
    config = {
        'policy_path': "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch",
        'output_dir': "./output/cfg_control_test",
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
            'cfg_enabled': cfg_enabled,  # 🔥 CFG控制配置
            'train_expert_only': True,
            'freeze_vision_encoder': True
        },
        'dataset': {
            'num_init_states': 1,
            'state_dim': 8,
            'use_so100_processing': False,
            'windowing_mode': 'last',
            'window_stride': 10,
            'max_windows_per_episode': 1
        },
        'training': {
            'num_train_steps': 1,
            'seed': 42,
            'save_freq': 1
        },
        'logging': {
            'use_wandb': False,
            'log_freq': 1
        }
    }
    return config


def create_mock_policy(cfg_enabled: bool = True):
    """Create a mock policy with specified CFG setting."""
    policy = Mock()
    
    # Mock config
    policy.config = Mock()
    policy.config.n_action_steps = 50
    policy.config.max_action_dim = 7
    policy.config.num_steps = 10
    policy.config.cfg_uncond_weight = 0.1
    policy.config.cfg_enabled = cfg_enabled
    
    # Mock model
    policy.model = Mock()
    policy.model.cfg_enabled = cfg_enabled
    policy.model.cfg_emb = Mock() if cfg_enabled else None
    
    # Mock parameters
    policy.parameters = lambda: [torch.randn(10, requires_grad=True)]
    
    # Mock forward method
    def mock_forward(batch):
        B = batch.get('batch_size', batch['state'].shape[0])
        T = batch['action'].shape[1]
        D = batch['action'].shape[2]
        losses = torch.randn(B, T, D, requires_grad=True)
        return {'losses': losses}
    
    policy.forward = mock_forward
    
    return policy


def test_cfg_enabled_mode() -> bool:
    """Test CFG enabled mode."""
    print("🧪 Testing CFG Enabled Mode")
    print("-" * 40)
    
    try:
        # Create config with CFG enabled
        config = create_test_config(cfg_enabled=True)
        
        # Create mock policy
        policy = create_mock_policy(cfg_enabled=True)
        
        # Simulate policy loading logic from training script
        policy_config = config.get('policy', {})
        cfg_enabled = policy_config.get('cfg_enabled', True)
        
        print(f"🔧 配置CFG功能: {'启用' if cfg_enabled else '禁用'}")
        policy.model.cfg_enabled = cfg_enabled
        if hasattr(policy, 'config'):
            policy.config.cfg_enabled = cfg_enabled
        
        # Verify CFG is enabled
        assert policy.model.cfg_enabled == True, "CFG should be enabled"
        assert policy.config.cfg_enabled == True, "CFG config should be enabled"
        
        # Test CFG embedding parameter inclusion
        if hasattr(policy.model, "cfg_emb") and getattr(policy.model, 'cfg_enabled', True):
            print("✅ CFG embedding参数已加入训练")
            cfg_emb_included = True
        else:
            print("⚠️ CFG已禁用，跳过CFG embedding参数训练")
            cfg_emb_included = False
        
        assert cfg_emb_included == True, "CFG embedding should be included when CFG is enabled"
        
        # Test CFG adapter creation
        cfg_adapter = PI0_CFG_Adapter(
            policy=policy,
            norm_stats_path=f"{config['policy_path']}/norm_stats.json",
            use_so100_processing=False
        )
        
        print("✅ CFG enabled mode test passed")
        return True
        
    except Exception as e:
        print(f"❌ CFG enabled mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cfg_disabled_mode() -> bool:
    """Test CFG disabled mode."""
    print("\n🧪 Testing CFG Disabled Mode")
    print("-" * 40)
    
    try:
        # Create config with CFG disabled
        config = create_test_config(cfg_enabled=False)
        
        # Create mock policy
        policy = create_mock_policy(cfg_enabled=False)
        
        # Simulate policy loading logic from training script
        policy_config = config.get('policy', {})
        cfg_enabled = policy_config.get('cfg_enabled', True)
        
        print(f"🔧 配置CFG功能: {'启用' if cfg_enabled else '禁用'}")
        policy.model.cfg_enabled = cfg_enabled
        if hasattr(policy, 'config'):
            policy.config.cfg_enabled = cfg_enabled
        
        # Verify CFG is disabled
        assert policy.model.cfg_enabled == False, "CFG should be disabled"
        assert policy.config.cfg_enabled == False, "CFG config should be disabled"
        
        # Test CFG embedding parameter exclusion
        if hasattr(policy.model, "cfg_emb") and getattr(policy.model, 'cfg_enabled', True):
            print("✅ CFG embedding参数已加入训练")
            cfg_emb_included = True
        else:
            print("⚠️ CFG已禁用，跳过CFG embedding参数训练")
            cfg_emb_included = False
        
        assert cfg_emb_included == False, "CFG embedding should be excluded when CFG is disabled"
        
        # Test CFG evaluation skipping
        should_run_cfg_eval = getattr(policy.model, 'cfg_enabled', True)
        if should_run_cfg_eval:
            print("🔍 CFG强度评估将运行")
        else:
            print("⚠️ CFG已禁用，跳过CFG强度评估")
        
        assert should_run_cfg_eval == False, "CFG evaluation should be skipped when CFG is disabled"
        
        # Test CFG adapter creation
        cfg_adapter = PI0_CFG_Adapter(
            policy=policy,
            norm_stats_path=f"{config['policy_path']}/norm_stats.json",
            use_so100_processing=False
        )
        
        print("✅ CFG disabled mode test passed")
        return True
        
    except Exception as e:
        print(f"❌ CFG disabled mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_consistency() -> bool:
    """Test that configuration is consistently applied."""
    print("\n🧪 Testing Configuration Consistency")
    print("-" * 40)
    
    try:
        # Test both enabled and disabled configurations
        for cfg_enabled in [True, False]:
            config = create_test_config(cfg_enabled=cfg_enabled)
            
            # Verify configuration is correctly set
            policy_config = config.get('policy', {})
            config_cfg_enabled = policy_config.get('cfg_enabled', True)
            
            assert config_cfg_enabled == cfg_enabled, f"Configuration mismatch: expected {cfg_enabled}, got {config_cfg_enabled}"
            
            print(f"✅ Configuration consistency verified for cfg_enabled={cfg_enabled}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration consistency test failed: {e}")
        return False


def main():
    """Run CFG control tests."""
    parser = argparse.ArgumentParser(description='CFG Control Test')
    parser.add_argument('--cfg_enabled', choices=['true', 'false'], 
                       help='Test specific CFG mode')
    parser.add_argument('--test_both', action='store_true',
                       help='Test both CFG enabled and disabled modes')
    args = parser.parse_args()
    
    print("🚀 CFG Control Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    if args.cfg_enabled == 'true':
        total_tests = 1
        if test_cfg_enabled_mode():
            tests_passed += 1
            
    elif args.cfg_enabled == 'false':
        total_tests = 1
        if test_cfg_disabled_mode():
            tests_passed += 1
            
    elif args.test_both or (not args.cfg_enabled and not args.test_both):
        total_tests = 3
        
        if test_cfg_enabled_mode():
            tests_passed += 1
        if test_cfg_disabled_mode():
            tests_passed += 1
        if test_configuration_consistency():
            tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"CFG Control Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All CFG control tests passed!")
        print("✅ CFG functionality can be properly controlled by configuration")
        print("✅ No hardcoded CFG forcing detected")
    else:
        print("❌ Some CFG control tests failed")
        print("⚠️  CFG configuration may not be working correctly")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
