#!/usr/bin/env python3
"""
Stage 11 Debug Script - Minimal test to identify the issue
"""
import os
import sys
import yaml
from pathlib import Path

# 修复tokenizers并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"=== Stage 11 Minimal Debug Test ===")
print(f"Script location: {current_file}")
print(f"Project root: {project_root}")
print()

def test_imports():
    """Test core module imports"""
    try:
        print("Testing imports...")
        
        # Core imports
        from pi0.modeling_pi0 import PI0Policy
        print("✓ PI0Policy")
        
        from pi0.ript.algos.rl_optimizers.pi0_cfg_interface import PI0_CFG_Adapter
        print("✓ PI0_CFG_Adapter")
        
        # Runner imports
        try:
            from pi0.ript.env.pi0_libero_runner import LIBEROEnvRunner
            print("✓ LIBEROEnvRunner")
        except ImportError as e:
            print(f"⚠️ LIBEROEnvRunner import failed: {e}")
            
        try:
            from pi0.ript.env.pi0_libero_runner_ript_vla import PI0LiberoRunner
            print("✓ PI0LiberoRunner")
        except ImportError as e:
            print(f"⚠️ PI0LiberoRunner import failed: {e}")
        
        print("✓ All basic imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading"""
    try:
        print("Testing config loading...")
        
        config_path = 'pi0/ript/config/stage11_test.yaml'
        if not Path(config_path).exists():
            print(f"❌ Config file not found: {config_path}")
            return False
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Config loaded, keys: {list(config.keys())}")
        
        # Check features
        use_ript_vla = config.get('features', {}).get('use_ript_vla_runner', False)
        print(f"✓ use_ript_vla_runner: {use_ript_vla}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_policy_loading():
    """Test policy loading"""
    try:
        print("Testing policy loading...")
        
        from pi0.modeling_pi0 import PI0Policy
        
        policy_path = "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch"
        print(f"Loading policy from: {policy_path}")
        
        if not Path(policy_path).exists():
            print(f"❌ Policy path not found: {policy_path}")
            return False
            
        policy = PI0Policy.from_pretrained(policy_path)
        print("✓ Policy loaded successfully")
        
        # Test CFG adapter creation
        from pi0.ript.algos.rl_optimizers.pi0_cfg_interface import PI0_CFG_Adapter
        
        cfg_adapter = PI0_CFG_Adapter(
            policy=policy,
            norm_stats_path=f"{policy_path}/norm_stats.json"
        )
        print("✓ CFG Adapter created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Policy loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Starting Stage 11 debug tests...")
    
    tests = [
        ("Module Imports", test_imports),
        ("Config Loading", test_config_loading), 
        ("Policy Loading", test_policy_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n=== {test_name} ===")
        success = test_func()
        results.append((test_name, success))
        
        if not success:
            print(f"❌ {test_name} failed, stopping tests")
            break
    
    print("\n=== Test Results ===")
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed")
        
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())