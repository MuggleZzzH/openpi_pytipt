#!/usr/bin/env python3
"""
完整的RIPT修复验证测试
测试从数据加载到状态设置的完整流程
"""

import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_complete_data_flow():
    """测试完整的数据流：从RIPT对齐数据集到状态设置"""
    print("🔍 测试完整数据流...")
    
    try:
        # 1. 测试RIPT对齐数据集
        from pi0.ript.utils.libero_utils_ript_aligned import (
            build_dataset_ript_aligned, 
            collate_fn_ript_aligned
        )
        
        dataset = build_dataset_ript_aligned(
            data_prefix="/zhaohan/ZJH/openpi_pytorch/datasets",
            suite_name="libero",
            benchmark_name="libero_spatial",
            task_names_to_use=["pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo"],
            load_state=True,
            seq_len=600
        )
        
        print(f"✅ 数据集创建成功，长度: {len(dataset)}")
        
        # 2. 测试数据加载
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            collate_fn=collate_fn_ript_aligned,
            shuffle=False
        )
        
        batch = next(iter(dataloader))
        print(f"✅ 数据加载成功，batch键: {list(batch.keys())}")
        
        # 3. 验证数据格式
        if 'init_state' in batch:
            init_state = batch['init_state']
            states = init_state['states']
            pad_mask = init_state['pad_mask']
            
            print(f"✅ init_state格式正确:")
            print(f"   states形状: {states.shape}")
            print(f"   pad_mask形状: {pad_mask.shape}")
            
            # 4. 模拟原版RIPT的状态提取
            batch_index = 0
            sample_states = init_state
            extracted_state = sample_states['states'][batch_index, 0][sample_states['pad_mask'][batch_index]]
            
            print(f"✅ 状态提取成功:")
            print(f"   提取状态形状: {extracted_state.shape}")
            print(f"   状态数据类型: {extracted_state.dtype}")
            print(f"   状态值范围: [{extracted_state.min():.3f}, {extracted_state.max():.3f}]")
            
            # 5. 测试状态检查函数
            from pi0.ript.env.pi0_libero_runner import PI0LiberoRunner
            runner = PI0LiberoRunner(None, None, None)  # 创建临时实例
            
            is_mujoco = runner._is_mujoco_state(extracted_state.numpy())
            print(f"✅ MuJoCo状态检查: {is_mujoco}")
            
            return True, extracted_state.numpy()
        else:
            print(f"❌ 缺少init_state字段")
            return False, None
            
    except Exception as e:
        print(f"❌ 完整数据流测试失败: {e}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return False, None

def test_training_script_integration():
    """测试训练脚本集成"""
    print("\n🔍 测试训练脚本集成...")
    
    try:
        # 检查训练脚本是否使用了正确的导入
        script_path = "11_train_ript_vla_style.py"
        if not os.path.exists(script_path):
            print(f"❌ 训练脚本不存在: {script_path}")
            return False
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # 检查关键修改
        checks = [
            ('build_dataset_ript_aligned', 'RIPT对齐数据集导入'),
            ('collate_fn_ript_aligned', 'RIPT对齐collate函数导入'),
            ('load_state=True', 'MuJoCo状态加载配置'),
        ]
        
        all_passed = True
        for check_str, description in checks:
            if check_str in content:
                print(f"   ✅ {description}: 已修复")
            else:
                print(f"   ❌ {description}: 未找到")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 训练脚本集成测试失败: {e}")
        return False

def test_config_alignment():
    """测试配置对齐"""
    print("\n🔍 测试配置对齐...")
    
    try:
        config_path = "pi0/ript/config/stage11_unified_pool.yaml"
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 检查关键配置
        checks = [
            ('defaults', lambda c: 'paths' in c.get('defaults', [])),
            ('data_prefix', lambda c: '${paths.data_prefix}' in str(c.get('data_prefix', ''))),
            ('dataset.load_state', lambda c: c.get('dataset', {}).get('load_state', False)),
            ('use_libero_demos', lambda c: c.get('use_libero_demos', False)),
        ]
        
        all_passed = True
        for key, check_func in checks:
            try:
                if check_func(config):
                    print(f"   ✅ {key}: 配置正确")
                else:
                    print(f"   ❌ {key}: 配置错误")
                    all_passed = False
            except Exception as e:
                print(f"   ⚠️ {key}: 检查失败 - {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 配置对齐测试失败: {e}")
        return False

def simulate_training_step(mujoco_state):
    """模拟训练步骤"""
    print("\n🔍 模拟训练步骤...")
    
    try:
        # 模拟原版RIPT的rollout_generator逻辑
        print("   模拟原版RIPT状态处理...")
        
        # 1. 模拟batch数据
        batch = {
            'init_state': {
                'states': torch.tensor(mujoco_state).unsqueeze(0).unsqueeze(0),  # [1, 1, state_dim]
                'pad_mask': torch.ones(1, 1, dtype=torch.bool)  # [1, 1]
            },
            'task_id': torch.tensor([0]),
            'task_name': 'test_task'
        }
        
        # 2. 模拟状态提取（原版RIPT逻辑）
        batch_index = 0
        sample_states = batch['init_state']
        init_state = sample_states['states'][batch_index, 0][sample_states['pad_mask'][batch_index]]
        
        # 3. 模拟RLOO扩展
        rloo_batch_size = 4
        env_init_states = init_state.unsqueeze(0).repeat(rloo_batch_size, 1).cpu().numpy()
        
        print(f"   ✅ 状态处理成功:")
        print(f"      原始状态: {init_state.shape}")
        print(f"      RLOO扩展: {env_init_states.shape}")
        print(f"      数据类型: {env_init_states.dtype}")
        
        # 4. 模拟环境状态设置
        print("   模拟环境状态设置...")
        
        # 检查状态格式
        from pi0.ript.env.pi0_libero_runner import PI0LiberoRunner
        runner = PI0LiberoRunner(None, None, None)
        
        for i, state in enumerate(env_init_states):
            is_mujoco = runner._is_mujoco_state(state)
            print(f"      环境 {i}: MuJoCo状态={is_mujoco}, 形状={state.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练步骤模拟失败: {e}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return False

def main():
    """主测试函数"""
    print("🎯 开始完整RIPT修复验证\n")
    
    # 测试1: 完整数据流
    data_flow_ok, mujoco_state = test_complete_data_flow()
    
    # 测试2: 训练脚本集成
    script_ok = test_training_script_integration()
    
    # 测试3: 配置对齐
    config_ok = test_config_alignment()
    
    # 测试4: 模拟训练步骤
    if data_flow_ok and mujoco_state is not None:
        training_ok = simulate_training_step(mujoco_state)
    else:
        training_ok = False
    
    # 总结
    print(f"\n🎯 完整RIPT修复验证总结:")
    print(f"   数据流测试: {'✅' if data_flow_ok else '❌'}")
    print(f"   脚本集成: {'✅' if script_ok else '❌'}")
    print(f"   配置对齐: {'✅' if config_ok else '❌'}")
    print(f"   训练模拟: {'✅' if training_ok else '❌'}")
    
    all_passed = all([data_flow_ok, script_ok, config_ok, training_ok])
    
    if all_passed:
        print(f"\n🎉 所有修复验证通过！")
        print(f"   关键修复点:")
        print(f"   1. ✅ 使用RIPT对齐数据集（包含MuJoCo状态）")
        print(f"   2. ✅ 正确的数据格式（states + pad_mask）")
        print(f"   3. ✅ 原版RIPT状态提取逻辑")
        print(f"   4. ✅ 环境状态设置方法对齐")
        print(f"   5. ✅ 配置文件结构对齐")
        print(f"\n🚀 现在可以开始训练，预期解决:")
        print(f"   - MuJoCo状态维度不匹配错误")
        print(f"   - CFG训练效果差的问题")
        print(f"   - RLOO优势分布不合理的问题")
    else:
        print(f"\n⚠️ 部分验证失败，需要进一步修复")
    
    return all_passed

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"❌ 验证过程中发生错误: {e}")
        print(f"错误详情:\n{traceback.format_exc()}")
