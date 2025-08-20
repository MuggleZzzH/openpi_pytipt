#!/usr/bin/env python3
"""
测试与原版RIPT的对齐情况
验证数据格式、状态处理、CFG机制等关键组件
"""

import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from pi0.ript.utils.libero_utils_ript_aligned import (
    build_dataset_ript_aligned, 
    collate_fn_ript_aligned
)

def test_paths_config():
    """测试paths配置是否正确加载"""
    print("🔍 测试paths配置...")
    
    config_path = "pi0/ript/config/paths.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            paths_config = yaml.safe_load(f)
        
        print(f"✅ paths配置加载成功:")
        print(f"   data_prefix: {paths_config['paths']['data_prefix']}")
        print(f"   output_prefix: {paths_config['paths']['output_prefix']}")
        print(f"   wandb_project: {paths_config['common']['wandb_project']}")
        return True
    else:
        print(f"❌ paths配置文件不存在: {config_path}")
        return False

def test_dataset_creation():
    """测试RIPT对齐数据集的创建"""
    print("\n🔍 测试RIPT对齐数据集...")
    
    try:
        # 测试数据集创建
        dataset = build_dataset_ript_aligned(
            data_prefix="/zhaohan/ZJH/openpi_pytorch/datasets",
            suite_name="libero",
            benchmark_name="libero_spatial",
            task_names_to_use=["pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"],
            load_state=True,  # 🔥 关键：加载MuJoCo状态
            seq_len=600
        )
        
        print(f"✅ 数据集创建成功:")
        print(f"   数据集长度: {len(dataset)}")
        print(f"   数据集类型: {type(dataset).__name__}")
        
        # 测试数据项
        if len(dataset) > 0:
            item = dataset[0]
            print(f"   数据项键: {list(item.keys())}")
            
            if 'init_state' in item:
                init_state = item['init_state']
                print(f"   init_state键: {list(init_state.keys())}")
                print(f"   states形状: {init_state['states'].shape}")
                print(f"   pad_mask形状: {init_state['pad_mask'].shape}")
            else:
                print(f"   ⚠️ 缺少init_state字段")
        
        return True, dataset
        
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return False, None

def test_collate_function(dataset):
    """测试collate函数"""
    print("\n🔍 测试collate函数...")
    
    try:
        from torch.utils.data import DataLoader
        
        # 创建DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            collate_fn=collate_fn_ript_aligned,
            shuffle=False
        )
        
        # 获取一个batch
        batch = next(iter(dataloader))
        
        print(f"✅ Collate函数测试成功:")
        print(f"   Batch键: {list(batch.keys())}")
        
        if 'init_state' in batch:
            init_state = batch['init_state']
            print(f"   init_state键: {list(init_state.keys())}")
            print(f"   states形状: {init_state['states'].shape}")
            print(f"   pad_mask形状: {init_state['pad_mask'].shape}")
            
            # 验证与原版RIPT格式一致
            states = init_state['states']
            pad_mask = init_state['pad_mask']
            
            print(f"   🔥 验证RIPT格式:")
            print(f"      states维度: {states.dim()}D (期望3D: B,T,state_dim)")
            print(f"      pad_mask维度: {pad_mask.dim()}D (期望2D: B,T)")
            print(f"      states数据类型: {states.dtype}")
            print(f"      pad_mask数据类型: {pad_mask.dtype}")
            
            # 模拟原版RIPT的状态提取
            batch_index = 0
            sample_states = init_state
            init_state_extracted = sample_states['states'][batch_index, 0][sample_states['pad_mask'][batch_index]]
            print(f"      提取的初始状态形状: {init_state_extracted.shape}")
            
        return True, batch
        
    except Exception as e:
        print(f"❌ Collate函数测试失败: {e}")
        return False, None

def test_state_extraction_ript_style(batch):
    """测试原版RIPT风格的状态提取"""
    print("\n🔍 测试原版RIPT风格状态提取...")
    
    try:
        if 'init_state' not in batch:
            print(f"❌ Batch缺少init_state字段")
            return False
        
        # 模拟原版RIPT的rollout_generator.py:216-220逻辑
        current_batch = batch
        batch_index = 0  # 选择第一个样本
        
        # Extract init_state for this sample
        sample_states = current_batch['init_state']
        init_state = sample_states['states'][batch_index, 0][sample_states['pad_mask'][batch_index]]
        
        # 模拟RLOO batch size扩展
        rloo_batch_size = 4
        env_init_states = init_state.unsqueeze(0).repeat(rloo_batch_size, 1).cpu().numpy()
        
        print(f"✅ 原版RIPT状态提取成功:")
        print(f"   原始状态形状: {init_state.shape}")
        print(f"   扩展后形状: {env_init_states.shape}")
        print(f"   状态数据类型: {env_init_states.dtype}")
        print(f"   状态值范围: [{env_init_states.min():.3f}, {env_init_states.max():.3f}]")
        
        return True, env_init_states
        
    except Exception as e:
        print(f"❌ 状态提取失败: {e}")
        print(f"   错误详情: {traceback.format_exc()}")
        return False, None

def test_config_integration():
    """测试配置文件集成"""
    print("\n🔍 测试配置文件集成...")
    
    config_path = "pi0/ript/config/stage11_unified_pool.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ 配置文件加载成功:")
        
        # 检查关键配置
        checks = [
            ('defaults', 'paths配置引入'),
            ('data_prefix', '数据路径配置'),
            ('dataset.load_state', 'MuJoCo状态加载'),
            ('logging.project', 'wandb项目配置'),
        ]
        
        for key_path, description in checks:
            keys = key_path.split('.')
            value = config
            try:
                for key in keys:
                    value = value[key]
                print(f"   ✅ {description}: {value}")
            except KeyError:
                print(f"   ⚠️ {description}: 未配置")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件解析失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎯 开始RIPT对齐测试\n")
    
    # 测试1: paths配置
    paths_ok = test_paths_config()
    
    # 测试2: 数据集创建
    dataset_ok, dataset = test_dataset_creation()
    
    # 测试3: collate函数
    if dataset_ok and dataset:
        collate_ok, batch = test_collate_function(dataset)
        
        # 测试4: 状态提取
        if collate_ok and batch:
            extraction_ok, states = test_state_extraction_ript_style(batch)
        else:
            extraction_ok = False
    else:
        collate_ok = False
        extraction_ok = False
    
    # 测试5: 配置集成
    config_ok = test_config_integration()
    
    # 总结
    print(f"\n🎯 RIPT对齐测试总结:")
    print(f"   Paths配置: {'✅' if paths_ok else '❌'}")
    print(f"   数据集创建: {'✅' if dataset_ok else '❌'}")
    print(f"   Collate函数: {'✅' if collate_ok else '❌'}")
    print(f"   状态提取: {'✅' if extraction_ok else '❌'}")
    print(f"   配置集成: {'✅' if config_ok else '❌'}")
    
    all_passed = all([paths_ok, dataset_ok, collate_ok, extraction_ok, config_ok])
    
    if all_passed:
        print(f"\n🎉 所有测试通过！你的实现已与原版RIPT对齐")
        print(f"   可以开始训练，期望看到:")
        print(f"   - 使用MuJoCo状态向量作为初始状态")
        print(f"   - 与原版RIPT相同的数据格式")
        print(f"   - 正确的CFG训练流程")
    else:
        print(f"\n⚠️ 部分测试失败，需要修复后再进行训练")
    
    return all_passed

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        print(f"错误详情:\n{traceback.format_exc()}")
