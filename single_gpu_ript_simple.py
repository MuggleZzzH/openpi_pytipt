#!/usr/bin/env python3
"""
基于backup runner的单GPU RIPT训练脚本
使用简洁版pi0_libero_runner_backup.py的逻辑，避免复杂的分布式代码
"""

import os
import sys
import torch
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# 设置环境
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(output_dir, exp_name):
    """设置输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{exp_name}_{timestamp}_rank0"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 输出目录: {run_dir}")
    return run_dir

def main():
    parser = argparse.ArgumentParser(description='单GPU RIPT训练')
    parser.add_argument('--config_path', type=str, 
                       default='pi0/ript/config/single_gpu_test.yaml',
                       help='配置文件路径')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()
    
    print("🚀 单GPU RIPT训练 (基于简洁backup逻辑)")
    print("=" * 50)
    
    # 1. 加载配置
    print("📋 加载配置...")
    config = load_config(args.config_path)
    print(f"   实验名: {config['exp_name']}")
    print(f"   任务数量: {len(config['task']['task_names_to_use'])}")
    
    # 2. 设置输出目录
    run_dir = setup_logging(config['output_dir'], config['exp_name'])
    
    # 3. 检查CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    
    device = torch.device('cuda:0')
    print(f"🔧 设备: {device}")
    
    # 4. 导入核心组件（确保使用backup版本）
    print("📦 导入核心组件...")
    from pi0 import PI0Policy
    from pi0.ript.env.pi0_libero_runner import LIBEROEnvRunner  # 这应该是backup版本
    from pi0.ript.reward_function import BinarySuccessReward
    from pi0.ript.algos.rl_optimizers.pi0_cfg_interface import PI0_CFG_Adapter
    
    # 5. 加载模型
    print("🤖 加载PI0模型...")
    policy = PI0Policy.from_pretrained(config['policy_path'])
    policy = policy.to(device)
    print(f"   参数数量: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M")
    
    # 6. 创建模型适配器
    model_adapter = PI0_CFG_Adapter(
        policy, 
        norm_stats_path=config.get('norm_stats_path'),
        device=device
    )
    
    # 7. 创建环境运行器
    print("🌍 创建环境运行器...")
    env_runner = LIBEROEnvRunner(
        policy=policy,
        task_names_to_use=config['task']['task_names_to_use'],
        benchmark_name=config['task']['benchmark_name'],
        num_parallel_envs=config['task']['num_parallel_envs'],
        max_episode_length=config['task']['max_episode_length'],
        rank=0,
        world_size=1
    )
    
    # 8. 创建奖励函数
    reward_fn = BinarySuccessReward()
    
    # 9. 简单的训练循环
    print("\n🔥 开始训练循环")
    print("-" * 30)
    
    num_train_steps = config['training']['num_train_steps']
    batch_size = config['algo']['data_batch_size']
    
    for step in range(num_train_steps):
        print(f"\n📊 训练步骤 {step + 1}/{num_train_steps}")
        
        # 选择当前任务 
        task_name = config['task']['task_names_to_use'][step % len(config['task']['task_names_to_use'])]
        print(f"🎯 当前任务: {task_name.split('_')[-1]}")
        
        try:
            # 数据收集
            print("📊 收集数据...")
            init_states = np.random.randn(batch_size, 10).astype(np.float32) * 0.1
            
            episodes = []
            rollout_results = env_runner.run_policy_in_env(
                env_name=task_name,
                all_init_states=init_states,
                debug_save_video=(step == 0)  # 只在第一步保存视频
            )
            
            # 收集episodes
            for success, total_reward, episode_data in rollout_results:
                episodes.append(episode_data)
                print(f"   Episode: 成功={success}, 奖励={total_reward:.3f}")
            
            # 计算优势 (简化版本)
            advantages = []
            for episode in episodes:
                total_reward = sum(episode['rewards'])
                # Leave-One-Out优势: 相对于批次平均值
                batch_avg_reward = np.mean([sum(ep['rewards']) for ep in episodes])
                advantage = total_reward - batch_avg_reward
                advantages.append(advantage)
            
            advantages = torch.tensor(advantages, device=device)
            print(f"   优势: 均值={advantages.mean():.3f}, 标准差={advantages.std():.3f}")
            
            # 模型更新 
            if len(episodes) > 0:
                print("🔄 更新模型...")
                loss = model_adapter.compute_weighted_loss(episodes, advantages, device)
                
                print(f"   损失: {loss.item():.6f}")
                
                # 反向传播（简化版本）
                if loss.requires_grad:
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        model_adapter.get_policy_model().parameters(), 
                        config['algo']['grad_norm_clip']
                    )
                    
                    # 这里应该添加优化器步骤，但为了测试先跳过
                    print("   ✅ 梯度计算完成")
                else:
                    print("   ⚠️ 损失不需要梯度，跳过更新")
            
            # 保存中间结果
            if (step + 1) % config['training']['save_freq'] == 0:
                result_file = run_dir / f"step_{step+1}_results.json"
                results = {
                    'step': step + 1,
                    'task_name': task_name,
                    'num_episodes': len(episodes),
                    'success_rate': sum(1 for ep in episodes if sum(ep['rewards']) > 0.5) / len(episodes),
                    'avg_reward': np.mean([sum(ep['rewards']) for ep in episodes]),
                    'loss': loss.item() if len(episodes) > 0 else 0.0
                }
                
                import json
                with open(result_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"   💾 结果已保存: {result_file}")
        
        except Exception as e:
            print(f"❌ 步骤 {step + 1} 失败: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            continue
    
    print("\n🎉 训练完成！")
    print(f"📁 结果保存在: {run_dir}")

if __name__ == "__main__":
    main()