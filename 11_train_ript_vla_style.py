#!/usr/bin/env python3
"""
Stage 11 RIPT-VLA风格简化版本
基于RIPT-VLA的直接架构模式，去除多余的抽象层

核心设计原则：
1. 直接在主循环中处理rollout收集和优化
2. 简化的组件架构，减少中间层
3. 直接使用SubprocVectorEnv进行并行
4. 模仿RIPT-VLA的成功模式


python 11_train_ript_vla_style.py --config_path pi0/ript/config/stage11_parallel_test.yaml 
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Dict, Any, Optional
import yaml
import traceback
import time
from tqdm import tqdm

# 修复tokenizers并行化警告和EGL错误
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["EGL_LOG_LEVEL"] = "fatal"  # 抑制EGL错误输出

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"=== Stage 11 RIPT-VLA风格简化训练 ===")
print(f"脚本位置: {current_file}")
print(f"项目根目录: {project_root}")
print()

# 导入配置管理
try:
    from omegaconf import OmegaConf, DictConfig
    OMEGACONF_AVAILABLE = True
    print("✓ OmegaConf配置管理已启用")
except ImportError:
    OMEGACONF_AVAILABLE = False
    print("⚠️ OmegaConf不可用，使用基础YAML加载")

# 导入核心模块
try:
    print("正在导入核心模块...")
    
    # PI0策略
    from pi0.modeling_pi0 import PI0Policy
    print("✓ PI0策略模块")
    
    # RIPT组件 - 只导入必需的
    from pi0.ript.reward_function import BinarySuccessReward
    from pi0.ript.algos.rl_optimizers.pi0_cfg_interface import PI0_CFG_Adapter
    print("✓ RIPT核心组件")
    
    # # 导入简化的环境runner
    # try:
    #     from pi0.ript.env.pi0_libero_runner_ript_vla import PI0LiberoRunner
    #     RIPT_VLA_RUNNER_AVAILABLE = True
    #     print("✓ RIPT-VLA Runner")
    # except ImportError as e:
    #     print(f"⚠️ RIPT-VLA runner导入失败: {e}")
    #     RIPT_VLA_RUNNER_AVAILABLE = False
    # 默认关闭RIPT-VLA Runner（若上方导入被注释或失败时保持为False）
    RIPT_VLA_RUNNER_AVAILABLE = False
        
    # 备用导入原有runner
    try:
        from pi0.ript.env.pi0_libero_runner import LIBEROEnvRunner
        ORIGINAL_RUNNER_AVAILABLE = True
        print("✓ 原有LIBEROEnvRunner")
    except ImportError as e:
        print("⚠️ 原有runner也不可用")
        ORIGINAL_RUNNER_AVAILABLE = False
    
    print("✓ 所有模块导入完成")
    
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

def load_config(config_path: str):
    """加载配置文件（优先使用OmegaConf，便于属性访问）"""
    print(f"正在加载配置文件: {config_path}")
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    if OMEGACONF_AVAILABLE:
        config = OmegaConf.load(config_path)
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # 简单的类型转换
    try:
        lr = config["algo"]["lr"]
        if isinstance(lr, str):
            if OMEGACONF_AVAILABLE:
                config.algo.lr = float(lr)
            else:
                config["algo"]["lr"] = float(lr)
    except Exception:
        pass
    
    print("✓ 配置文件加载成功")
    return config

def create_policy_and_optimizer(config: Dict[str, Any]):
    """创建策略和优化器（RIPT-VLA风格）"""
    print("正在加载PI0策略...")
    
    policy_path = config['policy_path']
    policy = PI0Policy.from_pretrained(policy_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = policy.to(device)
    print(f"✓ 策略加载成功，设备: {device}")
    
    # 直接创建优化器（像RIPT-VLA一样）
    print("正在创建优化器...")
    lr = config['algo'].get('lr', 1e-5)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=lr,
        weight_decay=0.01
    )
    print(f"✓ 优化器创建成功，学习率: {lr}")
    
    return policy, optimizer, device

def create_environment_runner(config: Dict[str, Any], policy):
    """创建环境runner（RIPT-VLA风格选择）"""
    use_ript_vla = config.get('features', {}).get('use_ript_vla_runner', False)
    
    print(f"🔍 Runner选择: use_ript_vla_runner = {use_ript_vla}")
    
    if use_ript_vla and RIPT_VLA_RUNNER_AVAILABLE:
        print("🚀 使用RIPT-VLA风格环境runner")
        
        runner = PI0LiberoRunner(
            policy=policy,
            benchmark_name=config['task']['benchmark_name'],
            rollouts_per_env=config['algo']['rloo_batch_size'],
            num_parallel_envs=config['task']['num_parallel_envs'],
            max_episode_length=config['task']['max_episode_length'],
            task_names_to_use=config['task'].get('task_names_to_use', []),
            rank=0
        )
        
    elif ORIGINAL_RUNNER_AVAILABLE:
        print("🔄 使用原有环境runner")
        
        # 确保norm_stats_path存在
        norm_stats_path = config.get('norm_stats_path')
        if not norm_stats_path:
            norm_stats_path = f"{config['policy_path']}/norm_stats.json"
        
        runner = LIBEROEnvRunner(
            policy=policy,
            benchmark_name=config['task']['benchmark_name'],
            rollouts_per_env=config['algo']['rloo_batch_size'],
            num_parallel_envs=config['task']['num_parallel_envs'],
            max_episode_length=config['task']['max_episode_length'],
            task_names_to_use=config['task'].get('task_names_to_use', []),
            norm_stats_path=norm_stats_path,
            config=config,
            rank=0,
            world_size=1
        )
    else:
        raise RuntimeError("❌ 没有可用的环境runner！")
    
    print("✓ 环境runner创建成功")
    return runner

def _dynamic_filter_rollouts(episodes: List[Dict], enable_dynamic_sampling: bool) -> List[Dict]:
    """按RIPT-VLA思路的最小动态采样：丢弃全0或全1成功率的批次"""
    if not enable_dynamic_sampling or not episodes:
        return episodes
    successes = [bool(ep.get('success', False)) for ep in episodes]
    if len(successes) > 0 and (all(successes) or not any(successes)):
        print(f"⚠️ 动态采样丢弃本批次 (uniform successes: {successes})")
        return []
    return episodes


def collect_rollouts_ript_vla_style(env_runner, task_name, num_rollouts, enable_dynamic_sampling: bool = False):
    """
    RIPT-VLA风格的rollout收集
    直接调用runner，无中间层
    """
    print(f"正在收集 {num_rollouts} 个rollouts...")
    
    try:
        # 获取任务的初始状态
        task_id = 0  # 简化处理，使用第一个任务
        if hasattr(env_runner, 'benchmark'):
            all_init_states = env_runner.benchmark.get_task_init_states(task_id)
        else:
            all_init_states = None
        
        # 直接调用环境runner的方法
        rollout_generator = env_runner.run_policy_in_env(
            env_name=task_name,
            all_init_states=all_init_states
        )
        
        # 收集所有rollouts
        collected_rollouts = []
        rollout_count = 0
        
        for success, total_reward, episode_data in rollout_generator:
            episode = {
                'success': success,
                'total_reward': total_reward,
                **episode_data
            }
            collected_rollouts.append(episode)
            rollout_count += 1
            
            if rollout_count >= num_rollouts:
                break
        
        # 最小动态采样过滤：丢弃全0或全1批次
        filtered = _dynamic_filter_rollouts(collected_rollouts, enable_dynamic_sampling)
        if not filtered:
            print("⚠️ 本批次被动态采样过滤，返回空集")
        else:
            print(f"✓ 成功收集了 {len(filtered)} 个rollouts (过滤后)")
        return filtered
        
    except Exception as e:
        print(f"❌ Rollout收集失败: {e}")
        traceback.print_exc()
        return []

def compute_advantages_simple(episodes: List[Dict]) -> torch.Tensor:
    """
    CFG原版优势计算逻辑 - 使用total_reward和Leave-One-Out baseline
    完全按照原版rl_optimizer_pi0_cfg.py的逻辑实现
    """
    if not episodes:
        return torch.tensor([])
    
    # 使用原版的total_reward，不做任何修改
    rewards = []
    for ep in episodes:
        reward = ep.get('total_reward', 0.0)
        rewards.append(float(reward))
    
    # 创建奖励张量
    rlhf_reward = torch.tensor(rewards, dtype=torch.float32)
    
    # 原版Leave-One-Out优势计算逻辑
    num_rollouts = len(episodes)
    rollouts_per_batch = len(episodes)  # 简化：批次大小等于总数
    
    if num_rollouts % rollouts_per_batch != 0 or rollouts_per_batch <= 1:
        # 简化的优势: reward - mean(reward)
        advantage = rlhf_reward - rlhf_reward.mean()
    else:
        num_batches = num_rollouts // rollouts_per_batch
        rlhf_reward_reshaped = rlhf_reward.reshape(num_batches, rollouts_per_batch)
        
        if rollouts_per_batch <= 1:
            advantage = rlhf_reward - rlhf_reward.mean()
        else:
            baseline = (rlhf_reward_reshaped.sum(1, keepdim=True) - rlhf_reward_reshaped) / (rollouts_per_batch - 1)
            advantage = rlhf_reward_reshaped - baseline
            advantage = advantage.flatten()
    
    print(f"🔧 CFG原版优势计算结果: {advantage}")
    return advantage

def update_policy_ript_vla_style(policy, optimizer, cfg_adapter, episodes, advantages, device):
    """
    RIPT-VLA风格的策略更新
    直接在主循环中处理，无复杂组件
    """
    if not episodes or len(advantages) == 0:
        print("⚠️ 没有有效数据进行策略更新")
        return 0.0
    
    print(f"正在更新策略（{len(episodes)} 个episodes）...")
    
    try:
        # 计算加权损失
        advantages = advantages.to(device)
        loss = cfg_adapter.compute_weighted_loss(episodes, advantages, device)
        
        # 梯度更新
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        
        optimizer.step()
        
        loss_value = loss.item()
        print(f"✓ 策略更新完成，损失: {loss_value:.6f}")
        
        return loss_value
        
    except Exception as e:
        print(f"❌ 策略更新失败: {e}")
        traceback.print_exc()
        return 0.0

def main_training_loop_ript_vla_style(config: Dict[str, Any]):
    """
    主训练循环（RIPT-VLA风格）
    直接在主函数中处理所有逻辑，减少抽象层
    """
    print("🚀 开始RIPT-VLA风格的训练循环")
    
    # 设置输出目录
    output_dir = Path(config['output_dir'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config.get('exp_name', 'ript_vla_style_train')
    output_dir = output_dir / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 创建策略和优化器
    policy, optimizer, device = create_policy_and_optimizer(config)
    
    # 创建CFG适配器（必需，用于损失计算）
    cfg_adapter = PI0_CFG_Adapter(
        policy=policy,
        norm_stats_path=f"{config['policy_path']}/norm_stats.json"
    )
    
    # 创建环境runner
    env_runner = create_environment_runner(config, policy)
    
    # 训练配置
    num_train_steps = config['training']['num_train_steps']
    # 与2_test_pi0_on_libero.py对齐：使用libero_goal基准默认task_id=1
    # 若YAML中明确给了task_names_to_use，则仍然使用第一个名称做显示，不影响环境内部task_id选择
    task_name = config['task'].get('task_names_to_use', ['libero_goal_default'])[0]
    rloo_batch_size = config['algo']['rloo_batch_size']
    
    print(f"\n开始训练循环:")
    print(f"  训练步数: {num_train_steps}")
    print(f"  任务: {task_name}")
    print(f"  批次大小: {rloo_batch_size}")
    print()
    
    all_training_metrics = []
    
    # 🔥 主训练循环 - RIPT-VLA风格
    for step in range(num_train_steps):
        step_start_time = time.time()
        
        print(f"=== 训练步骤 {step + 1}/{num_train_steps} ===")
        
        # 1. 收集rollouts（直接调用，无中间层）
        episodes = collect_rollouts_ript_vla_style(
            env_runner, task_name, rloo_batch_size,
            enable_dynamic_sampling=config['algo'].get('enable_dynamic_sampling', False)
        )
        
        if not episodes:
            print("⚠️ 未收集到有效episodes，跳过此步")
            continue
        
        # 2. 计算优势（简化计算）
        advantages = compute_advantages_simple(episodes)
        
        # 3. 更新策略（直接更新，无复杂组件）
        loss = update_policy_ript_vla_style(
            policy, optimizer, cfg_adapter, episodes, advantages, device
        )
        
        # 4. 记录指标
        avg_reward = np.mean([ep['total_reward'] for ep in episodes])
        success_rate = np.mean([ep['success'] for ep in episodes])
        step_time = time.time() - step_start_time
        
        step_metrics = {
            'step': step + 1,
            'num_episodes': len(episodes),
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'loss': loss,
            'step_time': step_time
        }
        all_training_metrics.append(step_metrics)
        
        # 5. 输出结果
        print(f"✓ 步骤 {step + 1} 完成:")
        print(f"  Episodes: {len(episodes)}")
        print(f"  平均奖励: {avg_reward:.4f}")
        print(f"  成功率: {success_rate:.2%}")
        print(f"  损失: {loss:.6f}")
        print(f"  耗时: {step_time:.2f}秒")
        
        # 6. 保存检查点
        if (step + 1) % config['training'].get('save_freq', 10) == 0:
            # 轻量权重（仅模型，便于部署与占用小）
            weights_path = output_dir / f"weights_step_{step + 1}.pt"
            torch.save({
                'step': step + 1,
                'policy_state_dict': policy.state_dict(),
                'config': config,
                'training_metrics': all_training_metrics,
            }, weights_path)
            print(f"✓ 轻量权重已保存: {weights_path}")

            # 可选：按较低频率保存含优化器的完整检查点，便于恢复训练
            save_opt_every = config.get('training', {}).get('save_optimizer_freq', None)
            if save_opt_every and ((step + 1) % int(save_opt_every) == 0):
                checkpoint_path = output_dir / f"checkpoint_step_{step + 1}.pt"
                torch.save({
                    'step': step + 1,
                    'policy_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'training_metrics': all_training_metrics,
                }, checkpoint_path)
                print(f"✓ 完整检查点已保存: {checkpoint_path}")
    
    # 保存最终结果
    final_results_path = output_dir / "final_training_results.json"
    # 将 OmegaConf 转为原生 dict 以便 JSON 序列化
    if OMEGACONF_AVAILABLE and isinstance(config, DictConfig):
        serializable_config = OmegaConf.to_container(config, resolve=True)
    else:
        serializable_config = config
    with open(final_results_path, 'w') as f:
        json.dump({
            'config': serializable_config,
            'training_metrics': all_training_metrics,
            'total_steps': len(all_training_metrics)
        }, f, indent=2)
    
    # 最终轻量权重（仅模型）
    final_weights_path = output_dir / "final_weights.pt"
    torch.save({
        'step': len(all_training_metrics),
        'policy_state_dict': policy.state_dict(),
        'config': config,
        'training_metrics': all_training_metrics,
    }, final_weights_path)
    print(f"✓ 最终轻量权重已保存: {final_weights_path}")

    # 可选：保存最终完整检查点（含优化器）便于恢复训练
    if config.get('training', {}).get('save_optimizer_final', False):
        final_checkpoint_path = output_dir / "final_checkpoint.pt"
        torch.save({
            'step': len(all_training_metrics),
            'policy_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'training_metrics': all_training_metrics,
        }, final_checkpoint_path)
        print(f"✓ 最终完整检查点已保存: {final_checkpoint_path}")

    print(f"\n🎉 RIPT-VLA风格训练完成!")
    print(f"📊 最终结果已保存: {final_results_path}")
    print(f"✨ 使用了简化的直接架构，减少了抽象层复杂度")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Stage 11 RIPT-VLA风格简化训练")
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True,
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config_path)
        
        # 显示配置
        print("\n====== 使用配置 ======")
        if OMEGACONF_AVAILABLE:
            print(OmegaConf.to_yaml(config))
        else:
            print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
        print("====================\n")
        
        # 开始RIPT-VLA风格的训练
        main_training_loop_ript_vla_style(config)
        
    except KeyboardInterrupt:
        print("\n⚠️ 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()