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
from collections import deque
import hashlib

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

def get_config_value(config, key, default=None, sources=['algo', 'features']):
    """
    统一配置读取函数，支持多级回退
    
    Args:
        config: 配置字典或对象
        key: 配置键名
        default: 默认值
        sources: 搜索的配置节点列表，按优先级排序
    
    Returns:
        配置值
    """
    # 处理OmegaConf和字典两种情况
    if hasattr(config, 'get'):
        # 字典式访问
        for source in sources:
            if source in config and config[source] is not None and key in config[source]:
                return config[source][key]
        # 根节点访问
        return config.get(key, default)
    else:
        # 属性式访问（针对一些旧的config对象）
        for source in sources:
            if hasattr(config, source):
                source_obj = getattr(config, source)
                if source_obj is not None and hasattr(source_obj, key):
                    return getattr(source_obj, key)
        # 根节点访问
        return getattr(config, key, default)

def set_config_value(config, key, value, target_source='algo'):
    """
    统一配置写回函数
    
    Args:
        config: 配置字典或对象
        key: 配置键名
        value: 要设置的值
        target_source: 目标配置节点
    """
    if hasattr(config, 'get'):
        # 字典式访问
        if target_source not in config:
            config[target_source] = {}
        config[target_source][key] = value
    else:
        # 属性式访问
        if not hasattr(config, target_source):
            setattr(config, target_source, type('Config', (), {})())
        target_obj = getattr(config, target_source)
        setattr(target_obj, key, value)

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
    
    # 🔥 关键修复：强制启用CFG（解决原始checkpoint兼容性问题）
    print("🔧 强制启用CFG功能...")
    policy.model.cfg_enabled = True
    if hasattr(policy, 'config'):
        policy.config.cfg_enabled = True
    print("✅ CFG已启用，训练和推理都将使用CFG分支")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = policy.to(device)
    print(f"✓ 策略加载成功，设备: {device}")
    
    # 🔥 修复：只训练专家头部，冻结PaliGemma前缀（提升稳定性）
    print("🔧 配置训练参数范围...")
    
    # 1. 冻结PaliGemma前缀
    for p in policy.model.paligemma_with_expert.parameters():
        p.requires_grad = False
    
    # 2. 只收集需要训练的参数
    trainable_params = []
    trainable_params += list(policy.model.action_in_proj.parameters())
    trainable_params += list(policy.model.action_time_mlp_in.parameters())
    trainable_params += list(policy.model.action_time_mlp_out.parameters())
    trainable_params += list(policy.model.action_out_proj.parameters())
    trainable_params += list(policy.model.state_proj.parameters())
    
    # 3. CFG embedding参数
    if hasattr(policy.model, "cfg_emb"):
        trainable_params += list(policy.model.cfg_emb.parameters())
        print("✅ CFG embedding参数已加入训练")
    
    # 4. 创建优化器
    print("正在创建优化器...")
    lr = config['algo'].get('lr', 1e-5)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    
    total_params = sum(p.numel() for p in trainable_params)
    print(f"✓ 优化器创建成功，学习率: {lr}")
    print(f"🎯 只训练专家头部，参数数量: {total_params:,}")
    
    return policy, optimizer, device

def compute_init_state_hash_from_obs(episode_data: Dict[str, Any]) -> str:
    """
    从原始观测计算初始状态哈希（备用方法）
    
    Args:
        episode_data: episode数据，包含observations
    
    Returns:
        初始状态的SHA256哈希字符串
    """
    try:
        # 提取第一个观测作为初始状态
        if 'observations' in episode_data and len(episode_data['observations']) > 0:
            first_obs = episode_data['observations'][0]
            
            # 提取关键状态信息用于哈希
            if isinstance(first_obs, dict):
                # 提取 end-effector 位置和姿态
                state_components = []
                if "robot0_eef_pos" in first_obs:
                    state_components.append(np.array(first_obs["robot0_eef_pos"], dtype=np.float32))
                if "robot0_eef_quat" in first_obs:
                    state_components.append(np.array(first_obs["robot0_eef_quat"], dtype=np.float32))
                if "robot0_gripper_qpos" in first_obs:
                    state_components.append(np.array(first_obs["robot0_gripper_qpos"], dtype=np.float32))
                
                if state_components:
                    # 合并所有状态组件
                    combined_state = np.concatenate(state_components).astype(np.float32)
                    # 计算哈希
                    return hashlib.sha256(combined_state.tobytes()).hexdigest()
        
        # 如果无法提取状态，返回默认哈希
        return "default_hash"
        
    except Exception as e:
        print(f"⚠️ 从观测计算初始状态哈希失败: {e}")
        return "error_hash"

def compute_init_state_hash(batch_states: torch.Tensor, batch_pad_mask: torch.Tensor, batch_idx: int) -> str:
    """
    计算初始状态的哈希值（完全对标 ript-vla_ori 的 compute_hash_from_state）
    
    Args:
        batch_states: (B, T, state_dim) 批量状态序列
        batch_pad_mask: (B, T) 批量有效掩码，True表示有效
        batch_idx: 在批次中的索引
    
    Returns:
        初始状态的SHA256哈希字符串
    """
    try:
        # 🔥 完全对标 RIPT: state_data = state['states'][bidx][0]
        if batch_idx < batch_states.shape[0] and batch_states.shape[1] > 0:
            # 取指定batch索引的第一个时间步状态
            state_data = batch_states[batch_idx, 0]  # (state_dim,)
            
            # 取对应的掩码
            if batch_pad_mask is not None and batch_idx < batch_pad_mask.shape[0]:
                state_mask = batch_pad_mask[batch_idx, 0]  # 第一个时间步的掩码
                
                # 🔥 完全对标 RIPT: state_data[state_mask].cpu().numpy().tobytes()
                if state_mask.item():  # 如果第一个时间步是有效的
                    state_bytes = state_data.cpu().numpy().astype(np.float32).tobytes()
                    return hashlib.sha256(state_bytes).hexdigest()
            else:
                # 如果没有掩码，直接使用状态数据
                state_bytes = state_data.cpu().numpy().astype(np.float32).tobytes()
                return hashlib.sha256(state_bytes).hexdigest()
        
        return "default_hash"
        
    except Exception as e:
        print(f"⚠️ 计算初始状态哈希失败: {e}")
        return "error_hash"

def load_rollout_stats(stats_path: str) -> Dict[str, List[int]]:
    """加载 rollout 统计信息"""
    if Path(stats_path).exists():
        try:
            with open(stats_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 加载 rollout_stats 失败: {e}")
    return {}

def save_rollout_stats(stats: Dict[str, List[int]], stats_path: str):
    """保存 rollout 统计信息"""
    try:
        Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"⚠️ 保存 rollout_stats 失败: {e}")

def should_skip_init_state(init_hash: str, rollout_stats: Dict[str, List[int]], 
                          rloo_batch_size: int, rollout_skip_cnt: Dict[str, int],
                          rollout_skip_threshold: int = 10) -> bool:
    """
    判断是否应该跳过某个初始状态（对标 ript-vla_ori 的跳过逻辑）
    
    Args:
        init_hash: 初始状态哈希
        rollout_stats: {init_hash: [success_list]} 历史记录
        rloo_batch_size: RLOO批次大小
        rollout_skip_cnt: 跳过计数器
        rollout_skip_threshold: 跳过阈值
    
    Returns:
        是否应该跳过
    """
    if init_hash in rollout_stats:
        recent_successes = rollout_stats[init_hash][-rloo_batch_size:]
        
        # 检查是否连续全成功或全失败
        if len(recent_successes) >= rloo_batch_size:
            if all(s == 1 for s in recent_successes):
                print(f"🔄 跳过样本: init_hash={init_hash[:8]}... (连续 {rloo_batch_size} 次全成功)")
                return True
            elif all(s == 0 for s in recent_successes):
                print(f"🔄 跳过样本: init_hash={init_hash[:8]}... (连续 {rloo_batch_size} 次全失败)")
                return True
    
    return False

def update_rollout_stats(init_hash: str, success: bool, rollout_stats: Dict[str, List[int]],
                        rollout_skip_cnt: Dict[str, int], max_history_length: int = 100):
    """更新 rollout 统计信息"""
    if init_hash not in rollout_stats:
        rollout_stats[init_hash] = []
        rollout_skip_cnt[init_hash] = 0
    
    # 添加新结果
    rollout_stats[init_hash].append(1 if success else 0)
    
    # 限制历史长度
    if len(rollout_stats[init_hash]) > max_history_length:
        rollout_stats[init_hash] = rollout_stats[init_hash][-max_history_length:]

def update_rollout_stats_with_correct_hash(episodes: List[Dict], batch_states: torch.Tensor, 
                                         batch_pad_mask: torch.Tensor, owner_indices: List[int],
                                         rollout_stats: Dict[str, List[int]], 
                                         rollout_skip_cnt: Dict[str, int]):
    """
    使用正确的状态哈希更新 rollout 统计信息（在 CFG adapter 处理后调用）
    
    Args:
        episodes: 原始 episodes
        batch_states: CFG adapter 处理后的状态 (B, state_dim)，注意这里是单个时间步
        batch_pad_mask: 对应的掩码，如果可用的话
        owner_indices: 窗口到 episode 的映射
        rollout_stats: rollout 统计字典
        rollout_skip_cnt: 跳过计数字典
    """
    if batch_states is None or len(episodes) == 0:
        return
    
    try:
        # 为每个 episode 计算正确的哈希
        episode_hash_map = {}  # episode_idx -> correct_hash
        
        # 根据 owner_indices 映射，为每个 episode 找到对应的状态
        for window_idx, episode_idx in enumerate(owner_indices):
            if window_idx < batch_states.shape[0] and episode_idx < len(episodes):
                # 使用窗口对应的状态计算哈希
                state_data = batch_states[window_idx]  # (state_dim,)
                
                # 对于单个状态（不是序列），直接计算哈希
                state_bytes = state_data.cpu().numpy().astype(np.float32).tobytes()
                correct_hash = hashlib.sha256(state_bytes).hexdigest()
                
                episode_hash_map[episode_idx] = correct_hash
        
        # 更新统计信息
        for episode_idx, episode in enumerate(episodes):
            if episode_idx in episode_hash_map:
                correct_hash = episode_hash_map[episode_idx]
                success = episode.get('success', False)
                
                # 如果之前有临时哈希，需要迁移统计
                temp_hash = episode.get('temp_init_hash')
                if temp_hash and temp_hash != correct_hash and temp_hash in rollout_stats:
                    # 迁移统计数据到正确的哈希
                    if correct_hash not in rollout_stats:
                        rollout_stats[correct_hash] = rollout_stats[temp_hash].copy()
                        rollout_skip_cnt[correct_hash] = rollout_skip_cnt.get(temp_hash, 0)
                    else:
                        # 合并统计数据
                        rollout_stats[correct_hash].extend(rollout_stats[temp_hash])
                    
                    # 清理临时哈希
                    del rollout_stats[temp_hash]
                    if temp_hash in rollout_skip_cnt:
                        del rollout_skip_cnt[temp_hash]
                    
                    # 由于已经迁移了统计，不需要再次更新
                    print(f"🔄 迁移统计: {temp_hash[:8]}... → {correct_hash[:8]}...")
                else:
                    # 🔥 关键修复：只有在没有迁移的情况下才更新统计，避免重复计数
                    # 检查是否已经用temp_hash更新过统计（在分组路径中可能发生）
                    if not temp_hash or temp_hash == correct_hash:
                        update_rollout_stats(correct_hash, success, rollout_stats, rollout_skip_cnt)
                    # 如果temp_hash存在且不等于correct_hash，说明之前没有更新过统计，现在更新
                    elif temp_hash and temp_hash not in rollout_stats:
                        update_rollout_stats(correct_hash, success, rollout_stats, rollout_skip_cnt)
                
                # 记录正确的哈希到 episode（用于调试）
                episode['correct_init_hash'] = correct_hash
                
    except Exception as e:
        print(f"⚠️ 更新正确哈希统计时出错: {e}")
        import traceback
        traceback.print_exc()

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

def _ript_dynamic_sampling(episodes: List[Dict], enable_dynamic_sampling: bool = True) -> List[Dict]:
    """
    🔥 RIPT原版动态采样：简单的全成功/全失败检查
    
    Args:
        episodes: 收集的episodes（通常是8个来自相同init_hash的样本）
        enable_dynamic_sampling: 是否启用动态采样
    
    Returns:
        过滤后的episodes（如果全成功/全失败则返回空列表）
    """
    if not enable_dynamic_sampling or not episodes:
        return episodes
    
    # 提取成功/失败状态
    successes = [bool(ep.get('success', False)) for ep in episodes]
    
    # 🔥 RIPT核心逻辑：检查是否全成功或全失败
    if len(set(successes)) == 1:  # 只有一种结果（全0或全1）
        uniform_result = "全成功" if successes[0] else "全失败"
        print(f"⚠️ RIPT动态采样拒绝: {uniform_result} ({len(episodes)}个样本)，缺乏学习价值")
        return []  # 丢弃整组
    
    # 混合结果，有学习价值
    success_count = sum(successes)
    print(f"✅ RIPT动态采样接受: {success_count}/{len(episodes)} 成功，结果混合有学习价值")
    return episodes


def _dynamic_filter_rollouts_legacy(episodes: List[Dict], dynamic_sampling_config: Dict, 
                             recent_success_rates: deque) -> List[Dict]:
    """
    🔄 原来的复杂动态采样（保留作为备用，但不推荐使用）
    升级版动态采样：区间策略 + 平滑窗口机制
    """
    if not dynamic_sampling_config.get('enabled', False) or not episodes:
        return episodes
    
    # 计算当前批次成功率
    successes = [bool(ep.get('success', False)) for ep in episodes]
    current_success_rate = np.mean(successes) if successes else 0.0
    
    p_min = dynamic_sampling_config.get('p_min', 0.1)
    p_max = dynamic_sampling_config.get('p_max', 0.9)
    
    # 第一层过滤：当前批次区间检查
    if current_success_rate < p_min or current_success_rate > p_max:
        print(f"⚠️ 复杂动态采样第一层拒绝: success_rate={current_success_rate:.3f} 不在 [{p_min}, {p_max}] 区间内")
        return []
    
    # 第二层过滤：平滑窗口检查（降低抖动）
    recent_success_rates.append(current_success_rate)
    if len(recent_success_rates) >= 2:  # 至少有2个样本才进行窗口检查
        window_avg = np.mean(recent_success_rates)
        if window_avg < p_min or window_avg > p_max:
            print(f"⚠️ 复杂动态采样第二层拒绝: 窗口平均={window_avg:.3f} 不在区间内 (窗口大小={len(recent_success_rates)})")
            return []
    
    print(f"✅ 复杂动态采样通过: 当前={current_success_rate:.3f}, 窗口平均={np.mean(recent_success_rates):.3f}")
    return episodes


def collect_rollouts_ript_vla_style_grouped(env_runner, task_name, demo_batch_size, rloo_batch_size,
                                           enable_ript_dynamic_sampling: bool = True,
                                           rollout_goal_per_step: int = None,
                                           rollout_stats: Dict[str, List[int]] = None,
                                           rollout_skip_cnt: Dict[str, int] = None):
    """
    🔥 完全对齐 RIPT 原版：按初始状态分组收集，确保 RLOO 优势计算的正确性
    
    核心逻辑：
    1. 确定需要收集的初始状态数量 (demo_batch_size)
    2. 对每个初始状态，并行收集 rloo_batch_size 个样本
    3. 确保每组 rloo_batch_size 个样本来自相同的 init_hash
    
    Args:
        demo_batch_size: 需要多少个不同的初始状态 (每个状态收集 rloo_batch_size 个样本)
        rloo_batch_size: 每个初始状态收集多少个样本
    
    Returns:
        episodes: 总共 demo_batch_size * rloo_batch_size 个样本，按组织排列
        valid_mask: 对应的有效性掩码
    """
    total_target_samples = demo_batch_size * rloo_batch_size
    print(f"🎯 RIPT风格收集：{demo_batch_size} 个初始状态 × {rloo_batch_size} 个样本 = {total_target_samples} 个样本")
    
    # 如果设置了全局样本目标，显示当前进度
    if rollout_goal_per_step and hasattr(env_runner, 'file_counter') and env_runner.file_counter:
        current_count = env_runner.file_counter.get()
        print(f"📊 当前全局样本数: {current_count}/{rollout_goal_per_step}")
    
    all_episodes = []
    collected_groups = 0
    
    try:
        # 每个训练步开始前重置计数器（按步早停，而非累计）
        if rollout_goal_per_step and hasattr(env_runner, 'file_counter') and env_runner.file_counter:
            try:
                env_runner.file_counter.set(0)
            except Exception:
                pass

        # 获取任务的初始状态池
        task_id = 0  # 简化处理，使用第一个任务
        if hasattr(env_runner, 'benchmark'):
            all_init_states = env_runner.benchmark.get_task_init_states(task_id)
            print(f"📋 可用初始状态数量: {len(all_init_states) if all_init_states else 0}")
        else:
            all_init_states = None
        
        # 🔥 核心循环：按初始状态分组收集
        for group_idx in range(demo_batch_size):
            # 检查全局早停条件
            if rollout_goal_per_step and hasattr(env_runner, 'file_counter') and env_runner.file_counter:
                current_global_count = env_runner.file_counter.get()
                if current_global_count >= rollout_goal_per_step:
                    print(f"🎯 达到全局样本目标 ({current_global_count}/{rollout_goal_per_step})，提前结束收集")
                    break
            
            print(f"🔄 收集第 {group_idx + 1}/{demo_batch_size} 组样本...")
            
            # 步骤1: 选择/生成一个初始状态
            if all_init_states is not None and len(all_init_states) > 0:
                # 从可用初始状态中随机选择
                init_state_idx = np.random.randint(0, len(all_init_states))
                init_state = all_init_states[init_state_idx]
                print(f"📍 使用初始状态 #{init_state_idx}")
            else:
                # 使用默认初始状态或随机生成
                init_state = None  # 让 runner 自己处理
                print(f"📍 使用默认初始状态")
            
            # 步骤2: 计算初始状态哈希（用于跳过检查）
            if init_state is not None:
                state_bytes = np.array(init_state).astype(np.float32).tobytes()
                init_hash = hashlib.sha256(state_bytes).hexdigest()
            else:
                init_hash = f"default_{group_idx}"  # 默认哈希
            
            # 步骤3: 检查是否应该跳过这个初始状态
            enable_init_skip = (rollout_stats is not None and rollout_skip_cnt is not None)
            if enable_init_skip and should_skip_init_state(init_hash, rollout_stats, rloo_batch_size, 
                                                          rollout_skip_cnt, rollout_skip_threshold=10):
                print(f"⏭️ 跳过初始状态 {init_hash[:8]}（历史表现一致）")
                rollout_skip_cnt[init_hash] = rollout_skip_cnt.get(init_hash, 0) + 1
                continue  # 跳过这个初始状态，不计入 collected_groups
            
            # 步骤4: 为这个初始状态收集 rloo_batch_size 个样本
            group_episodes = []
            
            # 🔥 关键：创建 rloo_batch_size 个相同的初始状态
            if init_state is not None:
                # 复制相同的初始状态
                env_init_states = np.tile(init_state, (rloo_batch_size, 1))
                print(f"🔄 第{group_idx+1}组并行运行 {rloo_batch_size} 个环境")
                print(f"   📍 初始状态哈希: {init_hash[:8]}...")
                print(f"   📊 状态值: {init_state[:4].round(3)}... (显示前4维)")
                print(f"   🔢 状态形状: {init_state.shape}, 复制到 {env_init_states.shape}")
                
                # 调用 runner 的并行执行（如果支持）
                if hasattr(env_runner, 'run_policy_in_env_batch'):
                    # 使用批量接口
                    batch_episodes = env_runner.run_policy_in_env_batch(
                        env_name=task_name,
                        init_states=env_init_states
                    )
                    group_episodes.extend(batch_episodes)
                else:
                    # 回退到逐个收集（但使用相同初始状态）
                    for sample_idx in range(rloo_batch_size):
                        rollout_generator = env_runner.run_policy_in_env(
                            env_name=task_name,
                            all_init_states=[init_state]  # 使用相同的初始状态
                        )
                        
                        # 收集一个样本
                        for success, total_reward, episode_data in rollout_generator:
                            episode = {
                                'success': success,
                                'total_reward': total_reward,
                                'init_hash': init_hash,  # 记录真实的初始状态哈希
                                **episode_data
                            }
                            group_episodes.append(episode)
                            break  # 只收集一个样本
                        
                        # 计数器在组被接受后统一更新
            else:
                # 没有具体初始状态，逐个收集（但会导致不同初始状态混合）
                print("⚠️ 没有具体初始状态，可能导致 RLOO 基线估计不准确")
                rollout_generator = env_runner.run_policy_in_env(
                    env_name=task_name,
                    all_init_states=all_init_states
                )
                
                sample_count = 0
                for success, total_reward, episode_data in rollout_generator:
                    episode = {
                        'success': success,
                        'total_reward': total_reward,
                        'init_hash': init_hash,
                        **episode_data
                    }
                    group_episodes.append(episode)
                    sample_count += 1
                    
                    # 计数器在组被接受后统一更新
                    
                    if sample_count >= rloo_batch_size:
                        break
            
            # 步骤5: RIPT风格动态采样过滤（优化版，支持多种策略）
            group_accepted = True
            if enable_ript_dynamic_sampling and len(group_episodes) == rloo_batch_size:
                # 🔥 使用RIPT原版的简单动态采样
                filtered_group = _ript_dynamic_sampling(group_episodes, enable_ript_dynamic_sampling)
                if not filtered_group:
                    group_accepted = False
            
            # 🔥 关键修复：为分组路径添加temp_init_hash字段
            if group_accepted and group_episodes:
                for ep in group_episodes:
                    if 'temp_init_hash' not in ep:
                        ep['temp_init_hash'] = init_hash  # 确保每个episode都有这个字段
                    if 'init_hash' not in ep:
                        ep['init_hash'] = init_hash
            
            # 步骤6: 延迟统计更新到correct_hash生成后（避免重复计数）
            # 注意：这里不再直接更新rollout_stats，而是延迟到CFG处理后统一更新
            
            # 步骤7: 添加到总列表（仅接受的组）
            if group_accepted and len(group_episodes) == rloo_batch_size:
                all_episodes.extend(group_episodes)
                collected_groups += 1
                print(f"✅ 第 {group_idx + 1} 组收集并接受：{len(group_episodes)} 个样本")
                # 🔥 对齐RIPT：每个被接受的组计数+1
                if rollout_goal_per_step and hasattr(env_runner, 'file_counter') and env_runner.file_counter:
                    try:
                        env_runner.file_counter.update(1)
                        current_global_count = env_runner.file_counter.get()
                        if current_global_count >= rollout_goal_per_step:
                            print(f"🎯 达到全局样本目标 ({current_global_count}/{rollout_goal_per_step} 组)，提前结束收集")
                            break
                    except Exception:
                        pass
            elif len(group_episodes) == rloo_batch_size:
                print(f"⚠️ 第 {group_idx + 1} 组收集但被过滤：{len(group_episodes)} 个样本")
            else:
                print(f"⚠️ 第 {group_idx + 1} 组样本不足：{len(group_episodes)}/{rloo_batch_size}")
                # 可以选择填充或跳过
                if len(group_episodes) > 0 and group_accepted:
                    # 填充到目标数量
                    while len(group_episodes) < rloo_batch_size:
                        group_episodes.append(group_episodes[-1])  # 用最后一个样本填充
                    all_episodes.extend(group_episodes)
                    collected_groups += 1
        
        print(f"📊 收集完成：{collected_groups} 组 × {rloo_batch_size} = {len(all_episodes)} 个样本")
        
        # 🔥 RIPT风格动态采样已在收集过程中实时应用
        if enable_ript_dynamic_sampling:
            print(f"📊 RIPT动态采样: 已过滤全成功/全失败的组，保留有学习价值的混合结果")
        
        # 步骤8: 生成有效性掩码
        valid_mask = [True] * len(all_episodes)
        
        return all_episodes, valid_mask
        
    except Exception as e:
        print(f"❌ 收集过程出错: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def collect_rollouts_ript_vla_style(env_runner, task_name, num_rollouts, 
                                     dynamic_sampling_config: Dict = None, 
                                     recent_success_rates: deque = None,
                                     rollout_goal_per_step: int = None,
                                     rollout_stats: Dict[str, List[int]] = None,
                                     rollout_skip_cnt: Dict[str, int] = None,
                                     rloo_batch_size: int = None):
    """
    兼容性包装器：将旧的接口转换为新的RIPT风格分组收集
    """
    if rloo_batch_size is None or rloo_batch_size <= 0:
        rloo_batch_size = 8  # 默认值
    
    # 计算需要多少组
    demo_batch_size = max(1, num_rollouts // rloo_batch_size)
    
    # 🔥 从复杂配置中提取简单的布尔开关
    if dynamic_sampling_config is not None:
        enable_ript_dynamic_sampling = dynamic_sampling_config.get('enabled', False)
        if enable_ript_dynamic_sampling:
            print(f"🔄 使用RIPT风格动态采样 (简单的全成功/全失败检查)")
    else:
        enable_ript_dynamic_sampling = True  # 默认启用
    
    print(f"🔄 兼容性转换: {num_rollouts} 个样本 → {demo_batch_size} 组 × {rloo_batch_size} 个样本")
    
    return collect_rollouts_ript_vla_style_grouped(
        env_runner=env_runner,
        task_name=task_name, 
        demo_batch_size=demo_batch_size,
        rloo_batch_size=rloo_batch_size,
        enable_ript_dynamic_sampling=enable_ript_dynamic_sampling,
        rollout_goal_per_step=rollout_goal_per_step,
        rollout_stats=rollout_stats,
        rollout_skip_cnt=rollout_skip_cnt
    )


def collect_rollouts_ript_vla_style_old(env_runner, task_name, num_rollouts, 
                                     dynamic_sampling_config: Dict = None, 
                                     recent_success_rates: deque = None,
                                     rollout_goal_per_step: int = None,
                                     rollout_stats: Dict[str, List[int]] = None,
                                     rollout_skip_cnt: Dict[str, int] = None,
                                     rloo_batch_size: int = None):
    """
    原有的逐个收集方式（保留作为备份）
    ⚠️ 注意：这种方式会导致 RLOO 基线估计不准确，因为不同初始状态的样本被混合在一起
    """
    print(f"⚠️ 使用旧版收集方式：逐个收集 {num_rollouts} 个rollouts（可能混合不同初始状态）")
    
    # 如果设置了全局样本目标，显示当前进度
    if rollout_goal_per_step and hasattr(env_runner, 'file_counter') and env_runner.file_counter:
        current_count = env_runner.file_counter.get()
        print(f"📊 当前全局样本数: {current_count}/{rollout_goal_per_step}")
    
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
        
        # 收集rollouts，支持文件计数器早停
        collected_rollouts = []
        rollout_count = 0
        
        for success, total_reward, episode_data in rollout_generator:
            episode = {
                'success': success,
                'total_reward': total_reward,
                **episode_data
            }
            
            # 🔥 新增：per-init 哈希跳过检查
            enable_init_skip = (rollout_stats is not None and rollout_skip_cnt is not None and 
                               rloo_batch_size is not None and rloo_batch_size > 0)
            
            # 🔥 修改：暂时使用观测哈希，后续在训练循环中用正确的状态哈希
            if enable_init_skip:
                # 使用备用方法计算哈希（基于观测）
                init_hash = compute_init_state_hash_from_obs(episode)
                
                # 检查是否应该跳过这个初始状态
                if should_skip_init_state(init_hash, rollout_stats, rloo_batch_size, 
                                        rollout_skip_cnt, rollout_skip_threshold=10):
                    rollout_skip_cnt[init_hash] = rollout_skip_cnt.get(init_hash, 0) + 1
                    
                    # 如果跳过次数过多，清理该哈希记录
                    if rollout_skip_cnt[init_hash] > 10:
                        if init_hash in rollout_stats:
                            del rollout_stats[init_hash]
                        del rollout_skip_cnt[init_hash]
                        print(f"🧹 清理过度跳过的 init_hash: {init_hash[:8]}...")
                    
                    continue  # 跳过这个 episode
                
                # 暂时记录哈希，后续会用正确的状态哈希更新
                episode['temp_init_hash'] = init_hash
            
            collected_rollouts.append(episode)
            rollout_count += 1
            
            # 🔥 新增：文件计数器更新和早停检查
            if hasattr(env_runner, 'file_counter') and env_runner.file_counter:
                env_runner.file_counter.update(1)
                current_global_count = env_runner.file_counter.get()
                
                # 早停检查：达到全局目标样本数
                if rollout_goal_per_step and current_global_count >= rollout_goal_per_step:
                    print(f"🎯 达到全局样本目标 ({current_global_count}/{rollout_goal_per_step})，提前结束收集")
                    break
            
            # 原有的数量限制
            if rollout_count >= num_rollouts:
                break
        
        # 🔄 传统的复杂动态采样过滤（不推荐，但保留兼容性）
        if dynamic_sampling_config and dynamic_sampling_config.get('use_legacy_filtering', False):
            # 使用复杂的传统过滤逻辑（需要平滑窗口）
            recent_success_rates = deque(maxlen=3)  # 临时创建窗口
            filtered = _dynamic_filter_rollouts_legacy(collected_rollouts, dynamic_sampling_config, recent_success_rates)
        else:
            # 🔥 默认使用RIPT风格的简单动态采样
            enable_ript_dynamic_sampling = dynamic_sampling_config.get('enabled', True) if dynamic_sampling_config else True
            filtered = _ript_dynamic_sampling(collected_rollouts, enable_ript_dynamic_sampling)
            
        if not filtered:
            print("⚠️ 本批次被动态采样过滤，返回空集")
            return filtered, []
        
        # 🔥 新增：padding + valid_mask 机制（对标 ript-vla_ori）
        target_rollouts = num_rollouts  # 目标数量
        valid_mask = [True] * len(filtered)
        
        if len(filtered) < target_rollouts:
            num_pad = target_rollouts - len(filtered)
            print(f"📦 Padding: 需要 {target_rollouts} 个rollouts，实际收集 {len(filtered)} 个，填充 {num_pad} 个")
            
            if len(filtered) > 0:
                # 用最后一个样本进行填充
                last_episode = filtered[-1].copy()
                for _ in range(num_pad):
                    padded_episode = last_episode.copy()
                    padded_episode['is_padded'] = True  # 标记为填充样本
                    filtered.append(padded_episode)
                    valid_mask.append(False)  # 标记为无效
        else:
                print("⚠️ 没有有效样本用于填充")
                return filtered, []
        
        print(f"✓ 成功收集了 {len(filtered)} 个rollouts (过滤+填充后，{sum(valid_mask)} 个有效)")
        return filtered, valid_mask
        
    except Exception as e:
        print(f"❌ Rollout收集失败: {e}")
        traceback.print_exc()
        return [], []

def compute_advantages_rloo_grouped(episodes: List[Dict], rloo_batch_size: int, 
                                   demo_batch_size: int = None) -> torch.Tensor:
    """
    🔥 增强版RLOO优势计算：验证分组结构，确保基线估计准确性
    
    专门为按初始状态分组的episodes设计，确保每组内的样本来自相同初始状态
    
    Args:
        episodes: 按分组排列的episodes列表 (每组rloo_batch_size个样本来自相同init_hash)
        rloo_batch_size: 每组的样本数量
        demo_batch_size: 组的数量（可选，用于验证）
    
    Returns:
        torch.Tensor: 计算得到的优势值
    """
    if not episodes:
        return torch.tensor([])
    
    num_rollouts = len(episodes)
    
    # 验证分组结构
    if demo_batch_size is not None:
        expected_total = demo_batch_size * rloo_batch_size
        if num_rollouts != expected_total:
            print(f"⚠️ 分组结构不匹配: 期望 {expected_total} 个样本 ({demo_batch_size}×{rloo_batch_size}), 实际 {num_rollouts} 个")
    
    # 提取奖励
    rewards = []
    init_hashes = []  # 用于验证分组
    for ep in episodes:
        reward = ep.get('total_reward', 0.0)
        rewards.append(float(reward))
        # 🔥 关键修复：优先使用correct_init_hash，其次temp_init_hash，最后init_hash
        hash_to_use = ep.get('correct_init_hash') or ep.get('temp_init_hash') or ep.get('init_hash', 'unknown')
        init_hashes.append(hash_to_use)
    
    rlhf_reward = torch.tensor(rewards, dtype=torch.float32)
    
    # 🔍 验证分组的正确性
    if len(init_hashes) >= rloo_batch_size:
        print("🔍 验证RLOO分组结构...")
        groups_verified = 0
        for group_start in range(0, num_rollouts, rloo_batch_size):
            group_end = min(group_start + rloo_batch_size, num_rollouts)
            if group_end - group_start == rloo_batch_size:
                group_hashes = init_hashes[group_start:group_end]
                unique_hashes = set(group_hashes)
                if len(unique_hashes) == 1:
                    groups_verified += 1
                else:
                    print(f"⚠️ 第 {group_start//rloo_batch_size + 1} 组包含多个不同初始状态: {unique_hashes}")
        
        total_groups = num_rollouts // rloo_batch_size
        if groups_verified == total_groups:
            print(f"✅ 分组验证通过: {groups_verified}/{total_groups} 组结构正确")
        else:
            print(f"⚠️ 分组验证失败: 仅 {groups_verified}/{total_groups} 组结构正确")
    
    # 🚀 RLOO优势计算
    if rloo_batch_size <= 1:
        print("⚠️ RLOO batch size过小，使用简单优势计算")
        advantage = rlhf_reward - rlhf_reward.mean()
    else:
        try:
            # 确保可以整除
            effective_rollouts = (num_rollouts // rloo_batch_size) * rloo_batch_size
            if effective_rollouts != num_rollouts:
                print(f"🔧 RLOO裁剪：{num_rollouts} → {effective_rollouts} rollouts")
                rlhf_reward = rlhf_reward[:effective_rollouts]
                num_rollouts = effective_rollouts
            
            num_batches = num_rollouts // rloo_batch_size
            rlhf_reward_reshaped = rlhf_reward.reshape(num_batches, rloo_batch_size)
            
            # 🔥 核心RLOO计算：每个样本与同组其他样本比较
            batch_sums = rlhf_reward_reshaped.sum(dim=1, keepdim=True)  # (num_batches, 1)
            baseline = (batch_sums - rlhf_reward_reshaped) / (rloo_batch_size - 1)  # (num_batches, rloo_batch_size)
            advantage = rlhf_reward_reshaped - baseline  # (num_batches, rloo_batch_size)
            advantage = advantage.flatten()  # 展平为一维
            
            # 安全性检查
            if torch.isnan(advantage).any() or torch.isinf(advantage).any():
                print("⚠️ RLOO计算产生NaN/Inf，使用安全替换")
                advantage = torch.nan_to_num(advantage, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 统计信息
            print(f"🎯 分组RLOO优势计算完成:")
            print(f"   分组配置: {num_rollouts} rollouts → {num_batches} 组 × {rloo_batch_size} 样本/组")
            print(f"   优势统计: mean={advantage.mean():.4f}, std={advantage.std():.4f}")
            print(f"   正优势比例: {(advantage > 0).float().mean():.2%}")
            
            # 🔍 按组显示优势分布（前几组）
            if num_batches <= 5:
                for batch_idx in range(num_batches):
                    batch_advantages = advantage[batch_idx * rloo_batch_size:(batch_idx + 1) * rloo_batch_size]
                    batch_rewards = rlhf_reward_reshaped[batch_idx]
                    print(f"   组 {batch_idx + 1}: 奖励 {batch_rewards.tolist()}, 优势 {batch_advantages.tolist()}")
            
        except Exception as e:
            print(f"❌ 分组RLOO计算失败: {e}，回退到简单方法")
            advantage = rlhf_reward - rlhf_reward.mean()
    
    return advantage


def compute_advantages_rloo(episodes: List[Dict], rloo_batch_size: int = None) -> torch.Tensor:
    """
    兼容性包装器：调用增强版分组RLOO计算
    """
    if rloo_batch_size is None or rloo_batch_size <= 0:
        rloo_batch_size = 8  # 默认值
    
    # 推断demo_batch_size
    demo_batch_size = len(episodes) // rloo_batch_size if len(episodes) >= rloo_batch_size else 1
    
    return compute_advantages_rloo_grouped(
        episodes=episodes,
        rloo_batch_size=rloo_batch_size,
        demo_batch_size=demo_batch_size
    )


def compute_advantages_rloo_old(episodes: List[Dict], rloo_batch_size: int = None) -> torch.Tensor:
    """
    原有的RLOO优势计算（保留作为备份）
    ⚠️ 注意：这个版本不验证分组结构，可能导致基线估计不准确
    """
    if not episodes:
        return torch.tensor([])
    
    # 提取奖励
    rewards = []
    for ep in episodes:
        reward = ep.get('total_reward', 0.0)
        rewards.append(float(reward))
    
    rlhf_reward = torch.tensor(rewards, dtype=torch.float32)
    num_rollouts = len(episodes)
    
    # 🔥 关键修复：使用真正的RLOO批次大小而不是总数
    if rloo_batch_size is None or rloo_batch_size <= 1:
        # 如果没有指定或batch size过小，退化为简单方法
        print("⚠️ RLOO batch size未指定或过小，使用简单优势计算")
        advantage = rlhf_reward - rlhf_reward.mean()
    else:
        # 🚀 正宗RLOO计算
        try:
            # 确保可以整除，如果不能整除则裁剪到最大可整除数量
            effective_rollouts = (num_rollouts // rloo_batch_size) * rloo_batch_size
            if effective_rollouts != num_rollouts:
                print(f"🔧 RLOO调整：{num_rollouts} → {effective_rollouts} rollouts (batch_size={rloo_batch_size})")
                rlhf_reward = rlhf_reward[:effective_rollouts]
                num_rollouts = effective_rollouts
            
            num_batches = num_rollouts // rloo_batch_size
            rlhf_reward_reshaped = rlhf_reward.reshape(num_batches, rloo_batch_size)
            
            # 标准RLOO：每个样本的baseline = 同批次其他样本的平均值
            # baseline[i,j] = (sum(batch[i]) - reward[i,j]) / (batch_size - 1)
            batch_sums = rlhf_reward_reshaped.sum(dim=1, keepdim=True)  # (num_batches, 1)
            baseline = (batch_sums - rlhf_reward_reshaped) / (rloo_batch_size - 1)  # (num_batches, rloo_batch_size)
            
            # 优势 = 自己的奖励 - 其他人的平均奖励
            advantage = rlhf_reward_reshaped - baseline  # (num_batches, rloo_batch_size)
            advantage = advantage.flatten()  # 展平为一维
            
            # NaN和Inf检查
            if torch.isnan(advantage).any() or torch.isinf(advantage).any():
                print("⚠️ RLOO计算产生NaN/Inf，使用安全替换")
                advantage = torch.nan_to_num(advantage, nan=0.0, posinf=1.0, neginf=-1.0)
            
            print(f"🎯 正宗RLOO优势计算完成:")
            print(f"   批次配置: {num_rollouts} rollouts → {num_batches} batches × {rloo_batch_size}")
            print(f"   优势统计: mean={advantage.mean():.4f}, std={advantage.std():.4f}")
            print(f"   正优势比例: {(advantage > 0).float().mean():.2%}")
            
        except Exception as e:
            print(f"❌ RLOO计算失败: {e}，回退到简单方法")
            advantage = rlhf_reward - rlhf_reward.mean()
    
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

def evaluate_with_cfg_sweep(policy, env_runner, task_name, config, eval_episodes=3):
    """🔥 新增：评估不同CFG强度的效果（完全配置化，无硬编码）"""
    # 从配置文件读取CFG扫描参数
    cfg_sweep_config = get_config_value(config, 'cfg_sweep_config', {}, ['features'])
    
    if not cfg_sweep_config.get('enabled', True):
        print("⚠️ CFG扫描评估已禁用")
        return get_config_value(config, 'collection_cfg_scale'), {}
    
    cfg_scales = cfg_sweep_config.get('scales', [1.0, 1.25, 1.5, 2.0, 3.0])
    best_cfg = cfg_scales[0] if cfg_scales else 1.0  # 使用第一个配置值作为默认
    best_success_rate = 0.0
    
    results = {}
    print(f"\n🔍 开始CFG强度扫描评估（完全配置化）...")
    print(f"   扫描范围: {cfg_scales}")
    print(f"   每个CFG评估轮数: {eval_episodes}")
    
    for cfg_scale in cfg_scales:
        print(f"📊 测试CFG={cfg_scale}...")
        # 临时设置CFG强度
        original_cfg = get_config_value(config, 'collection_cfg_scale')
        set_config_value(config, 'collection_cfg_scale', cfg_scale)
        env_runner.config.collection_cfg_scale = cfg_scale
        
        # 运行评估episodes
        success_count = 0
        for ep_idx in range(eval_episodes):
            try:
                # 使用现有的rollout收集函数 - 正确解包和传参
                episodes, valid_mask = collect_rollouts_ript_vla_style(
                    env_runner, task_name, 1,
                    dynamic_sampling_config=None,  # 评估时关闭动态采样
                    recent_success_rates=None,
                    rollout_goal_per_step=None,
                    rollout_stats=None,
                    rollout_skip_cnt=None,
                    rloo_batch_size=1
                )
                if episodes and len(episodes) > 0:
                    if episodes[0].get('success', False):
                        success_count += 1
            except Exception as e:
                print(f"   评估episode {ep_idx} 失败: {e}")
                continue
        
        success_rate = success_count / eval_episodes
        results[cfg_scale] = success_rate
        
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_cfg = cfg_scale
        
        # 恢复原设置
        set_config_value(config, 'collection_cfg_scale', original_cfg)
        env_runner.config.collection_cfg_scale = original_cfg
        
        print(f"   CFG={cfg_scale}: 成功率={success_rate:.2%} ({success_count}/{eval_episodes})")
    
    print(f"🎯 最佳CFG强度: {best_cfg} (成功率: {best_success_rate:.2%})")
    return best_cfg, results

def main_training_loop_ript_vla_style(config: Dict[str, Any]):
    """
    主训练循环（RIPT-VLA风格）+ 动态采样 + 文件计数器
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
    
    # 🔥 新增：读取增强配置
    features_config = config.get('features', {})
    dynamic_sampling_config = features_config.get('dynamic_sampling', {})
    enable_file_counter = features_config.get('enable_file_counter', False)
    adaptive_cfg_enabled = features_config.get('adaptive_cfg', False)
    
    # 🔥 修正：独立配置demo_batch_size，不再等于rloo_batch_size
    demo_batch_size = get_config_value(config, 'demo_batch_size', None, ['algo', 'dataset'])
    if demo_batch_size is None:
        # 尝试从dataset.num_init_states获取
        demo_batch_size = get_config_value(config, 'num_init_states', 22, ['dataset'])
        print(f"⚠️ demo_batch_size未配置，使用dataset.num_init_states={demo_batch_size}")
    else:
        print(f"✅ 使用配置的demo_batch_size={demo_batch_size}")
    
    # 🔥 修复：提前读取rloo_batch_size（在使用前定义）
    rloo_batch_size = config['algo']['rloo_batch_size']
    
    world_size = 1  # 当前单机，后续可从分布式环境读取
    early_stop_percentage = features_config.get('early_stop_percentage', 0.8)  # 新增配置项
    
    # 🔥 新增：统一配置读取 - rollout_stats相关
    enable_rollout_stats_tracking = get_config_value(config, 'enable_rollout_stats_tracking', False, ['features', 'algo'])
    rollout_stats_path = get_config_value(config, 'rollout_stats_path', str(output_dir / "rollout_stats.json"), ['features', 'algo'])
    
    if enable_file_counter:
        # 对齐RIPT：按组数早停（每个被接受的组计1），阈值按组计算
        total_target_groups = demo_batch_size * world_size
        rollout_goal_per_step = int(np.ceil(total_target_groups * early_stop_percentage))
        print(f"🎯 早停阈值计算: {rollout_goal_per_step} 组 = {demo_batch_size}组 × {world_size}GPU × {early_stop_percentage:.0%}")
    else:
        rollout_goal_per_step = None
    
    print(f"\n=== RIPT对齐验证 ===")
    print(f"demo_batch_size: {demo_batch_size} (每步收集的组数)")
    print(f"rloo_batch_size: {rloo_batch_size} (每组内样本数)") 
    print(f"目标样本数/步: {demo_batch_size * rloo_batch_size}")
    if rollout_goal_per_step:
        print(f"早停阈值: {rollout_goal_per_step}")
    else:
        print("早停阈值: 未启用")
    
    print(f"\n🔧 增强功能配置:")
    print(f"  动态采样: {'✅' if dynamic_sampling_config.get('enabled', False) else '❌'}")
    if dynamic_sampling_config.get('enabled', False):
        print(f"    模式: RIPT风格 (简单全成功/全失败检查)")
        print(f"    区间: [{dynamic_sampling_config.get('p_min', 0.1)}, {dynamic_sampling_config.get('p_max', 0.9)}] (备用)")
        print(f"    注：RIPT风格不使用平滑窗口，直接检查当前组的成功率")
    print(f"  文件计数器: {'✅' if enable_file_counter else '❌'}")
    if rollout_goal_per_step:
        print(f"    每步全局目标: {rollout_goal_per_step}")
    print(f"  自适应CFG: {'✅' if adaptive_cfg_enabled else '❌'}")
    
    # 🔥 简化：移除复杂的平滑窗口（RIPT原版不需要）
    # smooth_window_size = dynamic_sampling_config.get('smooth_window', 3)
    # recent_success_rates = deque(maxlen=smooth_window_size)
    
    # 🔥 新增：per-init 哈希跳过机制初始化
    
    rollout_stats = {}
    rollout_skip_cnt = {}
    
    if enable_rollout_stats_tracking:
        rollout_stats = load_rollout_stats(rollout_stats_path)
        rollout_skip_cnt = {k: 0 for k in rollout_stats.keys()}
        print(f"📊 已加载 {len(rollout_stats)} 个初始状态的历史记录")
        print(f"💾 rollout_stats 路径: {rollout_stats_path}")
    
    # 创建策略和优化器
    print("正在加载PI0策略...")
    
    # 🔥 正确的策略加载逻辑，严格遵守cfg_enabled配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_path = config['policy_path']
    
    # 加载策略
    policy = PI0Policy.from_pretrained(policy_path, local_files_only=True)
    print("Loading weights from local directory")
    policy = policy.to(device)
    
    # 🔥 关键修复：严格按照cfg_enabled设置CFG功能
    cfg_enabled = config.get('policy', {}).get('cfg_enabled', True)
    
    if cfg_enabled:
        print(f"🔧 启用CFG功能（按配置要求）...")
        # 同步到Policy、Model与Config三处，确保sample_actions走双分支路径
        policy.cfg_enabled = True
        if hasattr(policy, 'model'):
            setattr(policy.model, 'cfg_enabled', True)
        if hasattr(policy, 'config'):
            setattr(policy.config, 'cfg_enabled', True)
        # 从配置读取collection_cfg_scale
        policy.default_cfg_scale = config.get('algo', {}).get('collection_cfg_scale', 1.25)
        print(f"✅ CFG已启用，默认CFG强度: {policy.default_cfg_scale}")
    else:
        print(f"🔧 禁用CFG功能（按配置要求）...")
        policy.cfg_enabled = False
        if hasattr(policy, 'model'):
            setattr(policy.model, 'cfg_enabled', False)
        if hasattr(policy, 'config'):
            setattr(policy.config, 'cfg_enabled', False)
        policy.default_cfg_scale = 1.0  # 强制设为1.0，完全禁用CFG
        print(f"✅ CFG已禁用，强制CFG强度: {policy.default_cfg_scale}")
    
    print(f"✓ 策略加载成功，设备: {device}")
    
    # 创建优化器
    print("正在创建优化器...")
    if config.get('policy', {}).get('train_expert_only', False):
        # 只训练专家头部
        expert_params = []
        for name, param in policy.named_parameters():
            if 'expert' in name or 'cfg_embedding' in name:  # 包含CFG embedding参数
                expert_params.append(param)
        
        print("🔧 配置训练参数范围...")
        if hasattr(policy, 'cfg_embedding') and policy.cfg_embedding is not None:
            cfg_params = list(policy.cfg_embedding.parameters())
            expert_params.extend(cfg_params)
            print("✅ CFG embedding参数已加入训练")
        
        optimizer = torch.optim.AdamW(expert_params, lr=config['algo']['lr'])
        total_params = sum(p.numel() for p in expert_params)
        print(f"🎯 只训练专家头部，参数数量: {total_params:,}")
    else:
        # 训练所有参数
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config['algo']['lr'])
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"🎯 训练所有参数，参数数量: {total_params:,}")
    
    print(f"✓ 优化器创建成功，学习率: {config['algo']['lr']}")
    
    # 返回创建的组件
    # policy, optimizer, device = policy, optimizer, device
    
    # 创建CFG适配器（必需，用于损失计算）
    # 🔥 新增：窗口化配置支持
    dataset_config = config.get('dataset', {})
    windowing_mode = dataset_config.get('windowing_mode', 'last')
    window_stride = dataset_config.get('window_stride', 10)
    max_windows_per_episode = dataset_config.get('max_windows_per_episode', 1)
    
    print(f"\n🔧 CFG窗口化配置:")
    print(f"  模式: {windowing_mode}")
    print(f"  步长: {window_stride}")
    print(f"  每episode最大窗口数: {max_windows_per_episode}")
    
    cfg_adapter = PI0_CFG_Adapter(
        policy=policy,
        norm_stats_path=f"{config['policy_path']}/norm_stats.json",
        windowing_mode=windowing_mode,
        window_stride=window_stride,
        max_windows_per_episode=max_windows_per_episode
    )
    
    # 创建环境runner（使用标准工厂，确保并行与benchmark配置正确下发）
    env_runner = create_environment_runner(config, policy)

    # 🔥 同步CFG强度到runner（runner内部读取的是根级字段，这里明确设置）
    if hasattr(env_runner, 'config') and env_runner.config is not None:
        env_runner.config.collection_cfg_scale = (policy.default_cfg_scale if cfg_enabled else 1.0)
    else:
        env_runner.collection_cfg_scale = (policy.default_cfg_scale if cfg_enabled else 1.0)
    print(f"🔧 环境runner CFG强度设置为: {policy.default_cfg_scale if cfg_enabled else 1.0}")

    print("✓ 环境runner创建成功")
    
    # 🔥 新增：验证环境runner功能
    print(f"\n🔧 环境runner功能验证:")
    if hasattr(env_runner, 'run_policy_in_env_batch'):
        print("  批量执行: ✅")
    else:
        print("  批量执行: ❌ (将回退到串行)")

    # 验证初始状态池
    if hasattr(env_runner, 'get_task_init_states'):
        try:
            test_states = env_runner.get_task_init_states(0)
            if test_states:
                print(f"  初始状态池: ✅ ({len(test_states)}个状态)")
            else:
                print("  初始状态池: ⚠️ (为空)")
        except Exception as e:
            print(f"  初始状态池: ❌ (获取失败: {e})")
    else:
        print("  初始状态池: ❌ (不支持get_task_init_states)")
    
    # 🔥 新增：文件计数器初始化
    if enable_file_counter:
        file_counter = env_runner.setup_file_counter(counter_name="rollout", work_dir=str(output_dir))
        if file_counter:
            print(f"✅ 文件计数器已启用: {output_dir}/rollout_counter")
        else:
            print(f"⚠️ 文件计数器初始化失败，继续使用普通模式")
            enable_file_counter = False
    
    # 训练配置
    num_train_steps = config['training']['num_train_steps']
    # 与2_test_pi0_on_libero.py对齐：使用libero_goal基准默认task_id=1
    # 若YAML中明确给了task_names_to_use，则仍然使用第一个名称做显示，不影响环境内部task_id选择
    task_name = config['task'].get('task_names_to_use', ['libero_goal_default'])[0]
    
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
        
        # 1. 收集rollouts（RIPT原版风格：按初始状态分组）
        # 🔥 关键修改：使用优化的RIPT动态采样，支持多种策略
        enable_ript_dynamic_sampling = dynamic_sampling_config.get('enabled', True) if dynamic_sampling_config else True
        
        episodes, valid_mask = collect_rollouts_ript_vla_style_grouped(
            env_runner=env_runner,
            task_name=task_name, 
            demo_batch_size=demo_batch_size,  # 需要多少个不同的初始状态
            rloo_batch_size=rloo_batch_size,  # 每个初始状态收集多少个样本
            enable_ript_dynamic_sampling=enable_ript_dynamic_sampling,
            rollout_goal_per_step=rollout_goal_per_step,
            rollout_stats=rollout_stats if enable_rollout_stats_tracking else None,
            rollout_skip_cnt=rollout_skip_cnt if enable_rollout_stats_tracking else None
        )
        
        if not episodes:
            print("⚠️ 未收集到有效episodes，跳过此步")
            continue
        
        # 记录有效样本数量
        valid_count = sum(valid_mask) if valid_mask else len(episodes)
        step_metrics = {
            'step': step + 1,
            'num_episodes': len(episodes),
            'valid_episodes': valid_count,
            'padding_episodes': len(episodes) - valid_count
        }
        
        # 2. 🔥 新增：使用正确的状态哈希更新统计（在 CFG adapter 处理后）
        if enable_rollout_stats_tracking:
            try:
                # 通过 CFG adapter 处理 episodes 以获得正确的状态表示
                batch, owner_indices = cfg_adapter.process_episodes(episodes, device)
                batch_states = batch.get('state')  # (B, state_dim)
                
                # 使用正确的状态哈希更新统计
                update_rollout_stats_with_correct_hash(
                    episodes, batch_states, None, owner_indices, 
                    rollout_stats, rollout_skip_cnt
                )
                
                print(f"🔄 已用正确状态哈希更新 {len(episodes)} 个episodes的统计")
                
            except Exception as e:
                print(f"⚠️ 正确哈希更新失败: {e}")
        
        # 3. 计算优势（增强版分组RLOO方法，验证初始状态分组）
        advantages = compute_advantages_rloo_grouped(
            episodes=episodes, 
            rloo_batch_size=rloo_batch_size,
            demo_batch_size=demo_batch_size  # 传递分组信息用于验证
        )
        
        # 4. 更新策略（直接更新，无复杂组件）
        loss = update_policy_ript_vla_style(
            policy, optimizer, cfg_adapter, episodes, advantages, device
        )
        
        # 5. 记录指标
        avg_reward = np.mean([ep['total_reward'] for ep in episodes])
        success_rate = np.mean([ep['success'] for ep in episodes])
        step_time = time.time() - step_start_time
        
        # 更新step_metrics
        step_metrics.update({
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'loss': loss,
            'step_time': step_time
        })
        all_training_metrics.append(step_metrics)
        
        # 6. 输出结果
        print(f"✓ 步骤 {step + 1} 完成:")
        print(f"  Episodes: {len(episodes)} (有效: {valid_count}, 填充: {len(episodes) - valid_count})")
        print(f"  平均奖励: {avg_reward:.4f}")
        print(f"  成功率: {success_rate:.2%}")
        print(f"  损失: {loss:.6f}")
        print(f"  耗时: {step_time:.2f}秒")
        
        # 7. 自适应CFG调整（简化版本，基于当前步骤成功率）
        if adaptive_cfg_enabled and len(episodes) > 0:
            try:
                # 计算当前步骤的总体成功率
                current_successes = [ep.get('success', False) for ep in episodes]
                current_success_rate = np.mean(current_successes)
                current_cfg = get_config_value(config, 'collection_cfg_scale')
                
                # 🔥 完全配置化的CFG调整逻辑
                adaptive_cfg_config = get_config_value(config, 'adaptive_cfg_config', {}, ['features'])
                
                min_cfg = adaptive_cfg_config.get('min_cfg', 1.0)
                max_cfg = adaptive_cfg_config.get('max_cfg', 3.0)
                cfg_step = adaptive_cfg_config.get('cfg_step', 0.2)
                low_threshold = adaptive_cfg_config.get('low_success_threshold', 0.2)
                high_threshold = adaptive_cfg_config.get('high_success_threshold', 0.9)
                
                if current_success_rate < low_threshold:  # 成功率很低
                    new_cfg = max(min_cfg, current_cfg - cfg_step)
                    print(f"🔧 自适应CFG: 成功率过低({current_success_rate:.3f} < {low_threshold})，降低CFG {current_cfg:.2f} → {new_cfg:.2f}")
                    # 统一写回配置
                    set_config_value(config, 'collection_cfg_scale', new_cfg)
                    env_runner.config.collection_cfg_scale = new_cfg
                elif current_success_rate > high_threshold:  # 成功率很高
                    new_cfg = min(max_cfg, current_cfg + cfg_step)
                    print(f"🔧 自适应CFG: 成功率过高({current_success_rate:.3f} > {high_threshold})，提升CFG {current_cfg:.2f} → {new_cfg:.2f}")
                    # 统一写回配置
                    set_config_value(config, 'collection_cfg_scale', new_cfg)
                    env_runner.config.collection_cfg_scale = new_cfg
                else:
                    print(f"✅ 自适应CFG: 成功率适中({current_success_rate:.3f})，保持CFG={current_cfg:.2f}")
                
                step_metrics['adaptive_cfg_scale'] = get_config_value(config, 'collection_cfg_scale')
                step_metrics['current_success_rate'] = current_success_rate
                
            except Exception as e:
                print(f"⚠️ 自适应CFG调整失败: {e}")
        
        # 8. CFG评估（每10步进行一次）
        if (step + 1) % 10 == 0:
            try:
                # 从配置文件读取eval_episodes，如果没有配置则默认为2
                cfg_sweep_config = get_config_value(config, 'cfg_sweep_config', {}, ['features'])
                eval_episodes = cfg_sweep_config.get('eval_episodes', 2)
                best_cfg, cfg_results = evaluate_with_cfg_sweep(policy, env_runner, task_name, config, eval_episodes=eval_episodes)
                step_metrics['best_cfg_scale'] = best_cfg
                step_metrics['cfg_sweep_results'] = cfg_results
                print(f"🎯 推荐CFG强度: {best_cfg}")
                # 注意：如果启用了自适应CFG，这里不强制覆盖
                if not adaptive_cfg_enabled:
                    env_runner.config.collection_cfg_scale = best_cfg
            except Exception as e:
                print(f"⚠️ CFG评估失败: {e}")
        
        # 9. 定期保存 rollout_stats
        if enable_rollout_stats_tracking and (step + 1) % 5 == 0:  # 每5步保存一次
            save_rollout_stats(rollout_stats, rollout_stats_path)
            print(f"💾 已保存 rollout_stats ({len(rollout_stats)} 个初始状态)")
        
        # 10. 保存检查点
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

    # 最终保存 rollout_stats
    if enable_rollout_stats_tracking:
        save_rollout_stats(rollout_stats, rollout_stats_path)
        print(f"💾 最终保存 rollout_stats: {rollout_stats_path}")

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