#!/usr/bin/env python3
"""
Stage 11 RIPT-VLA风格简化版本
基于RIPT-VLA的直接架构模式，去除多余的抽象层

核心设计原则：
1. 直接在主循环中处理rollout收集和优化
2. 简化的组件架构，减少中间层
3. 直接使用SubprocVectorEnv进行并行
4. 模仿RIPT-VLA的成功模式


python 11_train_ript_vla_style.py --config_path pi0/ript/config/stage11_unified_pool.yaml 
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

# 🔥 早期设置multiprocessing，避免子进程重复设置
import multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

# 🔥 添加RIPT对齐的数据集工具
from pi0.ript.utils.libero_utils_ript_aligned import (
    build_dataset_ript_aligned,
    collate_fn_ript_aligned
)

import hashlib
import random

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

class RolloutStatsTracker:
    """
    每个初始状态的rollout统计跟踪器
    实现per-init跳过机制，与RIPT原版对齐
    """
    def __init__(self, rollout_skip_threshold: int = 3, stats_path: Optional[str] = None):
        self.rollout_stats = {}  # {(task_id, init_hash): [success_history]}
        self.rollout_skip_cnt = {}  # {(task_id, init_hash): skip_count}
        self.rollout_skip_threshold = rollout_skip_threshold
        self.stats_path = stats_path
        
        # 加载已有统计
        if stats_path and Path(stats_path).exists():
            self._load_stats()
        
        print(f"🔧 RolloutStatsTracker初始化:")
        print(f"  跳过阈值: {rollout_skip_threshold}")
        print(f"  统计路径: {stats_path}")
        print(f"  已有统计: {len(self.rollout_stats)} 个init")
    
    def _compute_init_hash(self, task_id: int, init_state_data: Any) -> str:
        """计算初始状态的哈希值"""
        if isinstance(init_state_data, torch.Tensor):
            data_bytes = init_state_data.cpu().numpy().tobytes()
        elif isinstance(init_state_data, np.ndarray):
            data_bytes = init_state_data.tobytes()
        else:
            data_bytes = str(init_state_data).encode()
        
        return hashlib.sha256(data_bytes).hexdigest()[:16]  # 短哈希
    
    def should_skip_init(self, task_id: int, init_hash: str, rloo_batch_size: int) -> bool:
        """
        判断是否应该跳过这个初始状态
        RIPT原版逻辑：最近K=rloo_batch_size次全成功则跳过
        """
        key = (task_id, init_hash)
        
        if key not in self.rollout_stats:
            return False
        
        history = self.rollout_stats[key]
        if len(history) < rloo_batch_size:
            return False
        
        # 检查最近K次是否全成功
        recent_k = history[-rloo_batch_size:]
        all_successful = all(s == 1 for s in recent_k)
        
        if all_successful:
            print(f"🚫 跳过init ({task_id}, {init_hash}): 最近{rloo_batch_size}次全成功")
            return True
        
        return False
    
    def update_stats(self, task_id: int, init_hash: str, successes: List[bool]):
        """更新统计信息"""
        key = (task_id, init_hash)
        
        if key not in self.rollout_stats:
            self.rollout_stats[key] = []
            self.rollout_skip_cnt[key] = 0
        
        # 添加新的成功记录
        success_ints = [1 if s else 0 for s in successes]
        self.rollout_stats[key].extend(success_ints)
        
        # 保持历史记录长度合理（最多保留100次）
        if len(self.rollout_stats[key]) > 100:
            self.rollout_stats[key] = self.rollout_stats[key][-100:]
        
        print(f"📊 更新统计 ({task_id}, {init_hash}): +{len(successes)} 次，"
              f"总计 {len(self.rollout_stats[key])} 次，"
              f"成功率 {np.mean(self.rollout_stats[key]):.2%}")
    
    def increment_skip_count(self, task_id: int, init_hash: str):
        """增加跳过计数"""
        key = (task_id, init_hash)
        if key not in self.rollout_skip_cnt:
            self.rollout_skip_cnt[key] = 0
        
        self.rollout_skip_cnt[key] += 1
        
        # 如果跳过次数过多，移除这个init（避免永久跳过）
        if self.rollout_skip_cnt[key] > self.rollout_skip_threshold:
            print(f"🗑️ 移除init ({task_id}, {init_hash}): 跳过次数超过阈值")
            if key in self.rollout_stats:
                del self.rollout_stats[key]
            del self.rollout_skip_cnt[key]
    
    def _load_stats(self):
        """加载统计数据"""
        try:
            with open(self.stats_path, 'r') as f:
                data = json.load(f)
                self.rollout_stats = data.get('rollout_stats', {})
                self.rollout_skip_cnt = data.get('rollout_skip_cnt', {})
                
                # 转换字符串键为元组
                new_stats = {}
                new_skip_cnt = {}
                for key, value in self.rollout_stats.items():
                    if isinstance(key, str) and ',' in key:
                        task_id, init_hash = key.strip('()').split(', ')
                        new_key = (int(task_id), init_hash.strip("'\""))
                        new_stats[new_key] = value
                    else:
                        new_stats[key] = value
                
                for key, value in self.rollout_skip_cnt.items():
                    if isinstance(key, str) and ',' in key:
                        task_id, init_hash = key.strip('()').split(', ')
                        new_key = (int(task_id), init_hash.strip("'\""))
                        new_skip_cnt[new_key] = value
                    else:
                        new_skip_cnt[key] = value
                
                self.rollout_stats = new_stats
                self.rollout_skip_cnt = new_skip_cnt
                
                print(f"✅ 加载统计数据: {len(self.rollout_stats)} 个init")
        except Exception as e:
            print(f"⚠️ 加载统计数据失败: {e}")
    
    def save_stats(self):
        """保存统计数据"""
        if not self.stats_path:
            return
        
        try:
            # 确保目录存在
            Path(self.stats_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 转换元组键为字符串以便JSON序列化
            serializable_stats = {}
            serializable_skip_cnt = {}
            
            for key, value in self.rollout_stats.items():
                str_key = str(key)
                serializable_stats[str_key] = value
            
            for key, value in self.rollout_skip_cnt.items():
                str_key = str(key)
                serializable_skip_cnt[str_key] = value
            
            data = {
                'rollout_stats': serializable_stats,
                'rollout_skip_cnt': serializable_skip_cnt,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.stats_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"💾 统计数据已保存: {self.stats_path}")
        except Exception as e:
            print(f"❌ 保存统计数据失败: {e}")

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

    # 🔥 使用RIPT对齐的数据加载器（修复MuJoCo状态问题）
    print("✓ 使用RIPT对齐数据加载器")
    
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
    
    # 🔧 根据配置控制CFG功能
    policy_config = config.get('policy', {})
    cfg_enabled = policy_config.get('cfg_enabled', True)  # 默认启用以保持兼容性

    print(f"🔧 配置CFG功能: {'启用' if cfg_enabled else '禁用'}")
    policy.model.cfg_enabled = cfg_enabled
    if hasattr(policy, 'config'):
        policy.config.cfg_enabled = cfg_enabled

    if cfg_enabled:
        print("✅ CFG已启用，训练和推理将使用CFG分支")
    else:
        print("⚠️ CFG已禁用，将使用标准训练模式")
    
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
    
    # 3. CFG embedding参数 (仅在CFG启用时训练)
    if hasattr(policy.model, "cfg_emb") and getattr(policy.model, 'cfg_enabled', True):
        trainable_params += list(policy.model.cfg_emb.parameters())
        print("✅ CFG embedding参数已加入训练")
    elif hasattr(policy.model, "cfg_emb"):
        print("⚠️ CFG已禁用，跳过CFG embedding参数训练")
    
    # 4. 创建优化器
    print("正在创建优化器...")
    lr = config['algo'].get('lr', 1e-5)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    
    total_params = sum(p.numel() for p in trainable_params)
    print(f"✓ 优化器创建成功，学习率: {lr}")
    print(f"🎯 只训练专家头部，参数数量: {total_params:,}")
    
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


# ❌ 已弃用：DemoStateSampler（时间步轮换逻辑）
# 新逻辑：直接使用demo第一帧，无时间步轮换，符合RIPT原版
"""
class DemoStateSampler:
    RIPT对齐的Demo状态采样器 - 同demo用同一状态，不同demo轮换状态
    已弃用：改为直接使用demo第一帧
    
    def __init__(self):
        self.demo_to_state_cache = {}  # 缓存：demo_id -> 选定的状态索引
        self.next_state_idx = 0  # 下一个新demo使用的状态索引
        
    def get_next_init_state(self, demo_initial_state):
        从demo中获取下一个初始状态（按顺序轮换）
        
        Args:
            demo_initial_state: LIBERO demo数据
            
        Returns:
            tuple: (selected_state_numpy, init_state_hash, state_description)
        
        if demo_initial_state is None:
            return None, None, "无demo数据"
            
        if 'init_state' not in demo_initial_state or demo_initial_state['init_state'] is None:
            return demo_initial_state['initial_obs'], "obs_fallback", "使用观测数据（无MuJoCo状态）"
            
        # 使用demo中的MuJoCo状态
        init_state_data = demo_initial_state['init_state']
        states = init_state_data['states']  # (T, state_dim)
        pad_mask = init_state_data['pad_mask']  # (T,)
        
        # 获取所有有效状态索引
        valid_indices = torch.where(pad_mask)[0]
        if len(valid_indices) == 0:
            return demo_initial_state['initial_obs'], "obs_fallback", "demo状态无有效数据，回退到观测"
        
        # 🔥 关键修复：同demo用同一状态，不同demo轮换状态
        # 构建 demo 唯一键：task_name|demo_id（均来自collate后的batch）
        task_name_str = str(demo_initial_state['task_name'][0])
        demo_id_val = demo_initial_state.get('demo_id', [None])[0]
        demo_uid = f"{task_name_str}|{demo_id_val}"
        
        if demo_uid not in self.demo_to_state_cache:
            # 新demo：分配下一个状态索引
            current_valid_idx = self.next_state_idx % len(valid_indices)
            self.demo_to_state_cache[demo_uid] = current_valid_idx
            self.next_state_idx += 1
            print(f"  🎯 新demo {demo_uid}: 分配状态索引 {current_valid_idx}")
        else:
            # 已知demo：复用之前分配的状态索引
            current_valid_idx = self.demo_to_state_cache[demo_uid]
            print(f"  🎯 复用demo {demo_uid}: 状态索引 {current_valid_idx}")
        
        selected_state_idx = valid_indices[current_valid_idx]
        selected_state = states[selected_state_idx]
        
        # 生成状态哈希用于追踪
        state_hash = f"demo_{demo_uid}_state_{selected_state_idx.item()}"
        state_desc = f"Demo {demo_uid} 状态 {current_valid_idx+1}/{len(valid_indices)} (索引: {selected_state_idx.item()})"
        
        print(f"  🎯 轮换选择: {state_desc}")
        
        return selected_state.numpy(), state_hash, state_desc
"""

# ❌ 已弃用：全局状态采样器实例
# global_demo_sampler = DemoStateSampler()


def collect_rollouts_ript_vla_style(env_runner, task_name, num_rollouts, enable_dynamic_sampling: bool = False, stats_tracker: Optional[RolloutStatsTracker] = None, demo_initial_state=None, n_video: int = 0):
    """
    RIPT-VLA风格的rollout收集（增强版：支持per-init跳过和demo初始状态）

    Args:
        demo_initial_state: 来自LIBERO数据集的demo初始状态（可选）
    """
    print(f"正在收集 {num_rollouts} 个rollouts...")

    try:
        # 🔥 多任务增强：处理demo初始状态（随机时间步策略）
        selected_state = None
        state_hash = None  # 用于统计跟踪
        if demo_initial_state is not None:
            print(f"  📋 使用LIBERO demo: 任务 {demo_initial_state['task_name'][0]}")
            task_id = demo_initial_state['task_id'][0].item()

            # 🔥 检查是否是基准初始状态（评估模式）
            if 'benchmark_init_state' in demo_initial_state:
                # 评估模式：直接使用benchmark提供的基准初始状态
                benchmark_state = demo_initial_state['benchmark_init_state']
                selected_state = np.ascontiguousarray(benchmark_state, dtype=np.float64)
                print(f"  ✅ 使用基准初始状态 (评估模式, dim={selected_state.shape[0]})")
            elif 'init_state' in demo_initial_state:
                # 训练模式：从demo序列中随机选择时间步
                init_state_data = demo_initial_state['init_state']
                states = init_state_data['states'][0]  # [T, D]
                pad_mask = init_state_data['pad_mask'][0]  # [T]
                
                if pad_mask.any():
                    valid_indices = torch.where(pad_mask)[0]  # 所有有效时间步
                    # 🔥 随机选择一个有效时间步（彻底随机，无首帧偏好）
                    random_idx = int(valid_indices[torch.randint(0, len(valid_indices), (1,))].item())
                    state_vec = states[random_idx]  # 随机时间步的状态
                    selected_state = np.ascontiguousarray(state_vec.numpy(), dtype=np.float64)
                    print(f"  ✅ 随机时间步采样 {random_idx}/{len(valid_indices)-1} (训练模式, 总长度T={len(states)}, 有效长度={len(valid_indices)}, dim={selected_state.shape[0]})")
                else:
                    selected_state = None
                    print(f"  ⚠️ 无有效时间步，回退环境默认初始化")
            else:
                selected_state = None
                print(f"  ⚠️ 缺少init_state数据，回退环境默认初始化")

            if selected_state is not None:
                all_init_states = [selected_state]      # 单向量 → 并行时由 runner 广播
                if stats_tracker is not None:
                    state_hash = stats_tracker._compute_init_hash(task_id, selected_state)
                print(f"  ✅ 使用随机时间步作为初始状态（dim={selected_state.shape[0]}）")
            else:
                all_init_states = None
                print(f"  ⚠️ 初始状态缺失，回退环境默认初始化")
        else:
            # 获取任务的初始状态和task_id
            task_id = 0  # 简化处理，使用第一个任务
            if hasattr(env_runner, 'benchmark'):
                all_init_states = env_runner.benchmark.get_task_init_states(task_id)
            else:
                all_init_states = None
        
        # 🔥 如果有统计跟踪器，先检查是否应该跳过这个任务
        if stats_tracker and all_init_states is not None:
            # 🔥 RIPT对齐：使用精确的状态哈希，而不是随机选择
            if state_hash is not None:
                # 使用从demo采样器获取的精确状态哈希
                init_hash = state_hash
            else:
                # 回退到第一个状态（避免随机性）
                sample_init_state = all_init_states[0]
                init_hash = stats_tracker._compute_init_hash(task_id, sample_init_state)
            
            if stats_tracker.should_skip_init(task_id, init_hash, num_rollouts):
                stats_tracker.increment_skip_count(task_id, init_hash)
                print(f"🚫 跳过此次收集：init ({task_id}, {init_hash}) 最近全成功")
                return []
        
        # 直接调用环境runner的方法
        # 🔥 将n_video数量参数转换为布尔值传递给环境运行器
        save_video_flag = n_video > 0
        rollout_generator = env_runner.run_policy_in_env(
            env_name=task_name,
            all_init_states=all_init_states,
            debug_save_video=save_video_flag  # 🔥 传递视频保存布尔标志
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
            
            # 🔥 添加init_hash信息（如果可用）
            if 'init_hash' not in episode and stats_tracker:
                # 尝试从episode_data中提取初始状态信息
                if 'init_state' in episode_data:
                    init_hash = stats_tracker._compute_init_hash(task_id, episode_data['init_state'])
                    episode['computed_init_hash'] = init_hash
            
            collected_rollouts.append(episode)
            rollout_count += 1
            
            if rollout_count >= num_rollouts:
                break
        
        # 🔥 更新统计跟踪器
        if stats_tracker and collected_rollouts:
            # 提取成功率信息
            successes = [ep.get('success', False) for ep in collected_rollouts]
            
            # 获取init_hash（使用episode中的或计算得到的）
            init_hash = None
            for ep in collected_rollouts:
                if 'init_hash' in ep:
                    init_hash = ep['init_hash']
                    break
                elif 'computed_init_hash' in ep:
                    init_hash = ep['computed_init_hash']
                    break
            
            if init_hash:
                stats_tracker.update_stats(task_id, init_hash, successes)
        
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

def compute_advantages_rloo(episodes: List[Dict], rloo_batch_size: int = None) -> torch.Tensor:
    """
    正宗的RLOO (Reward Ranked Leave-One-Out) 优势计算
    
    Args:
        episodes: 收集的episodes列表
        rloo_batch_size: RLOO批次大小，用于Leave-One-Out计算
    
    Returns:
        torch.Tensor: 计算得到的优势值
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

def update_policy_ript_vla_style(policy, optimizer, cfg_adapter, episodes, advantages, device, config=None):
    """
    RIPT-VLA风格的策略更新（支持梯度累积）
    """
    if not episodes or len(advantages) == 0:
        print("⚠️ 没有有效数据进行策略更新")
        return 0.0
    
    # 检查是否需要梯度累积
    gradient_accumulation_steps = 1
    if config:
        gradient_accumulation_steps = config.get('algo', {}).get('gradient_accumulation_steps', 1)
    
    if gradient_accumulation_steps > 1:
        return update_policy_with_gradient_accumulation(policy, optimizer, cfg_adapter, episodes, advantages, device, gradient_accumulation_steps, config)
    else:
        return update_policy_simple(policy, optimizer, cfg_adapter, episodes, advantages, device, config)

def update_policy_with_gradient_accumulation(policy, optimizer, cfg_adapter, episodes, advantages, device, gradient_accumulation_steps, config=None):
    """
    梯度累积版本的策略更新（AMP增强 + 窗口级微批处理）
    """
    total_episodes = len(episodes)

    print(f"🔧 窗口级微批梯度累积:")
    print(f"   总episodes: {total_episodes}")
    print(f"   累积步数: {gradient_accumulation_steps}")

    # 🔥 Phase 3: 数据利用率监控
    if hasattr(cfg_adapter, 'use_so100_processing') and cfg_adapter.use_so100_processing:
        # 估算训练样本数量 (基于平均轨迹长度)
        avg_episode_length = sum(len(ep.get('actions', [])) for ep in episodes) / len(episodes)
        estimated_samples = max(0, avg_episode_length - 50 + 1) * len(episodes)
        utilization_ratio = estimated_samples / len(episodes) if len(episodes) > 0 else 0
        print(f"📊 SO100数据利用率: {len(episodes)} episodes → ~{estimated_samples:.0f} samples ({utilization_ratio:.1f}x)")

    
    policy.train()
    
    # 🔥 关键：使用AMP的GradScaler（新版本API）
    try:
        scaler = torch.amp.GradScaler('cuda')  # 新版本API
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler()  # 旧版本兼容
    
    # 使用标准梯度累积方法
    return update_policy_with_gradient_accumulation_fallback(
        policy, optimizer, cfg_adapter, episodes, advantages, device, gradient_accumulation_steps, scaler, config=config
    )

def update_policy_with_gradient_accumulation_fallback(policy, optimizer, cfg_adapter, episodes, advantages, device, gradient_accumulation_steps, scaler, config=None):
    """
    回退版本的梯度累积（保持向后兼容）
    """
    total_episodes = len(episodes)
    mini_batch_size = max(1, total_episodes // gradient_accumulation_steps)
    
    # 🚀 统一样本池训练（无mini-batch分割）
    print(f"🔧 统一样本池训练:")
    print(f"   总episodes: {total_episodes}")
    print(f"   累积步数: {gradient_accumulation_steps}")
    
    total_loss = 0.0
    
    # 🔥 直接使用所有episodes，不再分割mini-batch
    try:
        # 🚀 使用autocast包裹forward计算
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            advantages = advantages.to(device)
            
            # 🚀 RIPT对齐：强制使用SO100处理，不允许回退
            if not (hasattr(cfg_adapter, 'use_so100_processing') and cfg_adapter.use_so100_processing):
                raise RuntimeError("❌ RIPT对齐要求：必须启用use_so100_processing配置")
            
            print("🚀 使用SO100统一样本池训练（RIPT对齐模式）...")
            # 从config读取参数，提供合理默认值
            batch_size_cfg = config.get('unified_pool_batch_size', 8) if config else 8
            shuffle_cfg = config.get('unified_pool_shuffle', True) if config else True
            print(f"  配置参数: batch_size={batch_size_cfg}, shuffle={shuffle_cfg}")
            
            loss = cfg_adapter.compute_weighted_loss_unified(
                episodes=episodes,  # 🔥 使用所有episodes
                advantages=advantages,  # 🔥 使用所有advantages
                device=device,
                batch_size=batch_size_cfg,
                shuffle_samples=shuffle_cfg,
                scaler=scaler,
                optimizer=optimizer,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            
            total_loss = loss.item()
            print(f"✅ SO100统一样本池训练完成，总损失: {total_loss}")
    
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        total_loss = 0.0
    
    print(f"✓ 统一样本池训练完成，总损失: {total_loss:.6f}")
    return total_loss

def update_policy_simple(policy, optimizer, cfg_adapter, episodes, advantages, device, config=None):
    """简单版本的策略更新（无梯度累积）"""
    print(f"正在更新策略（{len(episodes)} 个episodes）...")

    # 🔥 Phase 3: 数据利用率监控
    if hasattr(cfg_adapter, 'use_so100_processing') and cfg_adapter.use_so100_processing:
        # 估算训练样本数量 (基于平均轨迹长度)
        avg_episode_length = sum(len(ep.get('actions', [])) for ep in episodes) / len(episodes)
        estimated_samples = max(0, avg_episode_length - 50 + 1) * len(episodes)
        utilization_ratio = estimated_samples / len(episodes) if len(episodes) > 0 else 0
        print(f"📊 SO100数据利用率: {len(episodes)} episodes → ~{estimated_samples:.0f} samples ({utilization_ratio:.1f}x)")

    try:
        # 计算加权损失
        advantages = advantages.to(device)

        # 🚀 RIPT对齐：强制使用SO100处理，不允许回退
        if not (hasattr(cfg_adapter, 'use_so100_processing') and cfg_adapter.use_so100_processing):
            raise RuntimeError("❌ RIPT对齐要求：必须启用use_so100_processing配置")
            
        print("🚀 使用SO100统一样本池训练（RIPT对齐模式）...")
        # 从配置读取可调参数，提供合理默认值
        batch_size_cfg = config.get('unified_pool_batch_size', 8) if config else 8
        shuffle_cfg = config.get('unified_pool_shuffle', True) if config else True
        
        print(f"  配置参数: batch_size={batch_size_cfg}, shuffle={shuffle_cfg}")

        loss = cfg_adapter.compute_weighted_loss_unified(
            episodes=episodes,
            advantages=advantages,
            device=device,
            batch_size=batch_size_cfg,
            shuffle_samples=shuffle_cfg,
            optimizer=optimizer,
            scaler=None,
            gradient_accumulation_steps=1
        )

        loss_value = loss.item()
        print(f"✓ 策略更新完成，损失: {loss_value:.6f}")

        return loss_value
        
    except Exception as e:
        print(f"❌ 策略更新失败: {e}")
        traceback.print_exc()
        return 0.0

def evaluate_ript_style_all_tasks(policy, env_runner, config, rollouts_per_task=10, is_final_eval=False):
    """
    🎯 RIPT式全任务评估：与原版RIPT评估逻辑完全对齐
    
    特点：
    1. 遍历所有任务
    2. 使用基准初始状态池（benchmark.get_task_init_states）
    3. 按索引循环取初始状态（非随机）
    4. 禁用动态采样和状态跳过
    5. 从头开始完整评估任务能力
    
    Args:
        policy: 待评估的策略模型
        env_runner: 环境运行器
        config: 配置字典
        rollouts_per_task: 每个任务的rollout次数
        
    Returns:
        dict: 包含per-task和overall成功率的评估结果
    """
    print("\n🎯 开始RIPT式全任务评估...")
    
    # 获取任务列表
    task_names = config.get('task', {}).get('task_names_to_use', [])
    if not task_names:
        print("⚠️ 没有指定评估任务，跳过评估")
        return {}
    
    # 🔥 获取视频保存配置
    rollout_config = config.get('rollout', {})
    n_video_regular = rollout_config.get('n_video', 0)  # 常规评估视频数
    n_video_final = rollout_config.get('n_video_final', 1)  # 最终评估视频数
    n_video = n_video_final if is_final_eval else n_video_regular
    
    # 设置评估模式：固定随机种子确保可复现
    eval_seed = 42
    torch.manual_seed(eval_seed)
    np.random.seed(eval_seed)
    random.seed(eval_seed)
    
    all_results = {}
    overall_successes = []
    
    print(f"📊 评估配置:")
    print(f"  任务数量: {len(task_names)}")
    print(f"  每任务rollout数: {rollouts_per_task}")
    print(f"  评估随机种子: {eval_seed}")
    print(f"  视频保存: {'🎬最终评估' if is_final_eval else '📹常规评估'} - {n_video} 个视频/任务")
    
    # 遍历所有任务
    for task_idx, task_name in enumerate(task_names):
        print(f"\n🔄 评估任务 {task_idx+1}/{len(task_names)}: {task_name}")
        
        # 从benchmark获取任务名列表并将传入任务名映射到真实task_id
        resolved_task_id = None
        bench_task_names = []
        if hasattr(env_runner, 'benchmark') and env_runner.benchmark is not None:
            try:
                bench_task_names = env_runner.benchmark.get_task_names()
            except Exception:
                bench_task_names = []
        if bench_task_names:
            if task_name in bench_task_names:
                resolved_task_id = bench_task_names.index(task_name)
            else:
                # 名称不完全一致时做一次宽松匹配（忽略大小写与非字母数字字符）
                def _mk_key(s):
                    return ''.join(ch for ch in str(s).lower() if ch.isalnum())
                name_key = _mk_key(task_name)
                key_to_idx = {_mk_key(n): i for i, n in enumerate(bench_task_names)}
                if name_key in key_to_idx:
                    resolved_task_id = key_to_idx[name_key]
                else:
                    # 兜底：尝试子串匹配
                    candidates = [i for i, n in enumerate(bench_task_names) if _mk_key(n).find(name_key) != -1 or name_key.find(_mk_key(n)) != -1]
                    if candidates:
                        resolved_task_id = candidates[0]
        
        # 获取该任务的基准初始状态池（使用映射后的真实task_id）
        task_init_states = []
        if hasattr(env_runner, 'benchmark') and env_runner.benchmark is not None and resolved_task_id is not None:
            try:
                task_init_states = env_runner.benchmark.get_task_init_states(resolved_task_id)
                print(f"  ✅ 从benchmark获取 {len(task_init_states)} 个基准初始状态 (task_id={resolved_task_id})")
            except Exception as e:
                print(f"  ⚠️ 获取基准状态失败: {e}，使用环境默认")
                task_init_states = []
        else:
            if not bench_task_names:
                print(f"  ⚠️ Benchmark未初始化，使用环境默认重置")
            else:
                print(f"  ⚠️ 任务名无法映射到有效task_id，使用环境默认重置: {task_name}")
            task_init_states = []
        
        # 执行该任务的多次rollout
        task_successes = []
        for rollout_idx in range(rollouts_per_task):
            try:
                # 🔥 关键：使用基准初始状态池的循环索引（与RIPT对齐）
                has_task_init_states = False
                if isinstance(task_init_states, np.ndarray):
                    has_task_init_states = (task_init_states.size > 0)
                else:
                    has_task_init_states = (len(task_init_states) > 0)
                if has_task_init_states:
                    init_state_idx = rollout_idx % len(task_init_states)
                    init_state = task_init_states[init_state_idx]
                    print(f"    📍 Rollout {rollout_idx+1}: 使用基准状态 {init_state_idx}")
                    
                    # 构造demo_initial_state格式（兼容现有接口）
                    # 确保初始状态是连续的numpy数组
                    if isinstance(init_state, np.ndarray):
                        init_state = np.ascontiguousarray(init_state, dtype=np.float64)
                    
                    demo_initial_state = {
                        'task_name': [task_name],
                        'task_id': torch.tensor([resolved_task_id if resolved_task_id is not None else 0]),
                        'benchmark_init_state': init_state  # 确保是连续数组
                    }
                else:
                    print(f"    📍 Rollout {rollout_idx+1}: 使用环境默认重置")
                    demo_initial_state = None
                
                # 🔥 禁用动态采样，每次只收集1个rollout
                episodes = collect_rollouts_ript_vla_style(
                    env_runner, task_name, 1,
                    enable_dynamic_sampling=False,  # 🔥 评估时禁用
                    stats_tracker=None,             # 🔥 评估时禁用状态跟踪
                    demo_initial_state=demo_initial_state,
                    n_video=n_video if rollout_idx < n_video else 0  # 🔥 视频保存：前n个rollout保存视频
                )
                
                if episodes and len(episodes) > 0:
                    # 修复：处理可能的NumPy数组success值
                    success = episodes[0].get('success', False)
                    # 如果success是数组，使用.any()或.all()处理
                    if hasattr(success, 'shape') and len(getattr(success, 'shape', [])) > 0:
                        if hasattr(success, 'any'):
                            success = success.any()  # 任一元素为真则为真
                        else:
                            success = bool(success)  # 兜底转换
                    task_successes.append(success)
                    print(f"      结果: {'✅成功' if success else '❌失败'}")
                else:
                    print(f"      结果: ❌无效rollout")
                    task_successes.append(False)
                    
            except Exception as e:
                print(f"    ❌ Rollout {rollout_idx+1} 执行失败: {e}")
                task_successes.append(False)
        
        # 计算该任务的成功率 - 确保正确处理可能的NumPy数组
        task_successes_processed = []
        for success in task_successes:
            # 处理可能的NumPy数组或其他复杂类型
            if hasattr(success, 'shape') and len(getattr(success, 'shape', [])) > 0:
                if hasattr(success, 'any'):
                    success = success.any()
                else:
                    success = bool(success)
            task_successes_processed.append(bool(success))
            
        task_success_rate = np.mean(task_successes_processed) if task_successes_processed else 0.0
        all_results[task_name] = task_success_rate
        overall_successes.extend(task_successes_processed)
        
        print(f"  📊 任务 {task_name}: {task_success_rate:.2%} ({sum(task_successes_processed)}/{len(task_successes_processed)})")
    
    # 计算总体成功率 - 处理任意类型成功标记
    overall_successes_processed = []
    for success in overall_successes:
        # 处理可能的NumPy数组或其他复杂类型
        if hasattr(success, 'shape') and len(getattr(success, 'shape', [])) > 0:
            if hasattr(success, 'any'):
                success = success.any()
            else:
                success = bool(success)
        overall_successes_processed.append(bool(success))
        
    overall_success_rate = np.mean(overall_successes_processed) if overall_successes_processed else 0.0
    all_results['overall_success_rate'] = overall_success_rate
    
    print(f"\n🎉 RIPT式评估完成!")
    print(f"📊 总体成功率: {overall_success_rate:.2%} ({sum(overall_successes_processed)}/{len(overall_successes_processed)})")
    print(f"📋 Per-task 成功率:")
    for task_name, success_rate in all_results.items():
        if task_name != 'overall_success_rate':
            print(f"  {task_name}: {success_rate:.2%}")
    
    return all_results

def verify_random_sampling_effectiveness(config, task_names, max_samples=10):
    """
    🔍 验证随机采样的有效性
    收集样本并分析任务分布和时间步分布
    """
    print("\n🔍 验证RIPT-VLA随机采样对齐效果:")
    
    task_counts = {name: 0 for name in task_names}
    timestep_samples = []
    
    # 模拟采样过程
    for i in range(max_samples):
        # 任务选择验证
        if len(task_names) > 1:
            selected_task = random.choice(task_names)
            task_counts[selected_task] += 1
        
        # 时间步选择验证（模拟）
        simulated_valid_timesteps = torch.randint(1, 200, (torch.randint(50, 150, (1,)).item(),))
        if len(simulated_valid_timesteps) > 0:
            random_timestep = int(simulated_valid_timesteps[torch.randint(0, len(simulated_valid_timesteps), (1,))].item())
            timestep_samples.append(random_timestep)
    
    print(f"📊 任务选择分布 (共{max_samples}次采样):")
    for task, count in task_counts.items():
        percentage = (count / max_samples) * 100
        print(f"  {task}: {count} 次 ({percentage:.1f}%)")
    
    if timestep_samples:
        print(f"📊 时间步选择分布:")
        print(f"  范围: [{min(timestep_samples)}, {max(timestep_samples)}]")
        print(f"  平均: {np.mean(timestep_samples):.1f}")
        print(f"  标准差: {np.std(timestep_samples):.1f}")
    
    # 检查是否充分随机化
    task_distribution_uniform = all(abs(count - max_samples/len(task_names)) <= 3 for count in task_counts.values())
    timestep_range_good = len(timestep_samples) > 0 and (max(timestep_samples) - min(timestep_samples)) > 50
    
    if task_distribution_uniform and timestep_range_good:
        print("✅ 随机采样验证通过：分布充分随机化")
        return True
    else:
        print("⚠️ 随机采样可能需要调整：分布不够均匀")
        return False

def evaluate_with_cfg_sweep(policy, env_runner, task_name, eval_episodes=3):
    """🔥 新增：评估不同CFG强度的效果"""
    cfg_scales = [1.0, 1.5, 3.0, 5.0]
    best_cfg = 1.0
    best_success_rate = 0.0
    
    results = {}
    print(f"\n🔍 开始CFG强度扫描评估...")
    
    for cfg_scale in cfg_scales:
        print(f"📊 测试CFG={cfg_scale}...")
        # 临时设置CFG强度
        original_cfg = getattr(env_runner.config, 'collection_cfg_scale', None)
        if original_cfg is None and hasattr(env_runner.config, 'algo'):
            original_cfg = getattr(env_runner.config.algo, 'collection_cfg_scale', None)
        if original_cfg is None:
            print("⚠️ CFG扫描：未找到collection_cfg_scale配置，使用1.5")
            original_cfg = 1.5
        env_runner.config.collection_cfg_scale = cfg_scale
        
        # 运行评估episodes
        success_count = 0
        for ep_idx in range(eval_episodes):
            try:
                # 使用现有的rollout收集函数
                episodes = collect_rollouts_ript_vla_style(
                    env_runner, task_name, 1, enable_dynamic_sampling=False, n_video=0  # 🔥 CFG评估不保存视频
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
        env_runner.config.collection_cfg_scale = original_cfg
        
        print(f"   CFG={cfg_scale}: 成功率={success_rate:.2%} ({success_count}/{eval_episodes})")
    
    print(f"🎯 最佳CFG强度: {best_cfg} (成功率: {best_success_rate:.2%})")
    return best_cfg, results

def main_training_loop_ript_vla_style(config: Dict[str, Any]):
    """
    主训练循环（RIPT-VLA风格）
    直接在主函数中处理所有逻辑，减少抽象层
    """
    print("🚀 开始RIPT-VLA风格的训练循环")
    
    # 🔥 读取采样策略配置
    features = config.get('features', {})
    sampling_config = features.get('sampling_strategy', {})
    init_state_sampling = sampling_config.get('init_state_sampling', 'random')
    task_selection = sampling_config.get('task_selection', 'random')
    sampling_seed = sampling_config.get('sampling_seed', 42)
    
    print(f"🔧 RIPT-VLA对齐配置:")
    print(f"  初始状态采样策略: {init_state_sampling}")
    print(f"  任务选择策略: {task_selection}")
    print(f"  采样随机种子: {sampling_seed}")
    
    # 统一设置随机种子（保证可复现）
    try:
        training_seed = int(config.get('training', {}).get('seed', 42))
        random.seed(sampling_seed); np.random.seed(sampling_seed); torch.manual_seed(training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_seed)
        print(f"✅ 训练随机种子: {training_seed}, 采样随机种子: {sampling_seed}")
    except Exception as _e:
        print(f"⚠️ 随机种子设置失败: {_e}")
    
    # 🔥 验证随机采样效果
    task_names = config.get('task', {}).get('task_names_to_use', ['default_task'])
    if len(task_names) > 1:
        verify_random_sampling_effectiveness(config, task_names, max_samples=20)
    
    # 🔥 设置数值优化和显存管理
    print("🔧 设置数值优化...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    print("✅ TF32和显存优化已启用")
    
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
    # 🔥 Phase 3: 数据处理配置 (Legacy + SO100)
    dataset_config = config.get('dataset', {})
    data_processing_config = config.get('data_processing', {})
    policy_config = config.get('policy', {})

    # CFG状态检查
    cfg_enabled = getattr(policy.model, 'cfg_enabled', True)

    # SO100处理配置 (Phase 3新增) - 修复：从正确的配置路径读取
    use_so100_processing = data_processing_config.get('use_so100_processing', False)

    # Legacy窗口化配置 (向后兼容) - 修复：从正确的配置路径读取
    windowing_mode = data_processing_config.get('windowing_mode', 'last')
    window_stride = data_processing_config.get('window_stride', 10)
    max_windows_per_episode = data_processing_config.get('max_windows_per_episode', 1)

    print(f"\n🔧 训练配置:")
    print(f"  CFG模式: {'启用' if cfg_enabled else '禁用'}")
    print(f"  SO100处理: {'启用' if use_so100_processing else '禁用 (使用Legacy窗口化)'}")
    if not use_so100_processing:
        print(f"  窗口化模式: {windowing_mode}")
        print(f"  窗口步长: {window_stride}")
        print(f"  每episode最大窗口数: {max_windows_per_episode}")
    else:
        print(f"  数据利用率: 预期50-150x提升")
        print(f"  样本生成: 每个轨迹生成L-50+1个训练样本")

    cfg_adapter = PI0_CFG_Adapter(
        policy=policy,
        norm_stats_path=(config.get('norm_stats_path') if isinstance(config, dict) else getattr(config, 'norm_stats_path', None)) or f"{config['policy_path']}/norm_stats.json",
        use_so100_processing=use_so100_processing,  # 🔥 Phase 3: 新增SO100支持
        windowing_mode=windowing_mode,
        window_stride=window_stride,
        max_windows_per_episode=max_windows_per_episode
    )
    
    # 创建环境runner
    env_runner = create_environment_runner(config, policy)
    
    # 🔥 创建rollout统计跟踪器
    stats_path = config['algo'].get('rollout_stats_path', './output/stage11_ript_vla/rollout_stats.json')
    rollout_skip_threshold = config['algo'].get('rollout_skip_threshold', 3)
    stats_tracker = RolloutStatsTracker(
        rollout_skip_threshold=rollout_skip_threshold,
        stats_path=stats_path
    )
    
    # 🔥 解耦demo_batch_size与rloo_batch_size
    demo_batch_size = config['algo'].get('demo_batch_size', 6)  # 改为默认6，与原版RIPT一致
    rloo_batch_size = config['algo']['rloo_batch_size']
    num_train_steps = config['training']['num_train_steps']
    task_names = config['task'].get('task_names_to_use', ['LIBERO_SPATIAL_0'])

    # 🔥 创建多任务RIPT对齐的LIBERO demo数据加载器
    use_libero_demos = config.get('use_libero_demos', True)
    if use_libero_demos:
        try:
            # 从配置中获取数据路径
            libero_data_prefix = config.get('libero_data_prefix', '/zhaohan/ZJH/openpi_pytorch/datasets')
            benchmark_name = config.get('benchmark_name', 'libero_spatial')

            # 🔥 多任务支持：自动获取任务名列表
            if not task_names or task_names == ['LIBERO_SPATIAL_0']:
                try:
                    from libero.libero.benchmark import get_benchmark
                    bm = get_benchmark(benchmark_name.lower())()
                    task_names = bm.get_task_names()
                    print(f"🎯 自动获取任务列表: {len(task_names)} 个任务")
                except Exception as e:
                    # 回退：保留原来的一个任务
                    task_names = ['pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate']
                    print(f"⚠️ 自动获取任务失败，使用默认任务: {e}")

            # 🔥 构建每任务一个DataLoader/迭代器的字典结构
            from torch.utils.data import DataLoader
            task_to_loader = {}
            task_to_iter = {}

            for tname in task_names:
                # 为每个任务创建单独的数据集
                ds = build_dataset_ript_aligned(
                    data_prefix=libero_data_prefix,
                    suite_name="libero",
                    benchmark_name=benchmark_name,
                    task_names_to_use=[tname],   # 🔥 单任务过滤
                    load_state=True,
                    seq_len=600,
                    n_demos=50
                )
                # 关键：每任务loader只取一个子demo
                dl = DataLoader(
                    ds,
                    batch_size=1,
                    shuffle=True,                # 🔥 子demo随机打散，与RIPT-VLA完全对齐
                    collate_fn=collate_fn_ript_aligned,
                    num_workers=0
                )
                task_to_loader[tname] = dl
                task_to_iter[tname] = iter(dl)

            print(f"✅ 按任务构建DataLoader: {len(task_names)} 个任务")
            print(f"  数据路径: {libero_data_prefix}")
            print(f"  基准: {benchmark_name}")
            print(f"  任务列表: {task_names}")
            print(f"  🔥 包含MuJoCo状态: True")
            
            # 🔥 多任务模式提示
            if len(task_names) > 1:
                print(f"  🎯 多任务模式: 启用任务轮询，每组轮换不同任务")
                print(f"  📋 子demo轮换: 严格按顺序轮换（demo_0 → demo_1 → demo_2 ...）")
            else:
                print(f"  📍 单任务模式: 所有组使用同一任务，仅demo轮换")
                print(f"  📋 子demo轮换: 严格按顺序轮换（demo_0 → demo_1 → demo_2 ...）")
                print(f"  💡 提示: 要测试多任务轮换，请在配置中添加更多task_names_to_use")

            # 兼容性：保留原有变量（但会在后续逻辑中被task_to_*替代）
            demo_dataloader = None
            demo_data_iter = None
            
            # 🔥 为未来eval功能预留信息：存储到运行时配置
            if hasattr(env_runner, 'runtime') and env_runner.runtime is not None:
                env_runner.runtime.update({
                    'task_names': task_names,
                    'task_to_loader': task_to_loader,   # eval 如需也可复用
                })
            else:
                # 创建运行时配置字典
                env_runner.runtime = {
                    'task_names': task_names,
                    'task_to_loader': task_to_loader,   # eval 如需也可复用
                }
            print(f"✅ 多任务运行时配置已设置，支持未来eval功能")
            
        except Exception as e:
            print(f"⚠️ 多任务demo加载器创建失败: {e}")
            print("  将使用传统的环境重置方式")
            task_to_loader = {}
            task_to_iter = {}
            demo_dataloader = None
            demo_data_iter = None
    else:
        task_to_loader = {}
        task_to_iter = {}
        demo_dataloader = None
        demo_data_iter = None
    
    print(f"\n🔧 批次配置:")
    print(f"  demo_batch_size: {demo_batch_size} (每步收集的组数)")
    print(f"  rloo_batch_size: {rloo_batch_size} (每组内样本数)")
    print(f"  有效批次大小: {demo_batch_size * rloo_batch_size}")
    dynamic_sampling_enabled = config.get('features', {}).get('dynamic_sampling', {}).get('enabled', False)
    print(f"  动态采样: {'启用' if dynamic_sampling_enabled else '禁用'} (features.dynamic_sampling.enabled)")
    
    print(f"\n开始训练循环:")
    print(f"  训练步数: {num_train_steps}")
    print(f"  任务: {task_names}")
    print(f"  使用LIBERO demos: {'是' if demo_dataloader else '否'}")
    print()
    
    all_training_metrics = []
    
    # 🔥 多任务轮询指针（训练步外初始化）
    task_cursor = 0
    
    # 🔥 显存监控函数
    def print_gpu_memory(step_name: str):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            print(f"📊 {step_name} - GPU显存: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, 峰值: {max_allocated:.2f}GB")
    
    # 🔥 主训练循环 - 多任务轮询+按组收集模式（仅在成功收集后计步）
    steps_done = 0
    max_collection_retries = int(config.get('training', {}).get('max_collection_retries', 100))
    while steps_done < num_train_steps:
        step_start_time = time.time()
        torch.cuda.reset_peak_memory_stats()  # 重置峰值监控
        
        print(f"=== 训练步骤 {steps_done + 1}/{num_train_steps} ===")
        print_gpu_memory("步骤开始")
        
        # 1. 按组收集rollouts（带重试，解耦demo_batch_size与rloo_batch_size）
        all_collected_episodes = []
        successful_groups = 0
        collection_attempt = 0
        while not all_collected_episodes:
            collection_attempt += 1
            if collection_attempt > 1:
                print(f"🔁 收集重试 {collection_attempt}/{max_collection_retries}")
            
            # 每次尝试清空上一轮结果
            all_collected_episodes = []
            successful_groups = 0

            for group_idx in range(demo_batch_size):
                print(f"🔄 收集第 {group_idx + 1}/{demo_batch_size} 组...")

                # 🔥 智能任务选择：多任务随机选择 vs 单任务直选
                demo_batch = None
                if task_to_iter:
                    if len(task_names) > 1:
                        # 🔥 多任务环境：随机选择任务，与RIPT-VLA对齐
                        tname = random.choice(task_names)
                        print(f"  🎲 随机任务选择: 组{group_idx} -> 任务 {tname}")
                    else:
                        # 单任务环境：直接使用唯一任务
                        tname = task_names[0]
                        print(f"  📍 单任务模式: 组{group_idx} -> 任务 {tname}")
                
                    try:
                        demo_batch = next(task_to_iter[tname])
                        demo_id = demo_batch.get('demo_id', [None])[0]
                        print(f"  📋 使用子demo: {demo_id} (任务: {tname})")
                    except StopIteration:
                        # 重新初始化该任务的迭代器
                        task_to_iter[tname] = iter(task_to_loader[tname])
                        demo_batch = next(task_to_iter[tname])
                        demo_id = demo_batch.get('demo_id', [None])[0]
                        print(f"  📋 重新开始任务迭代: 子demo {demo_id} (任务: {tname})")
                    except Exception as e:
                        print(f"  ⚠️ Demo获取失败: {e}")
                        demo_batch = None
                        tname = task_names[0] if len(task_names) == 1 else task_names[(task_cursor + group_idx) % len(task_names)]
                else:
                    # 回退到原有逻辑（兼容性）
                    if demo_data_iter is not None:
                        try:
                            demo_batch = next(demo_data_iter)
                            demo_id = demo_batch.get('demo_id', [None])[0]
                            print(f"  📋 使用子demo: {demo_id} (任务{demo_batch['task_id'][0].item()})")
                        except StopIteration:
                            demo_data_iter = iter(demo_dataloader)
                            demo_batch = next(demo_data_iter)
                            demo_id = demo_batch.get('demo_id', [None])[0]
                            print(f"  📋 重新开始demo迭代: 子demo {demo_id} (任务{demo_batch['task_id'][0].item()})")
                        except Exception as e:
                            print(f"  ⚠️ Demo获取失败: {e}")
                            demo_batch = None
                    
                    tname = (task_names[step % len(task_names)]) if not demo_batch else demo_batch['task_name'][0]

                # 🔥 获取训练时的视频保存配置
                training_save_video = config.get('features', {}).get('save_video', False)
                training_n_video = config.get('features', {}).get('n_video_training', 1) if training_save_video else 0
                
                # 收集一组rollouts（传递任务名和demo初始状态）
                group_episodes = collect_rollouts_ript_vla_style(
                    env_runner, tname,  # 🔥 使用轮询选择的任务名
                    rloo_batch_size,
                    enable_dynamic_sampling=config.get('features', {}).get('dynamic_sampling', {}).get('enabled', False),
                    stats_tracker=stats_tracker,
                    demo_initial_state=demo_batch,  # 🔥 传递对应任务的demo初始状态
                    n_video=training_n_video  # 🔥 训练时视频保存：使用features配置
                )
            
            if group_episodes:
                successes = [ep.get('success', False) for ep in group_episodes]
                
                # 🔥 检查组级动态采样：只在启用时才丢弃全0或全1的组
                dynamic_sampling_enabled = config.get('features', {}).get('dynamic_sampling', {}).get('enabled', False)
                
                if dynamic_sampling_enabled and len(successes) > 0 and (all(successes) or not any(successes)):
                    print(f"⚠️ 组 {group_idx + 1} 被动态采样丢弃 (uniform successes: {all(successes)})")
                else:
                    all_collected_episodes.extend(group_episodes)
                    successful_groups += 1
                    
                    if dynamic_sampling_enabled:
                        print(f"✅ 组 {group_idx + 1} 收集成功：{len(group_episodes)} episodes，"
                              f"成功率 {np.mean(successes):.2%}")
                    else:
                        print(f"✅ 组 {group_idx + 1} 收集成功：{len(group_episodes)} episodes，"
                              f"成功率 {np.mean(successes):.2%} (动态采样已禁用)")

            else:
                print(f"❌ 组 {group_idx + 1} 收集失败")
            
            # 单次尝试结束
            
        # 🔥 定期保存统计数据（按已完成步数节奏）
        if steps_done % 5 == 0:
            stats_tracker.save_stats()
        
        print(f"📊 组收集完成(尝试 {collection_attempt}): {successful_groups}/{demo_batch_size} 组成功，总episodes: {len(all_collected_episodes)}")
        print_gpu_memory("收集完成")
        
        if not all_collected_episodes:
            if collection_attempt < max_collection_retries:
                print("⚠️ 未收集到有效episodes，继续重试收集")
                continue  # 回到收集重试while
            else:
                print("⚠️ 达到最大收集重试次数，仍未收集到有效episodes，将继续重试以满足严格条件")
                continue  # 持续重试，不增加steps_done
        
        # 2. 计算优势（正宗RLOO方法）
        advantages = compute_advantages_rloo(all_collected_episodes, rloo_batch_size=rloo_batch_size)
        print_gpu_memory("优势计算完成")
        
        # 3. 更新策略（带配置传递以支持梯度累积）
        loss = update_policy_ript_vla_style(
            policy, optimizer, cfg_adapter, all_collected_episodes, advantages, device, config
        )
        print_gpu_memory("策略更新完成")
        
        # 步数仅在成功收集并完成更新后递增
        steps_done += 1
        
        # 4. 记录指标
        avg_reward = np.mean([ep['total_reward'] for ep in all_collected_episodes])
        success_rate = np.mean([ep['success'] for ep in all_collected_episodes])
        step_time = time.time() - step_start_time
        
        step_metrics = {
            'step': steps_done,
            'demo_groups': successful_groups,
            'total_episodes': len(all_collected_episodes),
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'loss': loss,
            'step_time': step_time
        }
        all_training_metrics.append(step_metrics)
        
        # 5. 输出结果
        print(f"✓ 步骤 {steps_done} 完成:")
        print(f"  成功组数: {successful_groups}/{demo_batch_size}")
        print(f"  总Episodes: {len(all_collected_episodes)}")
        print(f"  平均奖励: {avg_reward:.4f}")
        print(f"  成功率: {success_rate:.2%}")
        print(f"  损失: {loss:.6f}")
        print(f"  耗时: {step_time:.2f}秒")
        print_gpu_memory("步骤结束")
        
        # 🔥 智能任务指针推进（仅在多任务环境下有意义）
        if task_to_iter and len(task_names) > 1:
            task_cursor = (task_cursor + demo_batch_size) % len(task_names)
            print(f"🔄 多任务指针推进到: {task_cursor} (下步起始任务: {task_names[task_cursor]})")
        elif task_to_iter and len(task_names) == 1:
            print(f"📍 单任务模式: 保持使用任务 {task_names[0]} (无需指针推进)")
        
        # 6. RIPT式全任务评估（使用rollout配置参数）
        rollout_config = config.get('rollout', {})
        eval_enabled = rollout_config.get('enabled', False)
        eval_steps = rollout_config.get('steps', 10)
        rollouts_per_env = rollout_config.get('rollouts_per_env', 5)
        
        # 🔍 调试信息：打印评估配置
        print(f"🔧 评估配置检查:")
        print(f"   rollout配置: {rollout_config}")
        print(f"   eval_enabled: {eval_enabled}")
        print(f"   eval_steps: {eval_steps}")
        print(f"   rollouts_per_env: {rollouts_per_env}")
        print(f"   当前步数: {steps_done}")
        
        # 🎬 视频保存配置检查
        training_save_video = config.get('features', {}).get('save_video', False)
        training_n_video = config.get('features', {}).get('n_video_training', 1) if training_save_video else 0
        eval_n_video = rollout_config.get('n_video', 0)
        eval_n_video_final = rollout_config.get('n_video_final', 1)
        
        print(f"🎬 视频保存配置:")
        print(f"   训练时视频: {'✅启用' if training_save_video else '❌禁用'} - {training_n_video} 个/rollout")
        print(f"   评估时视频: {eval_n_video} 个/rollout (常规), {eval_n_video_final} 个/rollout (最终)")
        
        should_eval = eval_enabled and ((steps_done == 1) or (eval_steps > 0 and steps_done % eval_steps == 0))
        print(f"   should_eval: {should_eval} (计算: {eval_enabled} and (({steps_done} == 1) or ({eval_steps} > 0 and {steps_done} % {eval_steps} == 0)))")
        
        if should_eval:
            try:
                print(f"\n🎯 开始第 {steps_done} 步的全任务评估...")
                print(f"   配置: 每 {eval_steps} 步评估, {rollouts_per_env} rollouts/任务")
                if steps_done == 1:
                    print(f"   📍 初始基线评估")
                    
                eval_results = evaluate_ript_style_all_tasks(
                    policy, 
                    env_runner, 
                    config, 
                    rollouts_per_task=rollouts_per_env,  # 使用配置参数
                    is_final_eval=False  # 🔥 常规评估，不保存视频（或保存少量视频）
                )
                
                if eval_results:
                    step_metrics['overall_success_rate'] = eval_results.get('overall_success_rate', 0.0)
                    step_metrics['task_success_rates'] = {k: v for k, v in eval_results.items() if k != 'overall_success_rate'}
                    print(f"📊 评估完成 - 总体成功率: {eval_results.get('overall_success_rate', 0.0):.2%}")
                
            except Exception as e:
                print(f"⚠️ RIPT式评估失败: {e}")
                traceback.print_exc()
        elif eval_enabled and eval_steps > 0:
            next_eval_step = ((steps_done // eval_steps) + 1) * eval_steps
            print(f"📊 第 {steps_done} 步: 跳过评估 (下次评估: 第 {next_eval_step} 步)")
        elif not eval_enabled:
            print(f"📊 第 {steps_done} 步: 评估已禁用")
        
        # 6.1 CFG参数调优（可选，每20步进行一次，仅在CFG启用时）
        if (steps_done) % 20 == 0 and getattr(policy.model, 'cfg_enabled', True):
            try:
                print(f"\n🔍 开始CFG强度调优...")
                best_cfg, cfg_results = evaluate_with_cfg_sweep(policy, env_runner, task_names[0], eval_episodes=2)
                step_metrics['best_cfg_scale'] = best_cfg
                step_metrics['cfg_sweep_results'] = cfg_results
                print(f"🎯 推荐CFG强度: {best_cfg}")
                # 可选：动态调整收集时使用的CFG强度
                # 🔥 修复：写入正确的algo路径
                if hasattr(env_runner.config, 'algo'):
                    env_runner.config.algo.collection_cfg_scale = best_cfg
                if isinstance(env_runner.config, dict) and 'algo' in env_runner.config:
                    env_runner.config['algo']['collection_cfg_scale'] = best_cfg
            except Exception as e:
                print(f"⚠️ CFG评估失败: {e}")
        elif (steps_done) % 20 == 0:
            print("⚠️ CFG已禁用，跳过CFG强度评估")
        
        # 7. 保存检查点
        if (steps_done) % config['training'].get('save_freq', 10) == 0:
            # 轻量权重（仅模型，便于部署与占用小）
            weights_path = output_dir / f"weights_step_{steps_done}.pt"
            torch.save({
                'step': steps_done,
                'policy_state_dict': policy.state_dict(),
                'config': config,
                'training_metrics': all_training_metrics,
            }, weights_path)
            print(f"✓ 轻量权重已保存: {weights_path}")

            # 可选：按较低频率保存含优化器的完整检查点，便于恢复训练
            save_opt_every = config.get('training', {}).get('save_optimizer_freq', None)
            if save_opt_every and ((steps_done) % int(save_opt_every) == 0):
                checkpoint_path = output_dir / f"checkpoint_step_{steps_done}.pt"
                torch.save({
                    'step': steps_done,
                    'policy_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'training_metrics': all_training_metrics,
                }, checkpoint_path)
                print(f"✓ 完整检查点已保存: {checkpoint_path}")
    
    # 🔥 保存最终统计数据
    stats_tracker.save_stats()
    print(f"📊 最终统计: {len(stats_tracker.rollout_stats)} 个不同的init状态")
    
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

    # 🎬 最终评估（保存视频记录）
    try:
        print(f"\n🎬 开始最终评估（保存视频记录）...")
        rollout_config = config.get('rollout', {})
        final_eval_results = evaluate_ript_style_all_tasks(
            policy, 
            env_runner, 
            config, 
            rollouts_per_task=rollout_config.get('rollouts_per_env', 10),  # 最终评估可用更多rollout
            is_final_eval=True  # 🔥 最终评估，保存视频
        )
        
        if final_eval_results:
            print(f"🏆 最终评估结果:")
            print(f"   总体成功率: {final_eval_results.get('overall_success_rate', 0.0):.2%}")
            for task_name, success_rate in final_eval_results.items():
                if task_name != 'overall_success_rate':
                    print(f"   {task_name}: {success_rate:.2%}")
            
            # 保存最终评估结果
            final_eval_path = output_dir / "final_evaluation_results.json"
            with open(final_eval_path, 'w') as f:
                json.dump(final_eval_results, f, indent=2)
            print(f"📄 最终评估结果已保存: {final_eval_path}")
        
    except Exception as e:
        print(f"⚠️ 最终评估失败: {e}")
        traceback.print_exc()

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