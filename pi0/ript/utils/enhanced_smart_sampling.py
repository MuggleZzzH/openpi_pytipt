#!/usr/bin/env python3
"""
增强的智能采样系统 - 补充缺失的初始状态选择逻辑
基于历史记录智能跳过全成功/全失败的初始状态
"""

import json
import hashlib
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path

class EnhancedSmartSampler:
    """
    增强版智能采样器 - 包含完整的初始状态选择逻辑
    实现基于历史记录的状态跳过机制
    """
    
    def __init__(self, config):
        self.config = config
        self.state_stats = {}  # 记录每个状态的统计信息
        self.sampling_history = []  # 采样历史
        
        # 新增: 状态跳过配置
        self.enable_rollout_stats_tracking = getattr(config, 'enable_rollout_stats_tracking', True)
        self.rollout_skip_threshold = getattr(config, 'rollout_skip_threshold', 3)
        self.rollout_stats_path = getattr(config, 'rollout_stats_path', './rollout_stats.json')
        
        # 加载历史统计数据
        self.load_rollout_stats()
    
    def load_rollout_stats(self):
        """从文件加载历史rollout统计数据"""
        try:
            if Path(self.rollout_stats_path).exists():
                with open(self.rollout_stats_path, 'r') as f:
                    saved_stats = json.load(f)
                    self.state_stats.update(saved_stats)
                print(f"✓ 加载了 {len(self.state_stats)} 个状态的历史统计数据")
            else:
                print("○ 未找到历史统计文件，从空白开始")
        except Exception as e:
            print(f"⚠️ 加载历史统计失败: {e}")
    
    def save_rollout_stats(self):
        """保存rollout统计数据到文件"""
        try:
            Path(self.rollout_stats_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.rollout_stats_path, 'w') as f:
                json.dump(self.state_stats, f, indent=2)
            print(f"✓ 保存了 {len(self.state_stats)} 个状态的统计数据")
        except Exception as e:
            print(f"⚠️ 保存统计数据失败: {e}")
    
    def should_skip_state(self, state_hash: str) -> bool:
        """
        核心功能: 检查是否应该跳过这个初始状态
        基于历史记录判断状态是否已经被"掌握"(全成功)或"无效"(全失败)
        """
        if not self.enable_rollout_stats_tracking:
            return False
        
        if state_hash not in self.state_stats:
            # 未测试过的状态，不跳过
            if self.config.debug_sampling:
                print(f"      新状态 {state_hash[:8]}: 首次测试，不跳过")
            return False
        
        stats = self.state_stats[state_hash]
        recent_successes = stats['successes']
        
        if len(recent_successes) < self.config.rollouts_per_batch:
            # 历史样本不够，不跳过
            if self.config.debug_sampling:
                print(f"      状态 {state_hash[:8]}: 样本不足({len(recent_successes)}), 不跳过")
            return False
        
        # 计算最近的成功率
        success_rate = sum(recent_successes) / len(recent_successes)
        
        # 检查是否是全成功或全失败的状态
        if success_rate == 1.0:
            # 全成功状态 - 可能已经"掌握"
            skip_count = stats.get('mastery_skip_count', 0)
            if skip_count < self.rollout_skip_threshold:
                stats['mastery_skip_count'] = skip_count + 1
                if self.config.debug_sampling:
                    print(f"      状态 {state_hash[:8]}: 已掌握(100%), 跳过 {skip_count+1}/{self.rollout_skip_threshold}")
                return True
            else:
                # 重新评估掌握状态
                stats['mastery_skip_count'] = 0
                if self.config.debug_sampling:
                    print(f"      状态 {state_hash[:8]}: 重新评估掌握状态")
                return False
                
        elif success_rate == 0.0:
            # 全失败状态 - 可能"无效"或太难
            skip_count = stats.get('failure_skip_count', 0)
            if skip_count < self.rollout_skip_threshold:
                stats['failure_skip_count'] = skip_count + 1
                if self.config.debug_sampling:
                    print(f"      状态 {state_hash[:8]}: 全失败(0%), 跳过 {skip_count+1}/{self.rollout_skip_threshold}")
                return True
            else:
                # 重新评估失败状态
                stats['failure_skip_count'] = 0
                if self.config.debug_sampling:
                    print(f"      状态 {state_hash[:8]}: 重新评估失败状态")
                return False
        else:
            # 混合结果状态 - 有训练价值，不跳过
            if self.config.debug_sampling:
                print(f"      状态 {state_hash[:8]}: 混合结果({success_rate:.2f}), 有价值")
            return False
    
    def smart_sample_init_state(self, init_dataset, max_attempts=50):
        """
        智能采样初始状态 - 避开已知的全成功/全失败状态
        🔥 新版本：优先使用环形索引，回退到智能跳过
        """
        for attempt in range(max_attempts):
            # 🔥 优先使用环形索引采样
            if hasattr(init_dataset, 'get_states_for_envs'):
                # 新的环形索引方法
                init_states = init_dataset.get_states_for_envs(
                    self.config.num_parallel_envs, 
                    getattr(self.config, 'rloo_batch_size', 2)
                )
            else:
                # 后备：使用原有随机采样
                init_states = init_dataset.sample_batch(self.config.num_parallel_envs)
            
            state_hash = self.compute_state_hash(init_states)
            
            # 检查是否应该跳过
            if not self.should_skip_state(state_hash):
                if self.config.debug_sampling:
                    print(f"    ✓ 选中状态 {state_hash[:8]} (尝试 {attempt+1})")
                return init_states, state_hash
            
            # 继续尝试下一个状态
            if self.config.debug_sampling and attempt < 5:  # 只显示前几次跳过
                print(f"    ○ 跳过状态 {state_hash[:8]} (尝试 {attempt+1})")
        
        # 如果所有尝试都被跳过，返回最后一个（避免死循环）
        print(f"    ⚠️ 达到最大尝试次数 {max_attempts}，使用最后采样的状态")
        return init_states, state_hash
    
    def compute_state_hash(self, state_array: np.ndarray) -> str:
        """计算状态数组的哈希值用于追踪"""
        return hashlib.sha256(np.ascontiguousarray(state_array).tobytes()).hexdigest()[:16]
    
    def is_batch_useful(self, rollouts: List[Tuple]) -> bool:
        """判断一个batch的轨迹是否有用（不是全0或全1）"""
        if not rollouts:
            return False
            
        successes = [r[0] for r in rollouts]  # 提取success标志
        
        # 检查是否全成功或全失败
        all_success = all(s == True for s in successes)
        all_failure = all(s == False for s in successes)
        
        if all_success:
            if self.config.debug_sampling:
                print(f"    跳过：全部成功的batch (成功率=100%)")
            return False
        elif all_failure:
            if self.config.debug_sampling:
                print(f"    跳过：全部失败的batch (成功率=0%)")
            return False
        else:
            success_rate = sum(successes) / len(successes)
            if self.config.debug_sampling:
                print(f"    ✓ 有用的batch：成功率={success_rate:.2f}")
            return True
    
    def update_state_stats(self, state_hash: str, rollouts: List[Tuple]):
        """更新状态统计信息"""
        if not self.enable_rollout_stats_tracking:
            return
            
        successes = [r[0] for r in rollouts]
        
        if state_hash not in self.state_stats:
            self.state_stats[state_hash] = {
                'attempts': 0,
                'successes': [],
                'last_success_rate': 0.0,
                'mastery_skip_count': 0,
                'failure_skip_count': 0,
                'first_seen': len(self.sampling_history)  # 记录首次出现的迭代
            }
        
        stats = self.state_stats[state_hash]
        stats['attempts'] += 1
        stats['successes'].extend(successes)
        
        # 只保留最近的记录 (可配置窗口大小)
        max_history = getattr(self.config, 'state_history_window', 20)
        if len(stats['successes']) > max_history:
            stats['successes'] = stats['successes'][-max_history:]
        
        # 计算最新成功率
        recent_successes = stats['successes']
        stats['last_success_rate'] = sum(recent_successes) / len(recent_successes)
        
        # 记录到采样历史
        self.sampling_history.append({
            'state_hash': state_hash,
            'iteration': len(self.sampling_history),
            'success_rate': stats['last_success_rate'],
            'rollout_count': len(rollouts)
        })
    
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """获取采样统计信息"""
        if not self.state_stats:
            return {}
        
        total_states = len(self.state_stats)
        mastered_states = sum(1 for stats in self.state_stats.values() 
                             if stats['last_success_rate'] == 1.0 and len(stats['successes']) >= 3)
        failed_states = sum(1 for stats in self.state_stats.values() 
                           if stats['last_success_rate'] == 0.0 and len(stats['successes']) >= 3)
        mixed_states = total_states - mastered_states - failed_states
        
        return {
            'total_states_tested': total_states,
            'mastered_states': mastered_states,
            'failed_states': failed_states,
            'mixed_states': mixed_states,
            'sampling_efficiency': mixed_states / total_states if total_states > 0 else 0.0,
            'total_sampling_attempts': len(self.sampling_history)
        }

def enhanced_collect_smart_batches(env_runner, reward_function, init_dataset, sampler, config, iteration_idx):
    """
    增强版智能轨迹收集 - 使用完整的初始状态选择逻辑
    """
    print(f"\n--- 增强智能轨迹收集 (迭代 {iteration_idx + 1}) ---")
    print(f"目标: 收集 {config.target_batches_per_iteration} 个有用的batches")
    print(f"每个batch包含 {config.rollouts_per_batch} 个轨迹")
    
    if hasattr(sampler, 'enable_rollout_stats_tracking') and sampler.enable_rollout_stats_tracking:
        stats = sampler.get_sampling_statistics()
        if stats:
            print(f"状态统计: 总计{stats['total_states_tested']}, "
                  f"已掌握{stats['mastered_states']}, "
                  f"失败{stats['failed_states']}, "
                  f"混合{stats['mixed_states']}")
    
    # 切换到推理模式收集轨迹
    env_runner.policy.eval()
    
    collected_batches = []
    total_attempts = 0
    state_skip_count = 0
    
    while len(collected_batches) < config.target_batches_per_iteration:
        if total_attempts >= config.max_sampling_attempts:
            print(f"⚠️ 达到最大采样次数 {config.max_sampling_attempts}，停止收集")
            break
            
        total_attempts += 1
        
        # 1. 智能采样初始状态 (核心改进)
        if hasattr(sampler, 'smart_sample_init_state'):
            init_states, state_hash = sampler.smart_sample_init_state(init_dataset)
        else:
            # 回退到原始随机采样
            init_states = init_dataset.sample_batch(config.num_parallel_envs)
            state_hash = sampler.compute_state_hash(init_states)
        
        if config.debug_sampling:
            print(f"\n尝试 {total_attempts}: 测试状态 {state_hash[:8]}")
        
        # 2. 在该初始状态下运行多个轨迹
        try:
            state_rollouts = []
            for rollout_idx in range(config.rollouts_per_batch):
                if config.debug_sampling:
                    print(f"    轨迹 {rollout_idx + 1}/{config.rollouts_per_batch} 从状态 {state_hash[:8]}...")
                
                # 为每个轨迹单独调用环境
                rollout_generator = env_runner.run_policy_in_env(
                    env_name=config.task_id,
                    all_init_states=init_states
                )
                
                try:
                    rollout_batch = list(rollout_generator)
                    if rollout_batch:
                        state_rollouts.extend(rollout_batch)
                    else:
                        if config.debug_sampling:
                            print(f"      轨迹 {rollout_idx + 1} 失败：无数据")
                        break
                except StopIteration:
                    if config.debug_sampling:
                        print(f"      轨迹 {rollout_idx + 1} 完成")
                    break
            
            if not state_rollouts:
                if config.debug_sampling:
                    print(f"    无效：没有收集到轨迹")
                continue
                
        except Exception as e:
            if config.debug_sampling:
                print(f"    错误：轨迹收集失败 - {e}")
            continue
        
        # 3. 智能过滤：检查是否有用
        if config.enable_smart_filtering:
            if not sampler.is_batch_useful(state_rollouts):
                sampler.update_state_stats(state_hash, state_rollouts)
                continue  # 跳过这个batch
        
        # 4. 转换为episode格式并计算奖励
        episodes = []
        for success, total_reward, episode_data in state_rollouts:
            episode = {
                'success': success,
                'total_reward': total_reward,
                **episode_data
            }
            
            # 使用奖励函数计算奖励
            try:
                computed_reward = reward_function.compute_reward(len(episodes), episode, None)
                episode['computed_reward'] = computed_reward
            except Exception as e:
                episode['computed_reward'] = 0.0
            
            episodes.append(episode)
        
        # 5. 收集有用的batch
        collected_batches.append(episodes)
        sampler.update_state_stats(state_hash, state_rollouts)
        
        if config.debug_sampling:
            success_count = sum(ep['success'] for ep in episodes)
            success_rate = success_count/len(episodes)
            print(f"    ✓ 收集batch {len(collected_batches)}: {len(episodes)} 个episodes, "
                  f"成功率={success_rate:.2f} ({'有用' if 0 < success_rate < 1 else '均匀'})")
    
    # 保存统计数据
    if hasattr(sampler, 'save_rollout_stats'):
        sampler.save_rollout_stats()
    
    print(f"\n✓ 增强智能采样完成:")
    print(f"  总尝试次数: {total_attempts}")
    print(f"  收集有用batches: {len(collected_batches)}")
    print(f"  总episodes: {sum(len(batch) for batch in collected_batches)}")
    print(f"  采样效率: {len(collected_batches)/total_attempts:.2%}")
    
    # 显示状态统计
    if hasattr(sampler, 'get_sampling_statistics'):
        stats = sampler.get_sampling_statistics()
        if stats and stats['total_states_tested'] > 0:
            print(f"  状态统计: 测试{stats['total_states_tested']}个, " 
                  f"掌握{stats['mastered_states']}个, "
                  f"失败{stats['failed_states']}个, "
                  f"有价值{stats['mixed_states']}个")
            print(f"  采样质量: {stats['sampling_efficiency']:.1%} (有价值状态比例)")
    
    return collected_batches

if __name__ == "__main__":
    print("增强智能采样系统 - 独立测试")
    
    # 创建测试配置
    class TestConfig:
        def __init__(self):
            self.debug_sampling = True
            self.enable_rollout_stats_tracking = True
            self.rollout_skip_threshold = 3
            self.rollout_stats_path = "./test_rollout_stats.json"
            self.rollouts_per_batch = 4
            self.num_parallel_envs = 1
            self.state_history_window = 20
    
    config = TestConfig()
    sampler = EnhancedSmartSampler(config)
    
    # 模拟一些状态测试
    print("\n=== 模拟状态测试 ===")
    
    # 测试状态1: 全成功状态
    state1 = np.array([1.0, 2.0, 3.0])
    hash1 = sampler.compute_state_hash(state1)
    rollouts1 = [(True, 1.0, {}), (True, 1.0, {}), (True, 1.0, {}), (True, 1.0, {})]
    sampler.update_state_stats(hash1, rollouts1)
    
    # 测试状态2: 全失败状态  
    state2 = np.array([4.0, 5.0, 6.0])
    hash2 = sampler.compute_state_hash(state2)
    rollouts2 = [(False, 0.0, {}), (False, 0.0, {}), (False, 0.0, {}), (False, 0.0, {})]
    sampler.update_state_stats(hash2, rollouts2)
    
    # 测试状态3: 混合结果状态
    state3 = np.array([7.0, 8.0, 9.0])
    hash3 = sampler.compute_state_hash(state3)
    rollouts3 = [(True, 1.0, {}), (False, 0.0, {}), (True, 1.0, {}), (False, 0.0, {})]
    sampler.update_state_stats(hash3, rollouts3)
    
    print("\n=== 测试状态跳过逻辑 ===")
    print(f"状态1 (全成功): 应该跳过 = {sampler.should_skip_state(hash1)}")
    print(f"状态2 (全失败): 应该跳过 = {sampler.should_skip_state(hash2)}")  
    print(f"状态3 (混合): 应该跳过 = {sampler.should_skip_state(hash3)}")
    
    print("\n=== 采样统计 ===")
    stats = sampler.get_sampling_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✓ 增强智能采样系统测试完成")