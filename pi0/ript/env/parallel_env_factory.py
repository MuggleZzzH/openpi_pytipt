"""
独立的并行环境工厂函数 - 解决SubprocVectorEnv序列化问题

这个模块提供完全独立的环境创建函数，不依赖任何包含模型的对象，
避免CloudPickle序列化时包含大模型导致的内存溢出问题。
"""

import os
import gym
import numpy as np
from typing import Optional

# 🔥 移除全局计数器，改用主进程传递episode_offset方式
# 避免多进程环境中全局变量重置问题


class SyncedInitStateWrapper:
    """
    RIPT对齐的初始状态包装器

    功能：
    1. 与原版RIPT完全一致的顺序轮换：selected_id = counter % num_init_states
    2. 统一调用接口：env.reset() + env.set_init_state(snapshot)
    3. 支持并行环境一致性：通过init_states参数广播相同状态
    4. 完全移除随机逻辑，确保可复现的评测结果
    """

    def __init__(self, env, fixed_init_state_id: int, init_states_array=None, episode_offset=0, worker_idx=0):
        """
        Args:
            env: 被包装的环境
            fixed_init_state_id: 保留兼容性，实际使用顺序轮换
            init_states_array: 从主进程传递的初始状态数组
            episode_offset: 主进程传递的episode偏移量，确保全局连续轮换
            worker_idx: 当前worker的索引，用于从batched init_states中切片
        """
        self.env = env
        self.fixed_init_state_id = fixed_init_state_id
        self.worker_idx = worker_idx  # 🔥 新增：worker索引用于状态切片
        
        # 🔥 RIPT对齐：使用主进程传递的episode偏移量
        self.episode_offset = episode_offset
        self.local_counter = 0  # 本次环境实例的局部计数器
        
        # 🔥 缓存机制：实现同demo内rloo batch复用同一snapshot
        self.cached_snapshot = None
        # 🔥 标志位：追踪是否已收到过带init_states的正式业务调用
        self.has_received_init_states = False
        
        # 🔥 RIPT对齐：使用主进程传递的初始状态数组
        self.init_states = init_states_array
        if self.init_states is not None:
            # 🔥 修复numpy数组类型判断问题
            if isinstance(self.init_states, np.ndarray):
                self.num_init_states = self.init_states.shape[0]
                print(f"✅ SyncedInitStateWrapper: 接收到 {self.num_init_states} 个初始状态 (numpy格式)")
                print(f"   状态维度: {self.init_states.shape}")
            elif isinstance(self.init_states, (list, tuple)) and len(self.init_states) > 0:
                self.num_init_states = len(self.init_states)
                print(f"✅ SyncedInitStateWrapper: 接收到 {self.num_init_states} 个初始状态 (list/tuple格式)")
                first_state = self.init_states[0]
                if isinstance(first_state, (list, tuple)):
                    print(f"   状态维度: {len(first_state)}")
                elif hasattr(first_state, 'shape'):
                    print(f"   状态维度: {first_state.shape}")
            else:
                self.init_states = None
                self.num_init_states = 0
                print(f"⚠️ SyncedInitStateWrapper: 初始状态格式错误 (类型: {type(self.init_states)})，轮换功能禁用")
        else:
            self.num_init_states = 0
            print(f"⚠️ SyncedInitStateWrapper: 未接收到初始状态数组，轮换功能禁用")

        # 代理所有属性到原始环境
        for attr in ['action_space', 'observation_space', 'task_description',
                     'num_init_states', 'step', 'close', 'seed']:
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

    def reset(self, init_state_id: Optional[int] = None, **kwargs):
        """
        RIPT对齐的环境重置：统一接口 + 顺序轮换
        
        优先级：
        1. 并行环境传入init_states数组（用于多worker一致性） → 缓存snapshot
        2. 缓存snapshot复用（同demo内rloo batch复用）
        3. 全局顺序轮换（与原版RIPT eval对齐）
        """
        # === 优先级1: 传入状态按worker切片处理（正式业务调用）===
        if 'init_states' in kwargs:
            # 🔥 关键修复：从kwargs中移除init_states，避免传递给不支持的底层环境
            init_states = kwargs.pop('init_states')  # 移除避免传递给底层
            
            # 先进行普通reset
            obs = self.env.reset(**kwargs)
            
            # 🔥 按worker切片处理状态
            if init_states is not None:
                # 确保是numpy数组
                if isinstance(init_states, (list, tuple)):
                    init_states_array = np.array(init_states)
                else:
                    init_states_array = np.asarray(init_states)
                
                # 按worker索引切片得到本worker的状态
                if len(init_states_array.shape) > 1:
                    # 多worker情况：(num_workers, 92)
                    snapshot = init_states_array[self.worker_idx]
                else:
                    # 单状态情况：(92,) 直接使用
                    snapshot = init_states_array
                
                # 确保格式正确
                snapshot = np.ascontiguousarray(snapshot, dtype=np.float64)
                
                # 🔥 缓存用于同demo内复用
                self.cached_snapshot = snapshot.copy()
                self.has_received_init_states = True
                print(f"🎯 优先级1: 使用传入状态worker_{self.worker_idx}, 形状={snapshot.shape}")
                
                # 🔥 使用CleanDiffuser支持的接口设置状态
                if hasattr(self.env, 'set_init_state'):
                    try:
                        print(f"🔧 调试: snapshot类型={type(snapshot)}, 形状={snapshot.shape}, dtype={snapshot.dtype}")
                        if hasattr(snapshot, 'ndim'):
                            print(f"🔧 调试: snapshot维度={snapshot.ndim}, 前5个值={snapshot[:5] if len(snapshot) > 5 else snapshot}")
                        
                        # 🔥 重要修复：确保状态是1D数组且为float64
                        if snapshot.ndim > 1:
                            snapshot = snapshot.flatten()
                        snapshot = np.ascontiguousarray(snapshot, dtype=np.float64)
                        print(f"🔧 处理后snapshot: 形状={snapshot.shape}, dtype={snapshot.dtype}")
                        
                        self.env.set_init_state(snapshot)
                    except Exception as e:
                        print(f"❌ set_init_state失败: {e}")
                        print(f"   snapshot类型: {type(snapshot)}")
                        print(f"   snapshot形状: {getattr(snapshot, 'shape', 'N/A')}")
                        # 尝试fallback：直接reset而不设置状态
                        print(f"⚠️ 回退到普通reset")
                        return self.env.reset(**kwargs)
            
            return obs
        
        # === 优先级2: 缓存snapshot复用（同demo内rloo batch复用）===
        if hasattr(self, 'cached_snapshot') and self.cached_snapshot is not None:
            print(f"🎯 优先级2: 复用缓存状态, 形状={self.cached_snapshot.shape}")
            obs = self.env.reset(**kwargs)
            
            # 🔥 使用缓存的状态直接调用set_init_state
            if hasattr(self.env, 'set_init_state'):
                try:
                    print(f"🔧 调试: cached_snapshot类型={type(self.cached_snapshot)}, 形状={self.cached_snapshot.shape}")
                    
                    # 🔥 重要修复：确保缓存状态格式正确
                    cached_state = self.cached_snapshot.copy()
                    if cached_state.ndim > 1:
                        cached_state = cached_state.flatten()
                    cached_state = np.ascontiguousarray(cached_state, dtype=np.float64)
                    
                    self.env.set_init_state(cached_state)
                except Exception as e:
                    print(f"❌ 缓存状态设置失败: {e}")
                    print(f"⚠️ 回退到普通reset")
                    return self.env.reset(**kwargs)
            
            return obs
        
        # === 优先级3: 全局顺序轮换（与原版RIPT完全对齐）===
        if self.num_init_states <= 0:
            # 无状态数组时使用环境默认初始化
            obs = self.env.reset(**kwargs)
            return obs
        
        # 🔥 正确逻辑：热身reset = 新环境且从未收到过带init_states的调用
        if self.cached_snapshot is None and not self.has_received_init_states:
            # 这是真正的热身reset：新环境且从未收到过正式业务调用
            obs = self.env.reset(**kwargs)
            print(f"[init] 预启动 reset（未设置 init_states）")
            return obs
            
        # 🔥 固定状态ID语义处理
        if self.fixed_init_state_id is not None and self.fixed_init_state_id >= 0:
            # 固定使用指定索引的snapshot
            selected_id = self.fixed_init_state_id % self.num_init_states
        else:
            # 顺序轮换（fixed_init_state_id为-1或None时）
            global_episode_idx = self.episode_offset + self.local_counter
            selected_id = global_episode_idx % self.num_init_states  # 🔥 等价于 initial_states[episode_idx]
            self.local_counter += 1
        
        # 🔥 全局轮换分支：有init_states时才进入
        if self.init_states is not None:
            # 🎯 完全对齐原版：snapshot = initial_states[episode_idx]
            
            # 获取对应的初始状态
            raw_snapshot = self.init_states[selected_id]
            
            # 🔥 确保状态格式正确（与优先级1路径保持一致）
            if isinstance(raw_snapshot, (list, tuple)):
                snapshot = np.ascontiguousarray(raw_snapshot, dtype=np.float64)
            elif hasattr(raw_snapshot, 'shape'):
                snapshot = np.ascontiguousarray(raw_snapshot, dtype=np.float64)
            else:
                snapshot = np.ascontiguousarray([raw_snapshot], dtype=np.float64)
            
            print(f"🎯 全局轮换: episode_{global_episode_idx} → state_id={selected_id}, 形状={snapshot.shape}")
            
            # 🔥 修复：先reset再手动set_init_state
            obs = self.env.reset(**kwargs)
            if hasattr(self.env, 'set_init_state'):
                self.env.set_init_state(snapshot)
            return obs
        
        # 🔥 兜底：如果没有init_states，使用默认reset
        obs = self.env.reset(**kwargs)
        return obs

    def __getattr__(self, name):
        """代理其他属性到原始环境"""
        return getattr(self.env, name)


def create_libero_env_independent(benchmark_name: str, env_name: str = None, task_id: int = None):
    """
    独立创建LIBERO环境的工厂函数 - 核心解决方案
    
    关键设计原则：
    1. 纯函数，不依赖任何外部对象
    2. 不引用包含模型的self对象
    3. 可以被CloudPickle安全序列化到子进程
    4. 与RIPT-VLA的成功模式保持一致
    
    Args:
        benchmark_name: LIBERO基准名称 (如 'libero_goal')
        env_name: 环境名称 (可选，用于兼容性)
        task_id: 任务ID (可选，自动推断)
    
    Returns:
        tuple: (environment, task_description)
    """
    # 在子进程中确保注册 Gym 环境
    try:
        from cleandiffuser.env import libero  # noqa: F401  确保注册
    except ImportError as e:
        raise ImportError("cleandiffuser is required for LIBERO environment creation") from e

    # 基准到环境ID的映射
    benchmark_to_env_id = {
        'libero_spatial': 'libero-spatial-v0',
        'libero_object': 'libero-object-v0',
        'libero_goal': 'libero-goal-v0',
        'libero_10': 'libero-10-v0',
        'libero_90': 'libero-90-v0'
    }

    if benchmark_name not in benchmark_to_env_id:
        raise ValueError(f"Unknown benchmark_name: {benchmark_name}")

    env_id = benchmark_to_env_id[benchmark_name]

    # 自动推断task_id (与串行实现一致)
    if task_id is None:
        if benchmark_name == "libero_goal":
            task_id = 1
        elif benchmark_name == "libero_spatial":
            task_id = 0
        else:
            task_id = 0

    # 使用与串行路径相同的 gym.make 配置，确保观测是 dict
    env = gym.make(
        env_id,
        task_id=task_id,
        image_size=224,
        camera_names=["agentview", "robot0_eye_in_hand"],
        seed=0,
    )

    task_description = getattr(env, 'task_description', env_name or f"{benchmark_name}_task_{task_id}")
    return env, task_description


def create_env_factory(benchmark_name: str, env_name: str = None, task_id: int = None,
                      fixed_init_state_id: int = None, init_states_array=None, episode_offset=0, worker_idx=0):
    """
    创建环境工厂函数 - 供SubprocVectorEnv使用

    Args:
        benchmark_name: LIBERO基准名称
        env_name: 环境名称 (可选)
        task_id: 任务ID (可选)
        fixed_init_state_id: 保留兼容性 (可选)
        init_states_array: 从主进程传递的初始状态数组
        episode_offset: 主进程传递的episode偏移量，确保全局连续轮换
        worker_idx: worker索引，用于从batched init_states中切片

    Returns:
        callable: 无参数的环境工厂函数
    """
    def env_factory():
        env, task_description = create_libero_env_independent(
            benchmark_name=benchmark_name,
            env_name=env_name,
            task_id=task_id
        )

        # 🔥 RIPT对齐：传递主进程管理的episode偏移量、初始状态数组和worker索引
        env = SyncedInitStateWrapper(env, fixed_init_state_id, 
                                   init_states_array=init_states_array,
                                   episode_offset=episode_offset,
                                   worker_idx=worker_idx)

        return env  # SubprocVectorEnv只需要环境对象

    return env_factory


# 测试函数 - 验证独立环境创建的正确性
def test_independent_env_creation():
    """测试独立环境创建是否正常工作"""
    print("🧪 测试独立环境创建...")
    
    try:
        # 测试libero_goal环境创建
        env, task_desc = create_libero_env_independent('libero_goal')
        print(f"✅ 成功创建环境: {task_desc}")
        
        # 测试基本环境功能
        obs = env.reset()
        print(f"✅ 环境重置成功，观测键: {list(obs.keys())}")
        
        # 测试动作执行
        dummy_action = [0, 0, 0, 0, 0, 0, -1]
        next_obs, reward, done, info = env.step(dummy_action)
        print(f"✅ 动作执行成功，奖励: {reward}")
        
        # 清理
        env.close()
        print("✅ 环境清理完成")
        
    except Exception as e:
        print(f"❌ 独立环境创建测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_independent_env_creation()