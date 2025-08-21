"""
独立的并行环境工厂函数 - 解决SubprocVectorEnv序列化问题

这个模块提供完全独立的环境创建函数，不依赖任何包含模型的对象，
避免CloudPickle序列化时包含大模型导致的内存溢出问题。
"""

import os
import gym
import numpy as np
from typing import Optional


class SyncedInitStateWrapper:
    """
    RIPT对齐的初始状态包装器

    功能：
    1. 与原版RIPT完全一致的顺序轮换：selected_id = counter % num_init_states
    2. 统一调用接口：env.reset() + env.set_init_state(snapshot)
    3. 支持并行环境一致性：通过init_states参数广播相同状态
    4. 完全移除随机逻辑，确保可复现的评测结果
    """

    def __init__(self, env, fixed_init_state_id: int, init_states_array=None):
        """
        Args:
            env: 被包装的环境
            fixed_init_state_id: 保留兼容性，实际使用顺序轮换
            init_states_array: 从主进程传递的初始状态数组
        """
        self.env = env
        
        # 🔥 RIPT对齐：全局顺序轮换计数器
        self.counter = 0  # 全局episode计数器
        
        # 🔥 RIPT对齐：使用主进程传递的初始状态数组
        self.init_states = init_states_array
        if self.init_states is not None:
            import numpy as np
            if isinstance(self.init_states, (list, tuple)):
                self.num_init_states = len(self.init_states)
                print(f"✅ SyncedInitStateWrapper: 接收到 {self.num_init_states} 个初始状态")
            else:
                self.init_states = None
                self.num_init_states = 50
                print(f"⚠️ SyncedInitStateWrapper: 初始状态格式错误，使用默认数量50")
        else:
            self.num_init_states = 50
            print(f"⚠️ SyncedInitStateWrapper: 未接收到初始状态数组，使用默认数量50")

        # 代理所有属性到原始环境
        for attr in ['action_space', 'observation_space', 'task_description',
                     'num_init_states', 'step', 'close', 'seed']:
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

    def reset(self, init_state_id: Optional[int] = None, **kwargs):
        """
        RIPT对齐的环境重置：统一接口 + 顺序轮换
        
        优先级：
        1. 并行环境传入init_states数组（用于多worker一致性）
        2. 全局顺序轮换（与原版RIPT eval对齐）
        """
        # === 优先级1: 并行环境传入init_states数组 ===
        if 'init_states' in kwargs:
            init_states = kwargs.pop('init_states')
            obs = self.env.reset(**kwargs)  # 🔥 RIPT对齐：先reset
            if hasattr(self.env, 'set_init_state') and init_states is not None:
                import numpy as np
                # 处理多维数组：取第一个状态并展平
                if isinstance(init_states, (list, np.ndarray)):
                    init_states_array = np.array(init_states)
                    if init_states_array.ndim > 1:
                        single_state = init_states_array.flatten() if init_states_array.shape[0] == 1 else init_states_array[0]
                    else:
                        single_state = init_states_array
                    self.env.set_init_state(single_state)  # 🔥 RIPT对齐：后set_init_state
                else:
                    self.env.set_init_state(init_states)
            return obs
        
        # === 优先级2: 全局顺序轮换（与原版RIPT完全对齐）===
        selected_id = self.counter % self.num_init_states  # 🔥 等价于 initial_states[episode_idx]
        self.counter += 1
        
        # 🔥 RIPT对齐：完全按原版方式 env.reset() + env.set_init_state(snapshot)
        obs = self.env.reset(**kwargs)
        
        if hasattr(self.env, 'set_init_state'):
            if self.init_states is not None:
                # 🎯 完全对齐原版：snapshot = initial_states[episode_idx]
                import numpy as np
                snapshot = np.array(self.init_states[selected_id])
                self.env.set_init_state(snapshot)
            else:
                # 无初始状态数组时的兜底：使用环境默认行为
                pass  # 环境会使用默认的随机初始化
        
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
                      fixed_init_state_id: int = None, init_states_array=None):
    """
    创建环境工厂函数 - 供SubprocVectorEnv使用

    Args:
        benchmark_name: LIBERO基准名称
        env_name: 环境名称 (可选)
        task_id: 任务ID (可选)
        fixed_init_state_id: 保留兼容性 (可选)
        init_states_array: 从主进程传递的初始状态数组 (关键！)

    Returns:
        callable: 无参数的环境工厂函数
    """
    def env_factory():
        env, task_description = create_libero_env_independent(
            benchmark_name=benchmark_name,
            env_name=env_name,
            task_id=task_id
        )

        # 🔥 RIPT对齐：传递主进程获取的初始状态数组给wrapper
        env = SyncedInitStateWrapper(env, fixed_init_state_id, init_states_array=init_states_array)

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