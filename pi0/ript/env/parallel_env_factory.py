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
    智能初始状态包装器

    支持两种模式：
    1. 固定模式：所有环境使用相同的初始状态ID（同步）
    2. 随机模式：每次重置随机选择初始状态ID
    """

    def __init__(self, env, fixed_init_state_id: int):
        """
        Args:
            env: 被包装的环境
            fixed_init_state_id: 初始状态ID配置
                - >= 0: 固定使用指定ID（同步模式）
                - -1: 随机选择ID（随机模式）
        """
        import random
        self.env = env
        self.fixed_init_state_id = fixed_init_state_id
        self.random_mode = (fixed_init_state_id == -1)
        self.random = random.Random()  # 独立的随机数生成器
        
        # 获取环境支持的初始状态数量
        self.num_init_states = getattr(env, 'num_init_states', 1)

        # 代理所有属性到原始环境
        for attr in ['action_space', 'observation_space', 'task_description',
                     'num_init_states', 'step', 'close', 'seed']:
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

    def reset(self, init_state_id: Optional[int] = None):
        """
        重置环境，根据模式选择初始状态ID

        Args:
            init_state_id: 传入的初始状态ID（在固定模式下忽略）

        Returns:
            observation: 环境观测
        """
        if self.random_mode:
            # 🎲 随机模式：从可用的初始状态中随机选择
            selected_id = self.random.randint(0, self.num_init_states - 1)
            print(f"🎲 随机选择初始状态ID: {selected_id}/{self.num_init_states}")
            return self.env.reset(init_state_id=selected_id)
        else:
            # 🔒 固定模式：使用指定的初始状态ID
            return self.env.reset(init_state_id=self.fixed_init_state_id)

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
                      fixed_init_state_id: int = None):
    """
    创建环境工厂函数 - 供SubprocVectorEnv使用

    这个函数返回一个lambda，该lambda调用独立的环境创建函数。
    关键在于返回的lambda不捕获任何外部变量引用。

    Args:
        benchmark_name: LIBERO基准名称
        env_name: 环境名称 (可选)
        task_id: 任务ID (可选)
        fixed_init_state_id: 固定的初始状态ID，用于确保并行环境同步 (可选)

    Returns:
        callable: 无参数的环境工厂函数
    """
    # 注意：这里使用闭包捕获参数，但不捕获任何包含模型的对象
    # 只捕获基本的字符串和整数参数，CloudPickle可以安全序列化
    def env_factory():
        env, task_description = create_libero_env_independent(
            benchmark_name=benchmark_name,
            env_name=env_name,
            task_id=task_id
        )

        # 🔥 智能初始状态包装器：
        # - fixed_init_state_id >= 0: 固定同步模式
        # - fixed_init_state_id == -1: 智能随机模式
        if fixed_init_state_id is not None:
            env = SyncedInitStateWrapper(env, fixed_init_state_id)

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