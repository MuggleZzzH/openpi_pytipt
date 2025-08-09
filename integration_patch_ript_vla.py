
# 集成补丁: 在训练脚本中添加RIPT-VLA runner选项

# 在配置文件中添加:
features:
  use_ript_vla_runner: true  # 启用RIPT-VLA风格的runner

# 在训练脚本中添加runner选择逻辑:
def create_env_runner(config, policy, rank=0, world_size=1):
    """根据配置选择合适的环境runner"""
    
    # 检查是否启用RIPT-VLA runner
    use_ript_vla = getattr(config.features, 'use_ript_vla_runner', False) if hasattr(config, 'features') else False
    
    if use_ript_vla:
        print("🚀 使用RIPT-VLA风格的环境runner")
        from pi0.ript.env.pi0_libero_runner_ript_vla import PI0LiberoRunner
        return PI0LiberoRunner(
            policy=policy,
            benchmark_name=config.task.benchmark_name,
            rollouts_per_env=config.algo.rollouts_per_env,
            num_parallel_envs=config.task.num_parallel_envs,
            rank=rank
        )
    else:
        print("🔄 使用原有的环境runner")  
        from pi0.ript.env.pi0_libero_runner import LIBEROEnvRunner
        return LIBEROEnvRunner(
            policy=policy,
            benchmark_name=config.task.benchmark_name,
            rollouts_per_env=config.algo.rollouts_per_env,
            num_parallel_envs=config.task.num_parallel_envs,
            config=config,
            rank=rank,
            world_size=world_size
        )
