#!/usr/bin/env python3
"""
第11阶段：RIPT-VLA Runner集成 (11_train_with_ript_vla.py) 
基于9_train_with_config.py，完全重构集成RIPT-VLA并行环境runner

核心特性：
1. 基于RIPT-VLA的真正多进程并行环境
2. 智能runner选择机制 (原有 vs RIPT-VLA)
3. 完全向后兼容的配置系统
4. 简化的训练流程和错误处理
5. 单模型多环境架构

使用方法:
    cd /zhaohan/ZJH/openpi_pytorch
    
    # 使用原有runner (向后兼容)
    python 11_train_with_ript_vla.py --config_path pi0/ript/config/debug_train_pi0.yaml
    
    # 使用RIPT-VLA runner (新功能)
    python 11_train_with_ript_vla.py --config_path pi0/ript/config/debug_train_ript_vla.yaml
    
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Dict, Any, Tuple, Optional, Union
import yaml
import traceback

# 修复tokenizers并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"=== 第11阶段：RIPT-VLA Runner集成训练 ===")
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
    
    # RIPT组件
    from pi0.ript.reward_function import BinarySuccessReward
    from pi0.ript.algos.rl_optimizers.rl_optimizer_pi0_cfg import RLOptimizerPI0_CFG
    from pi0.ript.algos.rl_optimizers.pi0_cfg_interface import PI0_CFG_Adapter
    from pi0.ript.algos.rl_optimizers.rollout_generator import RolloutGenerator
    print("✓ RIPT核心组件")
    
    # 🚀 智能Runner导入
    def import_runners():
        """智能导入runner类"""
        runners = {}
        
        # 原有runner
        try:
            from pi0.ript.env.pi0_libero_runner import LIBEROEnvRunner
            runners['original'] = LIBEROEnvRunner
            print("✓ 原有LIBEROEnvRunner")
        except ImportError as e:
            print(f"⚠️ 原有runner导入失败: {e}")
            runners['original'] = None
            
        # # RIPT-VLA runner
        # try:
        #     from pi0.ript.env.pi0_libero_runner_ript_vla import PI0LiberoRunner
        #     runners['ript_vla'] = PI0LiberoRunner
        #     print("✓ RIPT-VLA Runner")
        # except ImportError as e:
        #     print(f"⚠️ RIPT-VLA runner导入失败: {e}")
        #     runners['ript_vla'] = None
            
        return runners
    
    RUNNERS = import_runners()
    
    print("✓ 所有模块导入完成")
    
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

def load_config(config_path: str, overrides: List[str] = None) -> Dict[str, Any]:
    """加载并验证配置文件"""
    print(f"正在加载配置文件: {config_path}")
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    if OMEGACONF_AVAILABLE:
        # 使用OmegaConf
        config = OmegaConf.load(config_path)
        
        # 应用命令行覆盖
        if overrides:
            for override in overrides:
                key, value = override.split('=', 1)
                # 尝试转换数据类型
                try:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # 保持字符串
                
                OmegaConf.set(config, key, value)
        
        # 转换为普通字典以便兼容性
        config = OmegaConf.to_yaml(config)
        config = yaml.safe_load(config)
    else:
        # 使用基础YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # 配置验证和默认值
    config = validate_and_set_defaults(config)
    
    print("✓ 配置文件加载成功")
    return config

def validate_and_set_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """验证配置并设置默认值"""
    
    # 必需的配置项
    required_keys = [
        'policy_path', 'task.benchmark_name', 
        'algo.rloo_batch_size', 'training.num_train_steps'
    ]
    
    for key in required_keys:
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        if keys[-1] not in current:
            raise ValueError(f"配置缺少必需项: {key}")
    
    # 设置默认值
    defaults = {
        'output_dir': './output',
        'training.seed': 42,
        'task.num_parallel_envs': 1,
        'task.max_episode_length': 200,
        'algo.gradient_accumulation_steps': 1,
        'logging.use_wandb': False,
        'features.use_ript_vla_runner': False,
    }
    
    for key, default_value in defaults.items():
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        if keys[-1] not in current:
            current[keys[-1]] = default_value
    
    return config

def create_initial_state_dataset(config: Dict[str, Any]):
    """创建初始状态数据集（与Stage 9兼容）"""
    from torch.utils.data import Dataset
    
    class InitialStateDataset(Dataset):
        def __init__(self, config):
            self.config = config
            self.states = []
            
            # 获取数据集大小和状态维度
            num_states = config.get('dataset', {}).get('num_init_states', 50)
            state_dim = config.get('dataset', {}).get('state_dim', 8)
            
            np.random.seed(config.get('training', {}).get('seed', 42))
            
            # 生成多样化的初始状态
            for i in range(num_states):
                base_state = np.zeros(state_dim, dtype=np.float32)
                
                # 添加不同程度的噪声以增加多样性
                if i < num_states // 3:
                    noise = np.random.normal(0, 0.05, state_dim).astype(np.float32)
                elif i < 2 * num_states // 3:
                    noise = np.random.normal(0, 0.15, state_dim).astype(np.float32)
                else:
                    noise = np.random.normal(0, 0.25, state_dim).astype(np.float32)
                
                state = base_state + noise
                self.states.append(state)
            
            print(f"✓ 创建了包含 {len(self.states)} 个多样化初始状态的数据集")
        
        def __len__(self):
            return len(self.states)
        
        def __getitem__(self, idx):
            return torch.from_numpy(self.states[idx]).float()
        
        def sample_batch(self, batch_size=1):
            """兼容旧版本的采样接口"""
            if batch_size > len(self.states):
                indices = np.random.choice(len(self.states), batch_size, replace=True)
            else:
                indices = np.random.choice(len(self.states), batch_size, replace=False)
            
            sampled_states = np.array([self.states[i] for i in indices])
            return sampled_states
    
    return InitialStateDataset(config)

def create_env_runner(config: Dict[str, Any], policy, rank: int = 0, world_size: int = 1):
    """🚀 智能runner选择函数"""
    
    # 检查是否启用RIPT-VLA runner
    use_ript_vla = config.get('features', {}).get('use_ript_vla_runner', False)
    
    if rank == 0:
        print(f"🔍 Runner选择分析:")
        print(f"   配置中features: {config.get('features', {})}")
        print(f"   use_ript_vla_runner: {use_ript_vla}")
        print(f"   可用runners: {[k for k, v in RUNNERS.items() if v is not None]}")
    
    if use_ript_vla and RUNNERS['ript_vla'] is not None:
        if rank == 0:
            print("🚀 使用RIPT-VLA风格的环境runner")
        
        return RUNNERS['ript_vla'](
            policy=policy,
            benchmark_name=config['task']['benchmark_name'],
            rollouts_per_env=config['algo']['rloo_batch_size'],
            num_parallel_envs=config['task']['num_parallel_envs'],
            max_episode_length=config['task']['max_episode_length'],
            task_names_to_use=config['task'].get('task_names_to_use', []),
            rank=rank
        )
    
    elif RUNNERS['original'] is not None:
        if rank == 0:
            print("🔄 使用原有的环境runner")
        
        # 确保norm_stats_path存在
        norm_stats_path = config.get('norm_stats_path')
        if not norm_stats_path:
            # 推断默认路径
            policy_path = config['policy_path']
            if '/checkpoints/' in policy_path:
                norm_stats_path = f"{policy_path}/norm_stats.json"
            else:
                norm_stats_path = "./lerobot_dataset/norm_stats.json"
            config['norm_stats_path'] = norm_stats_path
            
        return RUNNERS['original'](
            policy=policy,
            benchmark_name=config['task']['benchmark_name'],
            rollouts_per_env=config['algo']['rloo_batch_size'],
            num_parallel_envs=config['task']['num_parallel_envs'],
            max_episode_length=config['task']['max_episode_length'],
            task_names_to_use=config['task'].get('task_names_to_use', []),
            norm_stats_path=norm_stats_path,
            config=config,
            rank=rank,
            world_size=world_size
        )
    
    else:
        raise RuntimeError("❌ 无可用的环境runner！")

def setup_training(config: Dict[str, Any]):
    """设置训练环境"""
    
    # 设置随机种子
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(config['output_dir'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config.get('exp_name', 'ript_vla_train')
    output_dir = output_dir / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config['output_dir'] = str(output_dir)
    print(f"输出目录: {output_dir}")
    
    return device, output_dir

def load_policy(config: Dict[str, Any], device):
    """加载PI0策略"""
    policy_path = config['policy_path']
    print(f"正在加载策略: {policy_path}")
    
    if not Path(policy_path).exists():
        raise FileNotFoundError(f"策略路径不存在: {policy_path}")
    
    policy = PI0Policy.from_pretrained(policy_path)

    # 🔧 根据配置控制CFG功能
    policy_config = config.get('policy', {})
    cfg_enabled = policy_config.get('cfg_enabled', True)  # 默认启用以保持兼容性

    print(f"🔧 配置CFG功能: {'启用' if cfg_enabled else '禁用'}")
    policy.model.cfg_enabled = cfg_enabled
    if hasattr(policy, 'config'):
        policy.config.cfg_enabled = cfg_enabled

    policy = policy.to(device)
    policy.eval()

    if cfg_enabled:
        print("✅ 策略加载成功，CFG已启用")
    else:
        print("✅ 策略加载成功，CFG已禁用")
    return policy

def create_trainer_components(config: Dict[str, Any], policy, env_runner, device):
    """创建训练组件（完整的RolloutGenerator版本）"""
    import torch
    from torch.utils.data import DataLoader
    
    # 创建奖励函数
    reward_function = BinarySuccessReward()
    
    # 🔧 创建CFG适配器（必需的model_adapter）
    print("正在创建CFG适配器...")

    # 🔥 Phase 3: SO100数据处理配置支持
    dataset_config = config.get('dataset', {})
    use_so100_processing = dataset_config.get('use_so100_processing', False)

    print(f"数据处理模式: {'SO100样本处理' if use_so100_processing else 'Legacy窗口化'}")

    cfg_adapter = PI0_CFG_Adapter(
        policy=policy,
        norm_stats_path=f"{config['policy_path']}/norm_stats.json",
        use_so100_processing=use_so100_processing  # 🔥 Phase 3: 新增SO100支持
    )
    
    # 🔧 为policy添加优化器（RLOptimizerPI0_CFG需要）
    print("正在为policy配置优化器...")
    lr = config['algo'].get('lr', 1e-5)
    # 确保学习率是float类型
    if isinstance(lr, str):
        lr = float(lr)
    
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=lr,
        weight_decay=0.01
    )
    policy.optimizer = optimizer  # 附加到policy上供RL优化器使用
    
    # 🔧 创建初始状态数据集（与Stage 9兼容）
    print("正在创建初始状态数据集...")
    init_dataset = create_initial_state_dataset(config)
    
    # 创建数据加载器
    init_state_dataloader = DataLoader(
        init_dataset,
        batch_size=config['task'].get('num_parallel_envs', 1),
        shuffle=True,
        drop_last=True
    )
    
    # 🔧 正确初始化RolloutGenerator
    print("正在创建RolloutGenerator...")
    rollout_generator = RolloutGenerator(
        env_runner=env_runner,
        rollouts_per_env=1,  # 每个环境生成1个rollout
        num_envs=config['task'].get('num_parallel_envs', 1),
        max_steps=config['task']['max_episode_length'],
        agent_gpus=[0],  # 使用单GPU
        init_state_dataloader=init_state_dataloader,
        init_state_dataset=init_dataset,
        enable_dynamic_sampling=config.get('algo', {}).get('enable_dynamic_sampling', True),
        rollout_skip_threshold=config.get('algo', {}).get('rollout_skip_threshold', 3),
        enable_rollout_stats_tracking=config.get('algo', {}).get('enable_rollout_stats_tracking', True),
        rollout_stats_path=config.get('algo', {}).get('rollout_stats_path', './rollout_stats.json'),
        use_val_init=config.get('algo', {}).get('use_val_init', False)
    )
    
    # 🔧 正确初始化RLOptimizerPI0_CFG（使用正确的参数）
    print("正在创建RL优化器...")
    rl_optimizer = RLOptimizerPI0_CFG(
        rollout_generator=rollout_generator,
        reward_function=reward_function,
        num_epochs=5,  # 默认epoch数
        batch_size=config['algo']['rloo_batch_size'],
        gradient_accumulation_steps=config['algo']['gradient_accumulation_steps'],
        grad_norm_clip=1.0  # 默认梯度裁剪
    )
    
    print("✓ 训练组件创建成功（使用完整RolloutGenerator逻辑）")
    return cfg_adapter, rl_optimizer

def train_loop(config: Dict[str, Any], cfg_adapter, rl_optimizer, output_dir):
    """主训练循环（使用完整RolloutGenerator逻辑）"""
    
    num_train_steps = config['training']['num_train_steps']
    log_freq = config.get('logging', {}).get('log_freq', 1)
    save_freq = config['training'].get('save_freq', 10)
    
    print(f"开始训练: {num_train_steps} 步")
    print(f"日志频率: {log_freq}, 保存频率: {save_freq}")
    print(f"RolloutGenerator配置:")
    print(f"  - 动态采样: {rl_optimizer.rollout_generator.enable_dynamic_sampling}")
    print(f"  - 统计跟踪: {rl_optimizer.rollout_generator.enable_rollout_stats_tracking}")
    print(f"  - 跳过阈值: {rl_optimizer.rollout_generator.rollout_skip_threshold}")
    
    all_training_metrics = []
    
    for step in range(num_train_steps):
        try:
            print(f"\n=== 训练步骤 {step + 1}/{num_train_steps} ===")
            
            # 🔧 使用RLOptimizerPI0_CFG的train_on_rollouts方法
            print("正在执行完整的训练周期（rollout + 优势计算 + 优化）...")
            
            training_metrics = rl_optimizer.train_on_rollouts(
                model_adapter=cfg_adapter,
                lr=config['algo'].get('lr', 1e-5),
                scaler=None,  # 暂不使用混合精度
                use_amp=False  # 暂不使用自动混合精度
            )
            
            if not training_metrics:
                print("⚠️ 训练未返回指标，跳过此步")
                continue
            
            print(f"训练指标: {training_metrics}")
            
            # 提取关键指标
            step_metrics = {
                'step': step + 1,
                **training_metrics  # 包含所有返回的指标
            }
            all_training_metrics.append(step_metrics)
            
            # 日志记录
            if (step + 1) % log_freq == 0:
                print(f"\n📊 步骤 {step + 1} 训练指标:")
                for key, value in step_metrics.items():
                    if key != 'step':
                        print(f"  {key}: {value}")
            
            # 保存检查点
            if (step + 1) % save_freq == 0:
                checkpoint_path = output_dir / f"checkpoint_step_{step + 1}.pt"
                policy = cfg_adapter.get_policy_model()
                torch.save({
                    'step': step + 1,
                    'policy_state_dict': policy.state_dict(),
                    'optimizer_state_dict': policy.optimizer.state_dict(),
                    'config': config,
                    'training_metrics': all_training_metrics,
                }, checkpoint_path)
                
                print(f"✓ 检查点已保存: {checkpoint_path}")
        
        except KeyboardInterrupt:
            print("\n⚠️ 训练被用户中断")
            break
        except Exception as e:
            print(f"❌ 训练步骤 {step + 1} 出错: {e}")
            traceback.print_exc()
            continue
    
    # 保存最终训练结果
    final_results_path = output_dir / "final_training_results.json"
    with open(final_results_path, 'w') as f:
        json.dump({
            'config': config,
            'training_metrics': all_training_metrics,
            'total_steps': len(all_training_metrics)
        }, f, indent=2)
    
    print(f"\n✓ 训练完成！")
    print(f"📊 完整训练结果已保存: {final_results_path}")
    print(f"🔧 使用了完整的RolloutGenerator逻辑，包括智能采样和统计跟踪")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="第11阶段：RIPT-VLA Runner集成训练")
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True,
        help="配置文件路径"
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="配置覆盖 (格式: key=value)"
    )
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config_path, args.override)
        
        # 显示配置
        print("\n====== 使用配置 ======")
        print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
        print("====================\n")
        
        # 设置训练环境
        device, output_dir = setup_training(config)
        
        # 加载策略
        policy = load_policy(config, device)
        
        # 创建环境runner
        env_runner = create_env_runner(config, policy, rank=0, world_size=1)
        
        # 创建训练组件
        cfg_adapter, rl_optimizer = create_trainer_components(
            config, policy, env_runner, device
        )
        
        # 开始训练
        train_loop(config, cfg_adapter, rl_optimizer, output_dir)
        
        print("\n🎉 训练成功完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️ 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()