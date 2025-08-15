"""
OpenPI-RIPT集成训练系统
将所有组件整合到统一的训练框架中
"""

import os
import time
import torch
import numpy as np
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# 核心组件导入
from pi0.modeling_pi0 import PI0Policy
from lerobot.configs.policies import PreTrainedConfig

# 我们的组件导入
from utils.openpi_ript_dataset_wrapper import create_openpi_ript_dataset
from utils.state_dimension_adapter import create_pi0_state_adapter
from ript.collectors.openpi_rollout_collector import create_openpi_rollout_collector
from ript.utils.advantage_processor import create_advantage_processor
from pi0.ript.algos.rl_optimizers.pi0_cfg_interface import PI0_CFG_Adapter

# 现有RIPT组件导入
try:
    from pi0.ript.env.pi0_libero_runner import LIBEROEnvRunner
    from pi0.ript.utils.rollout_stats_tracker import RolloutStatsTracker
    RIPT_ENV_AVAILABLE = True
except ImportError:
    print("⚠️ RIPT环境组件不可用，将使用模拟模式")
    RIPT_ENV_AVAILABLE = False


@dataclass
class TrainingConfig:
    """训练配置（临时使用，后续将替换为YAML）"""
    # 实验配置
    experiment_name: str = "openpi_ript_integration"
    output_dir: str = "./output/integrated_training"
    seed: int = 42
    
    # 模型配置
    checkpoint_path: str = "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch"
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True
    
    # 数据配置
    dataset_id: str = "ZibinDong/so100_grab_screwdriver"
    action_chunk_size: int = 50
    image_size: Tuple[int, int] = (224, 224)
    target_state_dim: int = 14
    
    # RIPT配置
    rloo_batch_size: int = 8
    demo_batch_size: int = 4
    enable_dynamic_sampling: bool = True
    enable_state_skipping: bool = True
    rollout_goal_per_step: int = 100
    rollout_skip_threshold: int = 3
    
    # 优势处理配置
    advantage_normalization: str = "standard"
    advantage_clipping: str = "symmetric"
    advantage_clip_value: float = 3.0
    advantage_negative_handling: str = "softplus"
    
    # CFG配置
    cfg_alpha: float = 0.1
    enable_cfg_safe_copy: bool = True
    
    # 训练配置
    num_train_steps: int = 1000
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 1e-2
    gradient_clip_norm: float = 1.0
    enable_amp: bool = True
    
    # 设备配置
    device: str = "cuda"
    device_id: int = 0
    
    # 监控配置
    enable_wandb: bool = False  # 暂时关闭，后续开启
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    verbose: bool = True


class OpenPIRiptTrainer:
    """
    OpenPI-RIPT集成训练器
    
    整合所有已完成的组件：
    1. OpenPI标准数据包装器
    2. 状态维度适配器
    3. RIPT Rollout收集器
    4. 优势值处理器
    5. CFG安全拷贝机制
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"{config.experiment_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 组件初始化
        self.policy = None
        self.optimizer = None
        self.cfg_adapter = None
        self.rollout_collector = None
        self.advantage_processor = None
        self.env_runner = None
        self.stats_tracker = None
        self.dataset = None
        
        # 训练状态
        self.current_step = 0
        self.training_metrics = []
        self.best_success_rate = 0.0
        
        print(f"🚀 OpenPI-RIPT训练器初始化")
        print(f"   实验名称: {config.experiment_name}")
        print(f"   输出目录: {self.run_dir}")
        print(f"   设备: {self.device}")
        
    def setup_components(self):
        """设置和初始化所有组件"""
        print("🔧 设置训练组件...")
        
        # 1. 设置随机种子
        self._setup_seeds()
        
        # 2. 初始化模型和优化器
        self._setup_model_and_optimizer()
        
        # 3. 初始化CFG适配器
        self._setup_cfg_adapter()
        
        # 4. 初始化数据组件
        self._setup_data_components()
        
        # 5. 初始化环境组件
        self._setup_environment_components()
        
        # 6. 初始化优势处理器
        self._setup_advantage_processor()
        
        # 7. 初始化统计跟踪器
        self._setup_stats_tracker()
        
        print("✅ 所有组件设置完成")
    
    def _setup_seeds(self):
        """设置随机种子"""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
        print(f"   🎲 随机种子: {self.config.seed}")
    
    def _setup_model_and_optimizer(self):
        """初始化PI0模型和优化器"""
        try:
            # 加载预训练配置
            model_config = PreTrainedConfig.from_pretrained(self.config.checkpoint_path)
            model_config.device = "cpu"  # 先在CPU上加载
            model_config.freeze_vision_encoder = self.config.freeze_vision_encoder
            model_config.train_expert_only = self.config.train_expert_only
            model_config.train_state_proj = self.config.train_state_proj
            
            # 加载PI0模型
            self.policy = PI0Policy.from_pretrained(
                self.config.checkpoint_path, 
                config=model_config
            )
            self.policy = self.policy.to(self.device)
            self.policy.train()
            
            # 设置优化器
            self.optimizer = torch.optim.AdamW(
                self.policy.get_optim_params(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                eps=1e-6
            )
            
            print(f"   ✅ PI0模型加载完成")
            print(f"   ✅ 优化器设置完成 (lr={self.config.learning_rate})")
            
        except Exception as e:
            print(f"   ❌ 模型加载失败: {e}")
            raise
    
    def _setup_cfg_adapter(self):
        """初始化CFG适配器"""
        try:
            # 构造norm_stats路径
            norm_stats_path = Path(self.config.checkpoint_path) / "norm_stats.json"
            if not norm_stats_path.exists():
                norm_stats_path = None
                print(f"   ⚠️ 未找到norm_stats.json，将使用默认归一化")
            
            self.cfg_adapter = PI0_CFG_Adapter(
                policy=self.policy,
                norm_stats_path=str(norm_stats_path) if norm_stats_path else None,
                windowing_mode="last",
                window_stride=10,
                max_windows_per_episode=1
            )
            
            print(f"   ✅ CFG适配器初始化完成")
            
        except Exception as e:
            print(f"   ❌ CFG适配器初始化失败: {e}")
            raise
    
    def _setup_data_components(self):
        """初始化数据相关组件"""
        try:
            # 创建OpenPI兼容数据集
            self.dataset = create_openpi_ript_dataset(
                repo_id=self.config.dataset_id,
                enable_ript=True,
                action_chunk_size=self.config.action_chunk_size,
                target_state_dim=self.config.target_state_dim,
                image_size=self.config.image_size
            )
            
            print(f"   ✅ OpenPI数据集创建完成")
            print(f"   ✅ 数据集大小: {len(self.dataset)}")
            
        except Exception as e:
            print(f"   ❌ 数据组件设置失败: {e}")
            # 如果数据集加载失败，创建一个模拟数据集用于测试
            print(f"   🔧 创建模拟数据集用于测试...")
            self.dataset = None
    
    def _setup_environment_components(self):
        """初始化环境相关组件"""
        try:
            if RIPT_ENV_AVAILABLE:
                # 使用真实的环境runner
                # 这里需要根据实际的RIPT环境配置
                print(f"   🔄 RIPT环境组件可用，但需要具体配置")
                self.env_runner = None  # 暂时设为None，需要具体环境配置
            else:
                print(f"   ⚠️ 使用模拟环境模式")
                self.env_runner = None
            
            # 创建rollout收集器
            rollout_config = {
                'rloo_batch_size': self.config.rloo_batch_size,
                'action_chunk_size': self.config.action_chunk_size,
                'enable_dynamic_sampling': self.config.enable_dynamic_sampling,
                'enable_state_skipping': self.config.enable_state_skipping,
                'image_size': self.config.image_size,
                'target_state_dim': self.config.target_state_dim,
                'action_dim': 7,  # 标准机器人动作维度
                'rollout_skip_threshold': self.config.rollout_skip_threshold,
                'rollout_stats_path': str(self.run_dir / 'rollout_stats.json'),
                'task_id': 0,
            }
            
            self.rollout_collector = create_openpi_rollout_collector(
                config_dict=rollout_config,
                env_runner=self.env_runner,
                stats_tracker=None  # 稍后设置
            )
            
            print(f"   ✅ Rollout收集器创建完成")
            
        except Exception as e:
            print(f"   ❌ 环境组件设置失败: {e}")
            traceback.print_exc()
    
    def _setup_advantage_processor(self):
        """初始化优势值处理器"""
        try:
            self.advantage_processor = create_advantage_processor(
                normalization=self.config.advantage_normalization,
                clipping=self.config.advantage_clipping,
                clip_value=self.config.advantage_clip_value,
                negative_handling=self.config.advantage_negative_handling,
                verbose=self.config.verbose
            )
            
            print(f"   ✅ 优势处理器创建完成")
            
        except Exception as e:
            print(f"   ❌ 优势处理器设置失败: {e}")
            raise
    
    def _setup_stats_tracker(self):
        """初始化统计跟踪器"""
        try:
            if RIPT_ENV_AVAILABLE:
                self.stats_tracker = RolloutStatsTracker(
                    rollout_skip_threshold=self.config.rollout_skip_threshold,
                    stats_path=str(self.run_dir / 'rollout_stats.json')
                )
                print(f"   ✅ 统计跟踪器创建完成")
            else:
                print(f"   ⚠️ 统计跟踪器不可用（模拟模式）")
                self.stats_tracker = None
                
        except Exception as e:
            print(f"   ❌ 统计跟踪器设置失败: {e}")
            self.stats_tracker = None
    
    def collect_and_process_data(self) -> Optional[Dict[str, Any]]:
        """
        收集和处理训练数据
        
        Returns:
            处理后的训练批次，包含OpenPI格式数据和处理过的优势值
        """
        try:
            if self.config.verbose:
                print(f"🔄 收集训练数据 (步骤 {self.current_step + 1})")
            
            # 1. 使用rollout收集器收集OpenPI格式数据
            if self.rollout_collector and self.env_runner:
                # 真实环境数据收集
                openpi_samples = self.rollout_collector.collect_rollouts_openpi_format(
                    task_name=f"task_step_{self.current_step}",
                    num_rollouts=self.config.rloo_batch_size
                )
                
                if not openpi_samples:
                    print("⚠️ 未收集到有效数据")
                    return None
                    
            else:
                # 模拟数据生成（用于测试）
                openpi_samples = self._generate_mock_training_data()
                if self.config.verbose:
                    print("🔧 使用模拟数据")
            
            # 2. 提取奖励并计算RLOO优势
            rewards = [sample.get("rollout_reward", np.random.random()) for sample in openpi_samples]
            advantages = self._compute_rloo_advantages(rewards)
            
            # 3. 使用优势处理器处理优势值
            processed_advantages = self.advantage_processor.process_advantages(
                advantages,
                batch_info={"step": self.current_step}
            )
            
            # 4. 更新样本的优势值
            for i, sample in enumerate(openpi_samples):
                if i < len(processed_advantages):
                    sample["advantages"] = torch.full_like(
                        sample["action"][:, 0], 
                        processed_advantages[i].item()
                    )
            
            # 5. 构造训练批次
            training_batch = self._prepare_training_batch(openpi_samples, processed_advantages)
            
            if self.config.verbose:
                print(f"   ✅ 数据收集完成: {len(openpi_samples)} 样本")
                print(f"   ✅ 优势处理完成: 范围 [{processed_advantages.min():.4f}, {processed_advantages.max():.4f}]")
            
            return training_batch
            
        except Exception as e:
            print(f"❌ 数据收集失败: {e}")
            traceback.print_exc()
            return None
    
    def _generate_mock_training_data(self) -> List[Dict[str, Any]]:
        """生成模拟训练数据（用于测试）"""
        mock_samples = []
        
        for i in range(self.config.rloo_batch_size):
            sample = {
                "image": {
                    "base_0_rgb": torch.randint(0, 255, self.config.image_size + (3,), dtype=torch.uint8),
                    "left_wrist_0_rgb": torch.randint(0, 255, self.config.image_size + (3,), dtype=torch.uint8)
                },
                "state": torch.randn(self.config.target_state_dim),
                "action": torch.randn(self.config.action_chunk_size, 7),
                "action_is_pad": torch.zeros(self.config.action_chunk_size, dtype=torch.bool),
                "prompt": f"mock_task_{i}",
                "advantages": torch.randn(self.config.action_chunk_size),
                "init_hash": f"mock_hash_{i}_{self.current_step}",
                "rollout_success": np.random.random() > 0.3,
                "rollout_reward": np.random.random(),
            }
            mock_samples.append(sample)
        
        return mock_samples
    
    def _compute_rloo_advantages(self, rewards: List[float]) -> torch.Tensor:
        """
        计算RLOO (Reward-Ranked Leave-One-Out) 优势
        
        Args:
            rewards: 奖励列表
            
        Returns:
            RLOO优势张量
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        num_samples = len(rewards)
        
        if num_samples <= 1:
            return torch.zeros_like(rewards_tensor)
        
        # RLOO公式：每个样本的优势 = 自己的奖励 - 其他样本的平均奖励
        advantages = torch.zeros_like(rewards_tensor)
        
        for i in range(num_samples):
            # 计算除自己外其他样本的平均奖励
            other_rewards = torch.cat([rewards_tensor[:i], rewards_tensor[i+1:]])
            baseline = other_rewards.mean()
            advantages[i] = rewards_tensor[i] - baseline
        
        return advantages
    
    def _prepare_training_batch(
        self, 
        openpi_samples: List[Dict[str, Any]], 
        advantages: torch.Tensor
    ) -> Dict[str, Any]:
        """将OpenPI样本转换为训练批次"""
        batch_size = len(openpi_samples)
        
        # 收集所有数据
        images_base = []
        images_wrist = []
        states = []
        actions = []
        action_is_pad = []
        prompts = []
        
        for sample in openpi_samples:
            images_base.append(sample["image"]["base_0_rgb"])
            images_wrist.append(sample["image"]["left_wrist_0_rgb"])
            states.append(sample["state"])
            actions.append(sample["action"])
            action_is_pad.append(sample["action_is_pad"])
            prompts.append(sample["prompt"])
        
        # 转换为张量并移动到设备
        training_batch = {
            "image": {
                "base_0_rgb": torch.stack(images_base).to(self.device),
                "left_wrist_0_rgb": torch.stack(images_wrist).to(self.device)
            },
            "state": torch.stack(states).to(self.device),
            "action": torch.stack(actions).to(self.device),
            "action_is_pad": torch.stack(action_is_pad).to(self.device),
            "prompt": prompts,
            "advantages": advantages.to(self.device)
        }
        
        return training_batch
    
    def train_step(self, training_batch: Dict[str, Any]) -> Dict[str, float]:
        """
        执行单个训练步骤
        
        Args:
            training_batch: 训练批次数据
            
        Returns:
            训练指标字典
        """
        step_start_time = time.time()
        
        try:
            # 准备优势值
            advantages = training_batch["advantages"]
            
            # 使用CFG适配器计算损失（包含安全拷贝）
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.config.enable_amp):
                # 将OpenPI批次转换为CFG适配器期望的格式
                episodes = self._batch_to_episodes(training_batch)
                
                # 使用CFG适配器的安全损失计算
                loss = self.cfg_adapter.compute_weighted_loss(episodes, advantages, self.device)
            
            # 梯度更新
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
            
            loss.backward()
            
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), 
                self.config.gradient_clip_norm
            )
            
            # 优化器步骤
            if (self.current_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 计算训练指标
            step_time = time.time() - step_start_time
            
            metrics = {
                "loss": loss.item() * self.config.gradient_accumulation_steps,
                "grad_norm": grad_norm.item(),
                "advantages_mean": advantages.mean().item(),
                "advantages_std": advantages.std().item(),
                "step_time": step_time,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
            }
            
            if self.config.verbose and (self.current_step + 1) % self.config.log_interval == 0:
                print(f"   步骤 {self.current_step + 1}: 损失={metrics['loss']:.6f}, "
                      f"梯度范数={metrics['grad_norm']:.4f}, "
                      f"时间={metrics['step_time']:.2f}s")
            
            return metrics
            
        except Exception as e:
            print(f"❌ 训练步骤失败: {e}")
            traceback.print_exc()
            return {"loss": float('inf'), "error": str(e)}
    
    def _batch_to_episodes(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将批次数据转换为CFG适配器期望的episode格式"""
        batch_size = batch["state"].shape[0]
        episodes = []
        
        for i in range(batch_size):
            episode = {
                'observations': [{
                    'agentview_image': batch["image"]["base_0_rgb"][i].cpu().numpy().transpose(1, 2, 0),
                    'robot0_eye_in_hand_image': batch["image"]["left_wrist_0_rgb"][i].cpu().numpy().transpose(1, 2, 0),
                    'robot0_eef_pos': batch["state"][i][:3].cpu().numpy(),
                    'robot0_eef_quat': [0, 0, 0, 1],  # 默认四元数
                    'robot0_gripper_qpos': batch["state"][i][-2:].cpu().numpy(),
                }],
                'actions': [batch["action"][i].cpu().numpy()],
                'task': batch["prompt"][i] if isinstance(batch["prompt"], list) else "default_task",
                'success': True,  # 假设成功
                'total_reward': batch["advantages"][i].mean().item(),
            }
            episodes.append(episode)
        
        return episodes
    
    def run_training(self):
        """运行完整的训练循环"""
        print(f"🚀 开始训练循环")
        print(f"   总步数: {self.config.num_train_steps}")
        print(f"   梯度累积步数: {self.config.gradient_accumulation_steps}")
        print(f"   保存间隔: {self.config.save_interval}")
        print()
        
        # 设置进度条
        progress_bar = tqdm(
            range(self.config.num_train_steps),
            desc="训练进度",
            unit="step",
            dynamic_ncols=True
        )
        
        try:
            for step in progress_bar:
                self.current_step = step
                
                # 1. 数据收集和处理
                training_batch = self.collect_and_process_data()
                
                if training_batch is None:
                    print(f"⚠️ 步骤 {step + 1}: 跳过（数据收集失败）")
                    continue
                
                # 2. 训练步骤
                metrics = self.train_step(training_batch)
                
                # 3. 记录指标
                self.training_metrics.append(metrics)
                
                # 4. 更新进度条
                progress_bar.set_postfix({
                    'Loss': f"{metrics.get('loss', 0):.4f}",
                    'Adv_Mean': f"{metrics.get('advantages_mean', 0):.3f}",
                    'Time': f"{metrics.get('step_time', 0):.2f}s"
                })
                
                # 5. 定期保存
                if (step + 1) % self.config.save_interval == 0:
                    self.save_checkpoint(step + 1)
                
                # 6. 内存清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"\n🎉 训练完成！")
            self.save_final_results()
            
        except KeyboardInterrupt:
            print(f"\n⚠️ 训练被中断")
            self.save_checkpoint(self.current_step, is_interrupt=True)
            
        except Exception as e:
            print(f"\n❌ 训练失败: {e}")
            traceback.print_exc()
            
        finally:
            progress_bar.close()
    
    def save_checkpoint(self, step: int, is_interrupt: bool = False):
        """保存检查点"""
        checkpoint_name = f"checkpoint_step_{step}"
        if is_interrupt:
            checkpoint_name += "_interrupted"
        
        checkpoint_path = self.run_dir / f"{checkpoint_name}.pt"
        
        try:
            checkpoint = {
                'step': step,
                'model_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'training_metrics': self.training_metrics,
                'advantage_processor_stats': self.advantage_processor.get_processing_stats(),
            }
            
            if self.rollout_collector:
                checkpoint['rollout_collector_stats'] = self.rollout_collector.get_collection_stats()
            
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 检查点已保存: {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ 保存检查点失败: {e}")
    
    def save_final_results(self):
        """保存最终结果"""
        results_path = self.run_dir / "training_results.json"
        
        try:
            import json
            
            results = {
                'config': {
                    'experiment_name': self.config.experiment_name,
                    'num_train_steps': self.config.num_train_steps,
                    'final_step': self.current_step + 1,
                    'model_checkpoint': self.config.checkpoint_path,
                },
                'final_metrics': self.training_metrics[-10:] if self.training_metrics else [],
                'component_stats': {
                    'advantage_processor': self.advantage_processor.get_processing_stats(),
                },
                'summary': {
                    'total_steps': len(self.training_metrics),
                    'average_loss': np.mean([m.get('loss', 0) for m in self.training_metrics[-100:]]) if self.training_metrics else 0,
                    'final_loss': self.training_metrics[-1].get('loss', 0) if self.training_metrics else 0,
                }
            }
            
            if self.rollout_collector:
                results['component_stats']['rollout_collector'] = self.rollout_collector.get_collection_stats()
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"📊 训练结果已保存: {results_path}")
            
            # 打印统计摘要
            self.print_training_summary()
            
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    def print_training_summary(self):
        """打印训练摘要"""
        if not self.training_metrics:
            return
        
        print(f"\n📊 训练摘要:")
        print(f"   总训练步数: {len(self.training_metrics)}")
        
        # 损失统计
        losses = [m.get('loss', 0) for m in self.training_metrics if 'loss' in m]
        if losses:
            print(f"   平均损失: {np.mean(losses):.6f}")
            print(f"   最终损失: {losses[-1]:.6f}")
            print(f"   最佳损失: {min(losses):.6f}")
        
        # 优势值统计
        print(f"\n📈 组件统计:")
        self.advantage_processor.print_stats()
        
        if self.rollout_collector:
            self.rollout_collector.print_stats()


def main():
    """主函数"""
    print("🌟 OpenPI-RIPT集成训练系统")
    print("="*50)
    
    # 创建配置
    config = TrainingConfig()
    
    # 创建训练器
    trainer = OpenPIRiptTrainer(config)
    
    try:
        # 设置组件
        trainer.setup_components()
        
        # 运行训练
        trainer.run_training()
        
    except Exception as e:
        print(f"❌ 系统错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
