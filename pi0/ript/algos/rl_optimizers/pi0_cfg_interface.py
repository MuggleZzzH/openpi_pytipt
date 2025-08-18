import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch.nn.functional as F
import json
from pathlib import Path
from pi0.modeling_pi0 import PI0Policy
# from lerobot.common.utils.utils import get_safe_dtype  # 暂时注释掉，使用本地实现

# Assuming the base interface is in a shared location
from pi0.ript.algos.rl_optimizers.model_interface import RLModelInterface

# 🔥 Phase 2: Import new SO100-style data processing classes
from pi0.ript.data import SO100StyleProcessor, TrajectoryToSampleGenerator

# 本地实现get_safe_dtype函数
def get_safe_dtype(tensor):
    """Get safe dtype for tensor operations"""
    if tensor.dtype in [torch.float16, torch.bfloat16]:
        return torch.float32
    return tensor.dtype


class PI0_CFG_Adapter(RLModelInterface):
    """
    Adapter for PI0Policy to work with the RIPT framework using an
    advantage-weighted loss, inspired by Classifier-Free Guidance.

    This adapter computes a weighted L2 loss from the policy's forward pass,
    which serves as a proxy for policy gradient updates, bypassing the need
    for explicit log_prob calculations or a value network.
    """

    def __init__(self, policy: PI0Policy, norm_stats_path: Optional[str] = None,
                 windowing_mode: str = 'last', window_stride: int = 10,
                 max_windows_per_episode: int = 1,
                 use_so100_processing: bool = False, **kwargs):
        """
        Initialize the adapter with a PI0Policy instance.
        Args:
            policy: An instance of PI0Policy.
            norm_stats_path: Path to norm_stats.json file for normalization.
                           If None, will try to find it automatically.
            windowing_mode: Windowing strategy - 'last'|'random'|'slide' (default: 'last' for compatibility)
            window_stride: Stride for sliding window mode (default: 10)
            max_windows_per_episode: Maximum windows per episode (default: 1)
            use_so100_processing: Whether to use SO100-style sample processing (Phase 2 feature)
        """
        # The base model is the PI0Policy itself.
        super().__init__(model=policy, **kwargs)
        self.policy = policy

        # 🔥 Phase 2: Processing method configuration
        self.use_so100_processing = use_so100_processing

        # 🔥 Legacy: 窗口化采样配置 (kept for backward compatibility)
        self.windowing_mode = windowing_mode
        self.window_stride = window_stride
        self.max_windows_per_episode = max_windows_per_episode

        if use_so100_processing:
            print(f"🚀 Using SO100-style sample processing (Phase 2)")
        else:
            print(f"🔧 Using legacy windowing: mode={windowing_mode}, stride={window_stride}, max_windows={max_windows_per_episode}")

        # 视频收集相关
        self.video_frames = {}  # {episode_idx: [frames]}
        self.video_save_enabled = True

        # 加载归一化统计信息
        self._load_norm_stats(norm_stats_path)

        # 🔥 Phase 2: Initialize SO100-style processors if enabled
        if self.use_so100_processing:
            self._initialize_so100_processors()
        
    def _load_norm_stats(self, norm_stats_path: Optional[str] = None):
        """Load normalization statistics from norm_stats.json"""
        if norm_stats_path is None:
            # 尝试在常见位置找到norm_stats.json
            possible_paths = [
                "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch/norm_stats.json",
                "/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json", 
                "./norm_stats.json"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    norm_stats_path = path
                    break
        
        if norm_stats_path and Path(norm_stats_path).exists():
            print(f"Loading norm_stats from: {norm_stats_path}")
            with open(norm_stats_path) as f:
                norm_stats = json.load(f)
                
            # 提取状态和动作的归一化参数
            self.state_mean = np.array(norm_stats["norm_stats"]["state"]["mean"][:8], dtype=np.float32)
            self.state_std = np.array(norm_stats["norm_stats"]["state"]["std"][:8], dtype=np.float32)
            self.action_mean = np.array(norm_stats["norm_stats"]["actions"]["mean"][:7], dtype=np.float32)
            self.action_std = np.array(norm_stats["norm_stats"]["actions"]["std"][:7], dtype=np.float32)
            
            print(f"✅ Loaded normalization stats:")
            print(f"  State mean shape: {self.state_mean.shape}")
            print(f"  State std shape: {self.state_std.shape}")
            print(f"  Action mean shape: {self.action_mean.shape}")
            print(f"  Action std shape: {self.action_std.shape}")
        else:
            print("⚠️  Warning: No norm_stats.json found, using identity normalization")
            # 使用单位归一化（不进行归一化）
            self.state_mean = np.zeros(8, dtype=np.float32)
            self.state_std = np.ones(8, dtype=np.float32)
            self.action_mean = np.zeros(7, dtype=np.float32)
            self.action_std = np.ones(7, dtype=np.float32)

    def _initialize_so100_processors(self):
        """
        Initialize SO100-style data processors with current normalization configuration.

        This method sets up the SO100StyleProcessor and TrajectoryToSampleGenerator
        using the same normalization statistics as the legacy windowing approach.
        """
        print("🔄 Initializing SO100-style data processors...")

        # Create configuration for SO100 processors
        so100_config = {
            'action_chunk_size': getattr(self.policy.config, 'n_action_steps', 50),
            'norm_stats_path': "/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch/norm_stats.json",  # 给真实路径
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std
        }

        # Initialize processors
        self.so100_processor = SO100StyleProcessor(so100_config)
        self.sample_generator = TrajectoryToSampleGenerator(self.so100_processor)

        print(f"✅ SO100 processors initialized:")
        print(f"  - Action chunk size: {so100_config['action_chunk_size']}")
        print(f"  - State normalization: mean={self.state_mean.shape}, std={self.state_std.shape}")
        print(f"  - Action normalization: mean={self.action_mean.shape}, std={self.action_std.shape}")

    def process_episodes_to_samples_so100(
        self,
        episodes: List[Dict[str, Any]],
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, Any], Dict[int, List[int]]]:
        """
        🔥 Phase 2: Process episodes using SO100-style sample generation.

        This method replaces the windowing approach with SO100-style processing:
        - Each trajectory of length L generates L-50+1 training samples
        - Each sample contains obs[t] → action[t:t+50] (relative actions)
        - Maintains episode-to-sample mapping for advantage computation

        Args:
            episodes: List of episode dictionaries
            device: Target device for tensors

        Returns:
            Tuple of:
                - batch: Training batch in OpenPI format
                - episode_to_samples_map: Mapping from episode indices to sample indices
        """
        if not self.use_so100_processing:
            raise ValueError("SO100 processing not enabled. Set use_so100_processing=True in constructor.")

        if device is None:
            device = next(self.policy.parameters()).device

        print(f"🚀 Processing {len(episodes)} episodes with SO100-style sampling...")

        # Convert episodes to the format expected by SO100 processors
        formatted_episodes = []
        for i, ep in enumerate(episodes):
            # Convert episode to SO100 format
            formatted_episode = self._convert_episode_to_so100_format(ep, i)
            formatted_episodes.append(formatted_episode)

        # Use SO100 processors to generate training batch
        training_batch, episode_to_samples_map = self.sample_generator.process_episodes_to_training_batch(
            formatted_episodes, device
        )

        # Add additional fields expected by CFG training
        training_batch['owner_indices'] = self._create_owner_indices(episode_to_samples_map)

        print(f"✅ SO100 processing complete:")
        print(f"  - Episodes: {len(episodes)}")
        print(f"  - Training samples: {training_batch['batch_size']}")
        print(f"  - Data utilization: {training_batch['batch_size'] / len(episodes):.1f}x")

        return training_batch, episode_to_samples_map

    def _convert_episode_to_so100_format(self, episode: Dict[str, Any], episode_idx: int) -> Dict[str, Any]:
        """
        Convert an episode from the current format to SO100 processor format.

        Args:
            episode: Episode dictionary from current system
            episode_idx: Episode index for identification

        Returns:
            Episode in SO100 processor format
        """
        # Extract episode data
        observations = episode.get('observations', [])
        actions = episode.get('actions', [])

        # Process observations and extract states
        processed_observations = []
        states = []

        for obs in observations:
            # Extract state (8-dimensional)
            unnorm_state = self._extract_state_from_obs(obs, episode_idx, 0)
            states.append(unnorm_state)

            # Create processed observation in OpenPI format
            processed_obs = self._create_processed_observation(obs, unnorm_state)
            processed_observations.append(processed_obs)

        # Create SO100 format episode
        so100_episode = {
            'id': f"episode_{episode_idx}",
            'processed_observations': processed_observations,
            'actions': actions,
            'states': states,
            'total_reward': episode.get('total_reward', 0.0)
        }

        return so100_episode

    def _create_owner_indices(self, episode_to_samples_map: Dict[int, List[int]]) -> List[int]:
        """
        Create owner indices array for tracking which samples belong to which episodes.

        Args:
            episode_to_samples_map: Mapping from episode indices to sample indices

        Returns:
            List where owner_indices[sample_idx] = episode_idx
        """
        # Find total number of samples
        total_samples = 0
        for sample_indices in episode_to_samples_map.values():
            total_samples = max(total_samples, max(sample_indices) + 1 if sample_indices else 0)

        # Create owner indices array
        owner_indices = [-1] * total_samples  # Initialize with -1 (invalid)

        for episode_idx, sample_indices in episode_to_samples_map.items():
            for sample_idx in sample_indices:
                owner_indices[sample_idx] = episode_idx

        return owner_indices

    def _create_processed_observation(self, obs: Dict[str, Any], unnorm_state: np.ndarray) -> Dict[str, Any]:
        """
        Create processed observation in OpenPI format.

        Args:
            obs: Raw observation dictionary
            unnorm_state: Unnormalized state (8-dimensional)

        Returns:
            Processed observation in OpenPI format
        """
        # Extract and process images
        base_image, wrist_image = self._extract_dual_images_from_obs(obs, 0, 0)

        # Apply image transformations (following 2_pi0_on_libero.py)
        def to_hwc_hmirror(arr: np.ndarray) -> np.ndarray:
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                # CHW -> HWC if needed
                if arr.shape[0] == 3 and arr.shape[-1] != 3:
                    arr = arr.transpose(1, 2, 0)
                # Horizontal mirror
                return arr[:, ::-1, :].copy()
            return arr

        base_image = to_hwc_hmirror(base_image)
        wrist_image = to_hwc_hmirror(wrist_image)

        # Normalize state
        normalized_state = self.normalize_state(unnorm_state)

        # Create processed observation
        processed_obs = {
            'image': {
                'base_0_rgb': torch.from_numpy(base_image).float(),
                'left_wrist_0_rgb': torch.from_numpy(wrist_image).float()
            },
            'state': torch.from_numpy(normalized_state).float(),
            'prompt': [obs.get('task_description', '')]
        }

        return processed_obs

    def map_episode_advantages_to_samples_so100(
        self,
        episode_advantages: torch.Tensor,
        episode_to_samples_map: Dict[int, List[int]]
    ) -> torch.Tensor:
        """
        🔥 Phase 2: Map episode-level RLOO advantages to sample-level advantages.

        This is critical for RIPT integration: episode-level RLOO advantages are computed
        correctly using RIPT mathematics, then propagated to all samples from that episode.

        Args:
            episode_advantages: Tensor of shape (num_episodes,) with RLOO advantages
            episode_to_samples_map: Mapping from episode indices to sample indices

        Returns:
            sample_advantages: Tensor of shape (num_samples,) with advantages for each sample
        """
        if not self.use_so100_processing:
            raise ValueError("SO100 processing not enabled.")

        print(f"🔄 Mapping {len(episode_advantages)} episode advantages to sample level...")

        # Use the sample generator's mapping method
        sample_advantages = self.sample_generator.map_episode_advantages_to_samples(
            episode_advantages, episode_to_samples_map
        )

        # 🔧 确保优势 tensor 与 batch 位于同一设备，避免 "different devices" 错误
        sample_advantages = sample_advantages.to(self.device)

        # Validate mapping consistency
        is_valid = self.sample_generator.validate_episode_to_sample_mapping(
            episode_advantages, episode_to_samples_map, len(sample_advantages)
        )

        if not is_valid:
            raise ValueError("Episode-to-sample advantage mapping validation failed")

        print(f"✅ Advantage mapping complete:")
        print(f"  - Episode advantages: {len(episode_advantages)}")
        print(f"  - Sample advantages: {len(sample_advantages)}")
        print(f"  - Positive samples: {(sample_advantages > 0).sum().item()}")
        print(f"  - Negative samples: {(sample_advantages <= 0).sum().item()}")

        return sample_advantages

    def create_unified_sample_pool(
        self,
        episodes: List[Dict[str, Any]],
        advantages: torch.Tensor,
        device: Optional[torch.device] = None,
        shuffle_samples: bool = True
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
        """
        🚀 统一样本池方法：你想要的理想架构

        将所有episodes一次性转换为统一的样本池，每个样本都有对应的优势值。
        这是标准深度学习训练范式，避免了复杂的episode-to-sample映射。

        Args:
            episodes: 原始episode列表
            advantages: episode级别的优势 (E,)
            device: 目标设备
            shuffle_samples: 是否打散样本顺序

        Returns:
            Tuple of:
                - unified_samples: 统一的样本列表，每个样本独立
                - sample_advantages: 对应的样本级优势 (N,)
        """
        if device is None:
            device = self.device

        print(f"🔄 Creating unified sample pool from {len(episodes)} episodes...")

        # 1. 生成所有样本并记录来源episode
        all_samples = []
        sample_episode_mapping = []  # 记录每个样本来自哪个episode

        for episode_idx, episode in enumerate(episodes):
            # 转换episode格式
            formatted_episode = self._convert_episode_to_so100_format(episode, episode_idx)

            # 生成该episode的所有样本
            episode_samples = self.so100_processor.process_trajectory_to_samples(formatted_episode)

            # 🔧 转换为OpenPI格式，确保包含'image'等字段
            for sample in episode_samples:
                openpi_sample = self.so100_processor.convert_to_openpi_format(sample)
                all_samples.append(openpi_sample)
                sample_episode_mapping.append(episode_idx)

        print(f"  Generated {len(all_samples)} samples from {len(episodes)} episodes")
        print(f"  Average samples per episode: {len(all_samples) / len(episodes):.1f}")

        # 2. 创建样本级优势（直接从episode优势复制）
        sample_advantages = torch.zeros(len(all_samples), device=device, dtype=advantages.dtype)
        for sample_idx, episode_idx in enumerate(sample_episode_mapping):
            sample_advantages[sample_idx] = advantages[episode_idx]

        # 3. 可选：打散样本顺序，破除相关性
        if shuffle_samples:
            import random
            # 创建索引列表并打散
            indices = list(range(len(all_samples)))
            random.shuffle(indices)

            # 重新排列样本和优势
            shuffled_samples = [all_samples[i] for i in indices]
            shuffled_advantages = sample_advantages[indices]

            all_samples = shuffled_samples
            sample_advantages = shuffled_advantages
            print(f"  ✅ Samples shuffled to break episode/temporal correlations")

        print(f"✅ Unified sample pool created:")
        print(f"  - Total samples: {len(all_samples)}")
        print(f"  - Positive samples: {(sample_advantages > 0).sum().item()}")
        print(f"  - Negative samples: {(sample_advantages <= 0).sum().item()}")
        print(f"  - Data utilization: {len(all_samples) / len(episodes):.1f}x")

        return all_samples, sample_advantages

    def compute_loss_from_sample_pool(
        self,
        samples: List[Dict[str, Any]],
        sample_advantages: torch.Tensor,
        batch_size: int = 8,  # 🔥 减少batch大小避免OOM
        device: Optional[torch.device] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        gradient_accumulation_steps: int = 1
    ) -> float:
        """
        🚀 从统一样本池计算损失：标准深度学习训练范式

        将样本池按固定batch_size切分，逐batch计算损失并累积。
        这是你想要的理想架构：固定batch大小，标准梯度累积。

        Args:
            samples: 统一样本池
            sample_advantages: 对应的样本优势 (N,)
            batch_size: 固定的batch大小
            device: 目标设备

        Returns:
            total_loss: 累积的总损失
        """
        if device is None:
            device = self.device

        total_samples = len(samples)
        num_batches = (total_samples + batch_size - 1) // batch_size  # 向上取整

        print(f"🔄 Computing loss from sample pool:")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Number of batches: {num_batches}")

        total_loss = 0.0
        processed_samples = 0
        gradient_step = 0  # 梯度累积计数器

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)

            # 提取当前batch的样本和优势
            batch_samples = samples[start_idx:end_idx]
            batch_advantages = sample_advantages[start_idx:end_idx]

            # 🔥 动态batch大小调整，避免OOM
            current_batch_size = len(batch_samples)

            try:
                # 将样本转换为模型输入格式
                batch_data = self._collate_samples_to_batch(batch_samples, device)

                # 计算CFG损失
                batch_loss = self._compute_cfg_loss_for_batch(batch_data, batch_advantages, device)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ Batch {batch_idx + 1} OOM，尝试减半batch大小...")

                    # 清理显存
                    torch.cuda.empty_cache()

                    # 分割batch为两个更小的batch
                    mid_point = len(batch_samples) // 2
                    if mid_point == 0:
                        print(f"❌ 单个样本都无法处理，跳过batch {batch_idx + 1}")
                        continue

                    # 处理前半部分
                    sub_batch1 = batch_samples[:mid_point]
                    sub_advantages1 = batch_advantages[:mid_point]
                    sub_data1 = self._collate_samples_to_batch(sub_batch1, device)
                    sub_loss1 = self._compute_cfg_loss_for_batch(sub_data1, sub_advantages1, device)
                    
                    # 立即处理第一个子batch的梯度
                    sub_weight1 = len(sub_batch1) / total_samples
                    sub_normalized_loss1 = sub_loss1 * sub_weight1 / gradient_accumulation_steps
                    sub_loss1_value = sub_loss1.detach().item()
                    
                    if scaler is not None:
                        scaler.scale(sub_normalized_loss1).backward()
                    else:
                        sub_normalized_loss1.backward()
                    
                    gradient_step += 1

                    # 清理并处理后半部分
                    del sub_data1, sub_loss1, sub_normalized_loss1
                    torch.cuda.empty_cache()

                    sub_batch2 = batch_samples[mid_point:]
                    sub_advantages2 = batch_advantages[mid_point:]
                    sub_data2 = self._collate_samples_to_batch(sub_batch2, device)
                    sub_loss2 = self._compute_cfg_loss_for_batch(sub_data2, sub_advantages2, device)
                    
                    # 立即处理第二个子batch的梯度
                    sub_weight2 = len(sub_batch2) / total_samples
                    sub_normalized_loss2 = sub_loss2 * sub_weight2 / gradient_accumulation_steps
                    sub_loss2_value = sub_loss2.detach().item()
                    
                    if scaler is not None:
                        scaler.scale(sub_normalized_loss2).backward()
                    else:
                        sub_normalized_loss2.backward()
                    
                    gradient_step += 1

                    # 计算加权平均loss值（仅用于日志）
                    batch_loss_value = (sub_loss1_value * len(sub_batch1) + sub_loss2_value * len(sub_batch2)) / current_batch_size

                    del sub_data2, sub_loss2, sub_normalized_loss2
                    torch.cuda.empty_cache()
                    
                    # 🔥 检查是否需要参数更新（OOM分割情况）
                    if gradient_step == gradient_accumulation_steps or batch_idx == num_batches - 1:
                        if optimizer is not None:
                            # 梯度裁剪
                            if scaler is not None:
                                scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                            
                            # 参数更新
                            if scaler is not None:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            
                            optimizer.zero_grad()
                            print(f"  ✓ 参数更新完成 (OOM分割, 步骤 {gradient_step}/{gradient_accumulation_steps}, 梯度范数: {grad_norm:.6f})")
                            gradient_step = 0
                    
                    # 跳过正常的梯度累积，因为已经处理过了
                    total_loss += batch_loss_value * batch_weight
                    processed_samples += len(batch_samples)
                    
                    print(f"  Batch {batch_idx + 1}/{num_batches}: {len(batch_samples)} samples, loss={batch_loss_value:.6f}")
                    continue  # 跳过正常处理流程
                else:
                    raise e

            # 🔥 梯度累积策略：类似ript-vla的做法
            batch_weight = len(batch_samples) / total_samples
            # 考虑梯度累积步数的归一化
            normalized_batch_loss = batch_loss * batch_weight / gradient_accumulation_steps
            
            # 保存loss数值用于日志
            batch_loss_value = batch_loss.detach().item()
            
            # 立即反向传播，累积梯度
            if scaler is not None:
                scaler.scale(normalized_batch_loss).backward()
            else:
                normalized_batch_loss.backward()
            
            gradient_step += 1
            
            # 累积数值（不保留梯度）
            total_loss += batch_loss_value * batch_weight
            
            # 🔥 参数更新逻辑：达到累积步数或最后一个batch时更新
            if gradient_step == gradient_accumulation_steps or batch_idx == num_batches - 1:
                if optimizer is not None:
                    # 梯度裁剪
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    
                    # 参数更新
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    print(f"  ✓ 参数更新完成 (步骤 {gradient_step}/{gradient_accumulation_steps}, 梯度范数: {grad_norm:.6f})")
                    gradient_step = 0
            
            # 清理batch计算图
            del batch_loss, normalized_batch_loss
            torch.cuda.empty_cache()

            processed_samples += len(batch_samples)

            if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                print(f"  Batch {batch_idx + 1}/{num_batches}: {len(batch_samples)} samples, loss={batch_loss_value:.6f}")

        print(f"✅ Sample pool processing complete:")
        print(f"  - Processed samples: {processed_samples}")
        print(f"  - Total loss: {total_loss:.6f}")

        return total_loss

    def _collate_samples_to_batch(self, samples: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
        """
        将样本列表整理成模型输入的batch格式。

        Args:
            samples: 样本列表
            device: 目标设备

        Returns:
            batch: 模型输入格式的batch
        """
        if not samples:
            raise ValueError("Empty samples list")

        # 使用sample_generator的collate方法
        return self.sample_generator.collate_samples_to_batch(samples, device)

    def _compute_cfg_loss_for_batch(
        self,
        batch: Dict[str, Any],
        advantages: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        为单个batch计算CFG损失（内存优化版本）。

        Args:
            batch: 模型输入batch
            advantages: 该batch的优势值
            device: 目标设备

        Returns:
            loss: 该batch的平均损失
        """
        # 🔥 内存优化：强制清理显存碎片
        torch.cuda.empty_cache()

        # 获取CFG参数
        cfg_alpha = getattr(self.policy.config, 'cfg_uncond_weight', 0.1)

        # 🔥 确保优势tensor在正确设备上
        advantages = advantages.to(device)

        # 二值化优势
        w_pos = (advantages > 0).float().to(device)

        B = batch.get('batch_size', batch['state'].shape[0])

        # CFG分支计算（内存优化）
        if getattr(self.policy.model, 'cfg_enabled', True):
            print(f"🔮 CFG双分支计算: batch_size={B}")

            # 🔥 分阶段计算避免内存峰值
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Step 1: 条件分支
                outputs = self.policy.forward(batch)
                # 兼容 (pred, dict) 或 dict 两种返回
                if isinstance(outputs, tuple):
                    loss_dict_pos = outputs[1]
                else:
                    loss_dict_pos = outputs
                per_step_per_dim_pos = loss_dict_pos['losses']  # (B, T, D)

                # 立即清理中间结果
                del outputs, loss_dict_pos
                torch.cuda.empty_cache()

                # Step 2: 无条件分支
                uncond_batch = batch.copy()
                uncond_batch['prompt'] = [''] * B
                uncond_outputs = self.policy.forward(uncond_batch)
                if isinstance(uncond_outputs, tuple):
                    loss_dict_uncond = uncond_outputs[1]
                else:
                    loss_dict_uncond = uncond_outputs
                per_step_per_dim_uncond = loss_dict_uncond['losses']

                # 立即清理
                del uncond_outputs, loss_dict_uncond, uncond_batch
                torch.cuda.empty_cache()

                # Step 3: CFG组合（在autocast内完成）
                combined_loss_per_step = w_pos.view(B, 1, 1) * per_step_per_dim_pos + cfg_alpha * per_step_per_dim_uncond

                # 立即清理分支结果
                del per_step_per_dim_pos, per_step_per_dim_uncond
                torch.cuda.empty_cache()
        else:
            print(f"📝 单分支计算: batch_size={B}")

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # 非CFG模式：只使用条件分支
                outputs = self.policy.forward(batch)
                if isinstance(outputs, tuple):
                    loss_dict = outputs[1]
                else:
                    loss_dict = outputs
                losses = loss_dict['losses']

                del outputs, loss_dict
                torch.cuda.empty_cache()

                combined_loss_per_step = w_pos.view(B, 1, 1) * losses

                del losses
                torch.cuda.empty_cache()

        # 计算平均损失
        loss = combined_loss_per_step.mean()

        # 最终清理
        del combined_loss_per_step
        torch.cuda.empty_cache()

        return loss

    def compute_weighted_loss_unified(
        self,
        episodes: List[Dict[str, Any]],
        advantages: torch.Tensor,
        device: Optional[torch.device] = None,
        batch_size: Optional[int] = None,
        shuffle_samples: Optional[bool] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        gradient_accumulation_steps: int = 1
    ) -> torch.Tensor:
        """
        🚀 统一样本池训练接口：你想要的理想架构

        这是完整的统一样本池训练方法，实现了：
        1. 统一样本生成
        2. 样本随机化
        3. 固定batch训练
        4. 标准梯度累积

        Args:
            episodes: episode列表
            advantages: episode级优势
            device: 目标设备
            batch_size: 固定batch大小
            shuffle_samples: 是否打散样本

        Returns:
            loss: 总损失
        """
        if not self.use_so100_processing:
            # 如果没有启用SO100，回退到原有方法
            return self.compute_weighted_loss(episodes, advantages, device)

        if device is None:
            device = next(self.policy.parameters()).device

        # 从配置读取可调参数，避免硬编码
        if batch_size is None:
            batch_size = getattr(self.policy.config, 'unified_pool_batch_size', 32)
        if shuffle_samples is None:
            shuffle_samples = getattr(self.policy.config, 'unified_pool_shuffle', True)

        print(f"🚀 Unified Sample Pool Training:")
        print(f"  - Episodes: {len(episodes)}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Shuffle samples: {shuffle_samples}")

        # 1. 创建统一样本池
        samples, sample_advantages = self.create_unified_sample_pool(
            episodes, advantages, device, shuffle_samples
        )

        # 2. 从样本池计算损失（内部已进行梯度累积）
        loss_value = self.compute_loss_from_sample_pool(
            samples, sample_advantages, batch_size, device, scaler, optimizer, gradient_accumulation_steps
        )

        print(f"✅ Unified training complete, loss: {loss_value:.6f}")

        # 返回零梯度tensor，因为梯度已在内部累积
        return torch.tensor(loss_value, device=device, requires_grad=False)

    def process_episodes(
        self,
        episodes: List[Dict[str, Any]],
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, Any], List[int]]:
        """
        🔥 Phase 2: Unified episode processing method.

        Routes to either SO100-style processing or legacy windowing based on configuration.

        Args:
            episodes: List of episode dictionaries
            device: Target device for tensors

        Returns:
            Tuple of:
                - batch: Training batch
                - owner_indices: List mapping batch indices to episode indices
        """
        if self.use_so100_processing:
            # Use SO100-style processing
            batch, episode_to_samples_map = self.process_episodes_to_samples_so100(episodes, device)
            owner_indices = batch['owner_indices']
            return batch, owner_indices
        else:
            # Use legacy windowing processing
            return self._extract_microbatch_data(episodes, device)
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using loaded statistics"""
        return (state - self.state_mean[:len(state)]) / (self.state_std[:len(state)] + 1e-6)
    
    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action using loaded statistics"""
        return action * (self.action_std + 1e-6) + self.action_mean

    def process_episodes(
        self,
        episodes: List[Dict[str, Any]],
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, Any], List[int]]:
        """
        将 episodes 打包为 PI0Policy 期望的 batch，支持窗口化采样：
        - 根据windowing_mode从每条轨迹产生多个窗口样本
        - action 形状保持 (B, T, 7)，B现在是总窗口数
        - 提供 action_is_pad: (B, T) 布尔，True 表示该时间步是 padding
        - 状态统一 8 维 (3 pos + 3 axis-angle + 2 gripper)
        
        Returns:
            batch: 窗口化后的训练批次，B维=总窗口数
            owner_indices: 每个窗口对应的原始episode索引，用于优势映射
        """
        if device is None:
            device = self.device
        if not episodes:
            raise ValueError("Empty episodes list provided")

        all_states, all_images, all_wrist_images, all_actions = [], [], [], []
        all_action_is_pad, all_tasks = [], []
        owner_indices = []  # 🔥 新增：记录每个窗口来自哪个episode

        target_seq_len = getattr(self.policy.config, 'n_action_steps', 50)  # 🔥 窗口大小=50
        diffusion_steps = getattr(self.policy.config, 'num_steps', 10)
        print(f"🔧 窗口化采样: target_seq_len={target_seq_len}, mode={self.windowing_mode}")

        for i, ep in enumerate(episodes):
            try:
                observations = ep['observations']
                actions = ep['actions']
                tasks = ep.get('task', "default task")
                if not observations:
                    raise ValueError(f"Episode {i} empty observations")

                # 🔥 新增：提取完整轨迹数据
                states_seq, base_images_seq, wrist_images_seq, actions_seq = [], [], [], []
                max_steps = min(len(observations), len(actions))
                
                # 提取完整序列数据
                for t in range(max_steps):
                    obs_t = observations[t] if t < len(observations) else {}
                    act_t = actions[t] if t < len(actions) else np.zeros(7, np.float32)

                    # 状态(8维)
                    states_seq.append(self._extract_state_from_obs(obs_t, i, t))
                    # 分别提取两个相机的图像
                    base_img, wrist_img = self._extract_dual_images_from_obs(obs_t, i, t)
                    base_images_seq.append(base_img)
                    wrist_images_seq.append(wrist_img)

                    # 动作(7维)
                    act_t = np.array(act_t[0] if (isinstance(act_t, list) and len(act_t) > 0) else act_t,
                                    dtype=np.float32)
                    if act_t.size != 7:
                        buf = np.zeros(7, dtype=np.float32)
                        buf[:min(7, act_t.size)] = act_t[:min(7, act_t.size)]
                        act_t = buf
                    actions_seq.append(act_t)

                # 🔥 窗口化采样：根据模式产生多个窗口
                windows = self._sample_windows_from_episode(
                    states_seq, base_images_seq, wrist_images_seq, actions_seq, target_seq_len
                )
                
                task_str = tasks[0] if isinstance(tasks, list) else str(tasks)
                
                # 为每个窗口添加数据和owner索引
                for window in windows:
                    all_states.append(window['state'])
                    all_images.append(window['base_image'])
                    all_wrist_images.append(window['wrist_image'])
                    all_actions.append(window['actions'])
                    all_action_is_pad.append(window['action_is_pad'])
                    all_tasks.append(task_str)
                    owner_indices.append(i)  # 记录窗口来源

            except Exception as e:
                print(f"Error processing episode {i}: {e}")
                # 兜底：至少产生一个默认窗口
                all_states.append(np.zeros(8, np.float32))
                all_images.append(np.ones((224, 224, 3), np.uint8) * 128)
                all_wrist_images.append(np.ones((224, 224, 3), np.uint8) * 128)
                all_actions.append(np.zeros((target_seq_len, 7), np.float32))
                all_action_is_pad.append(np.ones((target_seq_len,), dtype=bool))
                all_tasks.append("default task")
                owner_indices.append(i)

        batch = {
            "state": torch.from_numpy(np.stack(all_states)).to(device, dtype=torch.float32),            # (B,8)
            "image": {
                "base_0_rgb": torch.from_numpy(np.stack(all_images)).to(device, dtype=torch.uint8),       # (B,H,W,3)
                "left_wrist_0_rgb": torch.from_numpy(np.stack(all_wrist_images)).to(device, dtype=torch.uint8),  # (B,H,W,3) 🔥 新增
            },
            "action": torch.from_numpy(np.stack(all_actions)).to(device, dtype=torch.float32),          # (B,T,7)
            "action_is_pad": torch.from_numpy(np.stack(all_action_is_pad)).to(device),                  # (B,T) bool
            "prompt": all_tasks,
        }

        num_windows = len(all_states)
        assert batch["state"].shape[0] == num_windows
        assert batch["image"]["base_0_rgb"].shape[0] == num_windows
        assert batch["action"].shape[0] == num_windows
        assert batch["action_is_pad"].shape[0] == num_windows
        assert len(owner_indices) == num_windows
        
        print(f"🔧 窗口化批次: {len(episodes)} episodes → {num_windows} windows")
        print(f"   state {batch['state'].shape}, image {batch['image']['base_0_rgb'].shape}")
        print(f"   action {batch['action'].shape}, action_is_pad {batch['action_is_pad'].shape}")
        
        return batch, owner_indices
    
    def _sample_windows_from_episode(self, states_seq, base_images_seq, wrist_images_seq, actions_seq, target_seq_len):
        """🔥 新增：根据窗口化模式从一条轨迹中采样多个窗口"""
        if len(actions_seq) == 0:
            # 空轨迹，返回一个默认窗口
            return [self._create_empty_window(target_seq_len)]
        
        windows = []
        
        if self.windowing_mode == 'last':
            # 原有逻辑：只取最后一段
            window = self._create_window_from_range(
                states_seq, base_images_seq, wrist_images_seq, actions_seq, 
                -target_seq_len, len(actions_seq), target_seq_len
            )
            windows.append(window)
            
        elif self.windowing_mode == 'random':
            # 随机采样：从轨迹中随机取一个窗口
            if len(actions_seq) <= target_seq_len:
                # 轨迹太短，整条轨迹就是一个窗口
                window = self._create_window_from_range(
                    states_seq, base_images_seq, wrist_images_seq, actions_seq,
                    0, len(actions_seq), target_seq_len
                )
                windows.append(window)
            else:
                # 随机选择起始位置
                import random
                max_start = len(actions_seq) - target_seq_len
                start_idx = random.randint(0, max_start)
                window = self._create_window_from_range(
                    states_seq, base_images_seq, wrist_images_seq, actions_seq,
                    start_idx, start_idx + target_seq_len, target_seq_len
                )
                windows.append(window)
                
        elif self.windowing_mode == 'slide':
            # 滑动窗口：按步长采样多个窗口
            if len(actions_seq) <= target_seq_len:
                # 轨迹太短，只有一个窗口
                window = self._create_window_from_range(
                    states_seq, base_images_seq, wrist_images_seq, actions_seq,
                    0, len(actions_seq), target_seq_len
                )
                windows.append(window)
            else:
                # 滑动采样
                window_count = 0
                for start_idx in range(0, len(actions_seq) - target_seq_len + 1, self.window_stride):
                    if window_count >= self.max_windows_per_episode:
                        break
                    
                    window = self._create_window_from_range(
                        states_seq, base_images_seq, wrist_images_seq, actions_seq,
                        start_idx, start_idx + target_seq_len, target_seq_len
                    )
                    windows.append(window)
                    window_count += 1
                
                # 如果没有采样到任何窗口（理论上不会发生），回退到last模式
                if not windows:
                    window = self._create_window_from_range(
                        states_seq, base_images_seq, wrist_images_seq, actions_seq,
                        -target_seq_len, len(actions_seq), target_seq_len
                    )
                    windows.append(window)
        
        else:
            raise ValueError(f"Unknown windowing_mode: {self.windowing_mode}")
        
        return windows
    
    def _create_window_from_range(self, states_seq, base_images_seq, wrist_images_seq, actions_seq, start_idx, end_idx, target_seq_len):
        """从指定范围创建一个窗口 - 修复时序对齐问题
        
        正确的时序对齐：obs[t] → actions[t:t+H]
        即：用窗口起点的观测，去监督接下来这H步动作
        """
        # 处理负索引
        if start_idx < 0:
            start_idx = max(0, len(actions_seq) + start_idx)
        if end_idx < 0:
            end_idx = len(actions_seq) + end_idx
        
        # 确保范围有效
        start_idx = max(0, min(start_idx, len(actions_seq)))
        end_idx = max(start_idx, min(end_idx, len(actions_seq)))
        
        # 🔥 关键修复：正确的时序对齐
        # 窗口起点的obs -> 该obs之后的actions
        state_idx = start_idx  # 用窗口起点的obs
        
        # 提取窗口动作数据（与start_idx对应的obs之后的actions）
        window_actions = actions_seq[start_idx:end_idx]
        valid_len = len(window_actions)
        
        # 创建固定长度的动作序列
        final_actions = np.zeros((target_seq_len, 7), dtype=np.float32)
        if valid_len > 0:
            final_actions[:valid_len] = np.asarray(window_actions, dtype=np.float32)
        
        # 创建 padding 掩码
        action_is_pad = np.ones((target_seq_len,), dtype=bool)
        action_is_pad[:valid_len] = False
        
        # 🔥 使用窗口起点的状态和图像
        if state_idx < len(states_seq):
            final_state = np.asarray(states_seq[state_idx], dtype=np.float32)
            final_base_image = base_images_seq[state_idx] if state_idx < len(base_images_seq) else (np.ones((224, 224, 3), np.uint8) * 128)
            final_wrist_image = wrist_images_seq[state_idx] if state_idx < len(wrist_images_seq) else (np.ones((224, 224, 3), np.uint8) * 128)
        else:
            # 兜底：如果索引超出范围，使用默认值
            final_state = np.zeros(8, np.float32)
            final_base_image = np.ones((224, 224, 3), np.uint8) * 128
            final_wrist_image = np.ones((224, 224, 3), np.uint8) * 128
        
        return {
            'state': final_state,
            'base_image': final_base_image,
            'wrist_image': final_wrist_image,
            'actions': final_actions,
            'action_is_pad': action_is_pad
        }
    
    def _create_empty_window(self, target_seq_len):
        """创建一个空的默认窗口"""
        return {
            'state': np.zeros(8, np.float32),
            'base_image': np.ones((224, 224, 3), np.uint8) * 128,
            'wrist_image': np.ones((224, 224, 3), np.uint8) * 128,
            'actions': np.zeros((target_seq_len, 7), np.float32),
            'action_is_pad': np.ones((target_seq_len,), dtype=bool)
        }
    def compute_weighted_loss(
        self,
        episodes: List[Dict[str, Any]],
        advantages: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        CFG风格训练支持窗口化：同时计算条件和无条件损失
        
        严格契约模式：所有输入/输出必须满足预定义契约，违反则立即失败
        
        🔥 新增窗口化支持：
        - episodes: E个原始轨迹
        - advantages: (E,) episode级优势
        - 通过owner_indices映射到B个窗口级优势
        
        输入契约:
        - episodes: 非空list，每个episode包含必需字段
        - advantages: (E,) tensor，E与episodes数量匹配
        
        输出契约:
        - policy.forward()必须返回包含"losses"字段的dict，shape为(B,T,D)
        - action_is_pad必须为(B,T) bool tensor，至少有一个有效步
        """
        if device is None:
            device = self.device

        # === 严格输入契约验证 ===
        assert episodes and len(episodes) > 0, "episodes不能为空"
        assert advantages is not None and len(advantages) > 0, "advantages不能为空"
        assert len(episodes) == len(advantages), f"episodes数量({len(episodes)})必须与advantages数量({len(advantages)})匹配"
        assert advantages.dim() == 1, f"advantages必须是1维tensor，当前维度: {advantages.dim()}"
        assert isinstance(advantages, torch.Tensor), f"advantages必须是torch.Tensor类型，当前类型: {type(advantages)}"

        # 🔥 Phase 2: Choose processing method based on configuration
        if self.use_so100_processing:
            print("🚀 Using SO100-style sample processing...")
            batch, episode_to_samples_map = self.process_episodes_to_samples_so100(episodes, device)

            # Map episode advantages to sample advantages
            sample_advantages = self.map_episode_advantages_to_samples_so100(advantages, episode_to_samples_map)

            # Create owner indices for compatibility
            owner_indices = batch['owner_indices']

            # Use sample advantages instead of episode advantages
            window_advantages = sample_advantages

        else:
            print("🔧 Using legacy windowing processing...")
            # 🔥 Legacy: 窗口化批次处理
            batch, owner_indices = self.process_episodes(episodes, device)

            # 🔥 优势映射和归一化处理 (legacy approach)
            B = batch["state"].shape[0]
            window_advantages = torch.zeros(B, device=device, dtype=advantages.dtype)
            for window_idx, episode_idx in enumerate(owner_indices):
                window_advantages[window_idx] = advantages[episode_idx]
        
        # === 窗口化批次验证 ===
        assert "action_is_pad" in batch, "batch中必须包含action_is_pad字段"
        action_is_pad = batch["action_is_pad"]
        assert action_is_pad.dtype == torch.bool, f"action_is_pad必须是bool类型，当前类型: {action_is_pad.dtype}"
        assert action_is_pad.dim() == 2, f"action_is_pad必须是2维tensor (B,T)，当前维度: {action_is_pad.dim()}"
        
        # Get batch size (works for both processing methods)
        if self.use_so100_processing:
            B = batch["batch_size"]  # SO100 processing provides batch_size directly
        else:
            B = batch["state"].shape[0]  # Legacy windowing uses tensor shape

        assert len(owner_indices) == B, f"owner_indices长度({len(owner_indices)})必须与批次大小({B})匹配"
        
        # 验证至少有一个有效步
        valid_steps = (~action_is_pad).sum()
        assert valid_steps > 0, "action_is_pad显示所有步骤都是padding，必须至少有一个有效步骤"
        
        # === 关键修复：统一采样noise和time用于CFG双分支 ===
        n, d = self.policy.config.n_action_steps, self.policy.config.max_action_dim
        dtype = batch["state"].dtype
        
        # 采样一次noise和time，两个分支共享（与最初CFGRL实现对齐）
        noise = torch.randn(B, n, d, device=device, dtype=dtype)
        time = self.policy.model.sample_time(B, device).to(dtype)
        
        # Advantage mapping is now handled above based on processing method
        
        # 🔥 二值优势：简单判断正负（按用户要求保持二值化）
        w_pos = (window_advantages > 0).float()
        
        # 记录正优势窗口占比
        positive_ratio = w_pos.mean()
        
        processing_type = "samples" if self.use_so100_processing else "windows"
        print(f"🔧 优势映射: {len(episodes)} episodes → {B} {processing_type}")
        print(f"   episode优势: {advantages.shape} = {advantages[:3].tolist()[:3]}...")
        print(f"   {processing_type}优势: {window_advantages.shape} = {window_advantages[:3].tolist()[:3]}...")
        print(f"   二值优势: {w_pos[:3].tolist()[:3]}...")
        print(f"   正优势{processing_type}占比: {positive_ratio:.2%}")
        
        # === CFG风格双分支损失计算 ===
        
        # 1. 条件分支（正样本指示）- 使用共享的noise和time
        batch_positive = batch.copy()
        batch_positive["is_positive"] = torch.ones(B, device=device, dtype=torch.long)
        batch_positive["noise"] = noise
        batch_positive["time"] = time
        
        out_positive = self.policy.forward(batch_positive)
        
        # === 严格输出契约验证 - 条件分支 ===
        assert isinstance(out_positive, (dict, tuple)), f"policy.forward()必须返回dict或tuple，条件分支返回类型: {type(out_positive)}"
        
        if isinstance(out_positive, tuple):
            assert len(out_positive) >= 2, f"tuple返回值必须至少包含2个元素，当前长度: {len(out_positive)}"
            loss_scalar_pos, loss_dict_pos = out_positive[0], out_positive[1]
        else:
            loss_dict_pos = out_positive
            loss_scalar_pos = out_positive.get("loss")
        
        assert isinstance(loss_dict_pos, dict), f"loss_dict必须是dict类型，条件分支类型: {type(loss_dict_pos)}"
        assert "losses" in loss_dict_pos, "policy forward输出必须包含'losses'字段"
        
        per_step_per_dim_pos = loss_dict_pos["losses"]
        assert isinstance(per_step_per_dim_pos, torch.Tensor), f"losses必须是torch.Tensor，条件分支类型: {type(per_step_per_dim_pos)}"
        assert per_step_per_dim_pos.dim() == 3, f"losses必须是3维tensor (B,T,D)，条件分支维度: {per_step_per_dim_pos.dim()}"
        assert per_step_per_dim_pos.shape[0] == B, f"losses批次维度({per_step_per_dim_pos.shape[0]})必须与窗口数量({B})匹配"
        assert per_step_per_dim_pos.shape[1] == action_is_pad.shape[1], f"losses时间维度({per_step_per_dim_pos.shape[1]})必须与action_is_pad时间维度({action_is_pad.shape[1]})匹配"

        # 2. 无条件分支（无指示）- 使用相同的noise和time
        batch_uncond = batch.copy()
        batch_uncond["is_positive"] = torch.zeros(B, device=device, dtype=torch.long)
        batch_uncond["noise"] = noise  # 关键：与条件分支共享相同的noise
        batch_uncond["time"] = time    # 关键：与条件分支共享相同的time
        
        out_uncond = self.policy.forward(batch_uncond)
        
        # === 严格输出契约验证 - 无条件分支 ===
        assert isinstance(out_uncond, (dict, tuple)), f"policy.forward()必须返回dict或tuple，无条件分支返回类型: {type(out_uncond)}"
        
        if isinstance(out_uncond, tuple):
            assert len(out_uncond) >= 2, f"tuple返回值必须至少包含2个元素，当前长度: {len(out_uncond)}"
            loss_scalar_uncond, loss_dict_uncond = out_uncond[0], out_uncond[1]
        else:
            loss_dict_uncond = out_uncond
            loss_scalar_uncond = out_uncond.get("loss")
        
        assert isinstance(loss_dict_uncond, dict), f"loss_dict必须是dict类型，无条件分支类型: {type(loss_dict_uncond)}"
        assert "losses" in loss_dict_uncond, "policy forward输出必须包含'losses'字段"
        
        per_step_per_dim_uncond = loss_dict_uncond["losses"]
        assert isinstance(per_step_per_dim_uncond, torch.Tensor), f"losses必须是torch.Tensor，无条件分支类型: {type(per_step_per_dim_uncond)}"
        assert per_step_per_dim_uncond.dim() == 3, f"losses必须是3维tensor (B,T,D)，无条件分支维度: {per_step_per_dim_uncond.dim()}"
        assert per_step_per_dim_uncond.shape == per_step_per_dim_pos.shape, f"无条件分支losses形状({per_step_per_dim_uncond.shape})必须与条件分支({per_step_per_dim_pos.shape})完全匹配"

        # 3. 计算CFG组合损失
        per_step_pos = per_step_per_dim_pos.mean(dim=-1)  # (B,T)
        per_step_uncond = per_step_per_dim_uncond.mean(dim=-1)  # (B,T)
        
        # 获取有效步掩码，排除padding步
        mask = (~action_is_pad).float()  # (B,T)
        
        # 🔥 CFG权重计算：使用二值优势
        w_pos = w_pos.unsqueeze(1).expand_as(mask)  # (B,T) 二值化
        
        # 🔥 关键改进：标准CFGRL公式 - L = w_pos * L_pos + w_uncond * L_uncond
        cfg_alpha = getattr(self.policy.config, 'cfg_uncond_weight', 0.1)
        combined_loss_per_step = w_pos * per_step_pos + cfg_alpha * per_step_uncond
        
        # 🔥 关键修复：Padding感知的损失归约 - 按有效步数归一化
        # 每个窗口的有效损失 = 总损失 / 有效步数
        window_valid_steps = mask.sum(dim=1)  # (B,) 每个窗口的有效步数
        window_losses = (combined_loss_per_step * mask).sum(dim=1) / (window_valid_steps + 1e-8)  # (B,)
        
        # 最终损失：所有窗口的平均损失（每个窗口权重相等）
        final_loss = window_losses.mean()
        
        assert torch.isfinite(final_loss), f"CFG损失计算结果必须是有限数值，当前值: {final_loss}"
        assert not torch.isnan(final_loss), "CFG损失计算结果不能是NaN"
        assert not torch.isinf(final_loss), "CFG损失计算结果不能是Inf"
        
        return final_loss
    
    def compute_weighted_loss_microbatch(
        self,
        episodes: List[Dict[str, Any]],
        advantages: torch.Tensor,
        device: Optional[torch.device] = None,
        micro_batch_size: int = 8,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        optimizer = None,
        scaler = None
    ) -> torch.Tensor:
        """
        🔥 窗口级微批梯度累积版本 - 解决显存OOM问题
        
        核心思路：
        1. 先process_episodes得到所有窗口的batch (B_windows)
        2. 沿B_windows维度切分为micro_batch_size的小块
        3. 每个微批次：共享noise/time，分别计算条件/无条件分支，组合损失
        4. 损失/grad_accum_steps后backward（累积梯度）
        5. 满grad_accum_steps时统一step
        
        Args:
            episodes: E个原始轨迹
            advantages: (E,) episode级优势
            micro_batch_size: 微批次窗口数（控制显存峰值）
            grad_accum_steps: 梯度累积步数
            use_amp: 是否使用混合精度
            optimizer: 优化器（用于step）
            scaler: GradScaler（AMP模式下使用）
        
        Returns:
            平均损失值
        """
        if device is None:
            device = self.device
        
        print(f"🔧 窗口级微批处理: micro_batch_size={micro_batch_size}, grad_accum_steps={grad_accum_steps}, AMP={use_amp}")
        
        # 1. 处理episodes得到窗口batch
        batch, owner_indices = self.process_episodes(episodes, device)
        
        B_windows = batch["state"].shape[0]
        print(f"   总窗口数: {B_windows}")
        
        if B_windows == 0:
            print("⚠️ 无有效窗口数据")
            return torch.tensor(0.0, device=device)
        
        # 2. 准备优势映射
        window_advantages = torch.zeros(B_windows, device=device, dtype=advantages.dtype)
        for window_idx, episode_idx in enumerate(owner_indices):
            window_advantages[window_idx] = advantages[episode_idx]
        
        w_pos = (window_advantages > 0).float()
        
        # 3. 准备全局noise和time（所有微批次共享）
        n, d = self.policy.config.n_action_steps, self.policy.config.max_action_dim
        dtype = batch["state"].dtype
        
        # 🔥 关键：为整个batch生成统一的noise和time
        global_noise = torch.randn(B_windows, n, d, device=device, dtype=dtype)
        global_time = self.policy.model.sample_time(B_windows, device).to(dtype)
        
        # 4. 微批次处理
        total_loss = 0.0
        num_micro_batches = 0
        
        for start_idx in range(0, B_windows, micro_batch_size):
            end_idx = min(start_idx + micro_batch_size, B_windows)
            micro_B = end_idx - start_idx
            
            # 提取微批次数据
            micro_batch = {}
            for key, value in batch.items():
                if key == 'states':
                    # 处理嵌套的states字典
                    micro_batch[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor) and len(sub_value.shape) > 0:
                            micro_batch[key][sub_key] = sub_value[start_idx:end_idx]
                        else:
                            micro_batch[key][sub_key] = sub_value
                elif isinstance(value, torch.Tensor) and len(value.shape) > 0:
                    micro_batch[key] = value[start_idx:end_idx]
                else:
                    micro_batch[key] = value
            
            # 提取微批次的优势和noise/time
            micro_w_pos = w_pos[start_idx:end_idx]
            micro_noise = global_noise[start_idx:end_idx]
            micro_time = global_time[start_idx:end_idx]
            
            # 5. 🔥 使用autocast进行微批次的双分支计算
            try:
                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        micro_loss = self._compute_micro_batch_loss(
                            micro_batch, micro_w_pos, micro_noise, micro_time, device
                        )
                else:
                    micro_loss = self._compute_micro_batch_loss(
                        micro_batch, micro_w_pos, micro_noise, micro_time, device
                    )
                
                # 损失归一化并累积梯度
                normalized_loss = micro_loss / grad_accum_steps
                
                if optimizer is not None:
                    if use_amp and scaler is not None:
                        scaler.scale(normalized_loss).backward()
                    else:
                        normalized_loss.backward()
                
                total_loss += micro_loss.item()
                num_micro_batches += 1
                
                print(f"   微批次 {num_micro_batches}: B={micro_B}, loss={micro_loss.item():.6f}")
                
            except Exception as e:
                print(f"❌ 微批次 {start_idx}:{end_idx} 处理失败: {e}")
                continue
        
        # 6. 统一进行梯度更新（如果提供了optimizer）
        if optimizer is not None and num_micro_batches >= grad_accum_steps:
            try:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                print(f"✅ 窗口级微批更新完成: 梯度范数={grad_norm:.6f}")
            except Exception as e:
                print(f"❌ 梯度更新失败: {e}")
                optimizer.zero_grad()
        
        avg_loss = total_loss / max(1, num_micro_batches)
        print(f"🎯 窗口级微批处理完成: 平均损失={avg_loss:.6f}")
        
        return torch.tensor(avg_loss, device=device)
    
    def _compute_micro_batch_loss(
        self, 
        micro_batch: Dict[str, Any], 
        w_pos: torch.Tensor, 
        noise: torch.Tensor, 
        time: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        计算单个微批次的CFG损失
        
        Args:
            micro_batch: 微批次数据
            w_pos: 微批次正优势掩码
            noise: 微批次noise
            time: 微批次time
            device: 计算设备
        
        Returns:
            微批次损失
        """
        B = micro_batch["state"].shape[0]
        
        # 1. 条件分支
        batch_positive = micro_batch.copy()
        batch_positive["is_positive"] = torch.ones(B, device=device, dtype=torch.long)
        batch_positive["noise"] = noise
        batch_positive["time"] = time
        
        out_positive = self.policy.forward(batch_positive)
        if isinstance(out_positive, tuple):
            loss_dict_pos = out_positive[1]
        else:
            loss_dict_pos = out_positive
        
        per_step_per_dim_pos = loss_dict_pos["losses"]
        
        # 2. 无条件分支（共享noise和time）
        batch_uncond = micro_batch.copy()
        batch_uncond["is_positive"] = torch.zeros(B, device=device, dtype=torch.long)
        batch_uncond["noise"] = noise  # 🔥 共享相同的noise
        batch_uncond["time"] = time    # 🔥 共享相同的time
        
        out_uncond = self.policy.forward(batch_uncond)
        if isinstance(out_uncond, tuple):
            loss_dict_uncond = out_uncond[1]
        else:
            loss_dict_uncond = out_uncond
        
        per_step_per_dim_uncond = loss_dict_uncond["losses"]
        
        # 3. CFG组合损失计算
        per_step_pos = per_step_per_dim_pos.mean(dim=-1)  # (B,T)
        per_step_uncond = per_step_per_dim_uncond.mean(dim=-1)  # (B,T)
        
        # 获取有效步掩码
        mask = (~micro_batch["action_is_pad"]).float()  # (B,T)
        
        # CFG权重计算
        w_pos_expanded = w_pos.unsqueeze(1).expand_as(mask)  # (B,T)
        
        # 标准CFGRL公式
        cfg_alpha = getattr(self.policy.config, 'cfg_uncond_weight', 0.1)
        combined_loss_per_step = w_pos_expanded * per_step_pos + cfg_alpha * per_step_uncond
        
        # Padding感知的损失归约
        window_valid_steps = mask.sum(dim=1)  # (B,)
        window_losses = (combined_loss_per_step * mask).sum(dim=1) / (window_valid_steps + 1e-8)  # (B,)
        
        # 微批次最终损失
        micro_loss = window_losses.mean()
        
        return micro_loss
    
    def compute_act_logits(self, model, episodes: List[Dict[str, Any]], device: Optional[torch.device] = None):
        """
        This method is required by the RLModelInterface, but for our CFG-style
        optimizer, we use `compute_weighted_loss` instead.
        We can raise an error or return a placeholder.
        """
        raise NotImplementedError(
            "compute_act_logits is not used for PI0_CFG_Adapter. "
            "Use compute_weighted_loss instead."
        )

    @property
    def device(self):
        return next(self.policy.parameters()).device

    def get_policy_model(self):
        """Return the policy model, which is the PI0Policy instance itself."""
        return self.policy
        
    def _extract_state_from_obs(self, obs, episode_idx, step_idx):
        """提取 8 维状态: 3 pos + 3 axis-angle + 2 gripper，并按 norm_stats 归一化"""
        try:
            if isinstance(obs, np.ndarray) and obs.dtype == object:
                obs = (obs.item() if obs.size == 1 else obs[0]) if obs.size > 0 else {}
            if isinstance(obs, list) and len(obs) > 0:
                obs = obs[0]
            if not isinstance(obs, dict) or not obs:
                return np.zeros(8, np.float32)

            if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
                eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
                eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
                if eef_pos.size != 3: eef_pos = np.zeros(3, np.float32)
                if eef_quat.size != 4: eef_quat = np.array([0,0,0,1], np.float32)
                try:
                    from pi0.ript.utils.pi0_libero_utils import quat2axisangle
                    axis_angle = quat2axisangle(eef_quat).astype(np.float32)
                except Exception:
                    axis_angle = np.zeros(3, np.float32)

                gr = obs.get("robot0_gripper_qpos", [0.0, 0.0])
                gr = np.array(gr, dtype=np.float32)
                if gr.size < 2:
                    gr = np.pad(gr, (0, 2 - gr.size))
                else:
                    gr = gr[:2]

                unnorm = np.concatenate([eef_pos[:3], axis_angle[:3], gr[:2]], dtype=np.float32)  # (8,)
                state = self.normalize_state(unnorm)

            else:
                # 无关键字段则回退为 0（与统计相容）
                state = np.zeros(8, np.float32)

            return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        except Exception as e:
            print(f"提取状态时出错 (episode {episode_idx}, step {step_idx}): {e}")
            return np.zeros(8, np.float32)
    
    def _extract_dual_images_from_obs(self, obs, episode_idx, step_idx):
        """🔥 新增：从观测中提取两个相机的图像信息"""
        try:
            # 🔧 修复：处理SubprocVectorEnv返回的numpy.array包装的观测
            if isinstance(obs, np.ndarray) and obs.dtype == object:
                if obs.size == 1:
                    obs = obs.item()  # 提取单个元素
                elif obs.size > 0:
                    obs = obs[0]  # 取第一个元素
            
            if isinstance(obs, list) and len(obs) > 0:
                obs = obs[0]  # 取第一个环境的观测
            
            # 默认图像（兜底）
            default_img = np.ones((224, 224, 3), np.uint8) * 128
            base_img = default_img.copy()
            wrist_img = default_img.copy()
            
            if isinstance(obs, dict):
                # 提取base_0_rgb (agentview_image)
                if "agentview_image" in obs:
                    raw_img = obs["agentview_image"]
                    if isinstance(raw_img, np.ndarray) and raw_img.size > 0:
                        base_img = self._process_single_image(raw_img, "base")
                
                # 提取left_wrist_0_rgb (robot0_eye_in_hand_image)
                if "robot0_eye_in_hand_image" in obs:
                    raw_img = obs["robot0_eye_in_hand_image"]
                    if isinstance(raw_img, np.ndarray) and raw_img.size > 0:
                        wrist_img = self._process_single_image(raw_img, "wrist")
            
            return base_img, wrist_img
            
        except Exception as e:
            print(f"提取双图像时出错 (episode {episode_idx}, step {step_idx}): {e}")
            default_img = np.ones((224, 224, 3), np.uint8) * 128
            return default_img.copy(), default_img.copy()
    
    def _process_single_image(self, img, cam_type):
        """处理单个图像的通用逻辑，与"2_pi0_on_libero.py"的to_hwc_hmirror完全对齐"""
        try:
            # 🔧 统一图像处理：先规范到HWC，再做水平镜像（不做通道交换）
            if img.ndim == 3:
                # CHW -> HWC（如果需要）
                if img.shape[0] == 3 and img.shape[-1] != 3:
                    img = img.transpose(1, 2, 0)
                elif img.shape[-1] != 3:  # 既不是CHW也不是HWC
                    print(f"✗ 未知{cam_type}图像格式: {img.shape}")
                    raise ValueError(f"Unexpected {cam_type} image format: {img.shape}")
                
                # 水平镜像（翻转宽度维）
                img_processed = img[:, ::-1, :].copy()
                
                # 确保数据类型和范围正确
                if img_processed.dtype != np.uint8:
                    if img_processed.max() <= 1.0:  # 归一化的图像
                        img_processed = (img_processed * 255).astype(np.uint8)
                    else:
                        img_processed = img_processed.astype(np.uint8)
                
                # 确保图像尺寸正确
                if img_processed.shape[:2] != (224, 224):
                    try:
                        from skimage.transform import resize
                        img_processed = resize(img_processed, (224, 224), preserve_range=True).astype(np.uint8)
                    except ImportError:
                        # 如果没有skimage，使用简单的裁剪/填充
                        h, w = img_processed.shape[:2]
                        if h != 224 or w != 224:
                            # 简单居中裁剪或填充到224x224
                            resized = np.ones((224, 224, 3), dtype=np.uint8) * 128
                            start_h = max(0, (224 - h) // 2)
                            start_w = max(0, (224 - w) // 2)
                            end_h = min(224, start_h + h)
                            end_w = min(224, start_w + w)
                            src_h = min(h, 224)
                            src_w = min(w, 224)
                            resized[start_h:end_h, start_w:end_w] = img_processed[:src_h, :src_w]
                            img_processed = resized
                
                return img_processed
            else:
                print(f"✗ {cam_type}图像维度错误: {img.shape}")
                return np.ones((224, 224, 3), np.uint8) * 128
                
        except Exception as e:
            print(f"处理{cam_type}图像时出错: {e}")
            return np.ones((224, 224, 3), np.uint8) * 128

    def _extract_image_from_obs(self, obs, episode_idx, step_idx):
        """从观测中提取图像信息"""
        try:
            # 🔧 修复：处理SubprocVectorEnv返回的numpy.array包装的观测
            if isinstance(obs, np.ndarray) and obs.dtype == object:
                if obs.size == 1:
                    obs = obs.item()  # 提取单个元素
                elif obs.size > 0:
                    obs = obs[0]  # 取第一个元素
            
            if isinstance(obs, list) and len(obs) > 0:
                obs = obs[0]  # 取第一个环境的观测
            
            if isinstance(obs, dict) and "agentview_image" in obs:
                img = obs["agentview_image"]
                if isinstance(img, np.ndarray) and img.size > 0:
                    # 🔧 修复图像格式检查：处理CHW和HWC两种格式
                    if img.ndim == 3:
                        # 检查是CHW格式 (3, H, W) 还是HWC格式 (H, W, 3)
                        if img.shape[0] == 3 and img.shape[-1] != 3:  # CHW格式
                            # 转换CHW → HWC
                            img = img.transpose(1, 2, 0)
                        elif img.shape[-1] != 3:  # 既不是CHW也不是HWC
                            print(f"✗ 未知图像格式: {img.shape}")
                            raise ValueError(f"Unexpected image format: {img.shape}")
                        
                        # 现在img应该是HWC格式
                        # 静默处理图像（减少冗余日志）
                        
                        # 为模型推理准备图像（确保接收正确方向的图像）
                        # 根据您的需求，确保模型接收的是正着的图像
                        model_img = img.copy()  # 直接使用原始图像，不进行旋转
                        
                        # 收集图像用于视频生成（需要旋转以显示正确方向）
                        video_img = img[::-1, ::-1]  # 180度旋转用于视频显示
                        self._collect_image_for_video(video_img, episode_idx, step_idx)
                        
                        # 返回用于模型推理的图像（已进行180度旋转）
                        # 如果图像尺寸不是 224x224，需要调整大小
                        if model_img.shape[:2] != (224, 224):
                            try:
                                from skimage.transform import resize
                                model_img = resize(model_img, (224, 224), preserve_range=True).astype(np.uint8)
                            except ImportError:
                                # 如果没有 skimage，使用简单的插值
                                import cv2
                                model_img = cv2.resize(model_img, (224, 224))
                        return model_img
                    else:
                        print(f"✗ 图像维度异常: {img.shape}")
                else:
                    print(f"✗ 图像数据无效: type={type(img)}, size={getattr(img, 'size', 'N/A')}")
            else:
                available_keys = list(obs.keys()) if isinstance(obs, dict) else "非字典类型"
                print(f"✗ 观测中无agentview_image键, 可用键: {available_keys}")
            
            # 如果没有有效图像，返回占位符
            print(f"✗ 使用占位符图像 (episode {episode_idx}, step {step_idx})")
            return np.ones((224, 224, 3), dtype=np.uint8) * 128
            
        except Exception as e:
            print(f"提取图像时出错 (episode {episode_idx}, step {step_idx}): {e}")
            return np.ones((224, 224, 3), dtype=np.uint8) * 128

    def _collect_image_for_video(self, img, episode_idx, step_idx):
        """收集图像用于视频生成"""
        if not self.video_save_enabled:
            return
            
        try:
            if episode_idx not in self.video_frames:
                self.video_frames[episode_idx] = []
            
            # 保存原始方向的图像用于视频
            self.video_frames[episode_idx].append(img.copy())
            
        except Exception as e:
            print(f"收集视频帧时出错: {e}")

    def finalize_episode_video(self, episode_idx, task_description="default_task"):
        """完成episode时生成视频"""
        if not self.video_save_enabled or episode_idx not in self.video_frames:
            return None
            
        frames = self.video_frames[episode_idx]
        if len(frames) > 0:
            video_path = self._generate_episode_video(frames, episode_idx, task_description)
            # 清理已使用的帧
            del self.video_frames[episode_idx]
            return video_path
        return None

    def _generate_episode_video(self, episode_images, episode_idx, task_description):
        """从episode图像生成视频"""
        try:
            import os
            import imageio
            from datetime import datetime
            
            # 创建视频保存目录
            debug_dir = os.path.join(os.getcwd(), "ript", "debug_images")
            video_dir = os.path.join(debug_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            
            # 生成视频文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_str = str(task_description).replace(" ", "_").replace("/", "_")[:30]
            video_path = os.path.join(video_dir, f"episode_{episode_idx}_{timestamp}_{task_str}.mp4")
            
            # 确保所有图像尺寸一致
            processed_images = []
            for img in episode_images:
                if isinstance(img, np.ndarray) and img.size > 0:
                    if img.ndim == 3 and img.shape[-1] == 3:
                        # 调整到统一尺寸
                        if img.shape[:2] != (224, 224):
                            try:
                                from skimage.transform import resize
                                img = resize(img, (224, 224), preserve_range=True).astype(np.uint8)
                            except ImportError:
                                import cv2
                                img = cv2.resize(img, (224, 224))
                        
                        # 图像已经在pi0_libero_utils.py中处理过180度旋转
                        # 但是为了视频显示正确，我们需要使用原始方向的图像
                        processed_images.append(img)  # 使用原始方向的图像
            
            if len(processed_images) >= 5:  # 至少5帧才生成视频
                # 保存视频
                imageio.mimsave(video_path, processed_images, fps=10)
                print(f"✓ Episode视频已生成: {video_path} ({len(processed_images)} 帧)")
                return video_path
            else:
                print(f"⚠️ Episode {episode_idx} 图像数量不足，跳过视频生成 ({len(processed_images)} 帧)")
                return None
                
        except Exception as e:
            print(f"生成episode视频时出错: {e}")
            return None