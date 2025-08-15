"""
OpenPI标准格式的RIPT Rollout收集器
将RIPT环境收集的轨迹转换为OpenPI兼容格式，支持action chunking
"""

import os
import time
import torch
import numpy as np
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

# 导入现有的RIPT组件
try:
    from ..env.pi0_libero_runner import LIBEROEnvRunner
    from ..utils.rollout_stats_tracker import RolloutStatsTracker
    from ..reward_function import BinarySuccessReward
    RIPT_COMPONENTS_AVAILABLE = True
except ImportError:
    print("⚠️ RIPT组件导入失败，将使用模拟实现")
    RIPT_COMPONENTS_AVAILABLE = False
    LIBEROEnvRunner = None
    RolloutStatsTracker = None
    BinarySuccessReward = None


@dataclass
class OpenPIRolloutConfig:
    """OpenPI Rollout收集配置"""
    # 基础参数
    num_rollouts_per_collect: int = 8  # 每次收集的rollout数量
    action_chunk_size: int = 50        # 动作chunk大小
    enable_dynamic_sampling: bool = True  # 启用动态采样过滤
    enable_state_skipping: bool = True    # 启用状态跳过
    
    # 数据格式参数
    image_size: Tuple[int, int] = (224, 224)  # 图像尺寸
    target_state_dim: int = 14  # 目标状态维度
    action_dim: int = 7         # 动作维度
    
    # RIPT特定参数
    rollout_skip_threshold: int = 3  # 跳过阈值
    stats_save_path: str = "./ript_rollout_stats.json"  # 统计保存路径
    
    # 采样参数
    task_id: int = 0              # 当前任务ID
    use_random_init_states: bool = True  # 是否使用随机初始状态


class OpenPIRolloutCollectorOpenPIStandard:
    """
    OpenPI标准格式的RIPT Rollout收集器
    
    功能：
    1. 与现有RIPT环境runner集成
    2. 收集rollouts并转换为OpenPI标准格式
    3. 支持action chunking
    4. 保留RIPT的状态跳过和动态采样逻辑
    5. 提供丰富的统计信息
    """
    
    def __init__(
        self, 
        config: OpenPIRolloutConfig,
        env_runner: Optional[Any] = None,
        stats_tracker: Optional[Any] = None
    ):
        self.config = config
        self.env_runner = env_runner
        
        # 初始化统计跟踪器
        if stats_tracker is not None:
            self.stats_tracker = stats_tracker
        elif RIPT_COMPONENTS_AVAILABLE and RolloutStatsTracker:
            self.stats_tracker = RolloutStatsTracker(
                rollout_skip_threshold=config.rollout_skip_threshold,
                stats_path=config.stats_save_path
            )
        else:
            self.stats_tracker = None
            print("⚠️ 统计跟踪器不可用")
        
        # 收集统计
        self.collection_stats = {
            "total_collections": 0,
            "total_rollouts_collected": 0,
            "total_rollouts_filtered": 0,
            "successful_rollouts": 0,
            "skip_count": 0,
            "dynamic_filter_count": 0,
            "collection_times": [],
            "action_chunk_stats": {
                "chunks_generated": 0,
                "average_chunk_length": 0.0,
                "padding_ratio": 0.0
            }
        }
        
        print(f"✅ OpenPIRolloutCollector 初始化完成")
        print(f"   - Action chunk size: {config.action_chunk_size}")
        print(f"   - 图像尺寸: {config.image_size}")
        print(f"   - 状态维度: {config.target_state_dim}")
        print(f"   - 动态采样: {'启用' if config.enable_dynamic_sampling else '禁用'}")
        print(f"   - 状态跳过: {'启用' if config.enable_state_skipping else '禁用'}")
    
    def collect_rollouts_openpi_format(
        self, 
        task_name: str,
        num_rollouts: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        收集rollouts并转换为OpenPI标准格式
        
        Args:
            task_name: 任务名称
            num_rollouts: 收集的rollout数量，None则使用配置默认值
            **kwargs: 其他参数
            
        Returns:
            OpenPI标准格式的样本列表
        """
        start_time = time.time()
        
        if num_rollouts is None:
            num_rollouts = self.config.num_rollouts_per_collect
            
        print(f"🔄 开始收集 {num_rollouts} 个OpenPI格式rollouts...")
        print(f"   任务: {task_name}")
        
        try:
            # 1. 检查是否应该跳过（RIPT状态跳过逻辑）
            if self.config.enable_state_skipping and self._should_skip_collection(task_name, num_rollouts):
                self.collection_stats["skip_count"] += 1
                print(f"🚫 跳过此次收集：状态最近全成功")
                return []
            
            # 2. 收集原始rollouts
            raw_rollouts = self._collect_raw_rollouts(task_name, num_rollouts)
            
            if not raw_rollouts:
                print("❌ 未收集到原始rollouts")
                return []
            
            # 3. 应用动态采样过滤
            if self.config.enable_dynamic_sampling:
                filtered_rollouts = self._apply_dynamic_sampling_filter(raw_rollouts)
                if not filtered_rollouts:
                    self.collection_stats["dynamic_filter_count"] += 1
                    print("⚠️ 所有rollouts被动态采样过滤")
                    return []
            else:
                filtered_rollouts = raw_rollouts
            
            # 4. 转换为OpenPI标准格式
            openpi_samples = self._convert_to_openpi_format(filtered_rollouts)
            
            # 5. 更新统计信息
            collection_time = time.time() - start_time
            self._update_collection_stats(openpi_samples, collection_time)
            
            # 6. 更新RIPT统计跟踪器
            if self.stats_tracker:
                self._update_ript_stats(filtered_rollouts)
            
            print(f"✅ 收集完成: {len(openpi_samples)} 个OpenPI样本 (耗时: {collection_time:.2f}s)")
            return openpi_samples
            
        except Exception as e:
            print(f"❌ Rollout收集失败: {e}")
            traceback.print_exc()
            return []
    
    def _should_skip_collection(self, task_name: str, num_rollouts: int) -> bool:
        """检查是否应该跳过此次收集（RIPT状态跳过逻辑）"""
        if not self.stats_tracker:
            return False
            
        try:
            # 获取任务的初始状态（简化版本）
            task_id = self.config.task_id
            
            # 这里需要实际的初始状态，暂时使用占位符逻辑
            # 在实际集成中，应该从env_runner获取真实的初始状态
            sample_init_state = np.zeros(self.config.target_state_dim)  # 占位符
            init_hash = self.stats_tracker._compute_init_hash(task_id, sample_init_state)
            
            should_skip = self.stats_tracker.should_skip_init(task_id, init_hash, num_rollouts)
            
            if should_skip:
                self.stats_tracker.increment_skip_count(task_id, init_hash)
                
            return should_skip
            
        except Exception as e:
            print(f"⚠️ 状态跳过检查失败: {e}")
            return False
    
    def _collect_raw_rollouts(self, task_name: str, num_rollouts: int) -> List[Dict[str, Any]]:
        """收集原始rollouts（调用现有RIPT逻辑）"""
        if not self.env_runner:
            print("⚠️ 环境runner未设置，使用模拟数据")
            return self._generate_mock_rollouts(num_rollouts)
        
        try:
            # 获取任务的初始状态
            task_id = self.config.task_id
            if hasattr(self.env_runner, 'benchmark'):
                all_init_states = self.env_runner.benchmark.get_task_init_states(task_id)
            else:
                all_init_states = None
            
            # 调用环境runner生成rollouts
            rollout_generator = self.env_runner.run_policy_in_env(
                env_name=task_name,
                all_init_states=all_init_states
            )
            
            # 收集rollouts
            collected_rollouts = []
            rollout_count = 0
            
            for success, total_reward, episode_data in rollout_generator:
                episode = {
                    'success': success,
                    'total_reward': total_reward,
                    **episode_data
                }
                
                # 添加初始状态hash信息
                if 'init_state' in episode_data and self.stats_tracker:
                    init_hash = self.stats_tracker._compute_init_hash(task_id, episode_data['init_state'])
                    episode['init_hash'] = init_hash
                
                collected_rollouts.append(episode)
                rollout_count += 1
                
                if rollout_count >= num_rollouts:
                    break
            
            return collected_rollouts
            
        except Exception as e:
            print(f"❌ 原始rollout收集失败: {e}")
            traceback.print_exc()
            return []
    
    def _generate_mock_rollouts(self, num_rollouts: int) -> List[Dict[str, Any]]:
        """生成模拟rollouts用于测试"""
        mock_rollouts = []
        
        for i in range(num_rollouts):
            # 生成模拟轨迹数据
            episode_length = np.random.randint(10, 30)  # 随机轨迹长度
            
            mock_episode = {
                'success': np.random.random() > 0.3,  # 70%成功率
                'total_reward': np.random.random(),
                'states': np.random.randn(episode_length, self.config.target_state_dim),
                'actions': np.random.randn(episode_length, self.config.action_dim),
                'observations': {
                    'images': {
                        'base_camera': np.random.randint(0, 255, 
                            (episode_length,) + self.config.image_size + (3,), 
                            dtype=np.uint8),
                        'wrist_camera': np.random.randint(0, 255, 
                            (episode_length,) + self.config.image_size + (3,), 
                            dtype=np.uint8)
                    }
                },
                'task_description': f"mock_task_{i}",
                'init_state': np.random.randn(self.config.target_state_dim),
                'init_hash': f"mock_hash_{i}_{hash(str(i)) % 10000}"
            }
            
            mock_rollouts.append(mock_episode)
        
        print(f"🔧 生成了 {num_rollouts} 个模拟rollouts")
        return mock_rollouts
    
    def _apply_dynamic_sampling_filter(self, rollouts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用动态采样过滤（RIPT逻辑）"""
        if not rollouts:
            return rollouts
        
        # 提取成功率
        successes = [ep.get('success', False) for ep in rollouts]
        
        # 检查是否为uniform结果（全成功或全失败）
        if len(set(successes)) == 1:
            print(f"⚠️ 动态采样过滤: uniform结果 ({'全成功' if successes[0] else '全失败'})")
            return []  # 过滤掉uniform结果
        
        print(f"✅ 动态采样通过: 成功率 {np.mean(successes):.2%} ({sum(successes)}/{len(successes)})")
        return rollouts
    
    def _convert_to_openpi_format(self, rollouts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将RIPT rollouts转换为OpenPI标准格式"""
        openpi_samples = []
        
        for rollout_idx, rollout in enumerate(rollouts):
            try:
                # 提取轨迹数据
                states = rollout.get('states', [])
                actions = rollout.get('actions', [])
                observations = rollout.get('observations', {})
                
                if len(states) == 0 or len(actions) == 0:
                    print(f"⚠️ Rollout {rollout_idx}: 空轨迹，跳过")
                    continue
                
                # 转换为张量
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32)
                
                # 生成action chunk（OpenPI风格）
                action_chunk, action_padding = self._generate_action_chunk(actions)
                
                # 处理图像
                images = self._format_images_openpi(observations.get('images', {}), rollout_idx)
                
                # 获取当前状态（使用第一个时刻）
                current_state = states[0] if len(states) > 0 else torch.zeros(self.config.target_state_dim)
                
                # 任务描述
                task_prompt = rollout.get('task_description', f"rollout_{rollout_idx}")
                
                # 构造OpenPI标准样本
                openpi_sample = {
                    "image": images,
                    "state": current_state,
                    "action": action_chunk,
                    "action_is_pad": action_padding,
                    "prompt": task_prompt,
                    # RIPT扩展字段
                    "advantages": torch.tensor([rollout.get('total_reward', 0.0)] * len(action_chunk)),
                    "init_hash": rollout.get('init_hash', f"hash_{rollout_idx}"),
                    # 元数据
                    "rollout_success": rollout.get('success', False),
                    "rollout_reward": rollout.get('total_reward', 0.0),
                    "rollout_length": len(states),
                }
                
                openpi_samples.append(openpi_sample)
                
            except Exception as e:
                print(f"❌ Rollout {rollout_idx} 转换失败: {e}")
                continue
        
        return openpi_samples
    
    def _generate_action_chunk(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成action chunk和对应的padding mask
        
        Args:
            actions: 原始动作序列 (episode_length, action_dim)
            
        Returns:
            action_chunk: 固定长度的动作chunk (chunk_size, action_dim)
            action_padding: padding mask (chunk_size,)
        """
        episode_length, action_dim = actions.shape
        chunk_size = self.config.action_chunk_size
        
        # 确保动作维度匹配
        if action_dim != self.config.action_dim:
            print(f"⚠️ 动作维度不匹配: {action_dim} vs {self.config.action_dim}")
            # 简单的维度对齐
            if action_dim < self.config.action_dim:
                padding = torch.zeros(episode_length, self.config.action_dim - action_dim)
                actions = torch.cat([actions, padding], dim=-1)
            else:
                actions = actions[:, :self.config.action_dim]
        
        # 生成action chunk
        if episode_length >= chunk_size:
            # 轨迹长度足够，取前chunk_size个动作
            action_chunk = actions[:chunk_size]
            action_padding = torch.zeros(chunk_size, dtype=torch.bool)
        else:
            # 轨迹长度不足，需要padding
            action_chunk = torch.zeros(chunk_size, self.config.action_dim)
            action_chunk[:episode_length] = actions
            
            # 创建padding mask
            action_padding = torch.zeros(chunk_size, dtype=torch.bool)
            action_padding[episode_length:] = True  # padding位置为True
        
        # 更新统计信息
        self.collection_stats["action_chunk_stats"]["chunks_generated"] += 1
        valid_actions = (~action_padding).sum().item()
        self.collection_stats["action_chunk_stats"]["average_chunk_length"] = \
            (self.collection_stats["action_chunk_stats"]["average_chunk_length"] * 
             (self.collection_stats["action_chunk_stats"]["chunks_generated"] - 1) + valid_actions) / \
            self.collection_stats["action_chunk_stats"]["chunks_generated"]
        
        return action_chunk, action_padding
    
    def _format_images_openpi(self, images_dict: Dict[str, Any], rollout_idx: int) -> Dict[str, torch.Tensor]:
        """
        格式化图像为OpenPI标准格式
        
        Args:
            images_dict: 原始图像字典
            rollout_idx: rollout索引
            
        Returns:
            OpenPI标准格式的图像字典
        """
        formatted_images = {}
        
        try:
            # 处理基座相机
            if 'base_camera' in images_dict:
                base_img = images_dict['base_camera']
                if isinstance(base_img, np.ndarray):
                    # 取第一帧作为当前观测
                    current_frame = base_img[0] if len(base_img.shape) == 4 else base_img
                    formatted_images["base_0_rgb"] = torch.tensor(current_frame, dtype=torch.uint8)
                else:
                    formatted_images["base_0_rgb"] = torch.tensor(base_img, dtype=torch.uint8)
            
            # 处理手腕相机
            if 'wrist_camera' in images_dict:
                wrist_img = images_dict['wrist_camera']
                if isinstance(wrist_img, np.ndarray):
                    current_frame = wrist_img[0] if len(wrist_img.shape) == 4 else wrist_img
                    formatted_images["left_wrist_0_rgb"] = torch.tensor(current_frame, dtype=torch.uint8)
                else:
                    formatted_images["left_wrist_0_rgb"] = torch.tensor(wrist_img, dtype=torch.uint8)
            
            # 如果没有图像，创建默认图像
            if not formatted_images:
                print(f"⚠️ Rollout {rollout_idx}: 无图像数据，使用默认图像")
                default_img = torch.zeros(3, *self.config.image_size, dtype=torch.uint8)
                formatted_images["base_0_rgb"] = default_img
                formatted_images["left_wrist_0_rgb"] = default_img
            
            # 确保图像维度正确 (C, H, W)
            for key, img in formatted_images.items():
                if img.dim() == 3 and img.shape[0] not in [1, 3]:  # (H, W, C) -> (C, H, W)
                    formatted_images[key] = img.permute(2, 0, 1)
                elif img.dim() == 2:  # (H, W) -> (C, H, W)
                    formatted_images[key] = img.unsqueeze(0).repeat(3, 1, 1)
            
        except Exception as e:
            print(f"❌ 图像格式化失败 (rollout {rollout_idx}): {e}")
            # 创建安全的默认图像
            default_img = torch.zeros(3, *self.config.image_size, dtype=torch.uint8)
            formatted_images = {
                "base_0_rgb": default_img,
                "left_wrist_0_rgb": default_img
            }
        
        return formatted_images
    
    def _update_collection_stats(self, samples: List[Dict[str, Any]], collection_time: float):
        """更新收集统计信息"""
        self.collection_stats["total_collections"] += 1
        self.collection_stats["total_rollouts_collected"] += len(samples)
        self.collection_stats["collection_times"].append(collection_time)
        
        # 统计成功rollouts
        successful = sum(1 for sample in samples if sample.get("rollout_success", False))
        self.collection_stats["successful_rollouts"] += successful
        
        # 保持时间列表大小
        if len(self.collection_stats["collection_times"]) > 100:
            self.collection_stats["collection_times"] = self.collection_stats["collection_times"][-50:]
    
    def _update_ript_stats(self, rollouts: List[Dict[str, Any]]):
        """更新RIPT统计跟踪器"""
        if not self.stats_tracker or not rollouts:
            return
        
        try:
            # 提取成功率信息
            successes = [ep.get('success', False) for ep in rollouts]
            
            # 获取init_hash
            init_hash = None
            task_id = self.config.task_id
            
            for ep in rollouts:
                if 'init_hash' in ep:
                    init_hash = ep['init_hash']
                    break
            
            if init_hash:
                self.stats_tracker.update_stats(task_id, init_hash, successes)
                
        except Exception as e:
            print(f"⚠️ RIPT统计更新失败: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取收集统计信息"""
        stats = self.collection_stats.copy()
        
        # 计算平均值
        if stats["collection_times"]:
            stats["average_collection_time"] = np.mean(stats["collection_times"])
            stats["total_collection_time"] = sum(stats["collection_times"])
        else:
            stats["average_collection_time"] = 0.0
            stats["total_collection_time"] = 0.0
        
        # 计算成功率
        if stats["total_rollouts_collected"] > 0:
            stats["overall_success_rate"] = stats["successful_rollouts"] / stats["total_rollouts_collected"]
        else:
            stats["overall_success_rate"] = 0.0
        
        return stats
    
    def print_stats(self):
        """打印收集统计信息"""
        stats = self.get_collection_stats()
        
        print("\n📊 OpenPI Rollout收集统计:")
        print(f"   总收集次数: {stats['total_collections']}")
        print(f"   总rollout数: {stats['total_rollouts_collected']}")
        print(f"   成功rollout数: {stats['successful_rollouts']}")
        print(f"   整体成功率: {stats['overall_success_rate']:.2%}")
        print(f"   跳过次数: {stats['skip_count']}")
        print(f"   动态过滤次数: {stats['dynamic_filter_count']}")
        print(f"   平均收集时间: {stats['average_collection_time']:.2f}s")
        
        chunk_stats = stats["action_chunk_stats"]
        print(f"\n🎯 Action Chunk统计:")
        print(f"   生成chunk数: {chunk_stats['chunks_generated']}")
        print(f"   平均chunk长度: {chunk_stats['average_chunk_length']:.1f}")
    
    def reset_stats(self):
        """重置统计信息"""
        self.collection_stats = {
            "total_collections": 0,
            "total_rollouts_collected": 0,
            "total_rollouts_filtered": 0,
            "successful_rollouts": 0,
            "skip_count": 0,
            "dynamic_filter_count": 0,
            "collection_times": [],
            "action_chunk_stats": {
                "chunks_generated": 0,
                "average_chunk_length": 0.0,
                "padding_ratio": 0.0
            }
        }
        print("🔄 收集统计信息已重置")


# 便捷函数
def create_openpi_rollout_collector(
    config_dict: Dict[str, Any],
    env_runner: Optional[Any] = None,
    stats_tracker: Optional[Any] = None
) -> OpenPIRolloutCollectorOpenPIStandard:
    """
    工厂函数：创建OpenPI rollout收集器
    
    Args:
        config_dict: 配置字典
        env_runner: 环境runner
        stats_tracker: 统计跟踪器
        
    Returns:
        配置好的收集器实例
    """
    # 从配置字典创建配置对象
    rollout_config = OpenPIRolloutConfig(
        num_rollouts_per_collect=config_dict.get('rloo_batch_size', 8),
        action_chunk_size=config_dict.get('action_chunk_size', 50),
        enable_dynamic_sampling=config_dict.get('enable_dynamic_sampling', True),
        enable_state_skipping=config_dict.get('enable_state_skipping', True),
        image_size=tuple(config_dict.get('image_size', [224, 224])),
        target_state_dim=config_dict.get('target_state_dim', 14),
        action_dim=config_dict.get('action_dim', 7),
        rollout_skip_threshold=config_dict.get('rollout_skip_threshold', 3),
        stats_save_path=config_dict.get('rollout_stats_path', './ript_rollout_stats.json'),
        task_id=config_dict.get('task_id', 0),
    )
    
    collector = OpenPIRolloutCollectorOpenPIStandard(
        config=rollout_config,
        env_runner=env_runner,
        stats_tracker=stats_tracker
    )
    
    return collector
