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

    def __init__(self, policy: PI0Policy, norm_stats_path: Optional[str] = None, **kwargs):
        """
        Initialize the adapter with a PI0Policy instance.
        Args:
            policy: An instance of PI0Policy.
            norm_stats_path: Path to norm_stats.json file for normalization. 
                           If None, will try to find it automatically.
        """
        # The base model is the PI0Policy itself.
        super().__init__(model=policy, **kwargs)
        self.policy = policy
        
        # 视频收集相关
        self.video_frames = {}  # {episode_idx: [frames]}
        self.video_save_enabled = True
        
        # 加载归一化统计信息
        self._load_norm_stats(norm_stats_path)
        
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
    ) -> Dict[str, Any]:
        """
        将 episodes 打包为 PI0Policy 期望的 batch：
        - action 形状保持 (B, T, 7)，不在这里填到 32 维；
        - 提供 action_is_pad: (B, T) 布尔，True 表示该时间步是 padding；
        - 状态统一 8 维 (3 pos + 3 axis-angle + 2 gripper)；
        """
        if device is None:
            device = self.device
        if not episodes:
            raise ValueError("Empty episodes list provided")

        all_states, all_images, all_actions = [], [], []
        all_action_is_pad, all_tasks = [], []

        target_seq_len = getattr(self.policy.config, 'n_action_steps', 50)
        diffusion_steps = getattr(self.policy.config, 'num_steps', 10)
        print(f"Using target_seq_len={target_seq_len} (n_action_steps), diffusion_steps={diffusion_steps}")

        for i, ep in enumerate(episodes):
            try:
                observations = ep['observations']
                actions = ep['actions']
                tasks = ep.get('task', "default task")
                if not observations:
                    raise ValueError(f"Episode {i} empty observations")

                # 收集
                states_seq, images_seq, actions_seq = [], [], []
                max_steps = min(len(observations), len(actions))
                sample_idx = list(range(max_steps))  # 全量保存，便于视频

                for t in sample_idx:
                    obs_t = observations[t] if t < len(observations) else {}
                    act_t = actions[t] if t < len(actions) else np.zeros(7, np.float32)

                    # 状态(8维)
                    states_seq.append(self._extract_state_from_obs(obs_t, i, t))
                    # 图像(HWC)
                    images_seq.append(self._extract_image_from_obs(obs_t, i, t))

                    # 动作(7维)
                    act_t = np.array(act_t[0] if (isinstance(act_t, list) and len(act_t) > 0) else act_t,
                                    dtype=np.float32)
                    if act_t.size != 7:
                        buf = np.zeros(7, dtype=np.float32)
                        buf[:min(7, act_t.size)] = act_t[:min(7, act_t.size)]
                        act_t = buf
                    actions_seq.append(act_t)

                # 固定长度 + 掩码
                valid_len = min(len(actions_seq), target_seq_len)
                final_action = np.zeros((target_seq_len, 7), dtype=np.float32)
                if valid_len > 0:
                    final_action[:valid_len] = np.asarray(actions_seq[-valid_len:], dtype=np.float32)

                action_is_pad = np.ones((target_seq_len,), dtype=bool)
                action_is_pad[:valid_len] = False

                final_state = np.asarray(states_seq[-1], dtype=np.float32) if states_seq else np.zeros(8, np.float32)
                final_image = images_seq[-1] if images_seq else (np.ones((224, 224, 3), np.uint8) * 128)

                all_states.append(final_state)
                all_images.append(final_image)
                all_actions.append(final_action)
                all_action_is_pad.append(action_is_pad)

                task_str = tasks[0] if isinstance(tasks, list) else str(tasks)
                all_tasks.append(task_str)

            except Exception as e:
                print(f"Error processing episode {i}: {e}")
                all_states.append(np.zeros(8, np.float32))
                all_images.append(np.ones((224, 224, 3), np.uint8) * 128)
                all_actions.append(np.zeros((target_seq_len, 7), np.float32))
                all_action_is_pad.append(np.ones((target_seq_len,), dtype=bool))
                all_tasks.append("default task")

        batch = {
            "state": torch.from_numpy(np.stack(all_states)).to(device, dtype=torch.float32),            # (B,8)
            "image": {"base_0_rgb": torch.from_numpy(np.stack(all_images)).to(device, dtype=torch.uint8)},  # (B,H,W,3)
            "action": torch.from_numpy(np.stack(all_actions)).to(device, dtype=torch.float32),          # (B,T,7)
            "action_is_pad": torch.from_numpy(np.stack(all_action_is_pad)).to(device),                  # (B,T) bool
            "prompt": all_tasks,
        }

        bs = len(episodes)
        assert batch["state"].shape[0] == bs
        assert batch["image"]["base_0_rgb"].shape[0] == bs
        assert batch["action"].shape[0] == bs
        assert batch["action_is_pad"].shape[0] == bs
        print(f"Processed batch: state {batch['state'].shape}, image {batch['image']['base_0_rgb'].shape}, "
            f"action {batch['action'].shape}, action_is_pad {batch['action_is_pad'].shape}")
        return batch
    def compute_weighted_loss(
        self,
        episodes: List[Dict[str, Any]],
        advantages: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        CFG风格训练：同时计算条件和无条件损失
        - 条件损失：使用正优势加权，告诉模型这些是好样本
        - 无条件损失：提供基线，防止模式崩溃
        - 组合损失：条件 + α * 无条件 (α通常为0.1)
        """
        if device is None:
            device = self.device

        if not episodes or advantages is None or len(advantages) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        if len(episodes) != len(advantages):
            n = min(len(episodes), len(advantages))
            episodes = episodes[:n]
            advantages = advantages[:n]

        batch = self.process_episodes(episodes, device)
        
        # === CFG风格双分支损失计算 ===
        
        # 1. 条件分支（正样本指示）
        batch_positive = batch.copy()
        batch_positive["is_positive"] = torch.ones(len(episodes), device=device, dtype=torch.int32)
        
        out_positive = self.policy.forward(batch_positive)
        if isinstance(out_positive, tuple):
            loss_scalar_pos, loss_dict_pos = out_positive
        elif isinstance(out_positive, dict):
            loss_scalar_pos, loss_dict_pos = out_positive.get("loss", None), out_positive
        else:
            loss_scalar_pos, loss_dict_pos = out_positive, {}

        # 2. 无条件分支（无指示）
        batch_uncond = batch.copy()
        batch_uncond["is_positive"] = torch.zeros(len(episodes), device=device, dtype=torch.int32)
        
        out_uncond = self.policy.forward(batch_uncond)
        if isinstance(out_uncond, tuple):
            loss_scalar_uncond, loss_dict_uncond = out_uncond
        elif isinstance(out_uncond, dict):
            loss_scalar_uncond, loss_dict_uncond = out_uncond.get("loss", None), out_uncond
        else:
            loss_scalar_uncond, loss_dict_uncond = out_uncond, {}

        # 3. 计算CFG组合损失
        if isinstance(loss_dict_pos, dict) and "losses" in loss_dict_pos:
            # 使用详细损失信息
            per_step_per_dim_pos = loss_dict_pos["losses"]  # (B, T, D)
            per_step_per_dim_uncond = loss_dict_uncond.get("losses", per_step_per_dim_pos)
            
            if per_step_per_dim_pos.dim() == 3:
                per_step_pos = per_step_per_dim_pos.mean(dim=-1)  # (B,T)
                per_step_uncond = per_step_per_dim_uncond.mean(dim=-1)  # (B,T)

                # CFG权重计算：正优势样本用于条件训练
                w_traj = (advantages.to(device).float() > 0).float()  # (B,) 二值权重
                w_step_pos = w_traj.unsqueeze(1).expand_as(per_step_pos)  # (B,T)
                
                # 条件损失：优势加权
                denom_pos = w_step_pos.sum().clamp_min(1.0)
                loss_positive = (per_step_pos * w_step_pos).sum() / denom_pos
                
                # 无条件损失：所有样本均等权重
                loss_unconditional = per_step_uncond.mean()
                
                # CFG组合：条件 + α * 无条件
                cfg_alpha = getattr(self.policy.config, 'cfg_uncond_weight', 0.1)
                final_loss = loss_positive + cfg_alpha * loss_unconditional
                
                if torch.isnan(final_loss) or torch.isinf(final_loss):
                    final_loss = torch.tensor(0.0, device=device, requires_grad=True)
                return final_loss

        # 兜底：使用标量损失
        base_pos = loss_scalar_pos if (loss_scalar_pos is not None and isinstance(loss_scalar_pos, torch.Tensor)) \
            else torch.tensor(0.0, device=device, requires_grad=True)
        base_uncond = loss_scalar_uncond if (loss_scalar_uncond is not None and isinstance(loss_scalar_uncond, torch.Tensor)) \
            else torch.tensor(0.0, device=device, requires_grad=True)
            
        if base_pos.dim() != 0:
            base_pos = base_pos.mean()
        if base_uncond.dim() != 0:
            base_uncond = base_uncond.mean()
            
        # 优势权重
        w = (advantages.to(device).float() > 0).float()
        loss_positive = base_pos * (w.mean().clamp_min(1e-6))
        loss_unconditional = base_uncond
        
        cfg_alpha = getattr(self.policy.config, 'cfg_uncond_weight', 0.1)
        final_loss = loss_positive + cfg_alpha * loss_unconditional
        
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            final_loss = torch.tensor(0.0, device=device, requires_grad=True)
        return final_loss
    
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