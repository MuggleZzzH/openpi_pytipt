"""
LIBERO utilities for RIPT framework.
This is a simplified version that provides essential functions.
"""
import numpy as np
import imageio
import os
from datetime import datetime

def extract_image_from_obs(obs):
    """从观测中提取图像用于视频（使用正确的显示方向）"""
    try:
        # 处理dict类型观测
        if isinstance(obs, dict):
            # 优先使用agentview_image，但不旋转（用于显示和保存）
            for key in ["agentview_image", "robot0_eye_in_hand_image", "image"]:
                if key in obs and isinstance(obs[key], np.ndarray):
                    img = obs[key].copy()
                    
                    # 处理CHW格式转换为HWC格式
                    if img.ndim == 3 and img.shape[0] == 3:  # CHW格式 (3, H, W)
                        img = img.transpose(1, 2, 0)  # 转换为HWC格式 (H, W, 3)
                    elif img.ndim == 4:  # (batch, height, width, channels)
                        img = img[0]  # 取第一个batch
                    elif img.ndim == 1:  # 展平的图像
                        # 尝试重构为常见的图像尺寸
                        total_pixels = img.shape[0]
                        if total_pixels == 224 * 224 * 3:
                            img = img.reshape(224, 224, 3)
                        elif total_pixels == 256 * 256 * 3:
                            img = img.reshape(256, 256, 3)
                        else:
                            print(f"警告: 无法重构1D图像，尺寸: {img.shape}")
                            continue
                    
                    # 确保图像是uint8格式
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    
                    # 验证最终格式 - 应该是HWC格式
                    if img.ndim == 3 and img.shape[2] in [1, 3, 4]:  # RGB, RGBA, or grayscale
                        return img
                    elif img.ndim == 2:  # grayscale
                        return img
                    else:
                        print(f"警告: 图像格式不正确, 形状: {img.shape}, 维度: {img.ndim}")
                        
        return None
    except Exception as e:
        print(f"提取图像时出错: {e}")
        return None

def save_rollout_video(images, task_name, episode_idx=0, success=False, fps=10):
    """将图像序列保存为视频"""
    try:
        video_dir = os.path.join("pi0/ript/debug_images/videos")
        os.makedirs(video_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_str = task_name.replace(" ", "_")[:30] if isinstance(task_name, str) else f"episode_{episode_idx}"
        video_path = f"{video_dir}/{timestamp}_{task_str}_{'success' if success else 'failure'}.mp4"
        imageio.mimsave(video_path, images, fps=fps)
        print(f"已保存视频: {video_path}")
        return video_path
    except Exception as e:
        print(f"保存视频出错: {e}")
        return None