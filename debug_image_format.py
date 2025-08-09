#!/usr/bin/env python3
"""
调试LIBERO图像格式问题
检查实际的图像维度和格式
"""
import os
import sys
import numpy as np
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入LIBERO环境
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

print("=== LIBERO图像格式调试 ===")

try:
    # 设置LIBERO环境
    benchmark_dict = benchmark.get_benchmark_dict()
    libero_goal = benchmark_dict['libero_goal']()
    task_id = 1  # put_the_bowl_on_the_stove
    task = libero_goal.get_task(task_id)
    print(f"任务: {task.language}")
    
    # 创建环境
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), 
        task.problem_folder, 
        task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file, 
        "camera_heights": 224, 
        "camera_widths": 224
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    print("✓ 环境创建成功")
    
    # 重置环境
    obs = env.reset()
    print("✓ 环境重置完成")
    
    # 检查图像格式
    if "agentview_image" in obs:
        img = obs["agentview_image"]
        print(f"\n📸 agentview_image分析:")
        print(f"   原始形状: {img.shape}")
        print(f"   数据类型: {img.dtype}")
        print(f"   值范围: [{img.min()}, {img.max()}]")
        
        # 测试不同的处理方式
        print(f"\n🔧 不同处理方式的结果:")
        
        # 1. 仅BGR转RGB
        bgr_to_rgb = img[:, :, ::-1]
        print(f"   BGR->RGB: {bgr_to_rgb.shape}")
        
        # 2. 参考脚本的处理
        ref_process = img[:, :, ::-1].transpose(1, 2, 0)
        print(f"   参考脚本处理: {ref_process.shape}")
        
        # 3. 检查是否LIBERO输出的是CHW格式
        if len(img.shape) == 3:
            print(f"\n🤔 可能的格式分析:")
            if img.shape[0] == 3 or img.shape[0] == 1:
                print(f"   可能是CHW格式: (C={img.shape[0]}, H={img.shape[1]}, W={img.shape[2]})")
            elif img.shape[2] == 3 or img.shape[2] == 1:
                print(f"   可能是HWC格式: (H={img.shape[0]}, W={img.shape[1]}, C={img.shape[2]})")
        
        # 4. 尝试不同的transpose
        print(f"\n🔄 不同transpose的结果:")
        try:
            t012 = img.transpose(0, 1, 2)  # 不变
            print(f"   transpose(0,1,2): {t012.shape}")
        except:
            pass
            
        try:
            t120 = img.transpose(1, 2, 0)  # CHW -> HWC
            print(f"   transpose(1,2,0): {t120.shape}")
        except:
            pass
            
        try:
            t201 = img.transpose(2, 0, 1)  # HWC -> CHW
            print(f"   transpose(2,0,1): {t201.shape}")
        except:
            pass
    
    # 检查其他图像
    if "robot0_eye_in_hand_image" in obs:
        img2 = obs["robot0_eye_in_hand_image"]
        print(f"\n🤖 robot0_eye_in_hand_image:")
        print(f"   形状: {img2.shape}, 类型: {img2.dtype}")
    
    env.close()
    print("\n✅ 调试完成")
    
except Exception as e:
    print(f"❌ 调试失败: {e}")
    import traceback
    traceback.print_exc()