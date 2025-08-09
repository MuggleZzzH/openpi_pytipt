"""
Train PI0 policy using RIPT.
使用简化的直接导入系统

使用方法:
    cd /zhaohan/ZJH/openpi_pytorch
    python pi0/ript/scripts/train_ript_pi0.py --config_path pi0/ript/config/train_pi0_cfg_rl.yaml
"""

#!/usr/bin/env python3

# === 路径初始化 (必须在其他导入之前) ===
import os
import sys
from pathlib import Path

# 设置NCCL超时时间
os.environ["NCCL_TIMEOUT"] = "108000"

# 调试相关配置
DEBUG_SAVE_IMAGES = True  # 默认开启保存图像
DEBUG_SAVE_VIDEO = True   # 默认开启保存视频
DEBUG_IMAGE_DIR = "pi0/ript/debug_images"  # 调试图像保存目录

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # 上3级到openpi_pytorch目录
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=== PI0 + RIPT 训练脚本 ===")
print(f"脚本位置: {current_file}")
print(f"项目根目录: {project_root}")
print()

# 直接导入必要模块
try:
    print("正在导入必要模块...")
    
    # PI0 策略
    from pi0.modeling_pi0 import PI0Policy
    
    # RIPT 组件
    from pi0.ript.reward_function import BinarySuccessReward
    from pi0.ript.algos.rl_optimizers.rl_optimizer_pi0_cfg import RLOptimizerPI0_CFG
    from pi0.ript.algos.rl_optimizers.pi0_cfg_interface import PI0_CFG_Adapter
    from pi0.ript.algos.rl_optimizers.rollout_generator import RolloutGenerator
    
    # 🚀 智能Runner导入 - 支持RIPT-VLA模式
    def import_runner_classes():
        """智能导入runner类"""
        try:
            # 导入原有runner
            from pi0.ript.env.pi0_libero_runner import LIBEROEnvRunner
            print("✅ 原有LIBEROEnvRunner导入成功")
        except Exception as e:
            print(f"⚠️ 原有runner导入失败: {e}")
            LIBEROEnvRunner = None
            
        try:
            # 导入新的RIPT-VLA runner
            from pi0.ript.env.pi0_libero_runner_ript_vla import PI0LiberoRunner as RiptVlaRunner
            print("✅ RIPT-VLA Runner导入成功")
        except Exception as e:
            print(f"⚠️ RIPT-VLA runner导入失败: {e}")
            RiptVlaRunner = None
            
        return LIBEROEnvRunner, RiptVlaRunner
    
    # 执行导入
    OriginalRunner, RiptVlaRunner = import_runner_classes()
    
    # 🚀 Runner选择函数
    def create_env_runner(config, policy, rank=0, world_size=1):
        """根据配置选择合适的环境runner"""
        
        # 检查是否启用RIPT-VLA runner
        use_ript_vla = False
        # 修复：直接使用config.get()而不是hasattr，因为OmegaConf的特殊行为
        features = config.get('features', {})
        if features:
            use_ript_vla = features.get('use_ript_vla_runner', False)
        
        # 调试信息
        print(f"🔍 调试信息:")
        print(f"   hasattr(config, 'features'): {hasattr(config, 'features')}")
        print(f"   config['features']: {config.get('features', 'NOT_FOUND')}")
        print(f"   RiptVlaRunner is not None: {RiptVlaRunner is not None}")
        print(f"🔍 Runner选择: use_ript_vla_runner = {use_ript_vla}")
        
        if use_ript_vla and RiptVlaRunner is not None:
            print("🚀 使用RIPT-VLA风格的环境runner")
            return RiptVlaRunner(
                policy=policy,
                benchmark_name=config['task']['benchmark_name'],
                rollouts_per_env=config['algo']['rloo_batch_size'],
                num_parallel_envs=config['task']['num_parallel_envs'],
                max_episode_length=config['task']['max_episode_length'],
                task_names_to_use=config['task'].get('task_names_to_use', []),
                rank=rank
            )
        elif OriginalRunner is not None:
            print("🔄 使用原有的环境runner")
            # 构建与原有调用兼容的参数
            norm_stats_path = config.get('norm_stats_path', None)
            local_tasks = config['task'].get('task_names_to_use', [])
            
            return OriginalRunner(
                policy=policy,
                benchmark_name=config['task']['benchmark_name'],
                rollouts_per_env=config['algo']['rloo_batch_size'],
                num_parallel_envs=config['task']['num_parallel_envs'],
                max_episode_length=config['task']['max_episode_length'],
                task_names_to_use=local_tasks,
                norm_stats_path=norm_stats_path,
                config=config,
                rank=rank,
                world_size=world_size,
            )
        else:
            raise RuntimeError("❌ 无可用的环境runner！请检查导入。")
    
    # LIBERO benchmark (需要手动实现一个简单的接口)
    try:
        from libero.libero.benchmark import get_benchmark_dict
        benchmark = get_benchmark_dict()
    except ImportError:
        print("警告: LIBERO未安装，将使用模拟benchmark")
        # 创建一个简单的模拟benchmark
        class MockBenchmark:
            def get_task_names(self):
                return ["KITCHEN_SCENE1_put_the_black_bowl_on_the_plate"]
            def get_task_init_states(self, task_id):
                return [{"state": i} for i in range(10)]
        
        benchmark = {"libero_spatial": lambda: MockBenchmark()}
    
    print("✓ 所有模块导入成功")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 标准库导入
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import wandb
import yaml
import time
import json
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
import gc
import psutil
import threading
import signal
from typing import Optional
import io
import atexit
from functools import partial
from tqdm.auto import tqdm
#
# === Logging + Stdout Tee Utilities =========================================================
class _TeeIO(io.TextIOBase):
    """
    Tee stdout/stderr to one or more underlying text streams (console + log file).
    We open the log file in line-buffered mode so updates flush immediately.
    """
    def __init__(self, *streams):
        self._streams = tuple(s for s in streams if s is not None)

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

def setup_run_logging(output_dir: str, rank: int):
    """
    Create per-rank log file and tee sys.stdout/sys.stderr so *all* prints end up in the log.

    Returns: opened file handle (caller must NOT close; we register an atexit hook).
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = Path(output_dir) / f"train_rank{rank}.log"
    log_fh = open(log_path, mode="a", buffering=1, encoding="utf-8")  # line-buffered
    # Wrap existing stdout/stderr
    sys.stdout = _TeeIO(sys.stdout, log_fh)
    sys.stderr = _TeeIO(sys.stderr, log_fh)

    def _close_log():
        try:
            if hasattr(log_fh, 'closed') and not log_fh.closed:
                log_fh.flush()
                log_fh.close()
        except Exception as e:
            print(f"日志文件关闭警告: {e}")
    
    # 禁用自动atexit注册，改为手动管理
    # atexit.register(_close_log)
    
    # 将close函数附加到文件句柄，便于手动调用
    log_fh._manual_close = _close_log
    
    return log_fh

# === Action Progress Bar ====================================================================
class ActionProgressBar:
    """
    Lightweight textual progress indicator for action *chunks* generated by the policy.
    Each callback invocation == 1 chunk update.

    total_steps: intended maximum chunks per rollout episode (use max_episode_length).
    rank: only rank==0 renders a visible tqdm bar; others stay quiet but still count.
    """
    def __init__(self, total_steps: int, rank: int = 0, desc: str = "ActionGen"):
        self.total_steps = max(1, int(total_steps)) if total_steps is not None else None
        self.rank = rank
        self.count = 0
        self._bar = None
        if self.rank == 0:
            try:
                # dynamic_ncols adapts width; leave=False so it resets each episode
                self._bar = tqdm(total=self.total_steps, desc=desc, leave=False, dynamic_ncols=True)
            except Exception:
                self._bar = None

    def reset(self):
        self.count = 0
        if self._bar is not None:
            try:
                self._bar.reset(total=self.total_steps)
            except Exception:
                pass

    def update(self, actions_chunk=None):
        """
        Called each time the wrapped policy emits a new action chunk.
        We don't log the raw action tensor; we just tick progress.
        """
        self.count += 1
        if self._bar is not None:
            try:
                # If total unknown we set total on first call to something large; else clamp
                if self.total_steps is not None:
                    self._bar.update(1)
                    if self.count >= self.total_steps:
                        # auto refresh & newline
                        self._bar.refresh()
                else:
                    # fallback textual print
                    self._bar.write(f"[ActionChunk {self.count}]")
            except Exception:
                pass
        # 仅当 tqdm 进度条不可用时, 才回退到控制台打印, 并使用回车覆写同一行避免刷屏。
        if self._bar is None:
            print(f"[Rank{self.rank}] Action chunk #{self.count}\r", end="", flush=True)

# 添加图像保存相关导入
import os
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime


class LiberoInitStateDataset(Dataset):
    """A PyTorch Dataset to handle initial states from a Libero benchmark."""
    def __init__(self, benchmark_name, task_names_to_use, split='train'):
        self.benchmark_name = benchmark_name.lower()
        # 不在初始化时创建benchmark对象，而是在需要时创建
        # 存储所有需要的信息，而不是存储对象引用
        benchmark_obj = benchmark[self.benchmark_name]()
        
        # 如果没有指定任务名称，则使用所有任务
        if not task_names_to_use:
            task_names_to_use = benchmark_obj.get_task_names()
        
        self.task_names = task_names_to_use
        
        # 预先加载所有初始状态，避免在worker中重新加载
        all_init_states_list = []
        self.task_init_states = {}  # 存储每个任务的初始状态，用于任务分配
        
        for task_name in self.task_names:
            try:
                task_id = benchmark_obj.get_task_names().index(task_name)
                # 获取单个任务的初始状态
                task_init_states = benchmark_obj.get_task_init_states(task_id)
                all_init_states_list.append(task_init_states)
                self.task_init_states[task_name] = task_init_states
            except ValueError:
                # 处理配置中的任务名称可能不在基准中的情况
                print(f"Warning: Task '{task_name}' not found in benchmark '{benchmark_name}'. Skipping.")

        if not all_init_states_list:
             raise ValueError(f"No valid initial states found for tasks {self.task_names} in benchmark {benchmark_name}.")

        # ---- 修改 ----
        # 不再拼接成二维矩阵，改为扁平 list，支持不同任务不同维度
        from itertools import chain
        self.flat_states = list(chain.from_iterable(all_init_states_list))  # List[np.ndarray]

        # 向后兼容: 保留旧属性名，指向相同列表，防止其他模块引用 self.init_states 报错
        self.init_states = self.flat_states
        
        # 建立索引 -> 任务名映射（供任务分配或调试）
        self.init_state_to_task = []
        for task_name, states in self.task_init_states.items():
            self.init_state_to_task.extend([task_name] * len(states))
    
    # 添加序列化支持
    def __getstate__(self):
        # 返回需要序列化的状态
        return {
            'benchmark_name': self.benchmark_name,
            'task_names': self.task_names,
            'flat_states': self.flat_states,
            'task_init_states': self.task_init_states,
            'init_state_to_task': self.init_state_to_task
        }
    
    def __setstate__(self, state):
        # 从序列化状态恢复
        self.benchmark_name = state['benchmark_name']
        self.task_names = state['task_names']
        # 恢复 flat_states 与兼容别名
        self.flat_states = state['flat_states']
        self.init_states = self.flat_states
        self.task_init_states = state['task_init_states']
        self.init_state_to_task = state['init_state_to_task']

    def __len__(self):
        return len(self.flat_states)

    def __getitem__(self, idx):
        return self.flat_states[idx]
    
    def get_task_names(self):
        return self.task_names
    
    def get_task_init_states(self, task_name, num_states=None):
        """获取指定任务的初始状态"""
        if task_name not in self.task_init_states:
            raise ValueError(f"Task '{task_name}' not found in dataset")
        
        states = self.task_init_states[task_name]
        if num_states is not None:
            # 如果请求的状态数量超过可用数量，则循环使用
            indices = np.arange(num_states) % len(states)
            return states[indices]
        return states


def get_memory_usage():
    """获取当前内存使用情况"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
        cpu_memory = psutil.virtual_memory().percent
        return {
            'gpu_memory_allocated_gb': gpu_memory,
            'gpu_memory_cached_gb': gpu_memory_cached,
            'cpu_memory_percent': cpu_memory
        }
    else:
        cpu_memory = psutil.virtual_memory().percent
        return {'cpu_memory_percent': cpu_memory}

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def memory_cleanup_thread(stop_event, cleanup_interval=30):
    """后台内存清理线程"""
    while not stop_event.is_set():
        if stop_event.wait(cleanup_interval):
            break
        clear_gpu_memory()

def get_model_from_ddp(ddp_model):
    """从DDP模型中获取原始模型，兼容单GPU模式"""
    if hasattr(ddp_model, 'module'):
        return ddp_model.module
    else:
        return ddp_model

def setup_distributed():
    """设置分布式训练环境，支持单GPU模式"""
    # 检查是否在分布式环境中
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")
    
    # 如果没有设置分布式环境变量，则使用单GPU模式
    if rank_env is None or world_size_env is None:
        print("未检测到分布式环境变量，使用单GPU模式")
        local_rank = 0
        global_rank = 0
        world_size = 1
        
        # 设置设备
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)
            print(f"使用设备: {device}")
        else:
            device = torch.device("cpu")
            print("警告：未检测到CUDA设备，将使用CPU进行训练")
        
        # 设置随机种子
        base_seed = 42
        torch.manual_seed(base_seed)
        np.random.seed(base_seed)
        
        return local_rank, global_rank, world_size, device
    
    # 分布式模式
    print("检测到分布式环境变量，使用分布式训练模式")
    
    # 设置NCCL超时
    os.environ["NCCL_TIMEOUT"] = "108000"
    
    # 获取本地排名
    local_rank = int(local_rank_env)
    
    # 设置当前设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        print("警告：未检测到CUDA设备，将使用CPU进行训练")
    
    # 初始化进程组，添加超时设置以防止死锁
    init_params = {
        "backend": "nccl" if torch.cuda.is_available() else "gloo",
        "timeout": torch.distributed.default_pg_timeout if hasattr(torch.distributed, 'default_pg_timeout') else None
    }
    
    # 移除None值以避免参数错误
    init_params = {k: v for k, v in init_params.items() if v is not None}
    
    try:
        dist.init_process_group(**init_params)
    except Exception as e:
        print(f"分布式初始化失败: {e}")
        print("尝试使用gloo后端...")
        dist.init_process_group(backend="gloo")
    
    # 获取全局排名和世界大小
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 设置随机种子，确保每个进程使用不同的种子
    base_seed = 42
    torch.manual_seed(base_seed + global_rank)
    np.random.seed(base_seed + global_rank)
    
    return local_rank, global_rank, world_size, device

def save_checkpoint(policy, optimizer, step, config, filename, is_best=False, async_save=True):
    """保存检查点，包括模型、优化器状态和训练配置"""
    def _save_checkpoint_data():
        # 创建检查点数据
        checkpoint = {
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'step': step,
            'config': config,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 使用临时文件确保原子性写入
        temp_filename = filename + ".tmp"
        torch.save(checkpoint, temp_filename)
        os.rename(temp_filename, filename)
        
        # 如果是最佳模型，创建一个副本
        if is_best:
            best_filename = os.path.join(os.path.dirname(filename), "best_model.pth")
            temp_best_filename = best_filename + ".tmp"
            torch.save(checkpoint, temp_best_filename)
            os.rename(temp_best_filename, best_filename)
        
        # 清理内存
        del checkpoint
        clear_gpu_memory()
        
        print(f"检查点已保存到 {filename}")
    
    if async_save:
        # 异步保存以避免阻塞训练
        save_thread = threading.Thread(target=_save_checkpoint_data)
        save_thread.daemon = True
        save_thread.start()
    else:
        _save_checkpoint_data()

def load_checkpoint(policy, optimizer, filename):
    """加载检查点，恢复模型、优化器状态和训练配置"""
    if not os.path.exists(filename):
        print(f"检查点文件 {filename} 未找到。从头开始训练。")
        return 0
    
    # 加载检查点
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    
    # 恢复模型状态
    policy.load_state_dict(checkpoint['model_state_dict'])
    
    # 如果提供了优化器且检查点中有优化器状态，则恢复优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 获取训练步骤
    step = checkpoint.get('step', 0)
    
    print(f"已从 {filename} 加载检查点 (步骤 {step})")
    return step

def get_optimizer(model, config):
    """根据配置创建优化器"""
    optimizer_config = config['training'].get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'Adam')
    # 确保学习率是浮点数
    lr = float(config['algo']['lr'])
    
    if optimizer_type == 'Adam':
        beta1 = float(optimizer_config.get('beta1', 0.9))
        beta2 = float(optimizer_config.get('beta2', 0.999))
        weight_decay = float(optimizer_config.get('weight_decay', 0.0))
        return torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            betas=(beta1, beta2), 
            weight_decay=weight_decay
        )
    elif optimizer_type == 'AdamW':
        beta1 = float(optimizer_config.get('beta1', 0.9))
        beta2 = float(optimizer_config.get('beta2', 0.999))
        weight_decay = float(optimizer_config.get('weight_decay', 0.01))
        return torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            betas=(beta1, beta2), 
            weight_decay=weight_decay
        )
    elif optimizer_type == 'SGD':
        momentum = float(optimizer_config.get('momentum', 0.9))
        weight_decay = float(optimizer_config.get('weight_decay', 0.0))
        return torch.optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

def distribute_tasks(all_tasks, world_size, rank):
    """将任务分配给不同的GPU"""
    rank_to_tasks = {rank_i: [] for rank_i in range(world_size)}
    for task_i, task_name in enumerate(all_tasks):
        rank_to_tasks[task_i % world_size].append(task_name)
    return rank_to_tasks[rank]

# ========= 系统清理函数 =========

def cleanup_opengl_context():
    """简化的OpenGL/EGL上下文清理，避免SIGABRT崩溃"""
    try:
        # 简化的OpenGL清理，只做最基本的操作
        try:
            import OpenGL.GL as gl
            gl.glFinish()  # 完成所有OpenGL操作
        except ImportError:
            pass  # OpenGL未安装，跳过
        except Exception:
            pass  # 忽略OpenGL清理错误
            
        # 简化的EGL清理，移除可能引起问题的步骤
        try:
            import OpenGL.EGL as egl
            
            # 获取当前上下文
            current_context = egl.eglGetCurrentContext()
            current_display = egl.eglGetCurrentDisplay()
            
            # 只在确实有活动上下文时才尝试解绑
            if (current_context != egl.EGL_NO_CONTEXT and 
                current_display != egl.EGL_NO_DISPLAY):
                try:
                    egl.eglMakeCurrent(
                        current_display, 
                        egl.EGL_NO_SURFACE, 
                        egl.EGL_NO_SURFACE, 
                        egl.EGL_NO_CONTEXT
                    )
                except Exception:
                    pass  # 忽略解绑错误，这些在shutdown时是常见的
                        
        except ImportError:
            pass  # EGL不可用
        except Exception:
            pass  # 忽略所有EGL相关错误
            
        print("简化OpenGL清理完成")
        
    except Exception:
        pass  # 忽略所有清理错误

def safe_distributed_cleanup(timeout=10):
    """安全清理分布式训练，带超时保护"""
    if not dist.is_initialized():
        return
        
    def barrier_with_timeout():
        try:
            dist.barrier()
        except Exception as e:
            print(f"分布式barrier出错: {e}")
    
    try:
        # 在单独线程中运行barrier，带超时
        barrier_thread = threading.Thread(target=barrier_with_timeout)
        barrier_thread.daemon = True
        barrier_thread.start()
        barrier_thread.join(timeout=timeout)
        
        if barrier_thread.is_alive():
            print(f"警告: 分布式barrier超时 ({timeout}秒)")
        
        # 销毁进程组
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"分布式清理出错: {e}")

def safe_environment_cleanup(env):
    """安全清理环境，防止子进程泄漏"""
    if env is None:
        return
        
    try:
        # 确保环境关闭
        if hasattr(env, 'close'):
            env.close()
        
        # 等待子进程终止
        time.sleep(0.1)
        
        # 强制垃圾回收
        del env
        gc.collect()
        
    except Exception as e:
        print(f"环境清理出错: {e}")

def consolidated_exit_handler():
    """最简化的退出处理程序，避免任何可能的资源冲突"""
    try:
        print("正在执行最简化清理...")
        
        # 只做最基本的GPU内存清理
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        print("最简化清理完成")
        
    except Exception:
        pass  # 完全忽略所有错误

# 完全移除atexit注册以避免shutdown时的冲突
# 只使用signal handlers和手动清理

# ========= 信号处理程序 =========

def signal_handler(signum, frame):
    """简化的信号处理程序，优雅关闭"""
    print(f"\n收到信号 {signum}，开始优雅关闭...")
    try:
        # 执行简化的清理
        consolidated_exit_handler()
    except Exception:
        pass  # 忽略清理错误
    finally:
        # 退出程序
        os._exit(0)

# 注册信号处理程序
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def main():
    try:
        parser = argparse.ArgumentParser(
            description="Run RIPT online training for PI0 policy using a config file."
        )
        parser.add_argument(
            "--config_path",
            type=str,
            required=True,
            help="Path to the training config file. See `ript/config` for examples.",
        )
        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="Path to checkpoint to resume training from.",
        )
        args = parser.parse_args()

        # We don't yet know the final output_dir from config; create a temporary bootstrap log to capture early prints.
        _bootstrap_log_dir = Path.cwd() / "ript_bootstrap_logs"
        _bootstrap_log_dir.mkdir(parents=True, exist_ok=True)
        _bootstrap_log_fh = setup_run_logging(str(_bootstrap_log_dir), rank=int(os.environ.get("LOCAL_RANK", 0)))
        print(f"[BootstrapLogging] Early logs -> {_bootstrap_log_fh.name}")

        # 使用标准yaml配置加载
        print("正在加载配置文件...")
        import yaml
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ 配置文件加载成功")
        
        # === 强制指定统一的归一化参数路径，避免搜索顺序导致不一致 ===
        config['norm_stats_path'] = "/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json"

        # 设置分布式训练
        local_rank, global_rank, world_size, device = setup_distributed()
        
        # 创建实验ID，用于区分不同的训练运行
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{config['exp_name']}_{timestamp}"
        
        # 更新输出目录以包含实验ID
        config['output_dir'] = os.path.join(config['output_dir'], exp_id)

        # Reconfigure logging to the final experiment directory (overrides bootstrap tee)
        _final_log_fh = setup_run_logging(config['output_dir'], rank=global_rank)
        print(f"[RunLogging] Logging ALL stdout/stderr for rank{global_rank} -> {_final_log_fh.name}")

        if global_rank == 0:
            print("====== 使用配置 ======")
            print(yaml.dump(config))
            print("==========================")
            print(f"世界大小: {world_size} GPUs")
            print(f"使用设备: {device}")
            
            # 确保输出目录存在
            os.makedirs(config['output_dir'], exist_ok=True)
            
            # 初始化wandb
            if config['logging']['use_wandb']:
                wandb.init(
                    project=config['logging']['wandb_project'],
                    entity=config['logging']['wandb_entity'],
                    name=config['exp_name'],
                    config=config,
                    mode=config['logging'].get('wandb_mode', 'online'),  # 使用配置中的wandb_mode，默认为online
                )

        # 设置种子以确保可重复性
        torch.manual_seed(config['training']['seed'] + global_rank)
        np.random.seed(config['training']['seed'] + global_rank)

        # 1. 加载策略模型
        try:
            # 添加路径检查和强制本地加载

            
            # 获取当前工作目录
            cwd = os.getcwd()
            print(f"当前工作目录: {cwd}")
            
            # 解析模型路径
            raw_model_path = config['policy_path']
            print(f"原始模型路径配置: {raw_model_path}")
            
            # 如果是相对路径，尝试不同的解析方式
            if raw_model_path.startswith('./') or raw_model_path.startswith('../'):
                # 尝试相对于当前工作目录
                model_dir = Path(raw_model_path).expanduser().resolve()
                print(f"解析后的绝对路径: {model_dir}")
                
                # 如果不存在，尝试相对于脚本目录
                if not model_dir.exists():
                    script_dir = Path(__file__).parent.resolve()
                    alternative_path = (script_dir / '..' / '..' / '..' / '..' / '..' / raw_model_path[2:]).resolve()
                    print(f"尝试相对于脚本目录的路径: {alternative_path}")
                    
                    if alternative_path.exists():
                        model_dir = alternative_path
                        print(f"使用替代路径: {model_dir}")
            else:
                # 绝对路径或简单相对路径
                model_dir = Path(raw_model_path).expanduser().resolve()
                print(f"解析后的绝对路径: {model_dir}")
            
            # 检查路径是否存在
            if not model_dir.exists():
                # 尝试列出父目录的内容以便调试
                parent_dir = model_dir.parent
                if parent_dir.exists():
                    print(f"父目录 {parent_dir} 内容:")
                    for item in parent_dir.iterdir():
                        print(f"  - {item.name}")
                
                # 尝试使用硬编码的绝对路径作为最后的备选
                fallback_paths = [
                    # 原始配置路径和常见变体
                    Path('/zhaohan/ZJH/lerobot/lerobot/common/policies/pi0/checkpoints/pi0_libero_pytorch'),
                    Path('/zhaohan/ZJH/lerobot/common/policies/pi0/checkpoints/pi0_libero_pytorch'),
                    Path('/zhaohan/ZJH/lerobot/policies/pi0/checkpoints/pi0_libero_pytorch'),
                ]
                
                for fallback_path in fallback_paths:
                    print(f"尝试备选路径: {fallback_path}")
                    if fallback_path.exists():
                        model_dir = fallback_path
                        print(f"使用备选路径: {model_dir}")
                        break
                
                if not model_dir.exists():
                    raise FileNotFoundError(f"预训练模型目录不存在: {model_dir}")
            
            
                
            policy = PI0Policy.from_pretrained(
                str(model_dir),
                local_files_only=True      # 关键：只查本地
            ).to(device)
            
            if global_rank == 0:
                print(f"成功加载策略模型: {model_dir}")
        except Exception as e:
            print(f"加载策略模型失败: {e}")
            raise
        
        # 2. 启用梯度检查点以节省内存（如果配置中启用）
        if config['distributed'].get('gradient_checkpointing', False):
            # 检查策略是否支持梯度检查点
            if hasattr(policy, 'gradient_checkpointing_enable'):
                policy.gradient_checkpointing_enable()
                if global_rank == 0:
                    print("已启用梯度检查点")
            else:
                if global_rank == 0:
                    print("警告: 策略模型不支持梯度检查点，将不使用该功能")
        
        # 3. 根据模式决定是否使用DDP包装模型
        if world_size > 1:
            # 分布式训练模式
            if torch.cuda.is_available():
                ddp_policy = DDP(
                    policy, 
                    device_ids=[local_rank], 
                    output_device=local_rank, 
                    find_unused_parameters=False,
                    # 如果启用了ZeRO优化，则使用静态图
                    static_graph=config['distributed'].get('zero_optimization', False)
                )
            else:
                # CPU版本的DDP
                ddp_policy = DDP(
                    policy,
                    find_unused_parameters=False,
                    static_graph=config['distributed'].get('zero_optimization', False)
                )
            if global_rank == 0:
                print("使用分布式数据并行 (DDP) 模式")
        else:
            # 单GPU模式，不使用DDP
            ddp_policy = policy
            if global_rank == 0:
                print("使用单GPU模式（无DDP）")

        # 4. 获取任务列表
        task_name = config['task'].get('task_name') # 使用.get以确保安全
        if task_name:
            all_tasks = [task_name]
        else:
            # 获取基准中的所有任务
            benchmark_obj = benchmark[config['task']['benchmark_name'].lower()]()
            all_tasks = benchmark_obj.get_task_names()
        
        # 5. 将任务分配给不同的GPU
        local_tasks = distribute_tasks(all_tasks, world_size, global_rank)
        
        if global_rank == 0:
            print(f"任务分配:")
            for r in range(world_size):
                tasks = distribute_tasks(all_tasks, world_size, r)
                print(f"  GPU {r}: {tasks}")
        
        print(f"GPU {global_rank} 的任务: {local_tasks}")

        # 启动后台内存清理线程
        memory_cleanup_stop_event = threading.Event()
        memory_cleanup_thread_obj = threading.Thread(
            target=memory_cleanup_thread, 
            args=(memory_cleanup_stop_event,),
            daemon=True
        )
        memory_cleanup_thread_obj.start()
        
        # 6. 创建LIBERO环境运行器
        try:
            # 首先包装策略以提供更好的错误处理
            from pi0.ript.utils.pi0_libero_utils import Pi0PolicyWrapper
            
            wrapped_policy = Pi0PolicyWrapper(get_model_from_ddp(ddp_policy))

            # --- Action progress instrumentation --------------------------------------------------------
            # Use max_episode_length from config as an approximate total per rollout episode.
            _apb = ActionProgressBar(
                total_steps=config['task']['max_episode_length'],
                rank=global_rank,
                desc="Actions"
            )

            # Attach callback so every generated action chunk ticks the bar.
            if hasattr(wrapped_policy, "set_action_callback"):
                wrapped_policy.set_action_callback(_apb.update)
            else:
                # Fallback: monkeypatch attribute used by wrapper (we'll patch wrapper to call this)
                wrapped_policy.action_callback = _apb.update

            # 定义norm_stats_path供后续使用
            norm_stats_path = config.get('norm_stats_path', None)
            
            # 🚀 使用新的智能Runner选择逻辑
            libero_runner = create_env_runner(
                config=config,
                policy=wrapped_policy,  # 使用包装后的策略
                rank=global_rank,
                world_size=world_size
            )
            
            if global_rank == 0:
                print("成功创建LIBERO环境运行器")
        except Exception as e:
            print(f"创建LIBERO环境运行器失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 7. 创建初始状态数据集
        init_states_dataset = LiberoInitStateDataset(
            benchmark_name=config['task']['benchmark_name'],
            task_names_to_use=local_tasks,
            split='train',
        )
        
        # 根据模式选择采样器
        if world_size > 1:
            # 使用DistributedSampler确保每个GPU获取不同的数据
            init_states_sampler = DistributedSampler(
                init_states_dataset, 
                num_replicas=world_size,
                rank=global_rank,
                shuffle=True
            )
            shuffle_data = False  # 由DistributedSampler处理
        else:
            # 单GPU模式，不使用DistributedSampler
            init_states_sampler = None
            shuffle_data = True
        
        init_states_dataloader = DataLoader(
            init_states_dataset,
            batch_size=config['task']['num_parallel_envs'], # 使用配置中的并行环境数量
            shuffle=shuffle_data,
            sampler=init_states_sampler,
            num_workers=0,  # 设置为0以避免多进程序列化问题
            pin_memory=True,
            prefetch_factor=None,  # 当num_workers=0时不需要prefetch_factor
        )

        # 8. 创建Rollout生成器
        num_parallel_envs = config['task']['num_parallel_envs']  # 从配置中获取并保存为变量
        
        # 根据设备类型设置agent_gpus参数
        agent_gpus = [local_rank] if torch.cuda.is_available() else None
        
        rollout_generator = RolloutGenerator(
            env_runner=libero_runner,
            init_state_dataloader=init_states_dataloader,
            init_state_dataset=init_states_dataset,
            rollouts_per_env=config['algo']['rloo_batch_size'],
            num_envs=num_parallel_envs,  # 使用保存的变量
            max_steps=1_000_000, # 一个较大的数字，实际步数由runner控制
            agent_gpus=agent_gpus, # 根据设备类型设置
            enable_dynamic_sampling=config['algo'].get('enable_dynamic_sampling', False),
            enable_rollout_stats_tracking=config['algo'].get('enable_rollout_stats_tracking', False),
            rollout_skip_threshold=config['algo'].get('rollout_skip_threshold', 3),
            rollout_stats_path=config['algo'].get('rollout_stats_path', None),
            use_val_init=config['algo'].get('use_val_init', False),
        )

        # 9. 创建优化器
        optimizer = get_optimizer(ddp_policy, config)

        # 10. 创建RL优化器
        rl_optimizer = RLOptimizerPI0_CFG(
            rollout_generator=rollout_generator,
            reward_function=BinarySuccessReward(),
            num_epochs=config['algo']['num_epochs'],
            batch_size=config['algo']['data_batch_size'],
            gradient_accumulation_steps=config['algo']['gradient_accumulation_steps'],
            grad_norm_clip=config['algo'].get('grad_norm_clip', None),
        )

        # 11. 创建模型接口（传递归一化参数）
        model_interface = PI0_CFG_Adapter(
            policy=get_model_from_ddp(ddp_policy),
            norm_stats_path=norm_stats_path
        )
        
        # 为模型接口附加优化器，这样RL优化器可以访问它
        get_model_from_ddp(ddp_policy).optimizer = optimizer

        # 12. 创建混合精度训练的scaler（如果启用）
        use_mixed_precision = config['training'].get('use_mixed_precision', False)
        use_fp16 = config['distributed'].get('fp16', False)
        use_bf16 = config['distributed'].get('bf16', False)
        
        # 在CPU模式下禁用混合精度
        if not torch.cuda.is_available():
            use_mixed_precision = False
            use_fp16 = False
            use_bf16 = False
            if global_rank == 0:
                print("在CPU模式下禁用混合精度训练")
        
        scaler = None
        if use_mixed_precision or use_fp16:
            scaler = GradScaler()
            if global_rank == 0:
                print("启用混合精度训练 (FP16)")
        elif use_bf16 and torch.cuda.is_bf16_supported():
            if global_rank == 0:
                print("启用BF16精度训练")
        
        # 13. 恢复训练（如果提供了检查点）
        start_step = 0
        if args.resume:
            checkpoint_path = args.resume
            start_step = load_checkpoint(get_model_from_ddp(ddp_policy), optimizer, checkpoint_path)
            
            # 同步所有进程以确保它们从相同的步骤开始
            if dist.is_initialized():
                start_step_tensor = torch.tensor([start_step], device=device)
                dist.broadcast(start_step_tensor, src=0)
                start_step = int(start_step_tensor.item())

        # 14. 开始训练循环
        best_reward = float('-inf')
        consecutive_oom_errors = 0
        max_oom_retries = 3
        
        for step in range(start_step, config['training']['num_train_steps']):
            if global_rank == 0:
                print(f"========== 训练步骤 {step+1}/{config['training']['num_train_steps']} ==========")
                step_start_time = time.time()

            # Reset per-train-step action progress display
            if '_apb' in locals():
                _apb.reset()
            else:
                try:
                    _apb.reset()
                except Exception:
                    pass

            # 在每个epoch开始时设置数据加载器的epoch
            if hasattr(init_states_dataloader, 'sampler') and hasattr(init_states_dataloader.sampler, 'set_epoch'):
                init_states_dataloader.sampler.set_epoch(step)

            # 运行优化步骤
            try:
                # 在训练前清理内存
                if step % 5 == 0:  # 每5步清理一次
                    clear_gpu_memory()
                
                # 记录训练前的内存使用
                if global_rank == 0 and step % 10 == 0:
                    memory_stats = get_memory_usage()
                    print(f"训练前内存使用: {memory_stats}")
                
                # 添加训练前的策略状态检查
                if global_rank == 0 and step % 20 == 0:
                    print(f"检查策略状态: 策略类型={type(get_model_from_ddp(ddp_policy))}")
                    if hasattr(get_model_from_ddp(ddp_policy), '_action_queue'):
                        print(f"动作队列长度: {len(get_model_from_ddp(ddp_policy)._action_queue)}")
                    if hasattr(wrapped_policy, 'action_failure_count'):
                        print(f"动作失败次数: {wrapped_policy.action_failure_count}")
                
                # 使用统一的autocast设置
                autocast_enabled = use_mixed_precision or use_fp16 or (use_bf16 and torch.cuda.is_bf16_supported())
                autocast_dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_bf16_supported()) else torch.float16
                
                if autocast_enabled:
                    with autocast(dtype=autocast_dtype if torch.cuda.is_available() else torch.float32):
                        metrics = rl_optimizer.train_on_rollouts(
                            model_interface, 
                            float(config['algo']['lr']),
                            scaler=scaler,
                            use_amp=use_mixed_precision or use_fp16
                        )
                else:
                    metrics = rl_optimizer.train_on_rollouts(
                        model_interface, 
                        float(config['algo']['lr']),
                        scaler=None,
                        use_amp=False
                    )
                
                # 验证训练结果
                if metrics is None:
                    print("警告: 训练返回None metrics")
                    metrics = {'mean_reward': 0.0, 'success_rate': 0.0}
                
                # 重置OOM错误计数器
                consecutive_oom_errors = 0
                
            except torch.cuda.OutOfMemoryError as e:
                consecutive_oom_errors += 1
                print(f"CUDA内存不足错误 (第{consecutive_oom_errors}次): {e}")
                
                if consecutive_oom_errors <= max_oom_retries:
                    print("尝试清理内存并重试...")
                    clear_gpu_memory()
                    
                    # 重置策略状态
                    if hasattr(wrapped_policy, 'reset'):
                        wrapped_policy.reset()
                        # Reset action progress bar after OOM restart
                        try:
                            _apb.reset()
                        except Exception:
                            pass
                    
                    # 减少批次大小
                    if hasattr(rl_optimizer, 'batch_size'):
                        rl_optimizer.batch_size = max(1, rl_optimizer.batch_size // 2)
                        print(f"将批次大小减少到: {rl_optimizer.batch_size}")
                    
                    # 跳过当前步骤
                    metrics = {'mean_reward': 0.0, 'success_rate': 0.0, 'error': 'OOM'}
                    continue
                else:
                    print(f"连续{max_oom_retries}次内存不足错误，终止训练")
                    raise
                    
            except Exception as e:
                print(f"训练步骤出错: {e}")
                print(f"错误类型: {type(e).__name__}")
                if global_rank == 0:
                    import traceback
                    traceback.print_exc()
                    
                    # 提供更详细的错误信息
                    print(f"当前步骤: {step}")
                    print(f"配置信息: batch_size={config['algo']['data_batch_size']}, lr={config['algo']['lr']}")
                    
                    # 检查策略状态
                    if hasattr(wrapped_policy, 'action_failure_count'):
                        print(f"策略失败次数: {wrapped_policy.action_failure_count}")
                
                # 重置策略状态
                if hasattr(wrapped_policy, 'reset'):
                    try:
                        wrapped_policy.reset()
                        try:
                            _apb.reset()
                        except Exception:
                            pass
                        print("已重置策略状态")
                    except Exception as reset_e:
                        print(f"重置策略状态时出错: {reset_e}")
                
                metrics = {'mean_reward': 0.0, 'success_rate': 0.0, 'error': str(e)}
                
                # 清理内存以防止错误累积
                clear_gpu_memory()

            # 记录指标
            if global_rank == 0 and metrics is not None:
                step_duration = time.time() - step_start_time
                memory_stats = get_memory_usage()
                
                # 合并所有指标
                all_metrics = {
                    **metrics,
                    'step_time_seconds': step_duration,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    **memory_stats
                }
                
                if config['logging']['use_wandb']:
                    # 记录梯度和参数（如果启用且不会导致内存问题）
                    if config['logging'].get('log_gradients', False) and step % 10 == 0:  # 减少频率
                        try:
                            for name, param in ddp_policy.named_parameters():
                                if param.requires_grad and param.grad is not None:
                                    # 只记录统计信息而不是完整的直方图
                                    grad_norm = param.grad.norm().item()
                                    param_norm = param.norm().item()
                                    wandb.log({
                                        f"grad_norms/{name}": grad_norm,
                                        f"param_norms/{name}": param_norm
                                    }, step=step)
                        except Exception as e:
                            print(f"记录梯度信息时出错: {e}")
                    
                    wandb.log(all_metrics, step=step)
                
                print(f"步骤 {step+1} 指标: {all_metrics}")
                
                # 检查是否为最佳模型
                current_reward = metrics.get('mean_reward', float('-inf'))
                is_best = current_reward > best_reward
                if is_best:
                    best_reward = current_reward
                    print(f"新的最佳奖励: {best_reward:.4f}")

            # 保存检查点（内存优化版本）
            if global_rank == 0:
                # 定期保存
                if (step + 1) % config['training'].get('save_freq', 10) == 0:
                    checkpoint_path = os.path.join(config['output_dir'], f"checkpoint_step_{step+1}.pth")
                    save_checkpoint(
                        get_model_from_ddp(ddp_policy), 
                        optimizer, 
                        step+1, 
                        config, 
                        checkpoint_path,
                        is_best=is_best,
                        async_save=True  # 异步保存
                    )
                
                # 每隔较长时间保存一次最新检查点以避免频繁IO
                if (step + 1) % max(1, config['training'].get('save_freq', 10) // 2) == 0:
                    latest_path = os.path.join(config['output_dir'], "latest.pth")
                    save_checkpoint(
                        get_model_from_ddp(ddp_policy), 
                        optimizer, 
                        step+1, 
                        config, 
                        latest_path,
                        async_save=False  # 同步保存最新检查点
                    )
            
            # 使用更安全的同步方式，避免死锁
            try:
                if dist.is_initialized():
                    # 设置超时以避免无限等待
                    timeout = torch.distributed.default_pg_timeout if hasattr(torch.distributed, 'default_pg_timeout') else None
                    if timeout:
                        dist.barrier()
                    else:
                        # 对于没有超时支持的版本，使用all_reduce作为替代
                        dummy_tensor = torch.zeros(1, device=device)
                        dist.all_reduce(dummy_tensor)
            except Exception as e:
                print(f"同步过程中出错，继续训练: {e}")

        # 停止后台内存清理线程
        memory_cleanup_stop_event.set()
        if memory_cleanup_thread_obj.is_alive():
            memory_cleanup_thread_obj.join(timeout=5)  # 5秒超时
        
        # 保存最终的rollout统计信息
        try:
            rollout_generator.save_stats()
        except Exception as e:
            print(f"保存rollout统计信息时出错: {e}")
        
        # 清理rollout生成器（使用增强的cleanup方法）
        try:
            rollout_generator.cleanup()
        except Exception as e:
            print(f"清理rollout生成器时出错: {e}")
            # 手动清理环境资源
            try:
                safe_environment_cleanup(getattr(rollout_generator, 'env_runner', None))
            except Exception as fallback_error:
                print(f"环境清理回退失败: {fallback_error}")

        # Ensure progress bar is closed cleanly (rank0 only)
        try:
            if global_rank == 0 and hasattr(_apb, "_bar") and _apb._bar is not None:
                _apb._bar.close()
        except Exception:
            pass

        if global_rank == 0 and config['logging']['use_wandb']:
            wandb.finish()
        
        # 清理GPU内存
        clear_gpu_memory()
        
        # 使用安全的分布式清理（带超时保护）
        safe_distributed_cleanup()
        
        # 直接退出，避免任何cleanup引起的问题
        print("训练完成，直接退出...")
        os._exit(0)  # 使用_exit避免任何Python清理
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理资源
        try:
            # 停止后台线程
            if 'memory_cleanup_stop_event' in locals():
                memory_cleanup_stop_event.set()
            
            # 清理GPU内存
            clear_gpu_memory()
            
            # 清理rollout生成器（使用增强的cleanup方法）
            if 'rollout_generator' in locals():
                try:
                    rollout_generator.cleanup()
                except Exception as cleanup_error:
                    print(f"清理rollout生成器时出错: {cleanup_error}")
                    # 手动清理环境资源
                    safe_environment_cleanup(getattr(rollout_generator, 'env_runner', None))
        except Exception as cleanup_error:
            print(f"清理资源时出错: {cleanup_error}")
        
        # 使用安全的分布式清理（带超时保护）
        safe_distributed_cleanup()
        
        # 异常情况下也直接退出
        print("异常退出...")
        os._exit(1)  # 异常退出码


if __name__ == "__main__":
    main()

def save_observation_images(obs, step, save_dir):
    """收集图像帧用于生成视频，不再保存单独图像"""
    collected_frames = []
    
    try:
        # 确保是列表类型
        if not isinstance(obs, list):
            obs_list = [obs]
        else:
            obs_list = obs
            
        for env_idx, env_obs in enumerate(obs_list):
            # 检查obs类型
            if env_obs is None:
                continue
                
            # 处理不同类型的观测
            if isinstance(env_obs, dict):
                # 优先使用agentview_image
                if "agentview_image" in env_obs:
                    img_data = env_obs["agentview_image"].copy()  # 创建副本以避免修改原始数据
                    # 旋转图像使其方向正确（若图像已经倒置）
                    img_data = img_data[::-1, ::-1]  # 180度旋转
                    collected_frames.append(img_data)
                    break  # 只收集一个视图的图像
                elif "robot0_eye_in_hand_image" in env_obs:
                    img_data = env_obs["robot0_eye_in_hand_image"].copy()
                    img_data = img_data[::-1, ::-1]  # 180度旋转
                    collected_frames.append(img_data)
                    break
                elif "images" in env_obs and isinstance(env_obs["images"], list) and len(env_obs["images"]) > 0:
                    img_data = env_obs["images"][0].copy()
                    img_data = img_data[::-1, ::-1]  # 180度旋转
                    collected_frames.append(img_data)
                    break
            
            elif isinstance(env_obs, np.ndarray) and env_obs.ndim >= 2:
                img_data = env_obs.copy()
                img_data = img_data[::-1, ::-1]  # 180度旋转
                collected_frames.append(img_data)
        
        return collected_frames
    except Exception as e:
        print(f"收集图像帧时出错: {str(e)}")
        return []


def save_rollout_video(images, task_name, success=False, fps=10):
    """将一系列图像直接保存为视频，不保存中间图像文件"""
    if not images:
        return None
        
    try:
        # 创建视频保存目录
        video_dir = os.path.join(DEBUG_IMAGE_DIR, "videos")
        os.makedirs(video_dir, exist_ok=True)
        
        # 生成视频文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_str = task_name.replace(" ", "_").replace("/", "_")[:30]
        video_path = os.path.join(video_dir, f"{timestamp}_{task_str}_{'成功' if success else '失败'}.mp4")
        
        # 保存视频
        imageio.mimsave(video_path, images, fps=fps)
        print(f"已保存轨迹视频: {video_path}")
        return video_path
    except Exception as e:
        print(f"保存视频时出错: {str(e)}")
        return None