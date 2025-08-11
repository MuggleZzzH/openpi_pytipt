#!/usr/bin/env python3
"""
LIBERO-10 Checkpoint Evaluation Script
=====================================

用于评估PI0 checkpoint在LIBERO_GOAL全部10个任务上的性能
借鉴ript-vla_copy的评估框架，计算准确的成功率指标

使用方法:
    cd /zhaohan/ZJH/openpi_pytorch
    python evaluate_libero10.py --config_path pi0/ript/config/libero10_eval.yaml
    
    # 自定义checkpoint路径
    python evaluate_libero10.py --config_path pi0/ript/config/libero10_eval.yaml --checkpoint_path /path/to/your/checkpoint
"""

import os
import sys
import json
import time
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=== LIBERO-10 Checkpoint 评估脚本 ===")
print(f"项目根目录: {project_root}")
print()

# 导入必要模块
try:
    from pi0.modeling_pi0 import PI0Policy
    from pi0.ript.reward_function import BinarySuccessReward
    from pi0.ript.env.pi0_libero_runner import LIBEROEnvRunner
    print("✓ 成功导入所有必要模块")
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    sys.exit(1)

class LIBERO10Evaluator:
    """LIBERO-10任务评估器"""
    
    def __init__(self, config_path: str, checkpoint_path: str = None):
        self.config = self._load_config(config_path)
        self.checkpoint_path = checkpoint_path or self.config['policy_path']
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 评估配置
        self.eval_config = self.config['evaluation']
        self.rollouts_per_task = self.eval_config['rollouts_per_task']
        
        # 任务列表
        self.tasks = self.config['task']['task_names_to_use']
        print(f"📋 将评估 {len(self.tasks)} 个LIBERO_GOAL任务")
        for i, task in enumerate(self.tasks):
            print(f"  {i+1:2d}. {task}")
        
        # 初始化组件
        self.policy = None
        self.env_runner = None
        self.results = defaultdict(list)
        
        # 创建详细输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"eval_session_{timestamp}"
        self.session_dir.mkdir(exist_ok=True)
        
        # 保存评估配置
        with open(self.session_dir / "eval_config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
    def _load_config(self, config_path: str) -> Dict:
        """加载评估配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_policy(self):
        """加载PI0策略"""
        print(f"🔄 正在加载PI0模型: {self.checkpoint_path}")
        
        try:
            # 使用正确的PI0Policy加载方式
            self.policy = PI0Policy.from_pretrained(self.checkpoint_path)
            
            # 设置为评估模式
            self.policy.eval()
            
            # 移动到GPU
            if torch.cuda.is_available():
                self.policy = self.policy.cuda()
                print(f"✓ 模型已加载到GPU: {torch.cuda.get_device_name()}")
            else:
                print("⚠️ 使用CPU运行，可能较慢")
                
            print("✓ PI0策略加载成功")
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            print("💡 提示: 请确保checkpoint路径正确且包含所有必需文件:")
            print(f"   - {self.checkpoint_path}/config.json")
            print(f"   - {self.checkpoint_path}/model.safetensors")
            print(f"   - {self.checkpoint_path}/tokenizer配置文件")
            raise
    
    def setup_env_runner(self):
        """设置环境运行器"""
        print("🔄 正在设置环境运行器...")
        
        try:
            # 创建配置对象以传递CFG参数
            from types import SimpleNamespace
            runner_config = SimpleNamespace()
            runner_config.cfg_guidance_scale = self.config.get('algo', {}).get('cfg_guidance_scale', 3.0)
            
            self.env_runner = LIBEROEnvRunner(
                benchmark_name=self.config['task']['benchmark_name'],
                num_parallel_envs=self.config['task']['num_parallel_envs'],
                max_episode_length=self.config['task']['max_episode_length'],
                config=runner_config,  # 传递CFG配置
                video_dir=str(self.session_dir / "videos")  # 视频保存目录
            )
            print("✓ 环境运行器设置成功")
            
        except Exception as e:
            print(f"❌ 设置环境运行器失败: {e}")
            raise
    
    def evaluate_single_task(self, task_name: str) -> Dict:
        """评估单个任务"""
        print(f"\n🎯 评估任务: {task_name}")
        
        task_results = {
            'task_name': task_name,
            'success_count': 0,
            'total_rollouts': self.rollouts_per_task,
            'episode_rewards': [],
            'episode_lengths': [],
            'success_episodes': [],
            'failure_episodes': []
        }
        
        # 进度条
        pbar = tqdm(range(self.rollouts_per_task), desc=f"Evaluating {task_name.split('_')[-1]}")
        
        for episode_idx in pbar:
            try:
                # 运行单个episode
                episode_result = self.env_runner.run_policy_in_env(
                    policy=self.policy,
                    task_name=task_name,
                    episode_idx=episode_idx,
                    deterministic=True,  # 确定性评估
                    save_video=self.eval_config.get('save_videos', False)
                )
                
                # 记录结果
                reward = episode_result.get('reward', 0.0)
                length = episode_result.get('episode_length', 0)
                success = episode_result.get('success', False)
                
                task_results['episode_rewards'].append(reward)
                task_results['episode_lengths'].append(length)
                
                if success:
                    task_results['success_count'] += 1
                    task_results['success_episodes'].append(episode_idx)
                else:
                    task_results['failure_episodes'].append(episode_idx)
                
                # 更新进度条信息
                success_rate = task_results['success_count'] / (episode_idx + 1)
                pbar.set_postfix({
                    'Success Rate': f"{success_rate:.2%}",
                    'Success': f"{task_results['success_count']}/{episode_idx + 1}"
                })
                
            except Exception as e:
                print(f"❌ Episode {episode_idx} 评估失败: {e}")
                continue
        
        # 计算最终统计
        task_results['success_rate'] = task_results['success_count'] / self.rollouts_per_task
        task_results['avg_reward'] = np.mean(task_results['episode_rewards']) if task_results['episode_rewards'] else 0.0
        task_results['avg_length'] = np.mean(task_results['episode_lengths']) if task_results['episode_lengths'] else 0.0
        
        print(f"✅ {task_name}: {task_results['success_rate']:.2%} ({task_results['success_count']}/{self.rollouts_per_task})")
        
        return task_results
    
    def run_full_evaluation(self):
        """运行完整评估"""
        print("\n🚀 开始LIBERO-10完整评估")
        print("=" * 80)
        
        # 记录开始时间
        start_time = time.time()
        
        # 逐个评估所有任务
        all_task_results = []
        total_success = 0
        total_episodes = 0
        
        for task_idx, task_name in enumerate(self.tasks):
            print(f"\n[{task_idx + 1}/{len(self.tasks)}] 开始评估: {task_name}")
            
            task_result = self.evaluate_single_task(task_name)
            all_task_results.append(task_result)
            
            # 累积统计
            total_success += task_result['success_count']
            total_episodes += task_result['total_rollouts']
        
        # 计算总体统计
        overall_success_rate = total_success / total_episodes if total_episodes > 0 else 0.0
        evaluation_time = time.time() - start_time
        
        # 生成评估报告
        evaluation_report = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint_path': self.checkpoint_path,
            'evaluation_config': self.config,
            'overall_metrics': {
                'success_rate': overall_success_rate,
                'total_success': total_success,
                'total_episodes': total_episodes,
                'num_tasks': len(self.tasks),
                'evaluation_time_seconds': evaluation_time
            },
            'per_task_results': all_task_results
        }
        
        # 保存详细结果
        results_file = self.session_dir / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
        
        # 生成简洁报告
        self._generate_summary_report(evaluation_report)
        
        print("\n" + "=" * 80)
        print("🎉 LIBERO-10评估完成!")
        print(f"📊 总体成功率: {overall_success_rate:.2%} ({total_success}/{total_episodes})")
        print(f"⏱️  评估用时: {evaluation_time/60:.1f} 分钟")
        print(f"📁 详细结果保存至: {results_file}")
        
        return evaluation_report
    
    def _generate_summary_report(self, evaluation_report: Dict):
        """生成简洁的汇总报告"""
        summary_file = self.session_dir / "evaluation_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("LIBERO-10 Checkpoint 评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"评估时间: {evaluation_report['timestamp']}\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"每任务评估次数: {self.rollouts_per_task}\n\n")
            
            # 总体结果
            overall = evaluation_report['overall_metrics']
            f.write("总体结果:\n")
            f.write("-" * 20 + "\n")
            f.write(f"成功率: {overall['success_rate']:.2%}\n")
            f.write(f"成功次数: {overall['total_success']}/{overall['total_episodes']}\n")
            f.write(f"评估用时: {overall['evaluation_time_seconds']/60:.1f} 分钟\n\n")
            
            # 各任务详细结果
            f.write("各任务详细结果:\n")
            f.write("-" * 30 + "\n")
            
            for i, task_result in enumerate(evaluation_report['per_task_results']):
                task_short_name = task_result['task_name'].split('_')[-1]
                f.write(f"{i+1:2d}. {task_short_name:<15}: ")
                f.write(f"{task_result['success_rate']:6.1%} ")
                f.write(f"({task_result['success_count']:2d}/{task_result['total_rollouts']:2d})\n")
        
        print(f"📋 汇总报告保存至: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="LIBERO-10 Checkpoint Evaluation")
    parser.add_argument(
        '--config_path', 
        type=str, 
        default='pi0/ript/config/libero10_eval.yaml',
        help='评估配置文件路径'
    )
    parser.add_argument(
        '--checkpoint_path', 
        type=str, 
        default=None,
        help='PI0 checkpoint路径 (可选，默认使用配置文件中的路径)'
    )
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = LIBERO10Evaluator(args.config_path, args.checkpoint_path)
    
    # 加载模型和设置环境
    evaluator.load_policy()
    evaluator.setup_env_runner()
    
    # 运行完整评估
    results = evaluator.run_full_evaluation()
    
    # 输出最终结果
    overall_success_rate = results['overall_metrics']['success_rate']
    print(f"\n🏆 最终评估结果: {overall_success_rate:.2%}")

if __name__ == "__main__":
    main()