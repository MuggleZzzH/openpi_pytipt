"""
Training Performance Comparison Script

This script provides detailed performance comparison between Legacy windowing 
and SO100 sample processing modes for RIPT-VLA training.

Metrics compared:
1. Data utilization ratio (samples per episode)
2. Memory usage
3. Processing time
4. Training throughput
5. Gradient computation efficiency

Usage:
    python compare_training_performance.py --episodes 10 --episode_length 100
"""

import os
import sys
import time
import torch
import psutil
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from unittest.mock import Mock

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pi0.ript.algos.rl_optimizers.pi0_cfg_interface import PI0_CFG_Adapter


def create_mock_policy():
    """Create a mock PI0Policy for performance testing."""
    policy = Mock()
    
    # Mock config
    policy.config = Mock()
    policy.config.n_action_steps = 50
    policy.config.max_action_dim = 7
    policy.config.num_steps = 10
    policy.config.cfg_uncond_weight = 0.1
    
    # Mock forward method
    def mock_forward(batch):
        B = batch.get('batch_size', batch['state'].shape[0])
        T = batch['action'].shape[1]
        D = batch['action'].shape[2]
        
        # Simulate realistic computation time
        time.sleep(0.001 * B)  # 1ms per sample
        
        losses = torch.randn(B, T, D, requires_grad=True)
        return {'losses': losses}
    
    policy.forward = mock_forward
    policy.parameters = lambda: [torch.randn(100, requires_grad=True) for _ in range(10)]
    
    return policy


def create_test_episodes(num_episodes: int, episode_length: int) -> list:
    """Create test episodes with specified parameters."""
    episodes = []
    
    for i in range(num_episodes):
        episode = {
            'observations': [],
            'actions': [],
            'total_reward': np.random.uniform(0, 1)
        }
        
        for step in range(episode_length):
            # Create observation
            obs = {
                'robot0_eef_pos': np.random.randn(3).astype(np.float32),
                'robot0_eef_quat': np.array([0, 0, 0, 1], dtype=np.float32),
                'robot0_gripper_qpos': np.random.randn(2).astype(np.float32),
                'agentview_image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                'robot0_eye_in_hand_image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                'task_description': f'Performance test episode {i}'
            }
            episode['observations'].append(obs)
            
            # Create action
            action = np.random.randn(7).astype(np.float32)
            episode['actions'].append(action)
        
        episodes.append(episode)
    
    return episodes


def measure_memory_usage() -> float:
    """Measure current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def benchmark_processing_mode(
    use_so100_processing: bool,
    episodes: list,
    device: torch.device,
    num_runs: int = 3
) -> Dict[str, Any]:
    """Benchmark a specific processing mode."""
    
    mode_name = "SO100" if use_so100_processing else "Legacy"
    print(f"\n🔬 Benchmarking {mode_name} Processing Mode")
    print("-" * 40)
    
    # Initialize adapter
    policy = create_mock_policy()
    cfg_adapter = PI0_CFG_Adapter(
        policy=policy,
        norm_stats_path="/zhaohan/ZJH/openpi_pytorch/checkpoints/pi0_libero_pytorch/norm_stats.json",
        use_so100_processing=use_so100_processing,
        windowing_mode='last',
        window_stride=10,
        max_windows_per_episode=1
    )
    
    # Prepare data
    advantages = torch.randn(len(episodes))
    
    # Metrics storage
    processing_times = []
    memory_usages = []
    training_samples = []
    
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...")
        
        # Measure initial memory
        initial_memory = measure_memory_usage()
        
        # Measure processing time
        start_time = time.time()
        
        try:
            # Process episodes
            if use_so100_processing:
                batch, episode_to_samples_map = cfg_adapter.process_episodes_to_samples_so100(episodes, device)
                num_samples = batch['batch_size']
            else:
                batch, owner_indices = cfg_adapter.process_episodes(episodes, device)
                num_samples = batch['state'].shape[0]
            
            # Compute loss (simulates training step)
            loss = cfg_adapter.compute_weighted_loss(episodes, advantages, device)
            
            end_time = time.time()
            
            # Measure final memory
            final_memory = measure_memory_usage()
            
            # Record metrics
            processing_times.append(end_time - start_time)
            memory_usages.append(final_memory - initial_memory)
            training_samples.append(num_samples)
            
        except Exception as e:
            print(f"    ❌ Run {run + 1} failed: {e}")
            continue
    
    # Calculate statistics
    if processing_times:
        avg_time = np.mean(processing_times)
        std_time = np.std(processing_times)
        avg_memory = np.mean(memory_usages)
        avg_samples = np.mean(training_samples)
        data_utilization = avg_samples / len(episodes)
        throughput = avg_samples / avg_time  # samples per second
        
        results = {
            'mode': mode_name,
            'episodes': len(episodes),
            'avg_processing_time': avg_time,
            'std_processing_time': std_time,
            'avg_memory_usage': avg_memory,
            'avg_training_samples': avg_samples,
            'data_utilization_ratio': data_utilization,
            'throughput': throughput,
            'success_rate': len(processing_times) / num_runs
        }
        
        print(f"  ✅ {mode_name} Results:")
        print(f"    Processing time: {avg_time:.3f}±{std_time:.3f}s")
        print(f"    Memory usage: {avg_memory:.1f}MB")
        print(f"    Training samples: {avg_samples:.0f}")
        print(f"    Data utilization: {data_utilization:.1f}x")
        print(f"    Throughput: {throughput:.1f} samples/s")
        print(f"    Success rate: {results['success_rate']:.1%}")
        
        return results
    else:
        print(f"  ❌ All {mode_name} runs failed")
        return None


def compare_performance(episodes: list, device: torch.device) -> None:
    """Compare performance between Legacy and SO100 modes."""
    
    print(f"🚀 Training Performance Comparison")
    print(f"Episodes: {len(episodes)}")
    print(f"Average episode length: {np.mean([len(ep['actions']) for ep in episodes]):.1f}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Benchmark Legacy mode
    legacy_results = benchmark_processing_mode(
        use_so100_processing=False,
        episodes=episodes,
        device=device,
        num_runs=3
    )
    
    # Benchmark SO100 mode
    so100_results = benchmark_processing_mode(
        use_so100_processing=True,
        episodes=episodes,
        device=device,
        num_runs=3
    )
    
    # Performance comparison
    if legacy_results and so100_results:
        print(f"\n📊 Performance Comparison Summary")
        print("=" * 60)
        
        # Data utilization improvement
        utilization_improvement = so100_results['data_utilization_ratio'] / legacy_results['data_utilization_ratio']
        print(f"Data Utilization Improvement: {utilization_improvement:.1f}x")
        
        # Throughput comparison
        throughput_ratio = so100_results['throughput'] / legacy_results['throughput']
        print(f"Throughput Ratio (SO100/Legacy): {throughput_ratio:.2f}x")
        
        # Processing time comparison
        time_ratio = so100_results['avg_processing_time'] / legacy_results['avg_processing_time']
        print(f"Processing Time Ratio (SO100/Legacy): {time_ratio:.2f}x")
        
        # Memory usage comparison
        memory_ratio = so100_results['avg_memory_usage'] / legacy_results['avg_memory_usage']
        print(f"Memory Usage Ratio (SO100/Legacy): {memory_ratio:.2f}x")
        
        # Efficiency metrics
        legacy_efficiency = legacy_results['data_utilization_ratio'] / legacy_results['avg_processing_time']
        so100_efficiency = so100_results['data_utilization_ratio'] / so100_results['avg_processing_time']
        efficiency_improvement = so100_efficiency / legacy_efficiency
        
        print(f"Training Efficiency Improvement: {efficiency_improvement:.1f}x")
        
        # Summary
        print(f"\n🎯 Summary:")
        if utilization_improvement > 10:
            print(f"✅ SO100 achieves {utilization_improvement:.1f}x data utilization improvement")
        else:
            print(f"⚠️  SO100 data utilization improvement ({utilization_improvement:.1f}x) is below expected (>10x)")
        
        if efficiency_improvement > 1:
            print(f"✅ SO100 is {efficiency_improvement:.1f}x more efficient overall")
        else:
            print(f"⚠️  SO100 efficiency improvement ({efficiency_improvement:.1f}x) needs optimization")
    
    else:
        print(f"\n❌ Performance comparison failed - missing benchmark results")


def main():
    """Run performance comparison."""
    parser = argparse.ArgumentParser(description='Training Performance Comparison')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes')
    parser.add_argument('--episode_length', type=int, default=80, help='Length of each episode')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device to use')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    episodes = create_test_episodes(args.episodes, args.episode_length)
    
    # Run comparison
    compare_performance(episodes, device)


if __name__ == "__main__":
    main()
