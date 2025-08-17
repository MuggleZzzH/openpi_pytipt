"""
Test Script for SO100-Style Data Processing

This script validates the new data processing pipeline:
1. Creates mock trajectory data
2. Tests trajectory-to-sample conversion
3. Validates data utilization improvements
4. Checks episode-to-sample mapping for advantage computation

Run this script to verify the data processing works correctly before integration.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from pi0.ript.data.so100_style_processor import SO100StyleProcessor
from pi0.ript.data.sample_generator import TrajectoryToSampleGenerator

def create_mock_config() -> Dict[str, Any]:
    """Create a mock configuration for testing."""
    return {
        'action_chunk_size': 50,
        'norm_stats_path': "/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json",
        'state_mean': np.zeros(8, dtype=np.float32),
        'state_std': np.ones(8, dtype=np.float32),
        'action_mean': np.zeros(7, dtype=np.float32),
        'action_std': np.ones(7, dtype=np.float32)
    }


def create_mock_observation(step_idx: int) -> Dict[str, Any]:
    """Create a mock observation for testing."""
    return {
        'image': {
            'base_0_rgb': torch.randn(224, 224, 3),
            'left_wrist_0_rgb': torch.randn(224, 224, 3)
        },
        'state': np.random.randn(8).astype(np.float32),
        'prompt': [f'Test task step {step_idx}']
    }


def create_mock_trajectory(trajectory_id: str, length: int) -> Dict[str, Any]:
    """Create a mock trajectory for testing."""
    trajectory = {
        'id': trajectory_id,
        'processed_observations': [],
        'actions': [],
        'states': [],
        'total_reward': np.random.uniform(0, 1)
    }
    
    for step in range(length):
        # Create observation
        obs = create_mock_observation(step)
        trajectory['processed_observations'].append(obs)
        
        # Create state (8-dimensional)
        state = np.random.randn(8).astype(np.float32)
        trajectory['states'].append(state)
        
        # Create action (7-dimensional)
        action = np.random.randn(7).astype(np.float32)
        trajectory['actions'].append(action)
    
    return trajectory


def test_so100_style_processor():
    """Test the SO100StyleProcessor class."""
    print("=" * 60)
    print("Testing SO100StyleProcessor")
    print("=" * 60)
    
    # Create processor
    config = create_mock_config()
    processor = SO100StyleProcessor(config)
    
    # Test with different trajectory lengths
    test_lengths = [30, 50, 75, 100, 150]  # Some shorter than chunk_size, some longer
    
    for length in test_lengths:
        print(f"\n--- Testing trajectory length: {length} ---")
        
        # Create mock trajectory
        trajectory = create_mock_trajectory(f"test_traj_{length}", length)
        
        # Process trajectory
        samples = processor.process_trajectory_to_samples(trajectory)
        
        # Validate results
        expected_samples = max(0, length - 50 + 1)
        print(f"Expected samples: {expected_samples}, Generated: {len(samples)}")
        
        if len(samples) > 0:
            # Validate first sample
            sample = samples[0]
            assert processor.validate_sample_format(sample), "Sample format validation failed"
            
            # Check relative action computation
            original_action = trajectory['actions'][0]  # shape: (7,)
            current_state = trajectory['states'][0]    # shape: (8,)
            
            # 使用和SO100处理器相同的逻辑：只取前7维状态
            if current_state.shape[0] > original_action.shape[0]:
                state_for_action = current_state[:original_action.shape[0]]  # 取前7维
                expected_relative = original_action - state_for_action
            else:
                expected_relative = original_action - current_state
            
            actual_relative = sample['action'][0]  # First action in chunk
            
            np.testing.assert_allclose(expected_relative, actual_relative, rtol=1e-5)
            print("✓ Relative action computation verified")
            
            # Convert to OpenPI format
            openpi_sample = processor.convert_to_openpi_format(sample)
            print("✓ OpenPI format conversion successful")
    
    print("\n✅ SO100StyleProcessor tests passed!")


def test_trajectory_to_sample_generator():
    """Test the TrajectoryToSampleGenerator class."""
    print("\n" + "=" * 60)
    print("Testing TrajectoryToSampleGenerator")
    print("=" * 60)
    
    # Create processor and generator
    config = create_mock_config()
    processor = SO100StyleProcessor(config)
    generator = TrajectoryToSampleGenerator(processor)
    
    # Create multiple mock episodes
    episodes = []
    episode_lengths = [80, 120, 60, 100, 90]  # Different lengths
    
    for i, length in enumerate(episode_lengths):
        episode = create_mock_trajectory(f"episode_{i}", length)
        episodes.append(episode)
    
    print(f"Created {len(episodes)} test episodes with lengths: {episode_lengths}")
    
    # Test sample generation
    all_samples, episode_to_samples_map = generator.generate_samples_from_episodes(episodes)
    
    # Validate mapping
    print(f"\nEpisode-to-sample mapping:")
    for ep_idx, sample_indices in episode_to_samples_map.items():
        print(f"  Episode {ep_idx}: {len(sample_indices)} samples (indices {sample_indices[0] if sample_indices else 'N/A'}-{sample_indices[-1] if sample_indices else 'N/A'})")
    
    # Test advantage mapping
    episode_advantages = torch.randn(len(episodes))
    sample_advantages = generator.map_episode_advantages_to_samples(episode_advantages, episode_to_samples_map)
    
    print(f"\nAdvantage mapping:")
    print(f"  Episode advantages: {len(episode_advantages)}")
    print(f"  Sample advantages: {len(sample_advantages)}")
    
    # Validate mapping consistency
    is_valid = generator.validate_episode_to_sample_mapping(
        episode_advantages, episode_to_samples_map, len(all_samples)
    )
    assert is_valid, "Episode-to-sample mapping validation failed"
    
    # Test batch collation
    device = torch.device('cpu')
    
    # Convert samples to OpenPI format
    openpi_samples = []
    for sample in all_samples[:10]:  # Test with first 10 samples
        openpi_sample = processor.convert_to_openpi_format(sample)
        openpi_samples.append(openpi_sample)
    
    # Collate batch
    batch = generator.collate_samples_to_batch(openpi_samples, device)
    
    print(f"\nBatch collation test:")
    print(f"  Batch size: {batch['batch_size']}")
    print(f"  Image tensor shape: {batch['image']['base_0_rgb'].shape}")
    print(f"  Action tensor shape: {batch['action'].shape}")
    
    print("\n✅ TrajectoryToSampleGenerator tests passed!")


def test_data_utilization_improvement():
    """Test and demonstrate data utilization improvement."""
    print("\n" + "=" * 60)
    print("Testing Data Utilization Improvement")
    print("=" * 60)
    
    # Create processor
    config = create_mock_config()
    processor = SO100StyleProcessor(config)
    
    # Create realistic episode lengths
    episode_lengths = [120, 150, 80, 200, 90, 110, 160, 75]
    episodes = []
    
    for i, length in enumerate(episode_lengths):
        episode = create_mock_trajectory(f"episode_{i}", length)
        episodes.append(episode)
    
    # Calculate utilization stats
    stats = processor.get_data_utilization_stats(episodes)
    
    print(f"Data Utilization Analysis:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Valid episodes: {stats['valid_trajectories']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Improvement ratio: {stats['improvement_ratio']:.1f}x")
    print(f"  Avg samples per trajectory: {stats['avg_samples_per_trajectory']:.1f}")
    
    # Verify improvement is significant
    assert stats['improvement_ratio'] > 10, f"Expected >10x improvement, got {stats['improvement_ratio']:.1f}x"
    
    print("\n✅ Data utilization improvement verified!")


def main():
    """Run all tests."""
    print("🚀 Starting SO100-Style Data Processing Tests")
    print("=" * 80)
    
    try:
        # Test individual components
        test_so100_style_processor()
        test_trajectory_to_sample_generator()
        test_data_utilization_improvement()
        
        print("\n" + "=" * 80)
        print("🎉 All tests passed! Data processing pipeline is ready for integration.")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
