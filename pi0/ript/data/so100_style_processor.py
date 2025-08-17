"""
SO100-Style Data Processor

Converts trajectory data into training samples following the so100_train.py pattern.
Each trajectory of length L generates L-50+1 training samples, where each sample
contains one observation and the next 50 actions (relative to current state).

Key Features:
- Follows exact so100_train.py logic (line 65: action = action - state)
- Generates maximum training samples from each trajectory
- Maintains OpenPI format compatibility
- Handles padding for trajectories shorter than action_chunk_size
"""

import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class SO100StyleProcessor:
    """
    Processes trajectories into training samples following so100_train.py pattern.
    
    Core Logic:
    - Each trajectory of length L → L-50+1 training samples
    - Each sample: obs[t] → action[t:t+50] (relative actions)
    - Relative action: action - current_state (so100_train.py line 65)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the processor with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - action_chunk_size: Number of action steps per sample (default: 50)
                - normalization: Normalization statistics paths and settings
        """
        self.config = config
        self.action_chunk_size = config.get('action_chunk_size', 50)
        
        # Load normalization statistics (following 2_pi0_on_libero.py pattern)
        self._load_normalization_stats()
        
        print(f"✓ SO100StyleProcessor initialized:")
        print(f"  - Action chunk size: {self.action_chunk_size}")
        print(f"  - State mean shape: {self.state_mean.shape}")
        print(f"  - Action mean shape: {self.action_mean.shape}")
    
    def _load_normalization_stats(self):
        """Load normalization statistics following 2_pi0_on_libero.py pattern."""
        norm_stats_path = self.config.get('norm_stats_path', 
            "/zhaohan/ZJH/openpi_pytorch/lerobot_dataset/norm_stats.json")
        
        if not Path(norm_stats_path).exists():
            raise FileNotFoundError(f"Normalization stats not found: {norm_stats_path}")
        
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)
        
        # Extract statistics exactly as in 2_pi0_on_libero.py
        self.state_mean = np.array(
            norm_stats["norm_stats"]["state"]["mean"][:8], dtype=np.float32
        )
        self.state_std = np.array(
            norm_stats["norm_stats"]["state"]["std"][:8], dtype=np.float32
        )
        self.action_mean = np.array(
            norm_stats["norm_stats"]["actions"]["mean"][:7], dtype=np.float32
        )
        self.action_std = np.array(
            norm_stats["norm_stats"]["actions"]["std"][:7], dtype=np.float32
        )
    
    def process_trajectory_to_samples(self, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert a single trajectory into multiple training samples.
        
        Args:
            trajectory: Dictionary containing:
                - 'processed_observations': List of preprocessed observations
                - 'actions': List of action arrays
                - 'states': List of state arrays
                - 'id': Trajectory identifier
        
        Returns:
            List of training samples, each containing:
                - 'observation': Current observation
                - 'action': 50-step action chunk (relative actions)
                - 'action_is_pad': Padding mask
                - 'trajectory_id': Source trajectory ID
                - 'sample_index': Position in trajectory
        """
        obs_sequence = trajectory['processed_observations']
        action_sequence = trajectory['actions']
        state_sequence = trajectory['states']
        trajectory_id = trajectory.get('id', 'unknown')
        
        samples = []
        trajectory_length = len(obs_sequence)
        
        # Skip trajectories that are too short
        if trajectory_length < self.action_chunk_size:
            print(f"⚠ Skipping trajectory {trajectory_id}: length {trajectory_length} < {self.action_chunk_size}")
            return samples
        
        # Generate L-50+1 samples
        num_samples = trajectory_length - self.action_chunk_size + 1
        
        for t in range(num_samples):
            # Current observation and state
            current_obs = obs_sequence[t]
            current_state = state_sequence[t]  # shape: (state_dim,)
            
            # Extract action chunk
            action_chunk = action_sequence[t:t + self.action_chunk_size]  # shape: (50, action_dim)
            action_chunk = np.array(action_chunk, dtype=np.float32)
            
            # Key: Compute relative actions (so100_train.py line 65)
            # action = action - state
            # 注意：如果状态维度(8) > 动作维度(7)，只使用前7维状态
            if current_state.shape[0] > action_chunk.shape[1]:
                # 状态维度 > 动作维度，只使用前action_dim维状态
                state_for_action = current_state[:action_chunk.shape[1]]
                relative_actions = action_chunk - state_for_action[None, :]  # Broadcasting
            else:
                # 状态维度 <= 动作维度，直接计算
                relative_actions = action_chunk - current_state[None, :]  # Broadcasting
            
            # Create padding mask (all False for full-length chunks)
            action_is_pad = np.zeros(self.action_chunk_size, dtype=bool)
            
            # Create sample
            sample = {
                'observation': current_obs,
                'action': relative_actions,
                'action_is_pad': action_is_pad,
                'trajectory_id': trajectory_id,
                'sample_index': t,
                'current_state': current_state  # Keep for debugging/validation
            }
            
            samples.append(sample)
        
        print(f"✓ Trajectory {trajectory_id}: {trajectory_length} steps → {len(samples)} samples")
        return samples
    
    def convert_to_openpi_format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert sample to OpenPI format following so100_train.py pattern.
        
        Args:
            sample: Raw sample from process_trajectory_to_samples
            
        Returns:
            Sample in OpenPI format compatible with model training
        """
        # Extract observation components
        obs = sample['observation']
        
        # Convert to OpenPI format (following so100_train.py lines 71-75)
        openpi_sample = {
            "image": {
                "base_0_rgb": obs['image']['base_0_rgb'],
                "left_wrist_0_rgb": obs['image']['left_wrist_0_rgb']
            },
            "state": torch.from_numpy(obs['state']).float(),  # 确保state也是tensor
            "action": torch.from_numpy(sample['action']).float(),
            "action_is_pad": torch.from_numpy(sample['action_is_pad']),
            "prompt": obs.get('prompt', [''])
        }
        
        return openpi_sample
    
    def validate_sample_format(self, sample: Dict[str, Any]) -> bool:
        """
        Validate that a sample has the correct format.
        
        Args:
            sample: Sample to validate
            
        Returns:
            True if sample format is valid
        """
        required_keys = ['observation', 'action', 'action_is_pad', 'trajectory_id', 'sample_index']
        
        # Check required keys
        for key in required_keys:
            if key not in sample:
                print(f"✗ Missing key: {key}")
                return False
        
        # Check action shape
        if sample['action'].shape[0] != self.action_chunk_size:
            print(f"✗ Invalid action shape: {sample['action'].shape}, expected ({self.action_chunk_size}, action_dim)")
            return False
        
        # Check padding mask shape
        if sample['action_is_pad'].shape[0] != self.action_chunk_size:
            print(f"✗ Invalid padding mask shape: {sample['action_is_pad'].shape}")
            return False
        
        return True
    
    def get_data_utilization_stats(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate data utilization statistics.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            Statistics about data utilization improvement
        """
        total_episodes = len(trajectories)
        total_samples = 0
        valid_trajectories = 0
        
        for traj in trajectories:
            traj_length = len(traj.get('processed_observations', []))
            if traj_length >= self.action_chunk_size:
                samples_from_traj = traj_length - self.action_chunk_size + 1
                total_samples += samples_from_traj
                valid_trajectories += 1
        
        improvement_ratio = total_samples / total_episodes if total_episodes > 0 else 0
        
        stats = {
            'total_episodes': total_episodes,
            'valid_trajectories': valid_trajectories,
            'total_samples': total_samples,
            'improvement_ratio': improvement_ratio,
            'avg_samples_per_trajectory': total_samples / valid_trajectories if valid_trajectories > 0 else 0
        }
        
        return stats
