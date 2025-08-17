"""
Trajectory to Sample Generator

Manages the conversion of episode batches into training samples and maintains
the critical episode-to-sample mapping needed for RLOO advantage propagation.

Key Features:
- Batch processing of multiple trajectories
- Episode-to-sample mapping for advantage computation
- Memory-efficient sample generation
- Integration with RIPT advantage computation
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from .so100_style_processor import SO100StyleProcessor


class TrajectoryToSampleGenerator:
    """
    Manages trajectory-to-sample conversion and episode-sample mapping.
    
    This class handles:
    1. Batch processing of multiple episodes/trajectories
    2. Maintaining episode-to-sample mappings for advantage propagation
    3. Memory-efficient sample generation and collation
    4. Integration with RIPT RLOO advantage computation
    """
    
    def __init__(self, processor: SO100StyleProcessor):
        """
        Initialize the generator with a processor.
        
        Args:
            processor: SO100StyleProcessor instance for trajectory conversion
        """
        self.processor = processor
        self.action_chunk_size = processor.action_chunk_size
        
        print(f"✓ TrajectoryToSampleGenerator initialized with action_chunk_size={self.action_chunk_size}")
    
    def generate_samples_from_episodes(self, episodes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[int, List[int]]]:
        """
        Generate training samples from a batch of episodes.
        
        Args:
            episodes: List of episode dictionaries, each containing:
                - 'processed_observations': List of preprocessed observations
                - 'actions': List of action arrays
                - 'states': List of state arrays
                - 'id': Episode identifier
        
        Returns:
            Tuple of:
                - all_samples: List of all generated training samples
                - episode_to_samples_map: Dict mapping episode_idx to list of sample indices
        """
        all_samples = []
        episode_to_samples_map = {}
        
        print(f"🔄 Processing {len(episodes)} episodes into training samples...")
        
        for ep_idx, episode in enumerate(episodes):
            # Generate samples for this episode
            episode_samples = self.processor.process_trajectory_to_samples(episode)
            
            if len(episode_samples) == 0:
                # Skip episodes that are too short
                episode_to_samples_map[ep_idx] = []
                continue
            
            # Record mapping from episode to sample indices
            start_idx = len(all_samples)
            end_idx = start_idx + len(episode_samples)
            episode_to_samples_map[ep_idx] = list(range(start_idx, end_idx))
            
            # Add samples to global list
            all_samples.extend(episode_samples)
        
        # Calculate statistics
        total_samples = len(all_samples)
        valid_episodes = len([ep_idx for ep_idx, samples in episode_to_samples_map.items() if len(samples) > 0])
        
        print(f"✓ Sample generation complete:")
        print(f"  - Episodes processed: {len(episodes)}")
        print(f"  - Valid episodes: {valid_episodes}")
        print(f"  - Total samples generated: {total_samples}")
        print(f"  - Average samples per valid episode: {total_samples / valid_episodes if valid_episodes > 0 else 0:.1f}")
        
        return all_samples, episode_to_samples_map
    
    def map_episode_advantages_to_samples(self, episode_advantages: torch.Tensor, episode_to_samples_map: Dict[int, List[int]]) -> torch.Tensor:
        """
        Map episode-level advantages to sample-level advantages.
        
        This is critical for RIPT integration: episode-level RLOO advantages
        are computed correctly, then propagated to all samples from that episode.
        
        Args:
            episode_advantages: Tensor of shape (num_episodes,) with RLOO advantages
            episode_to_samples_map: Mapping from episode indices to sample indices
        
        Returns:
            sample_advantages: Tensor of shape (num_samples,) with advantages for each sample
        """
        sample_advantages = []
        
        print(f"🔄 Mapping {len(episode_advantages)} episode advantages to sample level...")
        
        for ep_idx, advantage in enumerate(episode_advantages):
            if ep_idx in episode_to_samples_map:
                sample_indices = episode_to_samples_map[ep_idx]
                
                # All samples from this episode get the same advantage value
                for _ in sample_indices:
                    sample_advantages.append(advantage.item())
            else:
                print(f"⚠ Episode {ep_idx} not found in sample mapping")
        
        sample_advantages_tensor = torch.tensor(sample_advantages, dtype=torch.float32)
        
        print(f"✓ Advantage mapping complete:")
        print(f"  - Episode advantages: {len(episode_advantages)}")
        print(f"  - Sample advantages: {len(sample_advantages_tensor)}")
        print(f"  - Positive samples: {(sample_advantages_tensor > 0).sum().item()}")
        print(f"  - Negative samples: {(sample_advantages_tensor <= 0).sum().item()}")
        
        return sample_advantages_tensor
    
    def collate_samples_to_batch(self, samples: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
        """
        Collate individual samples into a training batch.
        
        Args:
            samples: List of samples in OpenPI format
            device: Target device for tensors
        
        Returns:
            Collated batch ready for model training
        """
        if len(samples) == 0:
            raise ValueError("Cannot collate empty sample list")
        
        batch_size = len(samples)
        
        print(f"🔄 Collating {batch_size} samples into training batch...")
        
        # Extract and stack image data
        base_images = []
        wrist_images = []
        states = []
        actions = []
        action_is_pads = []
        prompts = []
        
        for sample in samples:
            base_images.append(sample['image']['base_0_rgb'])
            wrist_images.append(sample['image']['left_wrist_0_rgb'])
            states.append(sample['state'])
            actions.append(sample['action'])
            action_is_pads.append(sample['action_is_pad'])
            prompts.extend(sample['prompt'])  # Flatten prompt list
        
        # Stack tensors
        batch = {
            'image': {
                'base_0_rgb': torch.stack(base_images).to(device),
                'left_wrist_0_rgb': torch.stack(wrist_images).to(device)
            },
            'state': torch.stack(states).to(device),
            'action': torch.stack(actions).to(device),
            'action_is_pad': torch.stack(action_is_pads).to(device),
            'prompt': prompts,
            'batch_size': batch_size
        }
        
        print(f"✓ Batch collation complete:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Image shape: {batch['image']['base_0_rgb'].shape}")
        print(f"  - State shape: {batch['state'].shape}")
        print(f"  - Action shape: {batch['action'].shape}")
        print(f"  - Device: {device}")
        
        return batch
    
    def process_episodes_to_training_batch(self, episodes: List[Dict[str, Any]], device: torch.device) -> Tuple[Dict[str, Any], Dict[int, List[int]]]:
        """
        End-to-end processing: episodes → samples → training batch.
        
        Args:
            episodes: List of episode dictionaries
            device: Target device for tensors
        
        Returns:
            Tuple of:
                - training_batch: Collated batch ready for model training
                - episode_to_samples_map: Mapping for advantage propagation
        """
        print(f"🚀 Starting end-to-end processing of {len(episodes)} episodes...")
        
        # Step 1: Generate samples
        all_samples, episode_to_samples_map = self.generate_samples_from_episodes(episodes)
        
        if len(all_samples) == 0:
            raise ValueError("No valid samples generated from episodes")
        
        # Step 2: Convert to OpenPI format
        openpi_samples = []
        for sample in all_samples:
            openpi_sample = self.processor.convert_to_openpi_format(sample)
            openpi_samples.append(openpi_sample)
        
        # Step 3: Collate into batch
        training_batch = self.collate_samples_to_batch(openpi_samples, device)
        
        print(f"✅ End-to-end processing complete!")
        
        return training_batch, episode_to_samples_map
    
    def validate_episode_to_sample_mapping(self, episode_advantages: torch.Tensor, episode_to_samples_map: Dict[int, List[int]], total_samples: int) -> bool:
        """
        Validate the episode-to-sample mapping for consistency.
        
        Args:
            episode_advantages: Episode-level advantages
            episode_to_samples_map: Episode to sample mapping
            total_samples: Expected total number of samples
        
        Returns:
            True if mapping is valid
        """
        print("🔍 Validating episode-to-sample mapping...")
        
        # Check that all episodes are mapped
        mapped_episodes = set(episode_to_samples_map.keys())
        expected_episodes = set(range(len(episode_advantages)))
        
        if mapped_episodes != expected_episodes:
            missing = expected_episodes - mapped_episodes
            extra = mapped_episodes - expected_episodes
            print(f"✗ Episode mapping mismatch:")
            print(f"  - Missing episodes: {missing}")
            print(f"  - Extra episodes: {extra}")
            return False
        
        # Check sample index consistency
        all_sample_indices = []
        for ep_idx, sample_indices in episode_to_samples_map.items():
            all_sample_indices.extend(sample_indices)
        
        expected_indices = set(range(total_samples))
        actual_indices = set(all_sample_indices)
        
        if expected_indices != actual_indices:
            print(f"✗ Sample index mismatch:")
            print(f"  - Expected: {len(expected_indices)} indices")
            print(f"  - Actual: {len(actual_indices)} indices")
            return False
        
        print("✅ Episode-to-sample mapping validation passed!")
        return True
