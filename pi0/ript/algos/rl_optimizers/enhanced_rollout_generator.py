#!/usr/bin/env python3
"""
Enhanced Rollout Generator with FileGlobalCounter Integration

Based on original RIPT-VLA rollout generation with added features:
- File-based distributed coordination
- Dynamic task sampling
- Early stopping mechanisms
- Advanced memory management
"""

import os
import gc
import time
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Generator, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)

class EnhancedRolloutGenerator:
    """
    Enhanced rollout generator with distributed coordination and smart sampling.
    """
    
    def __init__(self, 
                 env_runner,
                 max_rollouts: int = 100,
                 early_stop_threshold: Optional[int] = None,
                 enable_file_counter: bool = True,
                 enable_smart_sampling: bool = True,
                 enable_dynamic_batching: bool = True,
                 max_step_batch_size: int = 32,
                 work_dir: str = "./"):
        """
        Initialize enhanced rollout generator.
        
        Args:
            env_runner: Environment runner instance
            max_rollouts: Maximum number of rollouts to generate
            early_stop_threshold: Early stop when counter reaches this value
            enable_file_counter: Use file-based counter for coordination
            enable_smart_sampling: Enable intelligent task/state sampling
            enable_dynamic_batching: Use dynamic batch processing
            max_step_batch_size: Maximum batch size for step processing
            work_dir: Working directory for counter files
        """
        self.env_runner = env_runner
        self.max_rollouts = max_rollouts
        self.early_stop_threshold = early_stop_threshold or max_rollouts
        self.enable_file_counter = enable_file_counter
        self.enable_smart_sampling = enable_smart_sampling
        self.enable_dynamic_batching = enable_dynamic_batching
        self.max_step_batch_size = max_step_batch_size
        self.work_dir = work_dir
        
        # Initialize file counter if enabled
        self.file_counter = None
        if enable_file_counter:
            self.file_counter = env_runner.setup_file_counter("rollout", work_dir)
        
        # Smart sampling state
        self.task_success_history = deque(maxlen=100)  # Track recent success rates
        self.state_sampling_weights = {}  # Adaptive state sampling weights
        
        # Performance metrics
        self.rollout_stats = {
            'total_generated': 0,
            'successful_rollouts': 0,
            'average_reward': 0.0,
            'tasks_attempted': set(),
            'generation_time': 0.0
        }
        
        logger.info(f"EnhancedRolloutGenerator initialized: max_rollouts={max_rollouts}")
    
    def reset_counter(self):
        """Reset the file counter for new rollout session."""
        if self.file_counter:
            try:
                from pi0.ript.algos.rl_optimizers.file_counter import reset_global_counter
                reset_global_counter(self.file_counter)
                if self.env_runner.rank == 0:
                    print("🔄 Rollout counter reset for new session")
            except ImportError:
                if self.env_runner.rank == 0:
                    print("⚠️ File counter reset not available")
    
    def should_early_stop(self) -> bool:
        """Check if early stopping conditions are met."""
        if not self.file_counter:
            return False
        
        try:
            current_count = self.file_counter.get()
            return current_count >= self.early_stop_threshold
        except Exception as e:
            logger.warning(f"Error checking early stop condition: {e}")
            return False
    
    def _select_next_task(self) -> Optional[str]:
        """Select next task using smart sampling if enabled."""
        if not self.env_runner.has_tasks():
            return None
        
        if self.enable_smart_sampling and hasattr(self.env_runner, 'get_best_performing_task'):
            # Try to get best performing task
            best_task = self.env_runner.get_best_performing_task()
            if best_task and best_task in self.env_runner.assigned_tasks:
                return best_task
        
        # Fall back to normal task polling
        return self.env_runner.get_current_task()
    
    def _generate_init_states(self, task_name: str, num_states: int) -> np.ndarray:
        """Generate initial states with smart sampling."""
        # Basic implementation - can be enhanced with learned state distributions
        if task_name in self.state_sampling_weights:
            # Use learned weights for this task
            weights = self.state_sampling_weights[task_name]
            # Generate states based on successful state history
            states = np.random.multivariate_normal(
                mean=weights['mean'], 
                cov=weights['cov'], 
                size=num_states
            ).astype(np.float32)
        else:
            # Default random sampling
            states = np.random.randn(num_states, 10).astype(np.float32) * 0.1
        
        return states
    
    def _update_sampling_weights(self, task_name: str, init_state: np.ndarray, success: bool):
        """Update state sampling weights based on rollout results."""
        if not self.enable_smart_sampling:
            return
        
        if task_name not in self.state_sampling_weights:
            self.state_sampling_weights[task_name] = {
                'successful_states': [],
                'failed_states': [],
                'mean': np.zeros_like(init_state),
                'cov': np.eye(len(init_state))
            }
        
        weights = self.state_sampling_weights[task_name]
        
        if success:
            weights['successful_states'].append(init_state.copy())
        else:
            weights['failed_states'].append(init_state.copy())
        
        # Update distribution parameters periodically
        if len(weights['successful_states']) >= 10:
            successful_array = np.array(weights['successful_states'])
            weights['mean'] = np.mean(successful_array, axis=0)
            weights['cov'] = np.cov(successful_array.T)
            
            # Keep only recent successful states
            if len(weights['successful_states']) > 50:
                weights['successful_states'] = weights['successful_states'][-30:]
    
    def generate_rollouts(self, 
                         policy_wrapper,
                         init_states: Optional[np.ndarray] = None,
                         debug_save_video: bool = False) -> Generator[Tuple[bool, float, Dict], None, None]:
        """
        Generate rollouts with enhanced features.
        
        Args:
            policy_wrapper: Policy wrapper for action generation
            init_states: Optional initial states
            debug_save_video: Whether to save debug videos
            
        Yields:
            Tuple of (success, reward, episode_data)
        """
        start_time = time.time()
        
        # Reset counter at start
        if self.file_counter and self.env_runner.rank == 0:
            self.reset_counter()
        
        rollout_count = 0
        
        try:
            while rollout_count < self.max_rollouts:
                # Check early stopping
                if self.should_early_stop():
                    if self.env_runner.rank == 0:
                        print(f"🛑 Early stopping triggered at {rollout_count} rollouts")
                    break
                
                # Select next task
                current_task = self._select_next_task()
                if not current_task:
                    logger.warning("No tasks available for rollout generation")
                    break
                
                # Generate initial states for this batch
                if init_states is None:
                    batch_size = min(self.max_step_batch_size, self.max_rollouts - rollout_count)
                    init_states_batch = self._generate_init_states(current_task, batch_size)
                else:
                    # Use provided states, but only take what we need
                    remaining = self.max_rollouts - rollout_count
                    init_states_batch = init_states[:remaining]
                
                # Generate rollouts for current task
                try:
                    rollout_results = self.env_runner.run_policy_in_env(
                        env_name=current_task,
                        all_init_states=init_states_batch,
                        debug_save_video=debug_save_video
                    )
                    
                    # Process results
                    for i, (success, total_reward, episode_data) in enumerate(rollout_results):
                        if rollout_count >= self.max_rollouts:
                            break
                        
                        # Update statistics
                        self.rollout_stats['total_generated'] += 1
                        self.rollout_stats['tasks_attempted'].add(current_task)
                        if success:
                            self.rollout_stats['successful_rollouts'] += 1
                        
                        # Update running average reward
                        old_avg = self.rollout_stats['average_reward']
                        n = self.rollout_stats['total_generated']
                        self.rollout_stats['average_reward'] = old_avg + (total_reward - old_avg) / n
                        
                        # Update task-specific statistics
                        if hasattr(self.env_runner, 'update_task_stats'):
                            self.env_runner.update_task_stats(current_task, success, total_reward)
                        
                        # Update sampling weights for smart sampling
                        if i < len(init_states_batch):
                            self._update_sampling_weights(current_task, init_states_batch[i], success)
                        
                        # Update file counter
                        if self.file_counter:
                            try:
                                new_count = self.file_counter.increment()
                                if new_count % 10 == 0 and self.env_runner.rank == 0:
                                    print(f"📊 Generated {new_count} rollouts globally")
                            except Exception as e:
                                logger.warning(f"Error updating file counter: {e}")
                        
                        rollout_count += 1
                        yield (success, total_reward, episode_data)
                        
                        # Memory management
                        if rollout_count % 20 == 0:
                            self._cleanup_memory()
                
                except Exception as e:
                    logger.error(f"Error generating rollout for task {current_task}: {e}")
                    continue
                
                # Advance to next task
                if self.env_runner.enable_task_polling:
                    self.env_runner.advance_task_cursor()
                
                # Clear init_states for next iteration if it was provided
                init_states = None
        
        finally:
            # Final cleanup and statistics
            self.rollout_stats['generation_time'] = time.time() - start_time
            self._log_final_stats()
            self._cleanup_memory()
    
    def _cleanup_memory(self):
        """Perform memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _log_final_stats(self):
        """Log final generation statistics."""
        if self.env_runner.rank == 0:
            stats = self.rollout_stats
            print(f"📈 Rollout Generation Complete:")
            print(f"   Total Generated: {stats['total_generated']}")
            print(f"   Success Rate: {stats['successful_rollouts'] / max(stats['total_generated'], 1):.1%}")
            print(f"   Average Reward: {stats['average_reward']:.3f}")
            print(f"   Tasks Attempted: {len(stats['tasks_attempted'])}")
            print(f"   Generation Time: {stats['generation_time']:.1f}s")
    
    def get_rollout_stats(self) -> Dict[str, Any]:
        """Get current rollout generation statistics."""
        return self.rollout_stats.copy()
    
    def cleanup(self):
        """Clean up resources."""
        if self.file_counter and hasattr(self.file_counter, 'cleanup'):
            self.file_counter.cleanup()
        self._cleanup_memory()

# Convenience function for creating enhanced rollout generator
def create_enhanced_rollout_generator(env_runner, 
                                    max_rollouts: int = 100,
                                    **kwargs) -> EnhancedRolloutGenerator:
    """Create an enhanced rollout generator with sensible defaults."""
    return EnhancedRolloutGenerator(
        env_runner=env_runner,
        max_rollouts=max_rollouts,
        **kwargs
    )