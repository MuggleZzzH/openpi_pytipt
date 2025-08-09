#!/usr/bin/env python3
"""
Comprehensive usage examples for FileGlobalCounter in OpenPI distributed training.

This file demonstrates how to integrate the FileGlobalCounter into different
aspects of the OpenPI training pipeline for distributed coordination.
"""

import os
import time
import torch
import torch.distributed as dist
import logging
from pathlib import Path
from pi0.ript.algos.rl_optimizers.file_counter import (
    FileGlobalCounter,
    setup_file_counter,
    setup_rollout_counter,
    setup_batch_counter,
    setup_episode_counter,
    reset_global_counter,
    cleanup_counter,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def example_basic_usage():
    """Basic FileGlobalCounter usage example."""
    print("=== Basic FileGlobalCounter Usage ===")
    
    # Setup a simple counter
    counter, filename = setup_file_counter(
        tmp_dir="/tmp/openpi_counters",
        counter_type="example",
        prefix="basic"
    )
    
    try:
        # Basic operations
        print(f"Initial value: {counter.get()}")
        
        # Increment counter
        new_value = counter.increment()
        print(f"After increment: {new_value}")
        
        # Add multiple values
        new_value = counter.add(5)
        print(f"After adding 5: {new_value}")
        
        # Reset counter
        counter.reset(10)
        print(f"After reset to 10: {counter.get()}")
        
        # Get and reset atomically
        old_value = counter.get_and_reset(0)
        print(f"Got {old_value}, new value: {counter.get()}")
        
    finally:
        # Cleanup
        cleanup_counter(filename)
        print("Cleaned up counter file")


def example_distributed_training_coordination():
    """Example of using FileGlobalCounter for distributed training coordination."""
    print("\n=== Distributed Training Coordination ===")
    
    # This would typically be called in a distributed training script
    if not dist.is_initialized():
        print("Not in distributed mode, simulating single process...")
        
    # Setup different types of counters for various coordination needs
    rollout_counter, rollout_file = setup_rollout_counter(
        tmp_dir="/tmp/openpi_counters"
    )
    
    batch_counter, batch_file = setup_batch_counter(
        tmp_dir="/tmp/openpi_counters"
    )
    
    episode_counter, episode_file = setup_episode_counter(
        tmp_dir="/tmp/openpi_counters"
    )
    
    try:
        # Simulate training coordination
        total_episodes_needed = 100
        episodes_per_batch = 8
        early_stop_threshold = int(total_episodes_needed * 0.8)  # 80% completion
        
        print(f"Training coordination example:")
        print(f"- Total episodes needed: {total_episodes_needed}")
        print(f"- Early stop threshold: {early_stop_threshold}")
        
        # Simulate rollout generation with early stopping
        for batch_idx in range(total_episodes_needed // episodes_per_batch):
            # Reset episode counter for this batch
            reset_global_counter(episode_counter, 0)
            
            # Simulate generating episodes in this batch
            for episode_idx in range(episodes_per_batch):
                # Each process would increment when completing an episode
                current_episodes = episode_counter.increment()
                
                # Update global rollout counter
                total_episodes = rollout_counter.increment()
                
                print(f"Batch {batch_idx}, Episode {episode_idx}: "
                      f"Total episodes completed: {total_episodes}")
                
                # Check for early stopping condition
                if total_episodes >= early_stop_threshold:
                    print(f"Early stopping triggered at {total_episodes} episodes")
                    break
            
            # Update batch counter
            batch_counter.increment()
            
            # Check early stopping between batches too
            if rollout_counter.get() >= early_stop_threshold:
                break
        
        print(f"Final stats:")
        print(f"- Total episodes: {rollout_counter.get()}")
        print(f"- Total batches: {batch_counter.get()}")
        
    finally:
        # Cleanup all counters
        cleanup_counter(rollout_file)
        cleanup_counter(batch_file)
        cleanup_counter(episode_file)
        print("Cleaned up all counter files")


def example_rollout_generator_integration():
    """Example showing how to integrate FileGlobalCounter into RolloutGenerator."""
    print("\n=== RolloutGenerator Integration Example ===")
    
    class MockRolloutGenerator:
        """Mock rollout generator showing integration pattern."""
        
        def __init__(self, batch_size=8, early_stop_percentage=0.8):
            self.batch_size = batch_size
            self.early_stop_percentage = early_stop_percentage
            
            # Setup file counter for coordination
            self.file_counter, self.counter_filename = setup_rollout_counter(
                tmp_dir="/tmp/openpi_counters"
            )
            
            # Calculate early stop threshold
            if dist.is_initialized():
                world_size = dist.get_world_size()
                total_batch_size = self.batch_size * world_size
                self.global_threshold = int(total_batch_size * self.early_stop_percentage)
            else:
                self.global_threshold = int(self.batch_size * self.early_stop_percentage)
            
            print(f"MockRolloutGenerator initialized:")
            print(f"- Batch size: {self.batch_size}")
            print(f"- Early stop threshold: {self.global_threshold}")
        
        def generate_rollouts(self, num_samples):
            """Mock rollout generation with early stopping."""
            print(f"\nGenerating rollouts for {num_samples} samples...")
            
            # Reset counter at start of rollout generation
            reset_global_counter(self.file_counter, 0)
            
            successful_rollouts = []
            early_stop = False
            
            for sample_idx in range(num_samples):
                if early_stop:
                    print(f"Early stopping at sample {sample_idx}")
                    break
                
                # Simulate rollout generation
                time.sleep(0.1)  # Simulate work
                
                # Mock success/failure
                success = (sample_idx % 3) != 0  # 2/3 success rate
                
                if success:
                    successful_rollouts.append(sample_idx)
                    # Update global counter for successful rollouts
                    current_count = self.file_counter.increment()
                    
                    print(f"Sample {sample_idx}: Success (global count: {current_count})")
                    
                    # Check early stopping condition
                    if current_count >= self.global_threshold:
                        print(f"Reached early stop threshold ({self.global_threshold})")
                        early_stop = True
                else:
                    print(f"Sample {sample_idx}: Failed")
            
            final_count = self.file_counter.get()
            print(f"\nRollout generation complete:")
            print(f"- Successful rollouts: {len(successful_rollouts)}")
            print(f"- Final global count: {final_count}")
            
            return successful_rollouts
        
        def cleanup(self):
            """Cleanup resources."""
            cleanup_counter(self.counter_filename)
            print("RolloutGenerator cleaned up")
    
    # Use the mock rollout generator
    generator = MockRolloutGenerator(batch_size=12, early_stop_percentage=0.6)
    
    try:
        # Generate rollouts
        results = generator.generate_rollouts(20)
        print(f"Generated {len(results)} successful rollouts")
    finally:
        generator.cleanup()


def example_error_handling_and_recovery():
    """Example showing error handling and recovery mechanisms."""
    print("\n=== Error Handling and Recovery ===")
    
    # Create a counter with custom retry settings
    counter, filename = setup_file_counter(
        tmp_dir="/tmp/openpi_counters",
        counter_type="robust",
        retry_attempts=5,
        retry_delay=0.05
    )
    
    try:
        # Test basic operations
        print("Testing basic operations...")
        counter.reset(0)
        counter.increment()
        counter.add(10)
        
        # Simulate file corruption by writing invalid content
        print("Simulating file corruption...")
        with open(filename, 'w') as f:
            f.write("invalid_number")
        
        # Counter should handle this gracefully
        try:
            value = counter.get()
            print(f"Got value despite corruption: {value}")
        except Exception as e:
            print(f"Error reading corrupted file: {e}")
        
        # Reset should fix the file
        print("Resetting counter to fix corruption...")
        counter.reset(42)
        value = counter.get()
        print(f"Value after reset: {value}")
        
        # Test directory creation
        print("Testing directory auto-creation...")
        deep_path = "/tmp/openpi_counters/deep/nested/path"
        deep_counter, deep_filename = setup_file_counter(
            tmp_dir=deep_path,
            counter_type="deep"
        )
        
        deep_counter.increment()
        print(f"Deep counter value: {deep_counter.get()}")
        cleanup_counter(deep_filename)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cleanup_counter(filename)
        print("Error handling test completed")


def example_performance_monitoring():
    """Example showing performance monitoring with counters."""
    print("\n=== Performance Monitoring ===")
    
    # Setup counters for different metrics
    counters = {}
    filenames = {}
    
    metrics = ["samples_processed", "successful_rollouts", "failed_rollouts", "batches_completed"]
    
    for metric in metrics:
        counter, filename = setup_file_counter(
            tmp_dir="/tmp/openpi_counters",
            counter_type=metric,
            prefix="perf"
        )
        counters[metric] = counter
        filenames[metric] = filename
    
    try:
        # Simulate training with performance monitoring
        print("Simulating training with performance monitoring...")
        
        num_batches = 5
        samples_per_batch = 10
        
        for batch_idx in range(num_batches):
            print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")
            
            batch_successful = 0
            batch_failed = 0
            
            for sample_idx in range(samples_per_batch):
                # Update samples processed
                counters["samples_processed"].increment()
                
                # Simulate processing with 70% success rate
                if (sample_idx + batch_idx) % 10 < 7:
                    counters["successful_rollouts"].increment()
                    batch_successful += 1
                else:
                    counters["failed_rollouts"].increment()
                    batch_failed += 1
            
            # Update batch counter
            counters["batches_completed"].increment()
            
            # Print batch statistics
            total_samples = counters["samples_processed"].get()
            total_success = counters["successful_rollouts"].get()
            total_failed = counters["failed_rollouts"].get()
            total_batches = counters["batches_completed"].get()
            
            success_rate = total_success / total_samples if total_samples > 0 else 0
            
            print(f"Batch {batch_idx + 1} completed:")
            print(f"  - Batch success: {batch_successful}/{samples_per_batch}")
            print(f"  - Total samples: {total_samples}")
            print(f"  - Total success rate: {success_rate:.2%}")
            print(f"  - Total batches: {total_batches}")
        
        # Final report
        print(f"\nFinal Performance Report:")
        for metric in metrics:
            value = counters[metric].get()
            print(f"  - {metric}: {value}")
    
    finally:
        # Cleanup all performance counters
        for filename in filenames.values():
            cleanup_counter(filename)
        print("Performance monitoring cleanup completed")


if __name__ == "__main__":
    print("FileGlobalCounter Usage Examples for OpenPI")
    print("=" * 50)
    
    # Ensure temp directory exists
    os.makedirs("/tmp/openpi_counters", exist_ok=True)
    
    try:
        # Run all examples
        example_basic_usage()
        example_distributed_training_coordination()
        example_rollout_generator_integration()
        example_error_handling_and_recovery()
        example_performance_monitoring()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        raise
    
    finally:
        # Final cleanup - remove temp directory if empty
        try:
            if os.path.exists("/tmp/openpi_counters"):
                if not os.listdir("/tmp/openpi_counters"):
                    os.rmdir("/tmp/openpi_counters")
                    print("Cleaned up temporary directory")
        except:
            pass