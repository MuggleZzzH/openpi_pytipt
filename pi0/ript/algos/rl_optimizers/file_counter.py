#!/usr/bin/env python3
"""
File-based Global Counter for Distributed Training

Enhanced implementation based on RIPT-VLA with improved error handling,
retry mechanisms, and PyTorch Distributed integration.
"""

import os
import fcntl
import time
import tempfile
import threading
from pathlib import Path
from typing import Optional, Union, Any
import torch
import torch.distributed as dist
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FileGlobalCounter:
    """
    File-based global counter for distributed training coordination.
    
    Features:
    - Thread-safe and process-safe operations using file locks
    - Automatic retry with exponential backoff
    - File corruption recovery
    - PyTorch Distributed integration
    - Detailed logging and error handling
    """
    
    def __init__(self, 
                 file_path: Optional[str] = None,
                 initial_value: int = 0,
                 max_retries: int = 5,
                 base_delay: float = 0.1,
                 timeout: float = 30.0):
        """
        Initialize FileGlobalCounter.
        
        Args:
            file_path: Path to counter file. If None, creates temp file.
            initial_value: Initial counter value
            max_retries: Maximum retry attempts for operations
            base_delay: Base delay for exponential backoff (seconds)
            timeout: Operation timeout (seconds)
        """
        self.initial_value = initial_value
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self._lock = threading.Lock()
        
        # Set up file path
        if file_path is None:
            # Create unique temp file for distributed coordination
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            self.file_path = os.path.join(
                tempfile.gettempdir(), 
                f"file_counter_{timestamp}_{rank}.txt"
            )
        else:
            self.file_path = file_path
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
        # Initialize file if it doesn't exist
        self._init_file()
        
        logger.info(f"FileGlobalCounter initialized: {self.file_path}")
    
    def _init_file(self) -> None:
        """Initialize counter file with initial value."""
        if not os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'w') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.write(str(self.initial_value))
                    f.flush()
                    os.fsync(f.fileno())
                logger.debug(f"Initialized counter file: {self.file_path}")
            except Exception as e:
                logger.error(f"Failed to initialize counter file: {e}")
                raise
    
    def _retry_with_backoff(self, operation, *args, **kwargs) -> Any:
        """Execute operation with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == self.max_retries - 1:
                    break
                    
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"Operation failed (attempt {attempt + 1}/{self.max_retries}), "
                             f"retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
        
        logger.error(f"Operation failed after {self.max_retries} attempts: {last_exception}")
        raise last_exception
    
    def _safe_read(self) -> int:
        """Safely read current counter value with shared lock."""
        with open(self.file_path, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock for reading
            content = f.read().strip()
            
        if not content:
            logger.warning("Counter file is empty, returning initial value")
            return self.initial_value
            
        try:
            return int(content)
        except ValueError as e:
            logger.error(f"Counter file corrupted, content: '{content}', returning initial value")
            return self.initial_value
    
    def _safe_write(self, value: int) -> None:
        """Safely write counter value with exclusive lock."""
        with open(self.file_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock for writing
            f.write(str(value))
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
    
    def _safe_update(self, increment: int) -> int:
        """Safely update counter with atomic read-modify-write."""
        with open(self.file_path, 'r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock for atomic update
            
            # Read current value
            f.seek(0)
            content = f.read().strip()
            current = int(content) if content else self.initial_value
            
            # Calculate new value
            new_value = current + increment
            
            # Write new value
            f.seek(0)
            f.write(str(new_value))
            f.truncate()
            f.flush()
            os.fsync(f.fileno())
            
            return new_value
    
    def get(self) -> int:
        """Get current counter value."""
        with self._lock:
            return self._retry_with_backoff(self._safe_read)
    
    def set(self, value: int) -> None:
        """Set counter to specific value."""
        with self._lock:
            self._retry_with_backoff(self._safe_write, value)
            logger.debug(f"Counter set to {value}")
    
    def update(self, increment: int) -> int:
        """Update counter by increment and return new value."""
        with self._lock:
            new_value = self._retry_with_backoff(self._safe_update, increment)
            logger.debug(f"Counter updated by {increment} to {new_value}")
            return new_value
    
    def increment(self) -> int:
        """Increment counter by 1 and return new value."""
        return self.update(1)
    
    def decrement(self) -> int:
        """Decrement counter by 1 and return new value.""" 
        return self.update(-1)
    
    def add(self, value: int) -> int:
        """Add value to counter and return new value."""
        return self.update(value)
    
    def reset(self, value: Optional[int] = None) -> None:
        """Reset counter to initial value or specified value."""
        reset_value = value if value is not None else self.initial_value
        self.set(reset_value)
        logger.info(f"Counter reset to {reset_value}")
    
    def cleanup(self) -> None:
        """Clean up counter file."""
        try:
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
                logger.info(f"Counter file removed: {self.file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove counter file: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Destructor with cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup

# Distributed training utilities
def setup_global_counter(counter_name: str, 
                        initial_value: int = 0,
                        work_dir: str = "./") -> FileGlobalCounter:
    """
    Set up a global counter for distributed training.
    
    Args:
        counter_name: Name of the counter (e.g., 'rollout', 'batch', 'episode')
        initial_value: Initial counter value
        work_dir: Working directory for counter files
        
    Returns:
        FileGlobalCounter instance
    """
    if not (dist.is_available() and dist.is_initialized()):
        logger.warning("PyTorch distributed not available, using local counter")
        rank, world_size = 0, 1
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    
    # Create unique counter file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    counter_dir = os.path.join(work_dir, "distributed_counters")
    os.makedirs(counter_dir, exist_ok=True)
    
    if rank == 0:
        # Master process creates the counter file path
        counter_file = os.path.join(counter_dir, f"{counter_name}_{timestamp}_{world_size}.txt")
        counter_path = [counter_file]
    else:
        counter_path = [None]
    
    # Broadcast counter file path to all processes
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(counter_path, src=0)
        dist.barrier()  # Ensure all processes have the path
    
    counter = FileGlobalCounter(
        file_path=counter_path[0],
        initial_value=initial_value
    )
    
    logger.info(f"Rank {rank}/{world_size} using global counter: {counter.file_path}")
    return counter

def reset_global_counter(counter: FileGlobalCounter) -> None:
    """Reset global counter across all distributed processes."""
    if not (dist.is_available() and dist.is_initialized()):
        counter.reset()
        return
    
    rank = dist.get_rank()
    
    # Only master process resets the counter
    if rank == 0:
        counter.reset()
        logger.info("Global counter reset by master process")
    
    # Synchronize all processes
    dist.barrier()

# Convenience functions for common use cases
def setup_rollout_counter(work_dir: str = "./") -> FileGlobalCounter:
    """Set up rollout counter for distributed training."""
    return setup_global_counter("rollout", initial_value=0, work_dir=work_dir)

def setup_batch_counter(work_dir: str = "./") -> FileGlobalCounter:
    """Set up batch counter for distributed training."""
    return setup_global_counter("batch", initial_value=0, work_dir=work_dir)

def setup_episode_counter(work_dir: str = "./") -> FileGlobalCounter:
    """Set up episode counter for distributed training."""
    return setup_global_counter("episode", initial_value=0, work_dir=work_dir)

if __name__ == "__main__":
    # Simple test
    with FileGlobalCounter() as counter:
        print(f"Initial value: {counter.get()}")
        print(f"After increment: {counter.increment()}")
        print(f"After adding 5: {counter.add(5)}")
        print(f"Final value: {counter.get()}")