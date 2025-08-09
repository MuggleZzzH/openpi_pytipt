"""
RIPT (Reinforcement Learning with Image-based Trajectory Prediction) for PI0.

This module provides RL fine-tuning capabilities for PI0 policies using CFG-style 
advantage-weighted optimization on LIBERO environment trajectories.
"""

from .reward_function import BinarySuccessReward

__version__ = "1.0.0"
__all__ = ["BinarySuccessReward"]