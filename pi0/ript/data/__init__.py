"""
RIPT-VLA Data Processing Module

This module provides SO100-style data processing for RIPT-VLA training,
converting trajectory data into training samples following the so100_train.py pattern.

Key Components:
- SO100StyleProcessor: Converts trajectories to training samples (L-50+1 samples per trajectory)
- TrajectoryToSampleGenerator: Manages batch processing and episode-sample mapping

Data Flow:
Episodes → Trajectory-to-Sample Conversion → Sample-based CFG Training
"""

from .so100_style_processor import SO100StyleProcessor
from .sample_generator import TrajectoryToSampleGenerator

__all__ = [
    'SO100StyleProcessor',
    'TrajectoryToSampleGenerator'
]
