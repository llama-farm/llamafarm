"""Training pipeline for vision models.

Provides:
- IncrementalTrainer: Fine-tuning with continual learning
- ReplayBuffer: Experience replay for corrections
- Training job management
"""

from .replay_buffer import ReplayBuffer, ReplaySample
from .trainer import IncrementalTrainer, TrainingConfig, TrainingJob

__all__ = [
    "IncrementalTrainer",
    "TrainingJob",
    "TrainingConfig",
    "ReplayBuffer",
    "ReplaySample",
]
