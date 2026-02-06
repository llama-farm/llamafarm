"""Training pipeline for vision models.

Provides:
- IncrementalTrainer: Fine-tuning with continual learning
- ReplayBuffer: Experience replay for corrections
- Training job management
"""

from .trainer import IncrementalTrainer, TrainingJob, TrainingConfig
from .replay_buffer import ReplayBuffer, ReplaySample

__all__ = [
    "IncrementalTrainer",
    "TrainingJob",
    "TrainingConfig",
    "ReplayBuffer",
    "ReplaySample",
]
