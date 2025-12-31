# =============================================================================
# Training Module
# =============================================================================
"""
Training pipelines for the language-conditioned agent.

This module provides:
- Behavior Cloning (BC) pre-training
- PPO fine-tuning
- Experiment tracking with Weights & Biases

Training Pipeline Overview:
---------------------------

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Expert Demos   │────▶│  BC Pre-train   │────▶│   PPO Fine-     │
│  (from BabyAI   │     │  (Supervised)   │     │   tune (RL)     │
│   bot)          │     │  ~10K steps     │     │   ~100K steps   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
    Demonstrations         Warm Policy          Optimized Policy
    (state, action)         ~60% SR             ~90% SR

Why Two-Stage Training?
-----------------------
1. BEHAVIOR CLONING (BC)
   - Learn from expert demonstrations
   - Supervised learning: predict expert's action
   - Gets agent to ~60% success quickly
   
2. PPO FINE-TUNING
   - Improve upon BC with trial-and-error
   - Learns from its own mistakes
   - Achieves higher performance than BC alone

This is called "Learning from Demonstrations + RL" and is 
very common in real robotics.
"""

from src.training.behavior_cloning import BehaviorCloning
from src.training.ppo_trainer import PPOTrainer, run_training
from src.training.experiment_config import ExperimentConfig, load_config

__all__ = [
    "BehaviorCloning",
    "PPOTrainer",
    "run_training",
    "ExperimentConfig",
    "load_config",
]
