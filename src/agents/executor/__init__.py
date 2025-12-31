# =============================================================================
# RL Executor Module
# =============================================================================
"""
RLlib-based policy executor for low-level control.

This module implements the RL policy that executes subgoals from the planner.

Architecture:
-------------
                    ┌─────────────────────┐
                    │    Current Subgoal   │
                    │   "navigate_to(red   │
                    │       ball)"         │
                    └──────────┬──────────┘
                               │
                               ▼
┌─────────────────┐   ┌─────────────────────┐   ┌─────────────────┐
│   Grid Image    │──▶│   Observation       │──▶│   Action        │
│    (7x7x3)      │   │   Encoder (CNN)     │   │   Head          │──▶ Action
│                 │   │   + Subgoal Embed   │   │   (Actor)       │   (0-6)
│   Direction     │──▶│                     │   │                 │
│    (one-hot)    │   │                     │   │   Value Head    │──▶ V(s)
└─────────────────┘   └─────────────────────┘   │   (Critic)      │
                                                └─────────────────┘

Why RLlib?
----------
1. PRODUCTION-GRADE: Used by Uber, Amazon, OpenAI
2. DISTRIBUTED: Can scale to many workers
3. FLEXIBLE: Supports custom models and environments
4. ALGORITHMS: PPO, A2C, DQN, SAC, and more
"""

from src.agents.executor.rllib_policy import (
    MiniGridRLlibEnv,
    create_ppo_config,
)
from src.agents.executor.observation_encoder import ObservationEncoder

__all__ = ["MiniGridRLlibEnv", "create_ppo_config", "ObservationEncoder"]
