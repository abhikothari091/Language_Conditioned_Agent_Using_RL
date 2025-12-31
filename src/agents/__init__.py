# =============================================================================
# Agents Module
# =============================================================================
"""
LLM Planner and RL Executor for language-conditioned control.

This module implements the hybrid architecture:

Architecture Overview:
----------------------

┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Instruction   │────▶│   LLM Planner    │────▶│   RL Executor   │
│  "go to red     │     │  (Llama 3.2)     │     │  (RLlib PPO)    │
│   ball, then    │     │                  │     │                 │
│   open door"    │     │  ↓ Generates     │     │  ↓ Executes     │
│                 │     │  Subgoals        │     │  Actions        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │                         │
                              ▼                         ▼
                        [navigate_to(        [turn_right, forward,
                         red_ball),           forward, pickup, ...]
                         pickup(),
                         navigate_to(door),
                         open()]

Why This Split?
---------------
1. LLM handles WHAT to do (language understanding)
2. RL handles HOW to do it (motor control)
3. LLM is frozen (no expensive fine-tuning)
4. RL is small and cheap to train

Subpackages:
- planner: LLM-based instruction parsing
- executor: RLlib PPO policy
"""

# These will be imported after we implement them
# from src.agents.planner import LLMPlanner
# from src.agents.executor import RLLibExecutor

__all__ = []  # Will be populated as we build modules
