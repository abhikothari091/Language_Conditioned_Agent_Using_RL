# =============================================================================
# Language-Conditioned Agent Using RL
# =============================================================================
"""
Main package for the language-conditioned RL agent.

This package implements a hybrid architecture:
- LLM Planner: Parses instructions into subgoals
- RL Executor: Learns to execute subgoals efficiently

Subpackages:
- environment: MiniGrid wrappers and trajectory logging
- agents: LLM planner and RL executor
- training: BC and PPO training pipelines
- evaluation: Metrics and analysis
"""

__version__ = "0.1.0"
__author__ = "Abhishek Kothari"

# Convenience imports (will be populated as we build modules)
# from src.environment import MiniGridWrapper
# from src.agents import LLMPlanner, RLLibExecutor
