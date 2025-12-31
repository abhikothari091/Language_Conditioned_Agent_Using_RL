# =============================================================================
# Environment Module
# =============================================================================
"""
Environment wrappers and utilities for MiniGrid/BabyAI.

This module provides:
- MiniGridWrapper: Custom Gymnasium wrapper with instruction support
- InstructionGenerator: BabyAI instruction sampling
- TrajectoryLogger: Episode logging in JSONL format

Why MiniGrid?
-------------
MiniGrid is a minimalist gridworld environment that's:
1. FAST: <1ms per step (vs 100ms+ for visual environments)
2. PROCEDURAL: Unlimited instruction/task combinations
3. GROUNDED: Natural language instructions are first-class
4. SIMPLE: Easy to debug and iterate quickly

The BabyAI extension adds language instructions like:
- "go to the red ball"
- "pick up the blue key, then open the yellow door"
- "put the green box next to the purple ball"
"""

from src.environment.minigrid_wrapper import MiniGridWrapper
from src.environment.trajectory_logger import TrajectoryLogger

__all__ = ["MiniGridWrapper", "TrajectoryLogger"]
