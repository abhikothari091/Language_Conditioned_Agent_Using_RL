# =============================================================================
# LLM Planner Module
# =============================================================================
"""
LLM-based instruction parsing and planning.

This module uses Llama 3.2 to convert natural language instructions
into structured subgoal sequences that the RL executor can follow.

Why Use an LLM for Planning?
----------------------------
1. ZERO-SHOT: No training required for language understanding
2. COMPOSITIONAL: Handles novel instruction combinations
3. COMMON SENSE: Knows that "pick up" requires being near the object
4. INTERPRETABLE: We can see the plan before execution

Example Flow:
-------------
Input:  "Pick up the blue key, then open the yellow door"
Output: [
    {"action": "navigate_to", "target": "blue key"},
    {"action": "pickup", "target": "blue key"},
    {"action": "navigate_to", "target": "yellow door"},
    {"action": "toggle", "target": "yellow door"}  # toggle = open/close
]

Why Llama 3.2 1B/3B?
--------------------
- SMALL: Fits on M3 Mac with 4-bit quantization
- FAST: ~100ms per generation (vs 500ms+ for 7B models)
- INSTRUCT: Trained to follow instructions, not just complete text
- OPEN: No API costs, runs locally
"""

from src.agents.planner.llm_planner import LLMPlanner
from src.agents.planner.prompts import PLANNING_PROMPTS

__all__ = ["LLMPlanner", "PLANNING_PROMPTS"]
