# =============================================================================
# Evaluation Module
# =============================================================================
"""
Evaluation metrics and analysis for the language-conditioned agent.

This module provides:
- Standard RL metrics (success rate, reward, episode length)
- Task-specific metrics (SPL, generalization)
- Failure mode analysis

Key Metrics Explained:
----------------------

1. SUCCESS RATE
   - % of episodes where task was completed
   - The most important metric for BabyAI
   - Target: >90% on training tasks

2. SPL (Success weighted by Path Length)
   - SPL = (1/N) Σ S_i * (L_i / max(P_i, L_i))
   - S_i: success (0 or 1)
   - L_i: shortest path length
   - P_i: actual path length
   - Measures efficiency, not just success
   - Perfect score = 1.0 (always succeeds optimally)

3. GENERALIZATION
   - Performance on unseen instruction templates
   - Tests compositional understanding
   - Example: train on "go to X", test on "pick up X"

4. SAMPLE EFFICIENCY
   - How many environment steps to reach X% success?
   - Lower is better
   - Comparing BC vs. RL vs. BC+RL
"""

from src.evaluation.metrics import (
    compute_success_rate,
    compute_spl,
    compute_generalization,
    EvaluationSuite,
)
from src.evaluation.failure_analysis import FailureAnalyzer

__all__ = [
    "compute_success_rate",
    "compute_spl",
    "compute_generalization",
    "EvaluationSuite",
    "FailureAnalyzer",
]
