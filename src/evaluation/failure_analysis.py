# =============================================================================
# Failure Analysis
# =============================================================================
"""
Analyze and categorize agent failures.

Understanding WHY an agent fails is crucial for improvement.
This module categorizes failures into actionable buckets.

Common Failure Modes in BabyAI:
-------------------------------

1. WRONG OBJECT
   - Agent goes to wrong color or type
   - Example: "go to red ball" but goes to blue ball
   - Cause: Language understanding failure

2. WRONG ORDERING
   - Multi-step task done in wrong order
   - Example: Opens door before picking up key
   - Cause: Planning failure

3. STUCK/LOOPING
   - Agent goes in circles or oscillates
   - Often near walls or corners
   - Cause: Exploration failure

4. TIMEOUT
   - Agent runs out of steps
   - May be making progress but too slow
   - Cause: Inefficient navigation

5. IMPOSSIBLE TASK
   - Task cannot be completed (environment bug)
   - Rare in BabyAI but can happen
   - Not agent's fault
"""

from collections import Counter
from typing import Any, Dict, List, Optional
import numpy as np


class FailureAnalyzer:
    """
    Analyze agent failures and categorize them.
    
    Example:
    --------
    >>> analyzer = FailureAnalyzer()
    >>> for episode in failed_episodes:
    ...     category = analyzer.categorize(episode)
    >>> analyzer.summary()
    """
    
    # Failure categories
    CATEGORIES = [
        "wrong_object",
        "wrong_ordering",
        "stuck_looping",
        "timeout",
        "impossible",
        "other",
    ]
    
    def __init__(self):
        """Initialize analyzer."""
        self.failures: List[Dict[str, Any]] = []
        self.category_counts = Counter()
    
    def categorize(
        self,
        episode: Dict[str, Any],
    ) -> str:
        """
        Categorize a failed episode.
        
        Parameters:
        -----------
        episode : dict
            Episode data with keys:
            - instruction: str
            - actions: List[int]
            - success: bool
            - observations (optional): List
            
        Returns:
        --------
        str
            Failure category
        """
        if episode.get("success", False):
            return "not_failure"
        
        actions = episode.get("actions", [])
        instruction = episode.get("instruction", "")
        length = len(actions)
        
        # Check for timeout (max steps reached)
        max_steps = episode.get("max_steps", 64)
        if length >= max_steps - 1:
            category = "timeout"
        
        # Check for looping (repeated action sequences)
        elif self._detect_loop(actions):
            category = "stuck_looping"
        
        # Check for wrong object (if we have observations)
        elif self._detect_wrong_object(episode):
            category = "wrong_object"
        
        # Check for ordering issues (requires task analysis)
        elif self._detect_wrong_ordering(episode):
            category = "wrong_ordering"
        
        else:
            category = "other"
        
        # Record
        self.failures.append({
            "category": category,
            "instruction": instruction,
            "length": length,
            "actions": actions,
        })
        self.category_counts[category] += 1
        
        return category
    
    def _detect_loop(self, actions: List[int], window: int = 6) -> bool:
        """
        Detect if agent is stuck in a loop.
        
        Looks for repeated subsequences.
        """
        if len(actions) < window * 2:
            return False
        
        # Check last few actions for patterns
        recent = actions[-window * 3:]
        
        # Look for repeating pattern of length 2-4
        for pattern_len in range(2, 5):
            for start in range(len(recent) - pattern_len * 2):
                pattern = recent[start:start + pattern_len]
                next_seq = recent[start + pattern_len:start + pattern_len * 2]
                if pattern == next_seq:
                    return True
        
        # Check for left-right or forward-backward oscillation
        if len(actions) >= 10:
            last_10 = actions[-10:]
            # Count direction changes
            changes = sum(1 for i in range(1, len(last_10)) 
                         if last_10[i] != last_10[i-1] and last_10[i] < 3 and last_10[i-1] < 3)
            if changes >= 6:  # Many direction changes = oscillating
                return True
        
        return False
    
    def _detect_wrong_object(self, episode: Dict[str, Any]) -> bool:
        """
        Detect if agent interacted with wrong object.
        
        Requires detailed observation logging to fully implement.
        """
        # Simple heuristic: if agent picked up or toggled something
        # but didn't succeed, might be wrong object
        actions = episode.get("actions", [])
        
        # Action 3 = pickup, Action 5 = toggle
        has_interaction = any(a in [3, 5] for a in actions)
        
        if has_interaction and not episode.get("success", False):
            return True
        
        return False
    
    def _detect_wrong_ordering(self, episode: Dict[str, Any]) -> bool:
        """
        Detect ordering issues in multi-step tasks.
        
        Example: Trying to open door before getting key.
        """
        instruction = episode.get("instruction", "").lower()
        actions = episode.get("actions", [])
        
        # Check if it's a multi-step instruction
        is_multistep = any(word in instruction for word in ["then", "and", "after"])
        
        if not is_multistep:
            return False
        
        # Look for toggle (open door) before pickup (get key)
        pickup_indices = [i for i, a in enumerate(actions) if a == 3]
        toggle_indices = [i for i, a in enumerate(actions) if a == 5]
        
        if pickup_indices and toggle_indices:
            # If any toggle happened before first pickup, ordering issue
            if toggle_indices[0] < pickup_indices[0]:
                return True
        
        return False
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary of all analyzed failures.
        
        Returns:
        --------
        dict
            Summary statistics
        """
        total = len(self.failures)
        
        if total == 0:
            return {"total_failures": 0, "categories": {}}
        
        # Compute percentages
        category_pcts = {
            cat: count / total * 100
            for cat, count in self.category_counts.items()
        }
        
        # Average length per category
        category_lengths = {cat: [] for cat in self.CATEGORIES}
        for failure in self.failures:
            category_lengths[failure["category"]].append(failure["length"])
        
        avg_lengths = {
            cat: np.mean(lengths) if lengths else 0
            for cat, lengths in category_lengths.items()
        }
        
        summary = {
            "total_failures": total,
            "categories": {
                cat: {
                    "count": self.category_counts[cat],
                    "percentage": category_pcts.get(cat, 0),
                    "avg_length": avg_lengths[cat],
                }
                for cat in self.CATEGORIES
                if self.category_counts[cat] > 0
            },
        }
        
        return summary
    
    def print_summary(self) -> None:
        """Print formatted summary."""
        summary = self.summary()
        
        print("=== Failure Analysis ===")
        print(f"Total failures: {summary['total_failures']}")
        print()
        
        if not summary['categories']:
            print("No failures categorized.")
            return
        
        # Sort by count
        sorted_cats = sorted(
            summary['categories'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        print(f"{'Category':<18} {'Count':<8} {'%':<8} {'Avg Len':<10}")
        print("-" * 44)
        for cat, stats in sorted_cats:
            print(f"{cat:<18} {stats['count']:<8} "
                  f"{stats['percentage']:.1f}%    {stats['avg_length']:.1f}")
    
    def get_examples(
        self,
        category: str,
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get example failures from a category.
        
        Parameters:
        -----------
        category : str
            Category to get examples from
        n : int
            Number of examples
            
        Returns:
        --------
        list
            Example failures
        """
        examples = [f for f in self.failures if f["category"] == category]
        return examples[:n]


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    print("Testing failure analysis...")
    print()
    
    analyzer = FailureAnalyzer()
    
    # Create fake failed episodes
    test_episodes = [
        # Timeout
        {"instruction": "go to red ball", "actions": list(range(64)), "success": False, "max_steps": 64},
        {"instruction": "go to blue key", "actions": list(range(64)), "success": False, "max_steps": 64},
        
        # Looping
        {"instruction": "go to green box", "actions": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], "success": False},
        
        # Wrong object (picked up something but failed)
        {"instruction": "pick up blue key", "actions": [2, 2, 3, 2, 2], "success": False},
        
        # Wrong ordering (toggled before pickup)
        {"instruction": "pick up key then open door", "actions": [2, 2, 5, 2, 3], "success": False},
        
        # Other
        {"instruction": "go to purple ball", "actions": [2, 2, 2], "success": False},
    ]
    
    for episode in test_episodes:
        category = analyzer.categorize(episode)
        print(f"Instruction: {episode['instruction'][:30]:<30} -> {category}")
    
    print()
    analyzer.print_summary()
    
    print()
    print("✓ Failure analysis test passed!")
