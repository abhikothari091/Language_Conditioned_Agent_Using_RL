# =============================================================================
# Evaluation Metrics
# =============================================================================
"""
Metrics for evaluating agent performance.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False


def compute_success_rate(
    successes: List[bool],
) -> float:
    """
    Compute success rate.
    
    Parameters:
    -----------
    successes : List[bool]
        List of success flags per episode
        
    Returns:
    --------
    float
        Success rate (0.0 to 1.0)
    """
    if not successes:
        return 0.0
    return sum(successes) / len(successes)


def compute_spl(
    successes: List[bool],
    optimal_lengths: List[int],
    actual_lengths: List[int],
) -> float:
    """
    Compute SPL (Success weighted by Path Length).
    
    This metric rewards both success AND efficiency.
    An agent that succeeds but takes too many steps scores lower.
    
    Parameters:
    -----------
    successes : List[bool]
        Success flag per episode
    optimal_lengths : List[int]
        Shortest path length per episode
    actual_lengths : List[int]
        Actual path length taken per episode
        
    Returns:
    --------
    float
        SPL score (0.0 to 1.0)
    """
    if not successes:
        return 0.0
    
    total = 0.0
    for success, optimal, actual in zip(successes, optimal_lengths, actual_lengths):
        if success:
            total += optimal / max(optimal, actual)
    
    return total / len(successes)


def compute_generalization(
    train_success_rate: float,
    test_success_rate: float,
) -> Dict[str, float]:
    """
    Compute generalization metrics.
    
    Parameters:
    -----------
    train_success_rate : float
        Success rate on training tasks
    test_success_rate : float
        Success rate on test tasks
        
    Returns:
    --------
    dict
        Generalization metrics
    """
    gap = train_success_rate - test_success_rate
    
    return {
        "train_success": train_success_rate,
        "test_success": test_success_rate,
        "generalization_gap": gap,
        "generalization_ratio": test_success_rate / max(train_success_rate, 0.01),
    }


class EvaluationSuite:
    """
    Comprehensive evaluation suite for the agent.
    
    Runs the agent on test episodes and computes all metrics.
    
    Example:
    --------
    >>> suite = EvaluationSuite(env_name="BabyAI-GoToObj-v0")
    >>> metrics = suite.evaluate(policy_fn, num_episodes=100)
    >>> print(f"Success rate: {metrics['success_rate']:.1%}")
    """
    
    def __init__(
        self,
        env_name: str = "BabyAI-GoToObj-v0",
        max_steps: int = 64,
        seed: int = 42,
    ):
        """
        Initialize evaluation suite.
        
        Parameters:
        -----------
        env_name : str
            Environment to evaluate on
        max_steps : int
            Maximum steps per episode
        seed : int
            Random seed for reproducibility
        """
        self.env_name = env_name
        self.max_steps = max_steps
        self.seed = seed
    
    def evaluate(
        self,
        policy_fn: Callable[[Dict[str, Any]], int],
        num_episodes: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run evaluation.
        
        Parameters:
        -----------
        policy_fn : callable
            Function that takes observation dict and returns action
        num_episodes : int
            Number of episodes to evaluate
        verbose : bool
            Print progress
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium not installed")
        
        from src.environment.minigrid_wrapper import MiniGridWrapper
        
        # Create environment
        env = MiniGridWrapper(
            env_name=self.env_name,
            max_steps=self.max_steps,
        )
        
        # Track results
        successes = []
        rewards = []
        lengths = []
        episodes_data = []
        
        for ep in range(num_episodes):
            obs, info = env.reset(seed=self.seed + ep)
            
            done = False
            ep_reward = 0.0
            ep_length = 0
            ep_actions = []
            
            while not done:
                # Get action from policy
                action = policy_fn(obs)
                ep_actions.append(action)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                ep_reward += reward
                ep_length += 1
            
            # Record results
            success = terminated and reward > 0
            successes.append(success)
            rewards.append(ep_reward)
            lengths.append(ep_length)
            
            episodes_data.append({
                "instruction": info.get("instruction", ""),
                "success": success,
                "reward": ep_reward,
                "length": ep_length,
                "actions": ep_actions,
            })
            
            if verbose and (ep + 1) % 20 == 0:
                current_sr = compute_success_rate(successes)
                print(f"Episode {ep+1}/{num_episodes}: Success rate = {current_sr:.1%}")
        
        env.close()
        
        # Compute metrics
        metrics = {
            "success_rate": compute_success_rate(successes),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "num_episodes": num_episodes,
            "episodes": episodes_data,
        }
        
        if verbose:
            print(f"\n=== Evaluation Results ===")
            print(f"Success rate: {metrics['success_rate']:.1%}")
            print(f"Mean reward:  {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
            print(f"Mean length:  {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
        
        return metrics
    
    def compare_policies(
        self,
        policies: Dict[str, Callable],
        num_episodes: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple policies.
        
        Parameters:
        -----------
        policies : dict
            Mapping from policy name to policy function
        num_episodes : int
            Episodes per policy
            
        Returns:
        --------
        dict
            Metrics per policy
        """
        results = {}
        
        for name, policy_fn in policies.items():
            print(f"\nEvaluating: {name}")
            print("-" * 40)
            results[name] = self.evaluate(policy_fn, num_episodes, verbose=True)
        
        # Print comparison
        print("\n=== Comparison ===")
        print(f"{'Policy':<20} {'Success':<12} {'Reward':<12} {'Length':<12}")
        print("-" * 56)
        for name, metrics in results.items():
            print(f"{name:<20} {metrics['success_rate']:.1%}        "
                  f"{metrics['mean_reward']:.3f}        "
                  f"{metrics['mean_length']:.1f}")
        
        return results


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    print("Testing evaluation metrics...")
    print()
    
    # Test metrics
    successes = [True, True, False, True, False]
    optimal_lengths = [10, 15, 20, 12, 18]
    actual_lengths = [12, 15, 25, 14, 30]
    
    sr = compute_success_rate(successes)
    spl = compute_spl(successes, optimal_lengths, actual_lengths)
    
    print(f"Success rate: {sr:.1%}")
    print(f"SPL: {spl:.3f}")
    
    gen = compute_generalization(0.9, 0.7)
    print(f"Generalization gap: {gen['generalization_gap']:.2f}")
    print()
    
    # Test evaluation suite with random policy
    if GYMNASIUM_AVAILABLE:
        print("=== Testing EvaluationSuite ===")
        
        suite = EvaluationSuite(env_name="BabyAI-GoToObj-v0")
        
        # Random policy
        def random_policy(obs):
            return np.random.randint(0, 7)
        
        metrics = suite.evaluate(random_policy, num_episodes=10, verbose=True)
    
    print("\n✓ Evaluation metrics test passed!")
