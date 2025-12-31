# =============================================================================
# PPO Trainer
# =============================================================================
"""
PPO (Proximal Policy Optimization) training with RLlib.

This module provides:
- PPO training loop
- Checkpoint management
- W&B experiment tracking

PPO Explained:
--------------
PPO is a policy gradient algorithm that's:
1. STABLE: Clips updates to prevent too-large changes
2. SAMPLE EFFICIENT: Reuses data with multiple SGD passes
3. SIMPLE: Easy to implement and tune

The key innovation is the "clipped objective":

L(θ) = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)

Where:
- r(θ) = π_new(a|s) / π_old(a|s)  (probability ratio)
- A = advantage (how much better was this action than average)
- ε = clip parameter (typically 0.2)

This prevents the policy from changing too much in one update.

Training Loop:
--------------
for iteration in range(num_iterations):
    1. Collect rollouts using current policy
    2. Compute advantages (using GAE)
    3. Update policy with PPO objective
    4. Log metrics, save checkpoints
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json

try:
    import ray
    from ray.rllib.algorithms.ppo import PPO, PPOConfig
    RLLIB_AVAILABLE = True
except ImportError:
    RLLIB_AVAILABLE = False

from src.training.experiment_config import ExperimentConfig


class PPOTrainer:
    """
    PPO training manager.
    
    Wraps RLlib's PPO algorithm with:
    - Easy configuration
    - Checkpoint management
    - Training logging
    
    Example:
    --------
    >>> config = ExperimentConfig(name="my_experiment")
    >>> trainer = PPOTrainer(config)
    >>> trainer.train(iterations=100)
    >>> trainer.save("checkpoints/ppo_final")
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize PPO trainer.
        
        Parameters:
        -----------
        config : ExperimentConfig
            Experiment configuration
        checkpoint_path : str, optional
            Path to load checkpoint from (for resuming)
        """
        if not RLLIB_AVAILABLE:
            raise ImportError("RLlib not installed. Run: pip install 'ray[rllib]'")
        
        self.config = config
        
        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Build PPO config
        from src.agents.executor.rllib_policy import MiniGridRLlibEnv
        
        ppo_config = (
            PPOConfig()
            .environment(
                env=MiniGridRLlibEnv,
                env_config={
                    "env_name": config.environment.name,
                    "max_steps": config.environment.max_steps,
                    "use_dense_reward": config.environment.use_dense_reward,
                },
            )
            .framework("torch")
            .env_runners(
                num_env_runners=config.training.num_workers,
                num_envs_per_env_runner=config.training.num_envs_per_worker,
            )
            .training(
                train_batch_size=config.training.train_batch_size,
                lr=config.training.learning_rate,
                gamma=config.training.gamma,
                clip_param=config.training.clip_param,
                num_sgd_iter=config.training.num_sgd_iter,
                entropy_coeff=config.training.entropy_coeff,
            )
            .resources(num_gpus=0)
        )
        
        # Build algorithm
        if checkpoint_path:
            self.algo = PPO.from_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint from: {checkpoint_path}")
        else:
            self.algo = ppo_config.build()
            print("Created new PPO algorithm")
        
        # Training history
        self.history: List[Dict[str, Any]] = []
        
        # Checkpoint dir
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        iterations: int = 100,
        checkpoint_freq: int = 10,
        log_freq: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Run PPO training.
        
        Parameters:
        -----------
        iterations : int
            Number of training iterations
        checkpoint_freq : int
            Save checkpoint every N iterations
        log_freq : int
            Log metrics every N iterations
            
        Returns:
        --------
        list
            Training history
        """
        print(f"Starting PPO training for {iterations} iterations...")
        print(f"Environment: {self.config.environment.name}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print()
        
        for i in range(iterations):
            # Train one iteration
            result = self.algo.train()
            
            # Extract key metrics
            metrics = {
                "iteration": i + 1,
                "episode_reward_mean": result.get("episode_reward_mean", 0),
                "episode_reward_max": result.get("episode_reward_max", 0),
                "episode_reward_min": result.get("episode_reward_min", 0),
                "episode_len_mean": result.get("episode_len_mean", 0),
                "episodes_this_iter": result.get("episodes_this_iter", 0),
                "timesteps_total": result.get("timesteps_total", 0),
            }
            
            # Add policy loss if available
            if "learner" in result:
                learner_stats = result["learner"].get("default_policy", {})
                metrics["policy_loss"] = learner_stats.get("policy_loss", 0)
                metrics["vf_loss"] = learner_stats.get("vf_loss", 0)
                metrics["entropy"] = learner_stats.get("entropy", 0)
            
            self.history.append(metrics)
            
            # Log progress
            if (i + 1) % log_freq == 0:
                print(f"Iter {i+1}/{iterations}: "
                      f"reward={metrics['episode_reward_mean']:.2f}, "
                      f"len={metrics['episode_len_mean']:.1f}, "
                      f"timesteps={metrics['timesteps_total']}")
            
            # Save checkpoint
            if (i + 1) % checkpoint_freq == 0:
                checkpoint_path = self.save(f"checkpoint_{i+1:05d}")
                print(f"  Saved checkpoint: {checkpoint_path}")
        
        print("\nTraining complete!")
        return self.history
    
    def save(self, name: str = "final") -> str:
        """
        Save checkpoint.
        
        Parameters:
        -----------
        name : str
            Checkpoint name
            
        Returns:
        --------
        str
            Path to saved checkpoint
        """
        checkpoint_path = self.algo.save(self.checkpoint_dir / name)
        
        # Also save history
        history_path = self.checkpoint_dir / f"{name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return checkpoint_path
    
    def evaluate(
        self,
        num_episodes: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate the trained policy.
        
        Parameters:
        -----------
        num_episodes : int
            Number of episodes to evaluate
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        print(f"Evaluating for {num_episodes} episodes...")
        
        # Use RLlib's built-in evaluation
        # For custom evaluation, we'll create our own loop
        from src.agents.executor.rllib_policy import MiniGridRLlibEnv
        
        env = MiniGridRLlibEnv({
            "env_name": self.config.environment.name,
            "max_steps": self.config.environment.max_steps,
        })
        
        total_reward = 0
        total_steps = 0
        successes = 0
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0
            ep_steps = 0
            
            while not done:
                # Get action from policy
                action = self.algo.compute_single_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_steps += 1
            
            total_reward += ep_reward
            total_steps += ep_steps
            if terminated and reward > 0:
                successes += 1
        
        metrics = {
            "success_rate": successes / num_episodes,
            "mean_reward": total_reward / num_episodes,
            "mean_episode_length": total_steps / num_episodes,
        }
        
        print(f"Results:")
        print(f"  Success rate: {metrics['success_rate']:.1%}")
        print(f"  Mean reward: {metrics['mean_reward']:.2f}")
        print(f"  Mean length: {metrics['mean_episode_length']:.1f}")
        
        return metrics
    
    def shutdown(self):
        """Clean up resources."""
        self.algo.stop()
        ray.shutdown()


def run_training(
    env_name: str = "BabyAI-GoToObj-v0",
    iterations: int = 100,
    experiment_name: str = "ppo_baseline",
    **kwargs,
) -> PPOTrainer:
    """
    Convenience function to run training.
    
    Parameters:
    -----------
    env_name : str
        Environment name
    iterations : int
        Training iterations
    experiment_name : str
        Experiment name for logging
    **kwargs
        Additional config overrides
        
    Returns:
    --------
    PPOTrainer
        Trained trainer instance
    """
    # Create config
    config = ExperimentConfig(name=experiment_name)
    config.environment.name = env_name
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config.training, key):
            setattr(config.training, key, value)
    
    # Create and run trainer
    trainer = PPOTrainer(config)
    trainer.train(iterations=iterations)
    
    return trainer


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    print("Testing PPO Trainer...")
    print()
    
    if not RLLIB_AVAILABLE:
        print("RLlib not available. Skipping test.")
        print("Install with: pip install 'ray[rllib]'")
    else:
        print("To test PPO training, run:")
        print("  python -m src.training.ppo_trainer")
        print()
        print("Or use the training script:")
        print("  python scripts/train_ppo.py --iterations 10")
        print()
        print("Note: Full training requires ~30 min to 2 hours")
        print("depending on compute resources.")
