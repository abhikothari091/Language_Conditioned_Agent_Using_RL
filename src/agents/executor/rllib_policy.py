# =============================================================================
# RLlib Policy Configuration
# =============================================================================
"""
RLlib setup for MiniGrid environments.

This module provides:
- Custom RLlib environment wrapper
- PPO configuration
- Custom model registration

RLlib Basics:
-------------
RLlib uses a few key concepts:

1. ALGORITHM: The RL method (PPO, A2C, DQN, etc.)
   - We use PPO (Proximal Policy Optimization)
   - Why? Stable, sample-efficient, works well for control

2. ENVIRONMENT: Where the agent learns
   - Must be Gymnasium-compatible
   - RLlib creates multiple copies for parallel rollouts

3. MODEL: The neural network
   - Policy: π(a|s) - probability of action given state
   - Value: V(s) - expected return from state

4. TRAINER: Orchestrates the training loop
   - Collects rollouts
   - Computes advantages
   - Updates policy

PPO Algorithm Overview:
-----------------------
PPO is a policy gradient method that:
1. Collects trajectories using current policy
2. Computes advantages (how much better was action than average)
3. Updates policy with clipped objective (prevents too-large updates)

Key hyperparameters:
- lr: Learning rate
- gamma: Discount factor (how much to value future rewards)
- clip_param: PPO clipping (typically 0.2)
- num_sgd_iter: SGD passes per batch
"""

from typing import Any, Dict, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.env_context import EnvContext
    RLLIB_AVAILABLE = True
except ImportError:
    RLLIB_AVAILABLE = False
    print("Warning: RLlib not installed. Run: pip install 'ray[rllib]'")


# =============================================================================
# RLlib Environment Wrapper
# =============================================================================
class MiniGridRLlibEnv(gym.Env):
    """
    RLlib-compatible wrapper for MiniGrid with FLAT observation space.
    
    Key Design Decision:
    --------------------
    RLlib's default encoders don't support Dict observation spaces.
    We flatten the observation to a single Box for compatibility:
    - Image: 7x7x3 = 147 values (normalized to [0,1])
    - Direction: 4 values (one-hot encoded)
    - Total: 151 values
    
    This allows using RLlib's built-in MLP encoder without custom models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize from RLlib config.
        
        RLlib passes configuration as a dict.
        This is how we customize the environment per experiment.
        """
        super().__init__()
        
        # Import here to avoid circular imports
        from src.environment.minigrid_wrapper import MiniGridWrapper
        
        # Extract config
        env_name = config.get("env_name", "BabyAI-GoToObj-v0")
        max_steps = config.get("max_steps", 64)
        use_dense_reward = config.get("use_dense_reward", True)
        seed = config.get("seed", None)
        
        # Create wrapped environment
        self.env = MiniGridWrapper(
            env_name=env_name,
            max_steps=max_steps,
            use_dense_reward=use_dense_reward,
            seed=seed,
        )
        
        # FLAT observation space for RLlib compatibility
        # 7*7*3 (image) + 4 (one-hot direction) = 151
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(151,), dtype=np.float32
        )
        
        # Action space
        self.action_space = self.env.action_space
        
        # Store instruction for reference
        self._current_instruction = ""
    
    def _flatten_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Convert MiniGrid dict observation to flat vector.
        
        Returns:
            np.ndarray of shape (151,):
                - First 147: Flattened 7x7x3 image, normalized to [0,1]
                - Last 4: One-hot encoded direction
        """
        # Normalize image to [0, 1] (max value in MiniGrid is ~10)
        image = obs["image"].flatten().astype(np.float32) / 10.0
        
        # One-hot encode direction (0-3)
        direction = np.zeros(4, dtype=np.float32)
        direction[obs["direction"]] = 1.0
        
        return np.concatenate([image, direction])
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Reset environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._current_instruction = obs.get("instruction", "")
        return self._flatten_obs(obs), info
    
    def step(self, action: int):
        """Take a step."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info
    
    @property
    def instruction(self) -> str:
        """Get the current instruction."""
        return self._current_instruction
    
    def render(self):
        """Render the environment."""
        return self.env.render()


# =============================================================================
# PPO Configuration
# =============================================================================
def create_ppo_config(
    env_name: str = "BabyAI-GoToObj-v0",
    num_workers: int = 2,
    num_envs_per_worker: int = 4,
    train_batch_size: int = 2048,
    lr: float = 3e-4,
    gamma: float = 0.99,
    clip_param: float = 0.2,
    num_sgd_iter: int = 10,
    framework: str = "torch",
    **kwargs,
) -> "PPOConfig":
    """
    Create a PPO configuration for training.
    
    Parameters:
    -----------
    env_name : str
        BabyAI environment name
    num_workers : int
        Number of parallel workers for rollout collection
        More workers = faster data collection = faster training
    num_envs_per_worker : int
        Environments per worker (vectorized)
    train_batch_size : int
        Number of timesteps per training batch
    lr : float
        Learning rate
    gamma : float
        Discount factor
    clip_param : float
        PPO clipping parameter
    num_sgd_iter : int
        SGD iterations per batch
    framework : str
        Deep learning framework ("torch" or "tf")
        
    Returns:
    --------
    PPOConfig
        RLlib PPO configuration object
    """
    if not RLLIB_AVAILABLE:
        raise ImportError("RLlib not installed. Run: pip install 'ray[rllib]'")
    
    config = (
        PPOConfig()
        .environment(
            env=MiniGridRLlibEnv,
            env_config={
                "env_name": env_name,
                "max_steps": 64,
                "use_dense_reward": True,
            },
        )
        .framework(framework)
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=num_envs_per_worker,
        )
        .training(
            train_batch_size=train_batch_size,
            lr=lr,
            gamma=gamma,
            clip_param=clip_param,
            num_sgd_iter=num_sgd_iter,
            # PPO-specific
            use_gae=True,  # Generalized Advantage Estimation
            lambda_=0.95,  # GAE lambda
            vf_loss_coeff=0.5,  # Value function loss weight
            entropy_coeff=0.01,  # Entropy bonus (encourages exploration)
        )
        .resources(
            num_gpus=0,  # Set to 1 if using GPU
        )
    )
    
    return config


def create_trainer(config: "PPOConfig"):
    """
    Create a PPO trainer from config.
    
    Returns an Algorithm object that can be used for training.
    """
    if not RLLIB_AVAILABLE:
        raise ImportError("RLlib not installed")
    
    # Initialize Ray if not already
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Build the algorithm
    algo = config.build()
    
    return algo


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    print("Testing RLlib configuration...")
    print()
    
    if not RLLIB_AVAILABLE:
        print("RLlib not available. Skipping test.")
        print("Install with: pip install 'ray[rllib]'")
    else:
        # Initialize Ray
        import ray
        ray.init(ignore_reinit_error=True, num_cpus=2)
        
        print("=== Testing Environment ===")
        env = MiniGridRLlibEnv({
            "env_name": "BabyAI-GoToObj-v0",
            "max_steps": 64,
        })
        
        obs, info = env.reset(seed=42)
        print(f"Observation keys: {obs.keys()}")
        print(f"Image shape: {obs['image'].shape}")
        print(f"Direction: {obs['direction']}")
        print(f"Subgoal: {obs['subgoal']}")
        print()
        
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        print(f"After action {action}: reward={reward:.3f}, done={done}")
        print()
        
        print("=== Testing PPO Config ===")
        config = create_ppo_config(
            env_name="BabyAI-GoToObj-v0",
            num_workers=0,  # Use 0 for testing
        )
        print(f"Config created: {type(config)}")
        print()
        
        # Note: Building the full trainer is slow, skip in quick test
        print("Skipping trainer build (slow). To test:")
        print("  algo = create_trainer(config)")
        print("  result = algo.train()")
        print()
        
        ray.shutdown()
        
        print("✓ RLlib configuration test passed!")
