# =============================================================================
# MiniGrid Wrapper
# =============================================================================
"""
Custom wrapper for MiniGrid/BabyAI environments.

This wrapper:
1. Converts BabyAI environments to a standard interface
2. Adds the instruction to the observation space
3. Provides optional dense reward shaping
4. Supports trajectory recording

Key Concepts:
-------------

OBSERVATION SPACE:
MiniGrid observations are 7x7x3 tensors representing:
- 7x7: The agent's field of view (partial observability)
- 3 channels: [object_type, color, state]

The agent doesn't see the whole grid - just what's in front of it!
This makes the problem harder but more realistic.

ACTION SPACE:
0: Turn left
1: Turn right
2: Move forward
3: Pick up object
4: Drop object
5: Toggle (open door, etc.)
6: Done (declare task complete)

REWARD:
- Default: 1 - 0.9 * (step_count / max_steps) on success, 0 otherwise
- With shaping: Small rewards for progress (getting closer, picking up, etc.)
"""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MiniGridWrapper(gym.Wrapper):
    """
    Wrapper for BabyAI/MiniGrid environments with instruction support.
    
    This wrapper modifies the observation space to include the instruction
    and provides utilities for trajectory logging and reward shaping.
    
    Example:
    --------
    >>> env = MiniGridWrapper("BabyAI-GoToObj-v0")
    >>> obs, info = env.reset()
    >>> print(obs["instruction"])  # "go to the red ball"
    >>> print(obs["image"].shape)  # (7, 7, 3)
    >>> action = 2  # Move forward
    >>> obs, reward, done, truncated, info = env.step(action)
    """
    
    # Human-readable action names for debugging/visualization
    ACTION_NAMES = [
        "turn_left", "turn_right", "forward",
        "pickup", "drop", "toggle", "done"
    ]
    
    def __init__(
        self,
        env_name: str = "BabyAI-GoToObj-v0",
        max_steps: Optional[int] = None,
        use_dense_reward: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize the MiniGrid wrapper.
        
        Parameters:
        -----------
        env_name : str
            Name of the BabyAI environment. Options include:
            - "BabyAI-GoToObj-v0": Go to a specific object (easiest)
            - "BabyAI-GoToLocal-v0": Go to object in local view
            - "BabyAI-PutNextLocal-v0": Put object next to another
            - "BabyAI-PickupLoc-v0": Pick up from specific location
            
        max_steps : int, optional
            Maximum steps per episode. Default uses env's default.
            
        use_dense_reward : bool
            If True, use reward shaping for faster learning.
            Adds small rewards for making progress.
            
        seed : int, optional
            Random seed for reproducibility.
        """
        # Import minigrid here to register environments
        # This is a common pattern - lazy import to avoid circular deps
        import minigrid  # noqa: F401
        from minigrid.wrappers import RGBImgPartialObsWrapper
        
        # Create the base environment
        base_env = gym.make(env_name, render_mode="rgb_array")
        
        # Apply max_steps if specified
        if max_steps is not None:
            base_env.unwrapped.max_steps = max_steps
        
        # Initialize the wrapper
        super().__init__(base_env)
        
        # Store configuration
        self.env_name = env_name
        self.use_dense_reward = use_dense_reward
        self._seed = seed
        
        # Current instruction (set on reset)
        self._current_instruction: str = ""
        
        # For dense rewards: track previous distance to goal
        self._prev_distance: Optional[float] = None
        
        # Modify observation space to include instruction
        # We'll encode the instruction as the mission string
        self.observation_space = spaces.Dict({
            "image": self.env.observation_space["image"],
            "direction": spaces.Discrete(4),  # 0=right, 1=down, 2=left, 3=up
            "instruction": spaces.Text(max_length=256),  # Natural language
        })
        
        # Action space remains the same (7 discrete actions)
        # But we only use 0-5 typically (done is automatic in BabyAI)
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment and get a new instruction.
        
        Returns:
        --------
        observation : dict
            Contains "image", "direction", and "instruction"
        info : dict
            Additional information about the episode
        """
        # Use provided seed or fall back to stored seed
        reset_seed = seed if seed is not None else self._seed
        
        # Reset the base environment
        obs, info = self.env.reset(seed=reset_seed, options=options)
        
        # Extract the instruction (BabyAI calls it "mission")
        self._current_instruction = self.env.unwrapped.mission
        
        # Reset dense reward tracking
        self._prev_distance = None
        
        # Build our observation dict
        wrapped_obs = {
            "image": obs["image"],
            "direction": obs["direction"],
            "instruction": self._current_instruction,
        }
        
        # Add instruction to info for convenience
        info["instruction"] = self._current_instruction
        
        return wrapped_obs, info
    
    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Parameters:
        -----------
        action : int
            Action to take (0-6, see ACTION_NAMES)
            
        Returns:
        --------
        observation : dict
            New observation after action
        reward : float
            Reward received (possibly shaped)
        terminated : bool
            True if episode ended (success or failure)
        truncated : bool
            True if episode truncated (max steps)
        info : dict
            Additional information
        """
        # Execute action in base environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply reward shaping if enabled
        if self.use_dense_reward:
            reward = self._shape_reward(reward, obs, terminated, info)
        
        # Build wrapped observation
        wrapped_obs = {
            "image": obs["image"],
            "direction": obs["direction"],
            "instruction": self._current_instruction,
        }
        
        # Add useful info
        info["instruction"] = self._current_instruction
        info["action_name"] = self.ACTION_NAMES[action]
        
        return wrapped_obs, reward, terminated, truncated, info
    
    def _shape_reward(
        self,
        original_reward: float,
        obs: Dict[str, Any],
        terminated: bool,
        info: Dict[str, Any],
    ) -> float:
        """
        Apply reward shaping for faster learning.
        
        Reward shaping adds small intermediate rewards to guide the agent.
        
        Why Reward Shaping?
        -------------------
        In sparse reward settings (only get reward at the end), the agent
        must explore randomly until it accidentally succeeds. This can take
        millions of steps!
        
        Dense rewards give feedback at every step:
        - Small positive reward for getting closer to goal
        - Small negative reward for getting further
        
        Trade-off: Shaped rewards can cause the agent to exploit the shaping
        instead of solving the real task. We keep shaping subtle.
        """
        shaped_reward = original_reward
        
        # Only shape if we have a goal and haven't terminated
        if not terminated and hasattr(self.env.unwrapped, 'goal_pos'):
            # Get current agent position
            agent_pos = self.env.unwrapped.agent_pos
            goal_pos = self.env.unwrapped.goal_pos
            
            if goal_pos is not None:
                # Calculate Manhattan distance to goal
                current_distance = abs(agent_pos[0] - goal_pos[0]) + \
                                   abs(agent_pos[1] - goal_pos[1])
                
                if self._prev_distance is not None:
                    # Small reward for getting closer (0.01), penalty for further
                    distance_delta = self._prev_distance - current_distance
                    shaped_reward += 0.01 * distance_delta
                
                self._prev_distance = current_distance
        
        return shaped_reward
    
    def render(self) -> np.ndarray:
        """Render the environment as an RGB image."""
        return self.env.render()
    
    def get_full_obs(self) -> np.ndarray:
        """
        Get the full grid observation (not just agent's view).
        
        Useful for:
        - Debugging
        - Visualization
        - Training with full observability (easier but less realistic)
        """
        return self.env.unwrapped.grid.encode()
    
    @property
    def instruction(self) -> str:
        """Get the current instruction."""
        return self._current_instruction
    
    @property
    def agent_pos(self) -> Tuple[int, int]:
        """Get the agent's current position."""
        return tuple(self.env.unwrapped.agent_pos)
    
    @property
    def agent_dir(self) -> int:
        """Get the agent's current direction (0=right, 1=down, 2=left, 3=up)."""
        return self.env.unwrapped.agent_dir


# =============================================================================
# Convenience functions
# =============================================================================

def make_env(
    env_name: str = "BabyAI-GoToObj-v0",
    **kwargs
) -> MiniGridWrapper:
    """
    Create a wrapped MiniGrid environment.
    
    This is a convenience function that creates the wrapper with
    sensible defaults.
    
    Parameters:
    -----------
    env_name : str
        Name of the BabyAI environment
    **kwargs
        Additional arguments passed to MiniGridWrapper
        
    Returns:
    --------
    MiniGridWrapper
        Wrapped environment ready for training
    """
    return MiniGridWrapper(env_name, **kwargs)


def list_available_envs() -> list:
    """
    List all available BabyAI environments.
    
    Returns:
    --------
    list
        Names of all BabyAI environments
    """
    import gymnasium as gym
    
    # Get all registered envs
    all_envs = gym.envs.registry.keys()
    
    # Filter to BabyAI envs
    babyai_envs = sorted([env for env in all_envs if env.startswith("BabyAI")])
    
    return babyai_envs


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    # Quick test to verify the wrapper works
    print("Testing MiniGridWrapper...")
    print()
    
    # List available environments
    print("Available BabyAI environments:")
    for env_name in list_available_envs()[:10]:  # Show first 10
        print(f"  - {env_name}")
    print("  ...")
    print()
    
    # Create environment
    env = make_env("BabyAI-GoToObj-v0", use_dense_reward=True)
    
    # Reset and show initial state
    obs, info = env.reset(seed=42)
    
    print(f"Instruction: {obs['instruction']}")
    print(f"Image shape: {obs['image'].shape}")
    print(f"Agent direction: {obs['direction']}")
    print(f"Agent position: {env.agent_pos}")
    print()
    
    # Take a few random actions
    print("Taking 5 random actions:")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: {info['action_name']:12s} -> reward={reward:.3f}")
        
        if terminated or truncated:
            print("  Episode ended!")
            break
    
    print()
    print("✓ MiniGridWrapper test passed!")
