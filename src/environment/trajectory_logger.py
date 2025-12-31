# =============================================================================
# Trajectory Logger
# =============================================================================
"""
Logging utilities for recording agent trajectories.

This module provides:
- TrajectoryLogger: Records episodes to JSONL files
- Trajectory dataclasses: Structured data for steps and episodes

Why Log Trajectories?
---------------------
1. DEBUGGING: Replay failed episodes to understand what went wrong
2. BEHAVIOR CLONING: Use logged expert trajectories to pre-train
3. ANALYSIS: Study agent behavior patterns across many episodes
4. DEMONSTRATIONS: Show the agent's decision-making process

Storage Format: JSONL (JSON Lines)
-----------------------------------
Each line is a complete JSON object representing one episode.
Why JSONL over CSV or Pickle?
- Human-readable (can open in text editor)
- Append-friendly (no need to load whole file to add)
- Language-agnostic (can read from Python, JS, etc.)
- Streaming-friendly (process line-by-line for large files)
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class Step:
    """
    A single step in a trajectory.
    
    Attributes:
    -----------
    observation : dict
        The observation at this step (image encoded as list)
    action : int
        The action taken (0-6)
    action_name : str
        Human-readable action name
    reward : float
        Reward received after taking this action
    terminated : bool
        Whether the episode ended after this step
    truncated : bool
        Whether the episode was truncated (max steps)
    info : dict
        Additional info from the environment
    """
    observation: Dict[str, Any]
    action: int
    action_name: str
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class Episode:
    """
    A complete episode (trajectory).
    
    Attributes:
    -----------
    episode_id : int
        Unique identifier for this episode
    instruction : str
        The natural language instruction
    env_name : str
        Name of the environment
    steps : list
        List of Step objects
    success : bool
        Whether the task was completed successfully
    total_reward : float
        Sum of rewards over the episode
    num_steps : int
        Number of steps taken
    timestamp : str
        When the episode was recorded
    metadata : dict
        Additional metadata (seed, agent version, etc.)
    """
    episode_id: int
    instruction: str
    env_name: str
    steps: List[Step]
    success: bool
    total_reward: float
    num_steps: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert Step objects to dicts
        d["steps"] = [s if isinstance(s, dict) else asdict(s) for s in self.steps]
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Episode":
        """Create Episode from dictionary."""
        steps = [Step(**s) if isinstance(s, dict) else s for s in d.get("steps", [])]
        d["steps"] = steps
        return cls(**d)


class TrajectoryLogger:
    """
    Logger for recording and replaying agent trajectories.
    
    This class handles:
    - Recording episodes step-by-step
    - Saving to JSONL files
    - Loading and replaying saved episodes
    - Filtering and querying saved data
    
    Example:
    --------
    >>> logger = TrajectoryLogger("data/trajectories")
    >>> 
    >>> # Start recording an episode
    >>> logger.start_episode(env_name="BabyAI-GoToObj-v0", instruction="go to red ball")
    >>> 
    >>> # Record each step
    >>> for obs, action, reward, terminated, truncated, info in episode_data:
    ...     logger.log_step(obs, action, reward, terminated, truncated, info)
    >>> 
    >>> # End and save the episode
    >>> episode = logger.end_episode(success=True)
    >>> 
    >>> # Later, load all episodes
    >>> episodes = logger.load_all()
    """
    
    def __init__(
        self,
        save_dir: str = "data/trajectories",
        filename: str = "trajectories.jsonl",
    ):
        """
        Initialize the trajectory logger.
        
        Parameters:
        -----------
        save_dir : str
            Directory to save trajectory files
        filename : str
            Name of the JSONL file
        """
        self.save_dir = Path(save_dir)
        self.filepath = self.save_dir / filename
        
        # Create directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Current episode being recorded
        self._current_episode: Optional[Dict[str, Any]] = None
        self._current_steps: List[Step] = []
        self._episode_counter = self._get_next_episode_id()
    
    def _get_next_episode_id(self) -> int:
        """Get the next episode ID based on existing file."""
        if not self.filepath.exists():
            return 0
        
        # Count existing episodes
        count = 0
        with open(self.filepath, "r") as f:
            for _ in f:
                count += 1
        return count
    
    def start_episode(
        self,
        env_name: str,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Start recording a new episode.
        
        Parameters:
        -----------
        env_name : str
            Name of the environment
        instruction : str
            The task instruction
        metadata : dict, optional
            Additional metadata to store
            
        Returns:
        --------
        int
            Episode ID
        """
        self._current_episode = {
            "episode_id": self._episode_counter,
            "env_name": env_name,
            "instruction": instruction,
            "metadata": metadata or {},
        }
        self._current_steps = []
        
        return self._episode_counter
    
    def log_step(
        self,
        observation: Dict[str, Any],
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a single step in the current episode.
        
        Parameters:
        -----------
        observation : dict
            The observation (will be converted for JSON)
        action : int
            Action taken
        reward : float
            Reward received
        terminated : bool
            Episode ended naturally
        truncated : bool
            Episode truncated (max steps)
        info : dict, optional
            Additional info
        """
        if self._current_episode is None:
            raise RuntimeError("No episode started! Call start_episode first.")
        
        # Convert observation for JSON serialization
        json_obs = self._convert_observation(observation)
        
        # Get action name from info if available
        action_name = info.get("action_name", f"action_{action}") if info else f"action_{action}"
        
        # Clean info for JSON (remove non-serializable items)
        clean_info = self._clean_for_json(info or {})
        
        step = Step(
            observation=json_obs,
            action=action,
            action_name=action_name,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=clean_info,
        )
        
        self._current_steps.append(step)
    
    def end_episode(self, success: bool) -> Episode:
        """
        End the current episode and save it.
        
        Parameters:
        -----------
        success : bool
            Whether the task was completed successfully
            
        Returns:
        --------
        Episode
            The completed episode object
        """
        if self._current_episode is None:
            raise RuntimeError("No episode to end! Call start_episode first.")
        
        # Calculate totals
        total_reward = sum(step.reward for step in self._current_steps)
        num_steps = len(self._current_steps)
        
        # Create episode object
        episode = Episode(
            episode_id=self._current_episode["episode_id"],
            instruction=self._current_episode["instruction"],
            env_name=self._current_episode["env_name"],
            steps=self._current_steps,
            success=success,
            total_reward=total_reward,
            num_steps=num_steps,
            metadata=self._current_episode["metadata"],
        )
        
        # Save to file
        self._save_episode(episode)
        
        # Update counter and reset state
        self._episode_counter += 1
        self._current_episode = None
        self._current_steps = []
        
        return episode
    
    def _save_episode(self, episode: Episode) -> None:
        """Append episode to JSONL file."""
        with open(self.filepath, "a") as f:
            json.dump(episode.to_dict(), f)
            f.write("\n")
    
    def _convert_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert observation to JSON-serializable format.
        
        NumPy arrays are converted to lists (with optional compression).
        """
        result = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                # For images, we store shape and flatten
                # In production, you might want to compress or store separately
                result[key] = {
                    "type": "ndarray",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "data": value.flatten().tolist(),
                }
            else:
                result[key] = value
        return result
    
    def _clean_for_json(self, obj: Any) -> Any:
        """Recursively clean object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items() 
                    if self._is_serializable(v)}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _is_serializable(self, obj: Any) -> bool:
        """Check if object can be JSON serialized."""
        try:
            json.dumps(self._clean_for_json(obj))
            return True
        except (TypeError, ValueError):
            return False
    
    def load_all(self) -> List[Episode]:
        """
        Load all episodes from the file.
        
        Returns:
        --------
        List[Episode]
            All saved episodes
        """
        if not self.filepath.exists():
            return []
        
        episodes = []
        with open(self.filepath, "r") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    episodes.append(Episode.from_dict(d))
        
        return episodes
    
    def load_successful(self) -> List[Episode]:
        """Load only successful episodes (for behavior cloning)."""
        return [ep for ep in self.load_all() if ep.success]
    
    def load_by_env(self, env_name: str) -> List[Episode]:
        """Load episodes from a specific environment."""
        return [ep for ep in self.load_all() if ep.env_name == env_name]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about saved trajectories.
        
        Returns:
        --------
        dict
            Statistics including counts, success rate, avg steps, etc.
        """
        episodes = self.load_all()
        
        if not episodes:
            return {"total_episodes": 0}
        
        successful = [ep for ep in episodes if ep.success]
        
        return {
            "total_episodes": len(episodes),
            "successful_episodes": len(successful),
            "success_rate": len(successful) / len(episodes),
            "avg_steps": sum(ep.num_steps for ep in episodes) / len(episodes),
            "avg_reward": sum(ep.total_reward for ep in episodes) / len(episodes),
            "envs": list(set(ep.env_name for ep in episodes)),
        }


# =============================================================================
# Compact Logger (for high-throughput training)
# =============================================================================

class CompactTrajectoryLogger:
    """
    Lightweight logger that stores only essential data.
    
    Use this during training when you need to log many episodes quickly
    but don't need full observation data.
    
    Stores:
    - Episode ID, instruction, success, total reward, num steps
    - Does NOT store full observations (much smaller file size)
    """
    
    def __init__(
        self,
        save_dir: str = "data/trajectories",
        filename: str = "training_log.jsonl",
    ):
        self.save_dir = Path(save_dir)
        self.filepath = self.save_dir / filename
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._episode_counter = 0
    
    def log_episode(
        self,
        instruction: str,
        success: bool,
        total_reward: float,
        num_steps: int,
        env_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a completed episode (compact format)."""
        entry = {
            "id": self._episode_counter,
            "instruction": instruction,
            "success": success,
            "reward": total_reward,
            "steps": num_steps,
            "env": env_name,
            "ts": datetime.now().isoformat(),
            **(metadata or {}),
        }
        
        with open(self.filepath, "a") as f:
            json.dump(entry, f)
            f.write("\n")
        
        self._episode_counter += 1


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    import tempfile
    
    print("Testing TrajectoryLogger...")
    print()
    
    # Create logger in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TrajectoryLogger(save_dir=tmpdir)
        
        # Record a fake episode
        logger.start_episode(
            env_name="BabyAI-GoToObj-v0",
            instruction="go to the red ball",
            metadata={"seed": 42, "agent": "random"},
        )
        
        # Log some fake steps
        for i in range(5):
            obs = {
                "image": np.random.rand(7, 7, 3),
                "direction": i % 4,
                "instruction": "go to the red ball",
            }
            logger.log_step(
                observation=obs,
                action=i % 7,
                reward=0.0 if i < 4 else 1.0,
                terminated=(i == 4),
                truncated=False,
                info={"action_name": f"action_{i % 7}"},
            )
        
        # End episode
        episode = logger.end_episode(success=True)
        
        print(f"Recorded episode {episode.episode_id}")
        print(f"  Instruction: {episode.instruction}")
        print(f"  Steps: {episode.num_steps}")
        print(f"  Success: {episode.success}")
        print(f"  Total reward: {episode.total_reward}")
        print()
        
        # Record another episode
        logger.start_episode(
            env_name="BabyAI-GoToObj-v0",
            instruction="go to the blue key",
        )
        for i in range(10):
            logger.log_step(
                observation={"image": np.zeros((7, 7, 3)), "direction": 0},
                action=2,
                reward=0.0,
                terminated=False,
                truncated=(i == 9),
                info={},
            )
        episode2 = logger.end_episode(success=False)
        
        # Get stats
        stats = logger.get_stats()
        print("Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()
        
        # Load successful only
        successful = logger.load_successful()
        print(f"Successful episodes: {len(successful)}")
    
    print()
    print("✓ TrajectoryLogger test passed!")
