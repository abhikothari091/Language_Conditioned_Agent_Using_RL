# =============================================================================
# Experiment Configuration
# =============================================================================
"""
Configuration management for experiments.

This module provides:
- YAML config loading
- Default configurations
- Experiment tracking setup

Why YAML Configs?
-----------------
1. REPRODUCIBILITY: Exact settings saved with each run
2. VERSIONING: Can track config changes in git
3. FLEXIBILITY: Override settings without code changes
4. DOCUMENTATION: Configs serve as documentation

Example config:
---------------
```yaml
experiment:
  name: "baseline_ppo"
  seed: 42

environment:
  name: "BabyAI-GoToObj-v0"
  max_steps: 64

training:
  bc_epochs: 50
  ppo_iterations: 500
  batch_size: 2048
  learning_rate: 3e-4
```
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    name: str = "BabyAI-GoToObj-v0"
    max_steps: int = 64
    use_dense_reward: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Grid encoder
    cnn_channels: tuple = (32, 64, 64)
    grid_output_dim: int = 128
    
    # Subgoal encoder
    subgoal_dim: int = 32
    
    # Policy head
    hidden_dim: int = 256
    output_dim: int = 256


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Behavior cloning
    bc_epochs: int = 50
    bc_batch_size: int = 64
    bc_learning_rate: float = 1e-3
    
    # PPO
    ppo_iterations: int = 500
    train_batch_size: int = 2048
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_param: float = 0.2
    num_sgd_iter: int = 10
    entropy_coeff: float = 0.01
    
    # Workers
    num_workers: int = 2
    num_envs_per_worker: int = 4


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.
    
    Combines all sub-configs into one object.
    Can be loaded from YAML or created programmatically.
    """
    # Experiment metadata
    name: str = "experiment"
    seed: int = 42
    
    # Sub-configs
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    checkpoint_dir: str = "experiments/checkpoints"
    log_dir: str = "experiments/logs"
    
    # Tracking
    use_wandb: bool = False
    wandb_project: str = "language_conditioned_agent"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        # Handle nested configs
        if "environment" in d and isinstance(d["environment"], dict):
            d["environment"] = EnvironmentConfig(**d["environment"])
        if "model" in d and isinstance(d["model"], dict):
            d["model"] = ModelConfig(**d["model"])
        if "training" in d and isinstance(d["training"], dict):
            d["training"] = TrainingConfig(**d["training"])
        
        return cls(**d)


def load_config(path: str) -> ExperimentConfig:
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    path : str
        Path to YAML config file
        
    Returns:
    --------
    ExperimentConfig
        Loaded configuration
    """
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    
    return ExperimentConfig.from_dict(d)


def create_default_config(name: str = "default") -> ExperimentConfig:
    """Create a default configuration."""
    return ExperimentConfig(name=name)


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    import tempfile
    
    print("Testing experiment configuration...")
    print()
    
    # Create default config
    config = create_default_config("test_experiment")
    
    print("=== Default Config ===")
    print(f"Name: {config.name}")
    print(f"Seed: {config.seed}")
    print(f"Environment: {config.environment.name}")
    print(f"Max steps: {config.environment.max_steps}")
    print(f"PPO iterations: {config.training.ppo_iterations}")
    print(f"Learning rate: {config.training.learning_rate}")
    print()
    
    # Save and reload
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        config.save(f.name)
        print(f"Saved to: {f.name}")
        
        # Load back
        loaded = load_config(f.name)
        print(f"Loaded name: {loaded.name}")
        assert loaded.name == config.name
        assert loaded.training.learning_rate == config.training.learning_rate
    
    print()
    print("✓ Experiment configuration test passed!")
