# =============================================================================
# Behavior Cloning
# =============================================================================
"""
Behavior Cloning (BC) pre-training.

BC is a simple imitation learning algorithm:
1. Collect expert demonstrations (state, action) pairs
2. Train a policy to predict the expert's action given state
3. This is just supervised learning!

Why BC Before RL?
-----------------
1. WARM START: Agent starts competent, not random
2. SAMPLE EFFICIENCY: RL needs fewer steps to reach good performance
3. SAFETY: Agent doesn't make dangerous random actions initially
4. FASTER ITERATION: BC training is quick (<1 hour)

Limitations of BC:
------------------
1. DISTRIBUTION SHIFT: Agent sees states it never saw in demos
2. COMPOUNDING ERRORS: Small mistakes lead to very wrong states
3. SUBOPTIMAL: Expert may not be optimal, BC copies mistakes

That's why we follow BC with RL fine-tuning!

BabyAI Demonstrations:
----------------------
BabyAI includes a "bot" that can solve tasks optimally.
We use this bot to generate expert demonstrations.
This is a huge advantage - no need for human annotators!
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.environment.trajectory_logger import TrajectoryLogger, Episode


# =============================================================================
# Expert Demonstration Generator
# =============================================================================
def generate_demonstrations(
    env_name: str = "BabyAI-GoToObj-v0",
    num_episodes: int = 1000,
    save_dir: str = "data/demos",
    seed: int = 42,
) -> List[Episode]:
    """
    Generate expert demonstrations using BabyAI bot.
    
    The BabyAI bot is a hand-coded optimal agent that knows
    how to solve all BabyAI tasks. We use it to generate
    training data for behavior cloning.
    
    Parameters:
    -----------
    env_name : str
        BabyAI environment name
    num_episodes : int
        Number of episodes to generate
    save_dir : str
        Where to save demonstrations
    seed : int
        Random seed
        
    Returns:
    --------
    List[Episode]
        Generated expert episodes
    """
    import gymnasium as gym
    import minigrid  # noqa: F401
    from minigrid.core.mission import MissionSpace
    
    # Import BabyAI bot
    try:
        from minigrid.envs.babyai import BabyAIMissionSpace
        from minigrid.utils.baby_ai_bot import BabyAIBot
    except ImportError:
        print("Warning: BabyAI bot not available. Using random actions.")
        BabyAIBot = None
    
    print(f"Generating {num_episodes} expert demonstrations...")
    print(f"Environment: {env_name}")
    
    # Create environment
    env = gym.make(env_name)
    
    # Create logger
    logger = TrajectoryLogger(save_dir=save_dir, filename=f"{env_name}_demos.jsonl")
    
    episodes = []
    successful = 0
    
    np.random.seed(seed)
    
    for ep_idx in range(num_episodes):
        obs, info = env.reset(seed=seed + ep_idx)
        
        # Create bot for this episode
        if BabyAIBot is not None:
            bot = BabyAIBot(env.unwrapped)
        
        # Start logging
        logger.start_episode(
            env_name=env_name,
            instruction=env.unwrapped.mission,
            metadata={"seed": seed + ep_idx, "episode": ep_idx},
        )
        
        done = False
        step_count = 0
        max_steps = env.unwrapped.max_steps
        
        while not done and step_count < max_steps:
            # Get expert action
            if BabyAIBot is not None:
                try:
                    action = bot.replan()
                    if action is None:
                        action = env.action_space.sample()
                except Exception:
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Log step
            logger.log_step(
                observation={
                    "image": obs["image"],
                    "direction": obs["direction"],
                    "mission": env.unwrapped.mission,
                },
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info={"action_name": f"action_{action}"},
            )
            
            obs = next_obs
            step_count += 1
        
        # End episode
        success = terminated and reward > 0
        episode = logger.end_episode(success=success)
        episodes.append(episode)
        
        if success:
            successful += 1
        
        if (ep_idx + 1) % 100 == 0:
            print(f"  Generated {ep_idx + 1}/{num_episodes} episodes "
                  f"(success rate: {successful/(ep_idx+1):.1%})")
    
    print(f"Done! Success rate: {successful/num_episodes:.1%}")
    print(f"Saved to: {logger.filepath}")
    
    env.close()
    return episodes


# =============================================================================
# BC Dataset
# =============================================================================
if TORCH_AVAILABLE:
    
    class BCDataset(Dataset):
        """
        PyTorch dataset for behavior cloning.
        
        Converts episodes into (observation, action) pairs
        for supervised learning.
        """
        
        def __init__(
            self,
            episodes: List[Episode],
            include_unsuccessful: bool = False,
        ):
            """
            Initialize BC dataset.
            
            Parameters:
            -----------
            episodes : List[Episode]
                List of episodes to use
            include_unsuccessful : bool
                If False (default), only use successful episodes
            """
            self.samples = []
            
            for episode in episodes:
                if not include_unsuccessful and not episode.success:
                    continue
                
                for step in episode.steps:
                    # Extract observation
                    obs = step.observation
                    
                    # Reconstruct image from stored format
                    if isinstance(obs.get("image"), dict):
                        img_data = obs["image"]
                        image = np.array(img_data["data"]).reshape(img_data["shape"])
                    else:
                        image = np.array(obs.get("image", np.zeros((7, 7, 3))))
                    
                    direction = obs.get("direction", 0)
                    
                    self.samples.append({
                        "image": image.astype(np.float32),
                        "direction": direction,
                        "action": step.action,
                    })
            
            print(f"Created BC dataset with {len(self.samples)} samples")
        
        def __len__(self) -> int:
            return len(self.samples)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            sample = self.samples[idx]
            return {
                "image": torch.tensor(sample["image"]),
                "direction": torch.tensor(sample["direction"], dtype=torch.long),
                "action": torch.tensor(sample["action"], dtype=torch.long),
            }
    
    
    class BCPolicy(nn.Module):
        """
        Simple BC policy network.
        
        Architecture:
        - CNN for grid encoding
        - MLP for action prediction
        """
        
        def __init__(
            self,
            num_actions: int = 7,
            hidden_dim: int = 256,
        ):
            super().__init__()
            
            # Import encoder
            from src.agents.executor.observation_encoder import GridEncoder
            
            self.grid_encoder = GridEncoder(output_dim=128)
            
            # Direction embedding
            self.dir_embed = nn.Embedding(4, 16)
            
            # Policy head
            self.policy = nn.Sequential(
                nn.Linear(128 + 16, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions),
            )
        
        def forward(
            self,
            image: torch.Tensor,
            direction: torch.Tensor,
        ) -> torch.Tensor:
            """
            Forward pass.
            
            Returns logits over actions.
            """
            # Encode grid
            grid_features = self.grid_encoder(image)
            
            # Encode direction
            dir_features = self.dir_embed(direction)
            
            # Concatenate and predict
            features = torch.cat([grid_features, dir_features], dim=-1)
            logits = self.policy(features)
            
            return logits


# =============================================================================
# BC Training
# =============================================================================
class BehaviorCloning:
    """
    Behavior Cloning trainer.
    
    Example:
    --------
    >>> # Load demonstrations
    >>> logger = TrajectoryLogger("data/demos")
    >>> episodes = logger.load_successful()
    >>> 
    >>> # Train BC
    >>> bc = BehaviorCloning(episodes)
    >>> bc.train(epochs=50)
    >>> bc.save("checkpoints/bc_policy.pt")
    """
    
    def __init__(
        self,
        episodes: List[Episode],
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):
        """
        Initialize BC trainer.
        
        Parameters:
        -----------
        episodes : List[Episode]
            Training episodes
        batch_size : int
            Training batch size
        learning_rate : float
            Learning rate
        device : str
            Device to train on
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")
        
        self.device = device
        self.batch_size = batch_size
        
        # Create dataset and loader
        self.dataset = BCDataset(episodes)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        
        # Create model
        self.model = BCPolicy().to(device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {"loss": [], "accuracy": []}
    
    def train(self, epochs: int = 50) -> Dict[str, List[float]]:
        """
        Train the BC policy.
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
            
        Returns:
        --------
        dict
            Training history with loss and accuracy
        """
        print(f"Training BC for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            self.model.train()
            for batch in self.dataloader:
                # Move to device
                image = batch["image"].to(self.device)
                direction = batch["direction"].to(self.device)
                action = batch["action"].to(self.device)
                
                # Forward pass
                logits = self.model(image, direction)
                loss = self.criterion(logits, action)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item() * len(action)
                predictions = logits.argmax(dim=-1)
                total_correct += (predictions == action).sum().item()
                total_samples += len(action)
            
            # Epoch metrics
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            
            self.history["loss"].append(avg_loss)
            self.history["accuracy"].append(accuracy)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"loss={avg_loss:.4f}, accuracy={accuracy:.1%}")
        
        print("Training complete!")
        return self.history
    
    def save(self, path: str) -> None:
        """Save trained model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved BC model to: {path}")
    
    def load(self, path: str) -> None:
        """Load trained model."""
        self.model.load_state_dict(torch.load(path))
        print(f"Loaded BC model from: {path}")
    
    def predict(
        self,
        image: np.ndarray,
        direction: int,
    ) -> int:
        """
        Predict action for given observation.
        
        Parameters:
        -----------
        image : np.ndarray
            Grid observation (7, 7, 3)
        direction : int
            Agent direction (0-3)
            
        Returns:
        --------
        int
            Predicted action
        """
        self.model.eval()
        with torch.no_grad():
            image_t = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
            direction_t = torch.tensor([direction], dtype=torch.long).to(self.device)
            logits = self.model(image_t, direction_t)
            return logits.argmax(dim=-1).item()


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    print("Testing behavior cloning...")
    print()
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping BC test.")
    else:
        import tempfile
        
        # Create fake episodes for testing
        from src.environment.trajectory_logger import Episode, Step
        
        fake_episodes = []
        for i in range(10):
            steps = []
            for j in range(5):
                steps.append(Step(
                    observation={"image": np.random.rand(7, 7, 3).tolist(), "direction": j % 4},
                    action=j % 7,
                    action_name=f"action_{j % 7}",
                    reward=1.0 if j == 4 else 0.0,
                    terminated=(j == 4),
                    truncated=False,
                ))
            fake_episodes.append(Episode(
                episode_id=i,
                instruction="test instruction",
                env_name="test",
                steps=steps,
                success=True,
                total_reward=1.0,
                num_steps=5,
            ))
        
        print("=== Testing BC Training ===")
        bc = BehaviorCloning(fake_episodes, batch_size=16, learning_rate=1e-3)
        history = bc.train(epochs=5)
        
        print(f"Final loss: {history['loss'][-1]:.4f}")
        print(f"Final accuracy: {history['accuracy'][-1]:.1%}")
        
        # Test prediction
        action = bc.predict(np.random.rand(7, 7, 3).astype(np.float32), direction=0)
        print(f"Predicted action: {action}")
        
        # Test save/load
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            bc.save(f.name)
            bc.load(f.name)
        
        print()
        print("✓ Behavior cloning test passed!")
