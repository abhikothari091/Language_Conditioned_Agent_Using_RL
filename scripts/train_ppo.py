#!/usr/bin/env python3
# =============================================================================
# Train PPO Agent
# =============================================================================
"""
Train a PPO agent on BabyAI environments.

Usage:
------
# Basic training
python scripts/train_ppo.py

# Custom settings
python scripts/train_ppo.py --env BabyAI-GoToLocal-v0 --iterations 500

# Resume from checkpoint
python scripts/train_ppo.py --checkpoint experiments/checkpoints/my_exp/checkpoint_00100
"""

import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agent on BabyAI")
    
    # Environment
    parser.add_argument(
        "--env",
        type=str,
        default="BabyAI-GoToObj-v0",
        help="Environment name (default: BabyAI-GoToObj-v0)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=64,
        help="Max steps per episode (default: 64)"
    )
    
    # Training
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Training batch size (default: 2048)"
    )
    
    # Workers
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)"
    )
    
    # Experiment
    parser.add_argument(
        "--name",
        type=str,
        default="ppo_baseline",
        help="Experiment name (default: ppo_baseline)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 60)
    print("PPO Training")
    print("=" * 60)
    print(f"Environment:    {args.env}")
    print(f"Iterations:     {args.iterations}")
    print(f"Learning rate:  {args.lr}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Workers:        {args.num_workers}")
    print(f"Experiment:     {args.name}")
    print("=" * 60)
    print()
    
    # Import here to avoid slow ray import if just checking help
    from src.training.experiment_config import ExperimentConfig
    from src.training.ppo_trainer import PPOTrainer
    
    # Create config
    config = ExperimentConfig(
        name=args.name,
        seed=args.seed,
    )
    config.environment.name = args.env
    config.environment.max_steps = args.max_steps
    config.training.learning_rate = args.lr
    config.training.train_batch_size = args.batch_size
    config.training.num_workers = args.num_workers
    
    # Save config
    config_path = f"experiments/configs/{args.name}.yaml"
    config.save(config_path)
    print(f"Saved config to: {config_path}")
    
    # Create trainer
    trainer = PPOTrainer(config, checkpoint_path=args.checkpoint)
    
    try:
        # Train
        history = trainer.train(
            iterations=args.iterations,
            checkpoint_freq=max(args.iterations // 10, 1),
        )
        
        # Save final checkpoint
        final_path = trainer.save("final")
        print(f"\nSaved final checkpoint to: {final_path}")
        
        # Evaluate
        print("\nRunning evaluation...")
        metrics = trainer.evaluate(num_episodes=100)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Final success rate: {metrics['success_rate']:.1%}")
        print(f"Final mean reward:  {metrics['mean_reward']:.3f}")
        
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    main()
