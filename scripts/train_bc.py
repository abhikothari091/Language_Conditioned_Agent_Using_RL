#!/usr/bin/env python3
# =============================================================================
# Train Behavior Cloning Agent
# =============================================================================
"""
Train a behavior cloning agent from expert demonstrations.

This script:
1. Generates expert demonstrations using BabyAI bot
2. Trains a BC policy
3. Evaluates the policy

Usage:
------
# Generate demos and train
python scripts/train_bc.py

# Custom settings
python scripts/train_bc.py --env BabyAI-GoToLocal-v0 --num-demos 2000 --epochs 100
"""

import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train BC agent on BabyAI")
    
    # Environment
    parser.add_argument(
        "--env",
        type=str,
        default="BabyAI-GoToObj-v0",
        help="Environment name (default: BabyAI-GoToObj-v0)"
    )
    
    # Demonstrations
    parser.add_argument(
        "--num-demos",
        type=int,
        default=1000,
        help="Number of demonstrations to generate (default: 1000)"
    )
    parser.add_argument(
        "--skip-demo-gen",
        action="store_true",
        help="Skip demo generation (use existing demos)"
    )
    
    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of BC training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    
    # Experiment
    parser.add_argument(
        "--name",
        type=str,
        default="bc_baseline",
        help="Experiment name (default: bc_baseline)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    return parser.parse_args()


def main():
    """Main BC training function."""
    args = parse_args()
    
    print("=" * 60)
    print("Behavior Cloning Training")
    print("=" * 60)
    print(f"Environment:  {args.env}")
    print(f"Demos:        {args.num_demos}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    print()
    
    # Import here to avoid slow imports if just checking help
    from src.training.behavior_cloning import (
        BehaviorCloning,
        generate_demonstrations,
    )
    from src.environment.trajectory_logger import TrajectoryLogger
    from src.evaluation.metrics import EvaluationSuite
    
    # Set up paths
    demo_dir = "data/demos"
    checkpoint_dir = f"experiments/checkpoints/{args.name}"
    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Step 1: Generate demonstrations
    if not args.skip_demo_gen:
        print("Step 1: Generating expert demonstrations...")
        print("-" * 40)
        episodes = generate_demonstrations(
            env_name=args.env,
            num_episodes=args.num_demos,
            save_dir=demo_dir,
            seed=args.seed,
        )
        print()
    else:
        print("Step 1: Loading existing demonstrations...")
        print("-" * 40)
        logger = TrajectoryLogger(
            save_dir=demo_dir,
            filename=f"{args.env}_demos.jsonl"
        )
        episodes = logger.load_successful()
        print(f"Loaded {len(episodes)} successful episodes")
        print()
    
    # Step 2: Train BC
    print("Step 2: Training behavior cloning policy...")
    print("-" * 40)
    bc = BehaviorCloning(
        episodes=episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    history = bc.train(epochs=args.epochs)
    
    # Save model
    model_path = os.path.join(checkpoint_dir, "bc_policy.pt")
    bc.save(model_path)
    print()
    
    # Step 3: Evaluate
    print("Step 3: Evaluating BC policy...")
    print("-" * 40)
    
    suite = EvaluationSuite(env_name=args.env)
    
    def bc_policy(obs):
        return bc.predict(obs["image"], obs["direction"])
    
    metrics = suite.evaluate(bc_policy, num_episodes=100)
    
    print()
    print("=" * 60)
    print("BC Training Complete!")
    print("=" * 60)
    print(f"Final training accuracy: {history['accuracy'][-1]:.1%}")
    print(f"Evaluation success rate: {metrics['success_rate']:.1%}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
