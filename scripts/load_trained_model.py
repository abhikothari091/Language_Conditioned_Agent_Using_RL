#!/usr/bin/env python3
"""
Load and evaluate the trained PPO model from Colab.

Usage:
1. Unzip trained_model.zip to experiments/
2. Run: python scripts/load_trained_model.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import argparse


class MiniGridFlatEnv(gym.Env):
    """Flat observation environment matching Colab training."""
    
    def __init__(self, config=None):
        super().__init__()
        import minigrid
        config = config or {}
        env_name = config.get("env_name", "BabyAI-GoToObj-v0")
        max_steps = config.get("max_steps", 64)
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.env.unwrapped.max_steps = max_steps
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(151,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.instruction = ""
    
    def _flatten_obs(self, obs):
        image = obs["image"].flatten().astype(np.float32) / 10.0
        direction = np.zeros(4, dtype=np.float32)
        direction[obs["direction"]] = 1.0
        return np.concatenate([image, direction])
    
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.instruction = self.env.unwrapped.mission
        return self._flatten_obs(obs), info
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._flatten_obs(obs), reward, term, trunc, info
    
    def render(self):
        return self.env.render()


def load_model(checkpoint_path: str):
    """Load trained PPO model from checkpoint."""
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    
    ray.init(ignore_reinit_error=True, num_cpus=2)
    
    def env_creator(config):
        return MiniGridFlatEnv(config)
    
    register_env("MiniGridFlat-v0", env_creator)
    
    # Create config matching training
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="MiniGridFlat-v0",
            env_config={"env_name": "BabyAI-GoToObj-v0", "max_steps": 64},
        )
        .framework("torch")
        .env_runners(num_env_runners=0)
        .resources(num_gpus=0)
    )
    
    algo = config.build()
    algo.restore(checkpoint_path)
    print(f"✓ Loaded model from {checkpoint_path}")
    
    return algo


def evaluate(algo, num_episodes: int = 100, render: bool = False):
    """Evaluate the trained model."""
    env = MiniGridFlatEnv({"env_name": "BabyAI-GoToObj-v0", "max_steps": 64})
    
    successes = 0
    total_reward = 0
    total_steps = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        done = False
        ep_reward = 0
        steps = 0
        
        if render and ep < 5:
            print(f"\nEpisode {ep+1}: {env.instruction}")
        
        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_reward += reward
            steps += 1
            
            if render and ep < 5:
                action_names = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
                print(f"  Step {steps}: {action_names[action]}")
        
        if term and ep_reward > 0:
            successes += 1
            if render and ep < 5:
                print(f"  ✓ SUCCESS in {steps} steps!")
        else:
            if render and ep < 5:
                print(f"  ✗ Failed")
        
        total_reward += ep_reward
        total_steps += steps
    
    return {
        "success_rate": successes / num_episodes,
        "mean_reward": total_reward / num_episodes,
        "mean_steps": total_steps / num_episodes,
    }


def demo_interactive(algo):
    """Interactive demo - watch the agent solve tasks."""
    import time
    
    env = MiniGridFlatEnv({"env_name": "BabyAI-GoToObj-v0", "max_steps": 64})
    action_names = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
    
    print("\n" + "="*60)
    print("INTERACTIVE DEMO")
    print("Press Enter to step, 'q' to quit, 'r' to reset")
    print("="*60)
    
    obs, _ = env.reset(seed=42)
    print(f"\nInstruction: {env.instruction}")
    
    step = 0
    while True:
        action = algo.compute_single_action(obs)
        print(f"Step {step+1}: Agent chooses '{action_names[action]}'")
        
        obs, reward, term, trunc, _ = env.step(action)
        step += 1
        
        if term:
            if reward > 0:
                print(f"\n🎉 SUCCESS! Completed in {step} steps!")
            else:
                print(f"\n❌ Failed after {step} steps")
            break
        elif trunc:
            print(f"\n⏰ Timeout after {step} steps")
            break
        
        user_input = input("  [Enter=next, r=reset, q=quit]: ").strip().lower()
        if user_input == 'q':
            break
        elif user_input == 'r':
            obs, _ = env.reset()
            step = 0
            print(f"\nNew instruction: {env.instruction}")


def main():
    parser = argparse.ArgumentParser(description="Load and evaluate trained model")
    parser.add_argument("--checkpoint", type=str, default="experiments/checkpoints/final",
                        help="Path to checkpoint directory")
    parser.add_argument("--episodes", type=int, default=100, help="Number of eval episodes")
    parser.add_argument("--render", action="store_true", help="Show first 5 episodes")
    parser.add_argument("--interactive", action="store_true", help="Interactive demo mode")
    args = parser.parse_args()
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        # Try finding it
        for path in [
            "experiments/checkpoints/final",
            "experiments/checkpoints",
            "checkpoints/final",
            "checkpoints",
        ]:
            if os.path.exists(path):
                checkpoint_path = path
                break
    
    print(f"Loading model from: {checkpoint_path}")
    algo = load_model(checkpoint_path)
    
    if args.interactive:
        demo_interactive(algo)
    else:
        print(f"\nEvaluating on {args.episodes} episodes...")
        results = evaluate(algo, args.episodes, args.render)
        
        print("\n" + "="*50)
        print("📊 EVALUATION RESULTS")
        print("="*50)
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Mean Reward:  {results['mean_reward']:.3f}")
        print(f"Mean Steps:   {results['mean_steps']:.1f}")
        print("="*50)
    
    algo.stop()


if __name__ == "__main__":
    main()
