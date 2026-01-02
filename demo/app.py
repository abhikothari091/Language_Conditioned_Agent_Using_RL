#!/usr/bin/env python3
"""
Interactive Demo - Flask server for the trained RL agent.

Run with: python demo/app.py
Then open: http://localhost:5000
"""

import os
import sys
import base64
from io import BytesIO
from flask import Flask, render_template, jsonify

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
import numpy as np

app = Flask(__name__)

# Global state
agent = None
env = None
current_obs = None
episode_actions = []
episode_done = False
stats = {"successes": 0, "episodes": 0}


class MiniGridFlatEnv(gym.Env):
    """Flat observation environment matching training."""
    
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


def load_agent():
    """Load the trained PPO agent."""
    global agent
    
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=2, log_to_driver=False)
    
    def env_creator(config):
        return MiniGridFlatEnv(config)
    
    register_env("MiniGridFlat-v0", env_creator)
    
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
    
    # Find checkpoint
    checkpoint_paths = [
        os.path.join(os.path.dirname(__file__), "..", "experiments", "checkpoints", "final"),
        os.path.join(os.path.dirname(__file__), "..", "checkpoints", "final"),
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            algo.restore(os.path.abspath(path))
            print(f"✓ Loaded model from {path}")
            break
    else:
        print("⚠️ No checkpoint found - using untrained model")
    
    agent = algo


def get_frame_base64():
    """Render environment and return as base64 PNG."""
    frame = env.render()
    
    from PIL import Image
    img = Image.fromarray(frame)
    # Resize for better display
    img = img.resize((400, 400), Image.NEAREST)
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reset")
def reset():
    global env, current_obs, episode_actions, episode_done
    
    env = MiniGridFlatEnv({"env_name": "BabyAI-GoToObj-v0", "max_steps": 64})
    seed = np.random.randint(0, 10000)
    current_obs, _ = env.reset(seed=seed)
    episode_actions = []
    episode_done = False
    
    return jsonify({
        "frame": get_frame_base64(),
        "instruction": env.instruction,
        "actions": [],
        "done": False,
        "success": False,
        "step": 0,
        "stats": stats,
    })


@app.route("/api/step")
def step():
    global current_obs, episode_actions, episode_done, stats
    
    if episode_done or agent is None:
        return jsonify({"error": "Episode done or agent not loaded"})
    
    # Get agent action
    action = agent.compute_single_action(current_obs)
    action_name = ACTION_NAMES[action]
    episode_actions.append(action_name)
    
    # Take step
    current_obs, reward, term, trunc, _ = env.step(action)
    done = term or trunc
    success = term and reward > 0
    
    if done:
        episode_done = True
        stats["episodes"] += 1
        if success:
            stats["successes"] += 1
    
    return jsonify({
        "frame": get_frame_base64(),
        "instruction": env.instruction,
        "actions": episode_actions,
        "done": done,
        "success": success,
        "step": len(episode_actions),
        "reward": float(reward),
        "stats": stats,
    })


@app.route("/api/stats")
def get_stats():
    return jsonify(stats)


if __name__ == "__main__":
    print("Loading agent...")
    load_agent()
    print("\n" + "="*50)
    print("🚀 Demo running at http://localhost:5001")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
