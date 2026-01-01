# 📚 Learning Guide: Language-Conditioned RL Agent

This document provides an in-depth explanation of everything in this project. By the end, you'll understand reinforcement learning, language grounding, and how all the components fit together.

---

## Table of Contents

1. [Core Concepts](#1-core-concepts)
2. [Project Architecture](#2-project-architecture)
3. [Environment Deep Dive](#3-environment-deep-dive)
4. [LLM Planner Explained](#4-llm-planner-explained)
5. [RL Executor & PPO](#5-rl-executor--ppo)
6. [Training Pipeline](#6-training-pipeline)
7. [Evaluation & Metrics](#7-evaluation--metrics)
8. [File-by-File Breakdown](#8-file-by-file-breakdown)
9. [Key Libraries](#9-key-libraries)
10. [Common Pitfalls & Tips](#10-common-pitfalls--tips)

---

## 1. Core Concepts

### 1.1 What is Reinforcement Learning?

**Reinforcement Learning (RL)** is learning by trial and error. An agent:
1. **Observes** the environment state
2. **Takes an action**
3. **Receives a reward** (positive or negative)
4. **Updates its behavior** to maximize future rewards

```
┌─────────────────────────────────────────────────────────┐
│                    RL Loop                              │
│                                                         │
│   ┌─────────┐    action    ┌─────────────┐             │
│   │  Agent  │──────────────▶│ Environment │             │
│   │ (Policy)│              │  (MiniGrid) │             │
│   └────▲────┘              └──────┬──────┘             │
│        │                          │                     │
│        │    observation, reward   │                     │
│        └──────────────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

**Key Terms:**
- **State (s)**: What the agent observes (e.g., grid image)
- **Action (a)**: What the agent does (e.g., move forward)
- **Reward (r)**: Feedback signal (e.g., +1 for success)
- **Policy π(a|s)**: Probability of taking action a in state s
- **Value V(s)**: Expected total reward from state s

### 1.2 What is Language Conditioning?

**Language conditioning** means the agent's behavior depends on a natural language instruction.

```
Without language conditioning:
  Agent → Always tries to reach any goal

With language conditioning:
  Instruction: "go to the RED ball"
  Agent → Specifically navigates to RED ball, ignores BLUE ball
```

This requires:
1. **Language understanding**: What does "red ball" mean?
2. **Grounding**: Mapping words to objects in the environment
3. **Planning**: Breaking "pick up X then open Y" into steps

### 1.3 Why Hybrid LLM + RL?

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| **Pure LLM** | Great language understanding | Slow inference, expensive to fine-tune |
| **Pure RL** | Fast, learns optimal control | Struggles with language, sample inefficient |
| **Hybrid** | Best of both worlds! | Slightly more complex architecture |

**Our approach:**
- **LLM (Llama 3.2)**: Parses instructions → subgoals (frozen, no training)
- **RL (PPO)**: Executes subgoals → actions (trained)

---

## 2. Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE SYSTEM ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐                                       │
│  │ Natural Language     │                                       │
│  │ Instruction          │                                       │
│  │ "pick up blue key,   │                                       │
│  │  open yellow door"   │                                       │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐     ┌────────────────────────────┐   │
│  │ LLM PLANNER          │     │ Subgoals:                  │   │
│  │ (Llama 3.2)          │────▶│ 1. navigate_to(blue key)   │   │
│  │ • Frozen weights     │     │ 2. pickup(blue key)        │   │
│  │ • Prompt engineering │     │ 3. navigate_to(yellow door)│   │
│  │ • JSON output        │     │ 4. toggle(yellow door)     │   │
│  └──────────────────────┘     └─────────────┬──────────────┘   │
│                                              │                   │
│             Current Subgoal: navigate_to(blue key)              │
│                                              │                   │
│                                              ▼                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ RL EXECUTOR (RLlib PPO)                                  │   │
│  │                                                          │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │   │
│  │  │ Grid Image  │   │ Direction   │   │ Subgoal     │    │   │
│  │  │ (7x7x3)     │   │ (one-hot)   │   │ Embedding   │    │   │
│  │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘    │   │
│  │         │                 │                  │           │   │
│  │         ▼                 ▼                  ▼           │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │            CNN + MLP Encoder                    │    │   │
│  │  └────────────────────┬────────────────────────────┘    │   │
│  │                       │                                  │   │
│  │         ┌─────────────┴─────────────┐                   │   │
│  │         ▼                           ▼                   │   │
│  │  ┌─────────────┐             ┌─────────────┐            │   │
│  │  │ Policy Head │             │ Value Head  │            │   │
│  │  │ π(a|s)      │             │ V(s)        │            │   │
│  │  └──────┬──────┘             └─────────────┘            │   │
│  │         │                                                │   │
│  └─────────┼────────────────────────────────────────────────┘   │
│            ▼                                                     │
│     ┌──────────────┐                                            │
│     │ Action       │                                            │
│     │ (0-6)        │                                            │
│     └──────┬───────┘                                            │
│            │                                                     │
│            ▼                                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ MINIGRID ENVIRONMENT                                     │   │
│  │                                                          │   │
│  │   ┌───┬───┬───┬───┬───┬───┬───┐                         │   │
│  │   │   │   │   │ 🔵 │   │   │   │  🔵 = Blue Key         │   │
│  │   ├───┼───┼───┼───┼───┼───┼───┤  🔴 = Red Ball         │   │
│  │   │   │   │   │   │   │   │   │  🚪 = Yellow Door       │   │
│  │   ├───┼───┼───┼───┼───┼───┼───┤  👤 = Agent             │   │
│  │   │   │ 🔴 │   │   │   │ 👤 │   │                         │   │
│  │   ├───┼───┼───┼───┼───┼───┼───┤                         │   │
│  │   │   │   │   │   │🚪 │   │   │                         │   │
│  │   └───┴───┴───┴───┴───┴───┴───┘                         │   │
│  │                                                          │   │
│  │   Returns: observation, reward, done                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Environment Deep Dive

### 3.1 MiniGrid Basics

**MiniGrid** is a minimalist gridworld environment designed for RL research.

**Key Properties:**
- **Fast**: <1ms per step (vs 100ms+ for visual environments)
- **Procedural**: Unlimited unique levels
- **Partial observability**: Agent only sees 7×7 area in front

### 3.2 Observation Space

The agent sees a **7×7×3 tensor**:

```python
observation["image"].shape = (7, 7, 3)
```

**Channels:**
| Channel | Meaning | Values |
|---------|---------|--------|
| 0 | Object type | 0=empty, 1=wall, 2=floor, 3=door, 4=key, 5=ball, 6=box |
| 1 | Color | 0=red, 1=green, 2=blue, 3=purple, 4=yellow, 5=grey |
| 2 | State | For doors: 0=open, 1=closed, 2=locked |

**Partial Observability:**
```
┌───────────────────────────────────┐
│ Full Grid (what exists)           │
│ ┌───┬───┬───┬───┬───┬───┬───┐    │
│ │   │   │   │ 🔵 │   │   │   │    │
│ ├───┼───┼───┼───┼───┼───┼───┤    │
│ │   │   │   │   │   │   │   │    │
│ ├───┼───┼───┼───┼───┼───┼───┤    │
│ │   │ 🔴 │   │   │   │ 👤▶│   │    │
│ ├───┼───┼───┼───┼───┼───┼───┤    │
│ │   │   │   │   │🚪 │   │   │    │
│ └───┴───┴───┴───┴───┴───┴───┘    │
│                                   │
│ Agent's View (what agent sees)    │
│         ┌───┬───┬───┐            │
│         │   │   │   │            │
│         ├───┼───┼───┤            │
│         │   │   │   │  Agent     │
│         ├───┼───┼───┤  faces     │
│         │   │   │   │  RIGHT →   │
│         ├───┼───┼───┤            │
│         │   │ 🚪 │   │            │
│         └───┴───┴───┘            │
│                                   │
│ Agent can't see the blue key!     │
│ It's behind and to the left.      │
└───────────────────────────────────┘
```

### 3.3 Action Space

| Action | ID | Effect |
|--------|-----|--------|
| Turn left | 0 | Rotate 90° counterclockwise |
| Turn right | 1 | Rotate 90° clockwise |
| Move forward | 2 | Move one cell in facing direction |
| Pick up | 3 | Pick up object in front (if any) |
| Drop | 4 | Drop held object |
| Toggle | 5 | Open/close door in front |
| Done | 6 | Declare task complete |

### 3.4 BabyAI Extension

**BabyAI** adds natural language instructions to MiniGrid:

```python
# BabyAI environment
env = gym.make("BabyAI-GoToObj-v0")
obs, info = env.reset()

print(obs["mission"])  # "go to the red ball"
```

**Task Hierarchy:**
| Environment | Difficulty | Example Instruction |
|-------------|------------|---------------------|
| GoToObj | Easy | "go to the red ball" |
| PickupLoc | Medium | "pick up the key on your left" |
| PutNextLocal | Medium | "put the box next to the ball" |
| GoToSeq | Hard | "go to the purple key, then go to the yellow door" |
| Synth | Very Hard | Compositional random instructions |

### 3.5 Our Wrapper: `minigrid_wrapper.py`

**Purpose**: Adapt MiniGrid/BabyAI for our training pipeline.

```python
class MiniGridWrapper(gym.Wrapper):
    """
    Adds:
    1. Instruction in observation dict
    2. Optional dense reward shaping
    3. Trajectory recording support
    """
```

**Code Walkthrough:**

```python
# Creating the wrapper
env = MiniGridWrapper(
    env_name="BabyAI-GoToObj-v0",  # Which environment
    max_steps=64,                   # Episode length limit
    use_dense_reward=True,          # Add shaped rewards
)

# Using it
obs, info = env.reset(seed=42)
# obs = {
#     "image": numpy array (7, 7, 3),
#     "direction": 0-3 (which way agent faces),
#     "instruction": "go to the red ball"
# }

obs, reward, done, truncated, info = env.step(action=2)  # Move forward
```

**Dense Reward Shaping:**
```python
def _shape_reward(self, original_reward, obs, terminated, info):
    """
    Problem: Sparse rewards only at episode end
    Solution: Small rewards for making progress
    
    +0.01 for getting closer to goal
    -0.01 for getting further from goal
    """
    # Calculate distance to goal
    current_distance = manhattan_distance(agent_pos, goal_pos)
    
    # Reward for getting closer
    shaped_reward = 0.01 * (prev_distance - current_distance)
    
    return original_reward + shaped_reward
```

---

## 4. LLM Planner Explained

### 4.1 Why Use an LLM?

**Problem**: RL agents don't naturally understand language.

**Traditional approach**: 
- Train end-to-end (slow, data-hungry)
- Hard-code instruction parsing (brittle)

**Our approach**: 
- Use pre-trained LLM for language (zero-shot)
- LLM outputs structured subgoals
- RL learns to execute subgoals

### 4.2 Llama 3.2 Setup

**Model Choice**: `meta-llama/Llama-3.2-1B-Instruct`

**Why this model?**
- **1B parameters**: Fits in ~4GB RAM with float16
- **Instruction-tuned**: Follows prompts well
- **Open source**: No API costs
- **MPS support**: Runs on M3 Mac GPU

**Loading the Model:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.float16,  # Half precision for Mac
    device_map="auto",           # Automatically use MPS
)
# Model loaded on: mps:0 (Metal GPU)
```

### 4.3 Prompt Engineering

**Prompt engineering** is crafting the input to get desired output.

**Our System Prompt** (`prompts.py`):
```python
SYSTEM_PROMPT = """You are a robot navigation planner in a grid world.

Your job is to convert natural language instructions into a sequence of subgoals.

Available actions:
- navigate_to(object): Move to be adjacent to the specified object
- pickup(object): Pick up an object (must be adjacent to it)
- drop(): Drop the currently held object
- toggle(object): Open/close a door (must be adjacent to it)

Rules:
1. To pick something up, you must first navigate to it
2. To open a door, you must first navigate to it
3. Some doors require keys - you must pick up the matching color key first

Always respond with a valid JSON array of subgoal objects."""
```

**Few-Shot Examples:**
```python
FEW_SHOT_EXAMPLES = [
    {
        "instruction": "go to the red ball",
        "plan": [{"action": "navigate_to", "target": "red ball"}]
    },
    {
        "instruction": "pick up the blue key",
        "plan": [
            {"action": "navigate_to", "target": "blue key"},
            {"action": "pickup", "target": "blue key"}
        ]
    },
    # ... more examples
]
```

**Why Few-Shot?**
- Shows LLM the exact format we want
- Dramatically improves accuracy
- Acts as "in-context learning"

### 4.4 Planner Code Walkthrough

```python
class LLMPlanner:
    def plan(self, instruction: str) -> List[Subgoal]:
        """
        Convert instruction to subgoal sequence.
        
        Input:  "pick up the green key and open the green door"
        Output: [
            Subgoal("navigate_to", "green key"),
            Subgoal("pickup", "green key"),
            Subgoal("navigate_to", "green door"),
            Subgoal("toggle", "green door")
        ]
        """
        # 1. Build chat messages with examples
        messages = build_chat_messages(instruction)
        
        # 2. Generate with LLM
        outputs = self.pipeline(messages, max_new_tokens=256)
        
        # 3. Parse JSON response
        subgoals = self._parse_plan(outputs)
        
        return subgoals
```

### 4.5 Fallback: Rule-Based Planner

For testing without LLM, we have `RuleBasedPlanner`:

```python
class RuleBasedPlanner:
    """Pattern matching for common instructions."""
    
    PATTERNS = [
        # "go to the <color> <object>"
        (r"go to (?:the )?(\w+) (\w+)", 
         lambda m: [Subgoal("navigate_to", f"{m.group(1)} {m.group(2)}")]),
        
        # "pick up the <color> <object>"
        (r"pick up (?:the )?(\w+) (\w+)",
         lambda m: [
             Subgoal("navigate_to", f"{m.group(1)} {m.group(2)}"),
             Subgoal("pickup", f"{m.group(1)} {m.group(2)}")
         ]),
        # ...
    ]
```

---

## 5. RL Executor & PPO

### 5.1 Why PPO?

**PPO (Proximal Policy Optimization)** is the most popular RL algorithm because:

| Property | Benefit |
|----------|---------|
| **Stable** | Clips updates to prevent too-large changes |
| **Sample efficient** | Reuses data with multiple gradient steps |
| **Simple** | Single algorithm, few hyperparameters |
| **Versatile** | Works for discrete and continuous actions |

### 5.2 PPO Algorithm Intuition

**The Problem with Vanilla Policy Gradient:**
```
If policy changes too much in one update:
- Performance can collapse
- Training becomes unstable
- Agent "forgets" what it learned
```

**PPO Solution: Clipped Objective**
```
Instead of:    L = r(θ) * A    (can change arbitrarily)

Use:           L = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)

Where:
  r(θ) = π_new(a|s) / π_old(a|s)    # How much policy changed
  A = advantage                      # How good was this action
  ε = 0.2                            # Clip range
```

**Visual Intuition:**
```
                    ┌──────────────────────────────────┐
                    │ PPO Clipping                     │
                    │                                  │
      Objective     │       ╱────────────              │
          ▲         │      ╱     clipped               │
          │         │     ╱                            │
          │         │    ╱                             │
          │─────────┼───╱┼─────────────────────────────│
          │         │  ╱ │  1-ε    1+ε                 │
          │         │ ╱  │                             │
          │         │╱   │                             │
          └─────────┴────┴─────────────────────────────┘
                         Probability Ratio r(θ)
                         
When r(θ) is outside [1-ε, 1+ε], the gradient is zero.
This prevents too-large policy updates.
```

### 5.3 Actor-Critic Architecture

Our policy has two heads:

```
┌─────────────────────────────────────┐
│ Observation                         │
│ (grid + direction + subgoal)        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Shared Encoder (CNN + MLP)          │
│ • Extracts features                 │
│ • Same features for both heads      │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌─────────────┐ ┌─────────────┐
│ Actor Head  │ │ Critic Head │
│ π(a|s)      │ │ V(s)        │
│             │ │             │
│ Outputs:    │ │ Outputs:    │
│ logits for  │ │ single      │
│ 7 actions   │ │ value       │
└─────────────┘ └─────────────┘

Actor: Chooses actions
Critic: Estimates value (for computing advantage)
```

### 5.4 Observation Encoder (`observation_encoder.py`)

**Grid Encoder (CNN):**
```python
class GridEncoder(nn.Module):
    """
    Extracts spatial features from 7x7x3 grid observation.
    
    Why CNN?
    - Grid is 2D spatial data (like a small image)
    - CNN captures local patterns (walls, objects)
    - Translation equivariant: object looks same anywhere
    """
    
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # ...
        )
        self.fc = nn.Linear(cnn_output_size, 128)
    
    def forward(self, x):
        # x: (batch, 7, 7, 3) -> (batch, 128)
        x = x.permute(0, 3, 1, 2)  # HWC -> CHW
        features = self.cnn(x)
        return self.fc(features.flatten(1))
```

**Subgoal Encoder:**
```python
class SubgoalEncoder(nn.Module):
    """
    Encodes current subgoal into a vector.
    
    Subgoal: "navigate_to(red ball)"
    -> action_idx=0, color_idx=0, object_idx=0
    -> embedding vector (32,)
    """
    
    def __init__(self):
        super().__init__()
        self.action_embed = nn.Embedding(4, 16)   # 4 action types
        self.color_embed = nn.Embedding(7, 8)     # 6 colors + unknown
        self.object_embed = nn.Embedding(6, 8)   # 5 objects + unknown
```

### 5.5 RLlib Configuration (`rllib_policy.py`)

**RLlib** is a production-grade RL library from Ray.

**Creating PPO Config:**
```python
config = (
    PPOConfig()
    .environment(
        env=MiniGridRLlibEnv,
        env_config={
            "env_name": "BabyAI-GoToObj-v0",
            "max_steps": 64,
        },
    )
    .training(
        lr=3e-4,              # Learning rate
        gamma=0.99,           # Discount factor
        clip_param=0.2,       # PPO clipping
        num_sgd_iter=10,      # SGD passes per batch
        train_batch_size=2048, # Samples per update
    )
    .env_runners(
        num_env_runners=2,    # Parallel workers
    )
)
```

**Key Hyperparameters:**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `lr` | 3e-4 | How fast to update weights |
| `gamma` | 0.99 | How much to value future rewards |
| `clip_param` | 0.2 | PPO clipping range |
| `num_sgd_iter` | 10 | Gradient updates per batch |
| `entropy_coeff` | 0.01 | Encourages exploration |

---

## 6. Training Pipeline

### 6.1 Two-Stage Training

```
Stage 1: Behavior Cloning (BC)
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Expert Demos    ────▶    Supervised    ────▶    Warm       │
│  (from BabyAI           Learning              Policy       │
│   bot)                                        (~60% SR)    │
│                                                             │
│  Loss = CrossEntropy(predicted_action, expert_action)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
Stage 2: PPO Fine-tuning
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Warm Policy    ────▶    Trial & Error   ────▶   Optimal   │
│  (~60% SR)              (PPO updates)          Policy      │
│                                                 (~90% SR)   │
│                                                             │
│  Loss = PPO Objective (clipped policy gradient)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Behavior Cloning (`behavior_cloning.py`)

**What is BC?**
- Supervised learning from expert demonstrations
- Learn to imitate the expert's actions
- Simple but effective warm-start

**Generating Expert Demos:**
```python
def generate_demonstrations(env_name, num_episodes):
    """Use BabyAI's built-in bot to generate demos."""
    
    from minigrid.utils.baby_ai_bot import BabyAIBot
    
    env = gym.make(env_name)
    bot = BabyAIBot(env.unwrapped)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        
        while not done:
            action = bot.replan()  # Expert computes optimal action
            obs, reward, done, _, _ = env.step(action)
            
            # Save (observation, action) pair
            save_to_dataset(obs, action)
```

**BC Training Loop:**
```python
class BehaviorCloning:
    def train(self, epochs):
        for epoch in range(epochs):
            for batch in dataloader:
                # Forward pass
                logits = self.model(batch["image"], batch["direction"])
                
                # Compute loss (cross-entropy)
                loss = F.cross_entropy(logits, batch["action"])
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
```

**Why BC Works:**
- BabyAI bot is optimal → demos are high quality
- No exploration needed → fast training
- Gets agent to ~60% success rate quickly

**Why BC Isn't Enough:**
- **Distribution shift**: Agent sees states expert never saw
- **Compounding errors**: Small mistakes lead to wrong states
- **Suboptimal**: Can't exceed expert's performance

### 6.3 PPO Fine-tuning (`ppo_trainer.py`)

**PPO Training Loop (Simplified):**
```python
for iteration in range(num_iterations):
    # 1. Collect trajectories with current policy
    trajectories = collect_rollouts(policy, env, num_steps=2048)
    
    # 2. Compute advantages (GAE)
    advantages = compute_gae(trajectories, value_function)
    
    # 3. Update policy with PPO objective
    for _ in range(num_sgd_iter):
        policy_loss = ppo_clip_loss(trajectories, advantages)
        value_loss = mse(value_predictions, returns)
        entropy_loss = -entropy(policy)
        
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
        
        optimizer.step(total_loss)
    
    # 4. Log metrics
    log_metrics(reward_mean, success_rate, loss)
```

**Generalized Advantage Estimation (GAE):**
```python
def compute_gae(rewards, values, gamma=0.99, lambda_=0.95):
    """
    GAE balances bias vs variance in advantage estimation.
    
    Low lambda (0): Low variance, high bias (uses value function)
    High lambda (1): High variance, low bias (uses full returns)
    Lambda=0.95: Good balance
    """
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)
    
    return advantages
```

---

## 7. Evaluation & Metrics

### 7.1 Key Metrics (`metrics.py`)

**1. Success Rate (Most Important)**
```python
def compute_success_rate(episodes):
    """Percentage of episodes completed successfully."""
    successes = sum(1 for ep in episodes if ep.success)
    return successes / len(episodes)
```

**2. SPL (Success weighted by Path Length)**
```python
def compute_spl(successes, optimal_lengths, actual_lengths):
    """
    SPL rewards efficiency, not just success.
    
    SPL = (1/N) Σ success_i * (optimal_length_i / actual_length_i)
    
    Perfect SPL = 1.0 (always succeeds with optimal path)
    """
    spl = 0
    for success, opt, actual in zip(successes, optimal_lengths, actual_lengths):
        if success:
            spl += opt / max(opt, actual)
    return spl / len(successes)
```

**3. Generalization**
```python
def compute_generalization(train_success, test_success):
    """
    How well does agent generalize to new instructions?
    
    Generalization gap = train_success - test_success
    
    Good: gap < 10%
    Concerning: gap > 20%
    """
    return {
        "train_success": train_success,
        "test_success": test_success,
        "gap": train_success - test_success,
    }
```

### 7.2 Failure Analysis (`failure_analysis.py`)

**Understanding WHY the agent fails:**

| Category | Description | Typical Cause |
|----------|-------------|---------------|
| `wrong_object` | Goes to wrong object | Language grounding failure |
| `wrong_ordering` | Does steps in wrong order | Planning failure |
| `stuck_looping` | Goes in circles | Exploration failure |
| `timeout` | Runs out of steps | Inefficient navigation |

**Loop Detection:**
```python
def _detect_loop(actions):
    """
    Check if agent is stuck repeating actions.
    
    Pattern: [left, right, left, right, left, right]
    This indicates the agent is oscillating.
    """
    for pattern_len in range(2, 5):
        pattern = actions[-pattern_len*2:-pattern_len]
        next_seq = actions[-pattern_len:]
        if pattern == next_seq:
            return True
    return False
```

---

## 8. File-by-File Breakdown

### Source Files (`src/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `environment/minigrid_wrapper.py` | Wraps MiniGrid for our pipeline | `MiniGridWrapper`, `make_env()` |
| `environment/trajectory_logger.py` | Logs episodes to JSONL | `TrajectoryLogger`, `Episode`, `Step` |
| `agents/planner/llm_planner.py` | Llama 3.2 instruction parsing | `LLMPlanner`, `RuleBasedPlanner`, `Subgoal` |
| `agents/planner/prompts.py` | Prompt templates | `SYSTEM_PROMPT`, `build_chat_messages()` |
| `agents/executor/observation_encoder.py` | Neural network encoders | `GridEncoder`, `SubgoalEncoder`, `ObservationEncoder` |
| `agents/executor/rllib_policy.py` | RLlib PPO configuration | `MiniGridRLlibEnv`, `create_ppo_config()` |
| `training/behavior_cloning.py` | BC training | `BehaviorCloning`, `generate_demonstrations()` |
| `training/ppo_trainer.py` | PPO training wrapper | `PPOTrainer`, `run_training()` |
| `training/experiment_config.py` | YAML config management | `ExperimentConfig`, `load_config()` |
| `evaluation/metrics.py` | Evaluation metrics | `EvaluationSuite`, `compute_spl()` |
| `evaluation/failure_analysis.py` | Failure categorization | `FailureAnalyzer` |

### Scripts (`scripts/`)

| Script | Purpose | Example Usage |
|--------|---------|---------------|
| `train_bc.py` | Train behavior cloning | `python scripts/train_bc.py --epochs 50` |
| `train_ppo.py` | Train with PPO | `python scripts/train_ppo.py --iterations 500` |

---

## 9. Key Libraries

### 9.1 MiniGrid / Gymnasium

```python
import gymnasium as gym
import minigrid  # Registers MiniGrid environments

# Create environment
env = gym.make("BabyAI-GoToObj-v0")

# Standard Gymnasium interface
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

**Key Concepts:**
- **Gymnasium**: Standard RL environment interface (successor to OpenAI Gym)
- **MiniGrid**: Gridworld environments
- **BabyAI**: Language extensions for MiniGrid

### 9.2 PyTorch

```python
import torch
import torch.nn as nn

# Define a neural network
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
    
    def forward(self, x):
        return torch.relu(self.fc(x))

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    output = model(batch["input"])
    loss = criterion(output, batch["target"])
    
    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
```

**MPS (Metal) Acceleration:**
```python
# Check if MPS is available (M1/M2/M3 Mac)
torch.backends.mps.is_available()  # True

# Use MPS device
device = torch.device("mps")
model = model.to(device)
data = data.to(device)
```

### 9.3 Transformers (HuggingFace)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Chat format for instruction-tuned models
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

# Generate response
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0])
```

### 9.4 Ray / RLlib

```python
import ray
from ray.rllib.algorithms.ppo import PPOConfig

# Initialize Ray
ray.init()

# Configure PPO
config = (
    PPOConfig()
    .environment("CartPole-v1")
    .training(lr=3e-4, gamma=0.99)
    .env_runners(num_env_runners=2)
)

# Build and train
algo = config.build()
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: reward = {result['episode_reward_mean']}")

# Save checkpoint
algo.save("checkpoints/ppo")
```

---

## 10. Common Pitfalls & Tips

### 10.1 Environment Issues

**Problem**: Environment import fails
```python
# Wrong
import gymnasium as gym
env = gym.make("BabyAI-GoToObj-v0")  # Error: not registered

# Right
import gymnasium as gym
import minigrid  # This registers the environments!
env = gym.make("BabyAI-GoToObj-v0")  # Works
```

**Problem**: Observation shape mismatch
```python
# MiniGrid uses HWC format (Height, Width, Channels)
obs["image"].shape  # (7, 7, 3)

# PyTorch expects CHW format (Channels, Height, Width)
# Convert before feeding to CNN:
x = obs["image"].permute(2, 0, 1)  # (3, 7, 7)
```

### 10.2 LLM Issues

**Problem**: Model too slow
```python
# Solution 1: Use smaller model
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # vs 3B

# Solution 2: Use half precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # or torch.bfloat16
)

# Solution 3: Cache the model (only load once)
planner = LLMPlanner()  # Keep reference, reuse
```

**Problem**: LLM outputs invalid JSON
```python
# Solution: Fallback parsing
def parse_plan(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            return json.loads(match.group())
        # Last resort: use rule-based parser
        return rule_based_parse(response)
```

### 10.3 Training Issues

**Problem**: Training too slow
```python
# Solution 1: Use more workers
config.env_runners(num_env_runners=4)

# Solution 2: Larger batch size (if memory allows)
config.training(train_batch_size=4096)

# Solution 3: Train on Colab with GPU
# (Create a Colab notebook for intensive training)
```

**Problem**: Reward doesn't improve
```python
# Solution 1: Check environment is working
env = make_env("BabyAI-GoToObj-v0")
obs, _ = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    _, reward, done, _, _ = env.step(action)
    print(f"action={action}, reward={reward}")

# Solution 2: Use BC warm-start
# Random policy takes forever to succeed
bc = BehaviorCloning(demos)
bc.train(epochs=50)  # Start with competent policy, THEN use RL

# Solution 3: Use dense rewards
env = MiniGridWrapper(use_dense_reward=True)
```

### 10.4 Memory Issues

**Problem**: Out of memory when loading LLM
```python
# Solution 1: Use smaller model (1B vs 3B)

# Solution 2: Offload to CPU when not using
model.to("cpu")  # Free GPU memory
# ... do other stuff ...
model.to("mps")  # Move back for generation

# Solution 3: Clear cache
torch.mps.empty_cache()
```

---

## 🎉 Congratulations!

You now understand:
- ✅ How reinforcement learning works
- ✅ What language conditioning means
- ✅ Why we use hybrid LLM + RL
- ✅ How MiniGrid/BabyAI environments work
- ✅ How the LLM planner parses instructions
- ✅ How PPO trains the RL policy
- ✅ How BC warm-start helps training
- ✅ What each file in the project does
- ✅ How to use each key library

**Next Steps:**

### Local Training (Quick Test)
```bash
cd Language_Conditioned_Agent_Using_RL
source venv/bin/activate
python scripts/train_bc.py --num-demos 200 --epochs 20
python scripts/train_ppo.py --iterations 50
```

### Full Training on Google Colab (Recommended)
1. Upload `notebooks/train_on_colab.ipynb` to Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells (~1-2 hours for 500 iterations)
4. Download `trained_model.zip` and extract locally

**Experiments to Try:**
1. Modify prompts in `prompts.py` and see how it affects planning
2. Experiment with different environments: `GoToObj` → `PickupLoc` → `PutNextLocal`
3. Add your own metrics or failure categories
4. Try different PPO hyperparameters (lr, clip_param, entropy_coeff)

Happy learning! 🚀
