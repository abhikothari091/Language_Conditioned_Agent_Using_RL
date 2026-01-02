# Language-Conditioned Agent Using RL

A language-conditioned reinforcement learning agent that follows natural language instructions in MiniGrid/BabyAI environments.

## 🎯 Project Overview

This project implements a **PPO-trained RL agent** that navigates to objects based on language instructions like "go to the red ball".

**Key Results:**
- ✅ **92.7% success rate** on BabyAI-GoToObj-v0
- ✅ **~5 steps average** to complete tasks
- ✅ **Trained on Google Colab** with T4 GPU (~30 min)

## 🎮 Try the Interactive Demo

```bash
# Activate Python 3.12 environment
source venv312/bin/activate

# Start the demo server
python demo/app.py

# Open in browser
open http://localhost:5001
```

![Demo](demo/demo_screenshot.png)

**Features:**
- 🔄 **New Episode** - Generate new task
- ▶️ **Step** - Watch one action at a time
- ⏩ **Auto-Play** - Animate full episode

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create Python 3.12 virtual environment (required for model compatibility)
python3.12 -m venv venv312
source venv312/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install 'ray[rllib]==2.51.2'
```

### 2. Use Pre-Trained Model

The trained model is in `experiments/checkpoints/final/`.

```bash
# Run evaluation
python scripts/load_trained_model.py --checkpoint experiments/checkpoints/final --episodes 100

# Interactive demo
python demo/app.py
```

### 3. Train Your Own (Optional)

**Option A: Google Colab (Recommended)**
1. Upload `notebooks/train_on_colab.ipynb` to Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells (~30-45 min for 200 iterations)
4. Download `trained_model.zip`

**Option B: Local Training**
```bash
python scripts/train_ppo.py --iterations 200
```

---

## 📁 Project Structure

```
├── demo/                   # Interactive web demo
│   ├── app.py              # Flask server
│   └── templates/          # HTML frontend
├── experiments/
│   └── checkpoints/        # Trained model weights
├── notebooks/
│   ├── train_on_colab.ipynb    # Colab training notebook
│   └── demo.ipynb              # Local demo notebook
├── scripts/
│   ├── load_trained_model.py   # Load & evaluate model
│   └── evaluate_model.py       # Simple evaluation
├── src/
│   ├── environment/        # MiniGrid wrappers
│   └── agents/executor/    # RLlib PPO policy
└── LEARNING.md             # Comprehensive learning guide
```

---

## 📊 Training Results

| Metric | Value |
|--------|-------|
| Final Success Rate | 92.7% |
| Average Episode Length | 5.2 steps |
| Training Iterations | 200 |
| Training Time (Colab T4) | ~30 minutes |

### Training Curve

The agent learns rapidly, reaching ~90% success within 50 iterations:

```
Iter  10: reward=0.366, len=42.4
Iter  50: reward=0.922, len=5.5
Iter 100: reward=0.924, len=5.4
Iter 200: reward=0.927, len=5.2
```

---

## 🛠️ Technical Details

### Environment
- **BabyAI-GoToObj-v0**: Navigate to a specified object
- **Observation**: 7×7×3 grid (flattened to 151 floats)
- **Actions**: 7 discrete (left, right, forward, pickup, drop, toggle, done)

### Model Architecture
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Network**: MLP with [256, 256] hidden layers
- **Framework**: RLlib + PyTorch

### Key Configuration (for training)
```python
PPOConfig()
.api_stack(enable_rl_module_and_learner=False, ...)  # Use stable OLD API
.environment(env="MiniGridFlat-v0")
.training(train_batch_size=2048, lr=3e-4, gamma=0.99)
.env_runners(num_env_runners=2, num_envs_per_env_runner=4)
```

---

## 📚 Learn More

See [LEARNING.md](LEARNING.md) for a comprehensive guide covering:
- Reinforcement Learning fundamentals
- MiniGrid/BabyAI environments
- PPO algorithm explained
- RLlib training pipeline
- Troubleshooting common issues

---

## 📝 License

MIT License - See LICENSE for details.
