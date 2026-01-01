# Language-Conditioned Agent Using RL

A language-conditioned reinforcement learning agent that follows multi-step natural language instructions in MiniGrid/BabyAI environments.

## 🎯 Project Overview

This project implements a **hybrid LLM + RL architecture**:
- **LLM Planner** (Llama 3.2): Parses natural language instructions into subgoals
- **RL Executor** (RLlib PPO): Learns to execute subgoals efficiently

```
Instruction: "Pick up the blue key, then open the yellow door"
    ↓
LLM Planner → ["navigate_to(blue_key)", "pickup()", "navigate_to(yellow_door)", "open()"]
    ↓
RL Executor → [↑, →, →, pickup, ←, ←, ↓, open] → Success!
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your HuggingFace token
```

### 2. Test the Environment

```bash
# Run environment exploration notebook
jupyter notebook notebooks/01_environment_exploration.ipynb
```

### 3. Train the Agent

#### Option A: Local Training (MacBook M3 Pro)
```bash
# Quick test (5-10 minutes)
python scripts/train_bc.py --num-demos 200 --epochs 20
python scripts/train_ppo.py --iterations 50
```

#### Option B: Full Training on Google Colab (Recommended)
```bash
# Upload notebooks/train_on_colab.ipynb to Colab
# Runtime → Change runtime type → T4 GPU
# Run all cells (~1-2 hours for 500 iterations)
# Download trained_model.zip when complete
```

**Training Times:**
| Task | MacBook M3 | Colab GPU |
|------|------------|-----------|
| BC (500 demos) | ~15 min | ~5 min |
| PPO (100 iter) | ~30 min | ~15 min |
| PPO (500 iter) | ~3 hours | ~1 hour |

### 4. Evaluate and Demo

```bash
# Launch evaluation dashboard
streamlit run dashboard/app.py

# Launch interactive demo
python demo/interactive_demo.py
```

## 📁 Project Structure

```
├── src/
│   ├── environment/      # MiniGrid wrappers & trajectory logging
│   ├── agents/
│   │   ├── planner/      # LLM-based instruction parser
│   │   └── executor/     # RLlib PPO policy
│   ├── training/         # BC + RL training pipelines
│   └── evaluation/       # Metrics and analysis
├── data/                 # Trajectories and demonstrations
├── experiments/          # Configs and checkpoints
├── dashboard/            # Streamlit evaluation UI
├── demo/                 # Gradio interactive demo
├── notebooks/            # Jupyter exploration
└── tests/                # Unit tests
```

## 🧠 Key Concepts

### Why Hybrid LLM + RL?

| Approach | Pros | Cons |
|----------|------|------|
| Pure LLM | Great language understanding | Slow, expensive, poor control |
| Pure RL | Fast, learns optimal control | Poor language grounding |
| **Hybrid** | **Best of both worlds** | Slightly more complex |

### Training Pipeline

1. **Behavior Cloning**: Learn from expert demonstrations (warm start)
2. **PPO Fine-tuning**: Optimize for efficiency and generalization
3. **Curriculum Learning**: Start easy, gradually increase difficulty

## 📊 Metrics

- **Success Rate**: % of episodes completed correctly
- **SPL**: Success weighted by Path Length (efficiency)
- **Generalization**: Performance on unseen instruction templates

## 🛠️ Development

```bash
# Run tests
pytest tests/ -v

# Format code
black src/ tests/
```

## 📚 References

- [MiniGrid Documentation](https://minigrid.farama.org/)
- [BabyAI Paper](https://arxiv.org/abs/1810.08272)
- [RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html)
- [Llama 3.2 on HuggingFace](https://huggingface.co/meta-llama)

## 📝 License

MIT License - See LICENSE for details.
