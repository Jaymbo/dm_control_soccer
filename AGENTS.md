# Repository Overview

## Project Description

**Multi-Agent Soccer with Deep Reinforcement Learning**

This project implements Deep Reinforcement Learning (DRL) agents for a 2-vs-2 soccer simulation using the DM Control Suite. It explores both centralized and multi-agent approaches to learn coordinated team behavior.

### Main Purpose and Goals
- Train AI agents to play soccer in a physics-based simulation
- Compare centralized PPO vs. Multi-Agent PPO (MAPPO) architectures
- Implement advanced techniques: Curriculum Learning, Reward Shaping, and Performance Optimizations
- Enable agents to learn: walking, ball approach, dribbling, and shooting

### Key Technologies
- **Deep Learning Framework**: PyTorch
- **Environment**: DM Control Suite (MuJoCo-based soccer simulation)
- **RL Algorithms**: PPO (Proximal Policy Optimization), MAPPO (Multi-Agent PPO)
- **Architecture**: CTDE (Centralized Training with Decentralized Execution)
- **Logging**: TensorBoard

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
├─────────────────────────────────────────────────────────────┤
│  Environment (DM Control Soccer)                             │
│    ↓                                                         │
│  Reward Wrapper (Shaped Rewards)                             │
│    ↓                                                         │
│  Agent (Actor-Critic)                                        │
│    ├── Actor Network (Policy) → Actions                      │
│    └── Critic Network (Value) → State Evaluation             │
│    ↓                                                         │
│  Replay Buffer (Trajectory Storage)                          │
│    ↓                                                         │
│  PPO Update (Policy Optimization)                            │
└─────────────────────────────────────────────────────────────┘
```

### Main Components

| Component | File(s) | Purpose |
|-----------|---------|---------|
| **Environment** | `env_wrapper.py`, `env_wrapper_optimized.py`, `env_wrapper_curriculum.py`, `env_wrapper_dynamic.py` | DM Control soccer with custom reward shaping / Dynamic Scoring |
| **Centralized Agent** | `agent.py` | Single network controlling all 4 players (12 action dims) |
| **MAPPO Agent** | `agent_mappo.py`, `agent_mappo_optimized.py` | Multi-agent with shared actor, centralized critic |
| **Training Scripts** | `train.py`, `train_mappo.py`, `train_mappo_*.py` | PPO/MAPPO training loops with various optimizations |
| **Testing Scripts** | `test.py`, `test_mappo.py`, `test_mappo_*.py` | Evaluation and visualization |
| **Viewer** | `train_live_viewer.py` | Real-time training visualization |

### Data Flow

1. **Observation**: Each player receives 119-dim ego-centric observation (ball position, goal positions, joint states)
2. **Action**: Each player outputs 3-dim continuous action (joint torques)
3. **Reward**: 
   - Base: +10 for goal, -10 for conceded
   - Shaped: Ball proximity, ball-to-goal distance, possession bonuses
4. **Training**: PPO updates with GAE (Generalized Advantage Estimation)

### System Interactions

```
Environment Step:
  obs (4x119) → Agent → actions (4x3) → Environment → rewards (4x1) + next_obs

Training Batch:
  Collect episodes_per_batch (default: 10-20)
  → Compute GAE advantages
  → PPO epochs (default: 10) with mini-batches (default: 256)
  → Update actor and critic networks
```

---

## Directory Structure

```
projekt soccer/
├── agent.py                        # Centralized Actor-Critic
├── agent_mappo.py                  # MAPPO Agent (CTDE)
├── agent_mappo_optimized.py        # Optimized MAPPO (vectorized)
├── env_wrapper.py                  # Reward shaping wrapper
├── env_wrapper_optimized.py        # Optimized rewards (ego-based)
├── env_wrapper_curriculum.py       # Curriculum learning wrapper
├── env_wrapper_dynamic.py          # Dynamic Scoring reward wrapper
├── train.py                        # Centralized PPO training
├── train_mappo.py                  # MAPPO training
├── train_mappo_optimized.py        # Optimized MAPPO training
├── train_mappo_curriculum.py       # Curriculum learning training
├── train_mappo_dynamic.py          # Dynamic Scoring MAPPO training
├── train_live_viewer.py            # Training with live visualization
├── test.py                         # Test centralized agent
├── test_mappo.py                   # Test MAPPO agent
├── test_mappo_curriculum.py        # Test curriculum agent
├── test_mappo_optimized.py         # Test optimized agent
├── debug_obs.py                    # Observation debugging
├── requirements.txt                # Python dependencies
├── logs/                           # Training checkpoints & TensorBoard logs
│   ├── soccer_ppo/
│   ├── soccer_mappo/
│   ├── soccer_mappo_optimized/
│   └── soccer_mappo_curriculum/
├── Documentation Files:
│   ├── CURRICULUM_LEARNING.md      # Curriculum learning guide
│   ├── MAPPO_VS_PPO.md             # Architecture comparison
│   ├── OPTIMIZATIONS_SUMMARY.md    # Performance optimizations
│   ├── TRAINING_USAGE.md           # Training usage guide
│   ├── README_DISTRIBUTED.md       # Distributed optimization setup (legacy)
│   ├── README_MLFLOW_MAPO.md       # MLflow + Optuna decentralized setup
│   └── README_DYNAMIC_SCORING.md   # Dynamic Scoring reward guide
├── Docker Files:
│   ├── Dockerfile                  # Worker container image
│   ├── docker-compose.master.yml   # PostgreSQL + MLflow + Dashboard stack
│   └── docker-compose.worker.yml   # Worker slave container
├── Distributed Optimization:
│   ├── optimize_curriculum.py      # Local hyperparameter optimization
│   └── worker_entrypoint.py        # Distributed worker entry point (MLflow support)
├── Configuration:
│   └── .env.example                # Environment variable template
└── Scripts:
    └── init_postgres.sql           # PostgreSQL initialization
```

### Key Configuration Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies (dm-control, mujoco, torch, tensorboard) |
| `logs/` | Automatic checkpoint saving and TensorBoard event files |

### Entry Points

**Training:**
- `python train.py` - Centralized PPO (baseline)
- `python train_mappo.py` - MAPPO with centralized critic
- `python train_mappo_optimized.py` - Optimized MAPPO (recommended)
- `python train_mappo_curriculum.py` - Curriculum learning (easiest to learn)

**Hyperparameter Optimization:**
- `python optimize_curriculum.py --n-trials 50` - Local optimization (SQLite)
- `python worker_entrypoint.py --storage postgresql://... --mlflow-tracking-uri http://... --infinite` - Distributed worker with MLflow
- `docker-compose -f docker-compose.master.yml up -d` - Start MLflow + Optuna master stack

**Testing:**
- `python test.py --checkpoint <path>` - Test centralized agent
- `python test_mappo.py --checkpoint <path>` - Test MAPPO agent
- Add `--viewer` flag for visualization

**Distributed (Docker):**
- `docker-compose -f docker-compose.master.yml up -d` - Start master (PostgreSQL + Dashboard)
- `docker-compose -f docker-compose.worker.yml up -d` - Start worker slave

---

## Development Workflow

### Prerequisites

**System Requirements:**
- Python 3.8+
- MuJoCo 3.1.6 (physics engine)
- GPU optional (CUDA, ROCm, or CPU fallback)

### Setup

1. **Install Dependencies:**
```bash
# Install core dependencies
pip install -r requirements.txt

# Install PyTorch (choose based on hardware):
# AMD GPU:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
# NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu118
# CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

2. **Verify Installation:**
```bash
python -c "import torch; import dm_control; print('OK')"
```

### Building/Running the Project

**Quick Start Training:**
```bash
# Test run (100 episodes)
python train_mappo_optimized.py --num-episodes 100 --viewer

# Full training (recommended)
python train_mappo_optimized.py --num-episodes 1000 --reward-config aggressive

# Curriculum learning (easiest convergence)
python train_mappo_curriculum.py --num-episodes 1000 --start-phase 0 --auto-advance
```

**Key Training Parameters:**
```bash
--num-episodes      # Total training episodes (500-2000)
--episodes-per-batch # Episodes per PPO update (10-20)
--ppo-epochs        # PPO epochs per batch (4-10)
--hidden-dim        # Network size (256-1024)
--lr                # Learning rate (1e-4 to 3e-4)
--entropy-coef      # Exploration bonus (0.01-0.05)
--reward-config     # 'balanced' or 'aggressive'
--viewer            # Enable visualization
--eval-at-end       # Show viewer after training
```

### Testing Approach

**Unit Testing:**
- Agent forward pass tests (included in optimized scripts)
- Reward wrapper tests with mock environments
- Buffer and GAE computation tests

**Integration Testing:**
```bash
# Short training run
python train_mappo_optimized.py --num-episodes 50 --eval-at-end

# Load and test checkpoint
python test_mappo_optimized.py --checkpoint logs/soccer_mappo_optimized/final_agent.pt --viewer
```

**Performance Metrics:**
- Steps per second (CPU: ~1500-2000, GPU: ~3000-5000)
- Episodes to first goal (50-150 with optimized rewards)
- Final average reward (15-40+ per 100 episodes)

### Development Environment Setup

**Recommended IDE:** VS Code or PyCharm with Python support

**Useful Commands:**
```bash
# Syntax check
python -m py_compile *.py

# Run with debugging
python -m pdb train_mappo_optimized.py --num-episodes 10

# Monitor TensorBoard
tensorboard --logdir logs/
# Open http://localhost:6006
```

### Lint and Format

**Current Status:** No linting/formatting configuration present

**Recommended Setup:**
```bash
# Install linting tools
pip install flake8 black isort mypy

# Format code
black *.py
isort *.py

# Lint
flake8 *.py --max-line-length=100

# Type checking (optional)
mypy *.py --ignore-missing-imports
```

### Common Development Tasks

**1. Modify Reward Function:**
- Edit `env_wrapper_optimized.py` → `_compute_shaped_reward()`
- Adjust weights in `REWARD_CONFIGS` dict
- Test with: `python train_mappo_optimized.py --num-episodes 100 --viewer`

**2. Change Network Architecture:**
- Edit `agent_mappo_optimized.py` → `ActorCritic` class
- Adjust `--hidden-dim`, `--actor-layers`, `--critic-layers` flags
- Monitor `Loss/policy` and `Loss/value` in TensorBoard

**3. Tune Hyperparameters:**
```bash
# Higher exploration
python train_mappo_optimized.py --entropy-coef 0.05 --entropy-decay 0.8

# More stable training
python train_mappo_optimized.py --episodes-per-batch 10 --mini-batch-size 64

# Larger network
python train_mappo_optimized.py --hidden-dim 512 --actor-layers 3
```

**4. Debug Observations:**
```bash
python debug_obs.py
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| MuJoCo license error | Use `mujoco==3.1.6` (no license needed) |
| GPU not detected | Install PyTorch with correct CUDA/ROCm version |
| Training diverges | Lower learning rate, increase `--entropy-coef` |
| Agents don't move | Check reward shaping, use curriculum learning |
| Slow training | Use `train_mappo_optimized.py`, enable GPU |
| Worker cannot connect to storage | Check network, credentials, Cloudflare tunnel status |
| Trials not distributed | Ensure same `--study-name` and storage URL |
| Out of memory in Docker | Reduce `--episodes-per-batch` or set CPU/memory limits |
| MLflow not logging | Check `--mlflow-tracking-uri`, verify server is running |
| MLflow connection timeout | Use SSH tunnel or Cloudflare Zero Trust for remote access |

---

## Distributed Optimization

For large-scale hyperparameter search across multiple machines, see `README_DISTRIBUTED.md`.

**Quick Setup:**

1. **Master (Server):**
   ```bash
   docker-compose -f docker-compose.master.yml up -d
   # Access dashboard at http://localhost:8080
   ```

2. **Worker (Any machine):**
   ```bash
   docker-compose -f docker-compose.worker.yml up -d
   # Or run directly: python worker_entrypoint.py --storage postgresql://... --infinite
   ```

**Security:** Never expose PostgreSQL directly. Use Cloudflare Zero Trust, WireGuard, or SSH tunnels.

---

## Quick Reference

### Training Commands

```bash
# Fastest convergence (Curriculum Learning)
python train_mappo_curriculum.py --num-episodes 1000 --auto-advance

# Best performance (Optimized MAPPO)
python train_mappo_optimized.py --num-episodes 1000 --reward-config aggressive

# Baseline (Centralized PPO)
python train.py --num-episodes 1000

# With visualization
python train_mappo_optimized.py --num-episodes 500 --viewer --viewer-interval 50 --eval-at-end
```

### Checkpoint Management

```bash
# Save location: logs/<experiment_name>/checkpoint_ep<N>.pt
# Best model: logs/<experiment_name>/best_agent.pt
# Final model: logs/<experiment_name>/final_agent.pt

# Test checkpoint
python test_mappo_optimized.py --checkpoint logs/soccer_mappo_optimized/best_agent.pt --viewer
```

### TensorBoard Monitoring

```bash
tensorboard --logdir logs/
# Metrics: Reward/avg_100, Loss/policy, Loss/value, Loss/entropy
```
