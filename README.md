# 🚀 G1 Koopman RL Policy

Welcome to the **G1 Koopman RL Policy** project! This repository implements Koopman operator-based reinforcement learning (RL) for the G1 robot, enabling efficient terrain adaptation (e.g., sand, hard ground) using PPO and Dyna frameworks. 🌟

---

## 🎯 Project Overview

The G1 Koopman RL Policy project focuses on:
- Modeling G1 robot dynamics using **Koopman operators** for terrains like sand (friction 0.9) and hard ground (friction 1.2).
- Training **PPO policies** with `stable-baselines3` for robust control.
- Using the **Dyna framework** to fit Koopman models and tune PPO policies simultaneously.
- Supporting both CPU and GPU (CUDA) for faster training.

### Key Features
- 🧠 Koopman modeling with `pykoop` for dynamics prediction.
- 🤖 PPO policy training with `stable-baselines3`.
- 🔄 Dyna framework for model-based RL.
- 🌍 Terrain-specific data collection (sand, hard ground).

---

## 📂 Directory Structure

| Path                          | Description                       |
|-------------------------------|-----------------------------------|
| `koopman_rl/`                | Core scripts for training and tuning |
| `└── train_koopman_g1_cuda.py` | Train Koopman model with CUDA support |
| `└── dyna_koopman_ppo_tune_sand.py` | Dyna framework for PPO tuning on sand |
| `└── tune_ppo_with_koopman_sand.py` | PPO tuning with pre-trained Koopman model |
| `└── g1_utils.py`            | Utility functions (e.g., `get_data_path`) |
| `data/`                      | Terrain data files               |
| `└── g1_koopman_data_sand.csv` | Sand terrain data               |
| `resources/koopman_model/`   | Saved models and validation plots |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (optional, for faster training)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Koopman_optimization_rl_policy.git
   cd Koopman_optimization_rl_policy
