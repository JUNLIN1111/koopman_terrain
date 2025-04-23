🚀 G1 Koopman RL Policy

Welcome to the G1 Koopman RL Policy project! This repository implements Koopman operator-based reinforcement learning (RL) for the G1 robot, enabling efficient terrain adaptation (e.g., sand, hard ground) using PPO and Dyna frameworks. 🌟

🎯 Project Overview
The G1 Koopman RL Policy project focuses on:

Modeling G1 robot dynamics using Koopman operators for terrains like sand (friction 0.9) and hard ground (friction 1.2).
Training PPO policies with stable-baselines3 for robust control.
Using the Dyna framework to fit Koopman models and tune PPO policies simultaneously.
Supporting both CPU and GPU (CUDA) for faster training.

Key Features

🧠 Koopman modeling with pykoop for dynamics prediction.
🤖 PPO policy training with stable-baselines3.
🔄 Dyna framework for model-based RL.
🌍 Terrain-specific data collection (sand, hard ground).


📂 Directory Structure



Path
Description



koopman_rl/
Core scripts for training and tuning


└── train_koopman_g1_cuda.py
Train Koopman model with CUDA support


`└── dyna_koopman



