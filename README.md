# Deep Reinforcement Learning — DQN for CartPole and Pong

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-green)
![Status](https://img.shields.io/badge/Project-Portfolio--Ready-brightgreen)

## Overview
This project implements **Deep Q-Networks (DQN)** and its popular extensions:
- **Double DQN**
- **Dueling DQN**
- **Prioritized Experience Replay (PER)**
- **N-step returns**

Environments:
- Classic Control: **CartPole-v1**
- Atari: **PongNoFrameskip-v4** (with preprocessing & wrappers)

The implementation is designed to be **reproducible, modular, and portfolio-friendly**.  
It includes logging with **TensorBoard / MLflow**, hyperparameter sweeps with **Optuna**, and gameplay **video recordings**.

---

## Project Structure
```
.
├─ envs/               # environment wrappers (CartPole, Atari)
├─ models/             # DQN variants (CNN, MLP, Dueling, etc.)
├─ memory/             # replay buffer (standard, PER, N-step)
├─ train/              # training scripts (cartpole, pong, evaluation)
├─ utils/              # logging, schedulers, seeds
├─ configs/            # YAML configs for reproducible runs
├─ artifacts/          # checkpoints, logs, videos, metrics
├─ requirements.txt
└─ README.md
```

---

## Features
- Modular DQN implementation (toggle Double, Dueling, PER, N-step)
- Atari preprocessing: gray-scaling, resizing to 84×84, frame stacking
- Replay buffer with Prioritized Sampling (SumTree)
- ε-greedy and Noisy Nets exploration
- Target soft updates (Polyak averaging)
- Logging with TensorBoard & MLflow
- Hyperparameter tuning with Optuna
- Video generation for evaluation episodes

---

## Results
- **CartPole**: Solved consistently (avg. return > 195 over 100 episodes).
- **Pong**: Stable training with DQN extensions; reward improves towards ~+18 after ~3–5M frames (depending on config).
Graphs and videos will be added here.

---

## Installation
```bash
git clone https://github.com/USERNAME/rl-dqn-pong-cartpole.git
cd rl-dqn-pong-cartpole
pip install -r requirements.txt
```

Dependencies:
- Python 3.10+
- torch
- gymnasium[atari,accept-rom-license]
- opencv-python
- optuna
- mlflow
- tensorboard

---

## Usage
Train CartPole:
```bash
python train/train_cartpole.py --config configs/cartpole.yaml
```

Train Pong:
```bash
python train/train_pong.py --config configs/pong.yaml
```

Evaluate and record videos:
```bash
python train/eval.py --env PongNoFrameskip-v4 --checkpoint artifacts/checkpoints/best.pth --record artifacts/videos/
```

---

## Reproducibility
- All configs stored in `configs/`.
- Seeds are fixed for determinism across PyTorch/NumPy/envs.
- Set RNG seed (default: 42).

---

## Repository Topics
`reinforcement-learning, deep-reinforcement-learning, dqn, double-dqn, dueling-dqn, prioritized-experience-replay, atari, cartpole, pong, pytorch, machine-learning, portfolio-project`

---

## License
MIT License — free to use for research and portfolio purposes.
