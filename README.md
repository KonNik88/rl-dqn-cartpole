# Deep Reinforcement Learning — DQN for CartPole and Pong

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-green)
![Status](https://img.shields.io/badge/Status-In_Progress-yellow)

## Overview

This project implements **Deep Q-Networks (DQN)** and advanced extensions for two environments:

- **CartPole-v1** — baseline task for tabular control.  
- **PongNoFrameskip-v4** — pixel-based Atari environment using CNN.

Extensions implemented:
- **Double DQN (DDQN)**
- **Dueling DQN**
- **Prioritized Experience Replay (PER)**
- **N-step returns**
- **Soft target update (τ)**
- **TensorBoard logging and video evaluation**

The implementation is **modular and reproducible**, designed for educational and portfolio use.

---

## Project Structure

```
project/
│
├─ configs/
│    ├─ cartpole.yaml
│    ├─ pong_ddqn_duel_per_n3.yaml
│    └─ pong_ddqn_duel_per_n3_long.yaml     # final 3M-run
│
├─ envs/
│    ├─ cartpole_wrappers.py
│    └─ atari_wrappers.py                   # preprocessing: 84×84 gray, FrameStack(4), ActionMap([2,5])
│
├─ models/
│    ├─ dqn_mlp.py                          # MLP for CartPole
│    └─ dqn_cnn.py                          # NatureCNN for Pong (dueling=True/False)
│
├─ memory/
│    ├─ replay_buffer.py                    # uniform replay
│    └─ per_buffer.py                       # prioritized replay (α, β) + n-step
│
├─ train/
│    ├─ train_cartpole.py
│    ├─ train_pong.py
│    ├─ eval.py
│    └─ eval_pong.py
│
├─ artifacts_pong/
│    ├─ tensorboard/                        # TB logs per run
│    ├─ checkpoints/                        # model checkpoints
│    ├─ plots/                              # generated training plots
│    └─ videos/                             # evaluation videos
│
└─ notebooks/
     ├─ 01_cartpole_experiments.ipynb
     └─ 02_pong_experiments.ipynb
```

---

## Environment & Dependencies

```bash
conda create -n rl_env python=3.10
conda activate rl_env
pip install -r requirements.txt
```

Main packages:
- torch>=2.5.1
- gymnasium[atari,accept-rom-license]
- opencv-python
- numpy, matplotlib
- tensorboard
- optuna, mlflow (for hyperparameter sweeps)

---

## Usage

Train **CartPole**:
```bash
python -m train.train_cartpole --config configs/cartpole.yaml
```

Train **Pong (DDQN+Dueling+PER+n-step)**:
```bash
python -m train.train_pong --config configs/pong_ddqn_duel_per_n3_long.yaml
```

Evaluate and record video:
```bash
python -m train.eval_pong --ckpt artifacts_pong/checkpoints/.../final.pth --episodes 10 --epsilon 0.01
```

View TensorBoard logs:
```bash
tensorboard --logdir artifacts_pong/tensorboard
```

---

## Results Summary

| Date | Config | Frames | Mean Return | Comment |
|:--:|:--|:--:|:--:|:--|
| 09.11 | pong_ddqn_duel_per_n3_probe | 800k | +4.6 | First “alive” signal |
| 10.11 | same config (new seed) | 800k | −21 | stochastic dead run |
| 12.11 | pong_ddqn_duel_per_n3_long | 3M | — | checkpoints not saved (under investigation) |

- CartPole consistently solved (MA100 > 400).  
- Pong pipeline operational; first signs of learning observed after ~0.8M frames.

---

## Key Insights

- Pong requires long runs (>1M frames) for learning stability.  
- PER + n-step improves training but needs proper β schedule and long warmup.  
- Soft target update (τ=0.005) stabilizes DDQN.  
- Logging and evaluation fully integrated.

---

## Next Steps

- Fix checkpoint saving logic for long runs.  
- Add episode-level MA100 logging in TensorBoard.  
- Complete long-run training (~3M frames).  
- Finalize plots, videos, and summary notebook for GitHub.

---

## Repository Topics

`reinforcement-learning, deep-reinforcement-learning, dqn, ddqn, dueling-dqn, prioritized-replay, atari, pong, cartpole, pytorch, portfolio-project`

---

## License

MIT License — Free for educational and research purposes.
