# Deep Reinforcement Learning — DQN for CartPole

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.4-red)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Overview

This repository contains a **clean, minimal implementation of Deep Q-Networks (DQN) and extensions for CartPole-v1**.

Implemented algorithms and tricks:

- **DQN** (baseline)
- **Double DQN (DDQN)**
- **Dueling DQN**
- **Prioritized Experience Replay (PER)**
- **N-step returns**
- **Soft target updates (τ)**
- **TensorBoard logging and evaluation script**

The code is designed as an **educational / portfolio project**:
- modular structure (`envs/`, `models/`, `memory/`, `train/`, `utils/`);
- experiment notebook with several variants;
- reproducible CLI entrypoints.

---

## Project Structure

```text
.
├─ configs/
│   └─ cartpole.yaml              # base config for DQN/DDQN/PER/N-step and others
│
├─ envs/
│   └─ cartpole_wrappers.py       # make_cartpole(...), seeding, wrappers
│
├─ models/
│   ├─ dqn_mlp.py                 # MLPQ and DuelingMLP for CartPole
│   └─ dqn_cnn.py                 # (legacy, not used in CartPole pipeline)
├─ memory/
│   ├─ replay_buffer.py           # uniform replay, supports n-step
│   └─ per_buffer.py              # prioritized replay (α, β) + n-step
│
├─ train/
│   ├─ train_cartpole.py          # main training loop
│   └─ eval.py                    # evaluation of a saved checkpoint
│
├─ utils/
│   ├─ schedule.py                # LinearSchedule for ε, etc.
│   └─ seed.py                    # set_seed(...) for reproducibility
│
├─ scripts/
│   ├─ run_cartpole.sh            # convenience launcher (Linux/macOS)
│   └─ run_cartpole.bat           # convenience launcher (Windows)
│
├─ notebooks/
│   └─ 01_cartpole_experiments.ipynb  # experiments & analysis
│
├─ LICENSE
├─ .gitignore
├─ requirements.txt
└─ README.md
```

---

## Environment & Installation

Create a fresh environment (example with conda):

```bash
conda create -n rl_env python=3.10
conda activate rl_env
pip install -r requirements.txt
```

Main dependencies:

- `torch` — neural networks and optimization
- `gymnasium` — CartPole-v1 environment
- `numpy`, `pandas`, `matplotlib`, `tqdm` — utilities & plots
- `PyYAML` — configs
- `tensorboard` — logging
- `imageio` — optional, for video recording in eval

---

## Configuration

All hyperparameters are stored in `configs/cartpole.yaml`, e.g.:

- environment: id, max episode length, seed
- algorithm: `algo.name` = `dqn` / `ddqn`
- model: hidden layer sizes, dueling head on/off
- replay buffer: type (`uniform` / `per`), capacity, n-step, γ
- optimization: learning rate, batch size, target update τ
- logging: TensorBoard logdir, evaluation frequency
- checkpointing: directory, filename

The training script reads the config and can be patched from the notebook using
simple YAML updates (see `01_cartpole_experiments.ipynb`).

---

## Usage

### Train CartPole from CLI

Default run (uses `configs/cartpole.yaml`):

```bash
python -m train.train_cartpole --config configs/cartpole.yaml
```

or via helper script:

```bash
# Linux / macOS
./run_cartpole.sh

# Windows
run_cartpole.bat
```

During training you will see a progress bar with:

- current step
- episode index
- moving average return (MA100)
- current loss and ε.

### Evaluate a Saved Checkpoint

After training, you can evaluate a model using `train/eval.py`:

```bash
python -m train.eval   --env CartPole-v1   --checkpoint path/to/cartpole_best_ma100.pth   --episodes 10   --record artifacts/videos   --seed 42
```

The script will:

- load the MLP model (Dueling or plain, depending on the config);
- run several episodes with greedy/ε-greedy policy (here ε=0, fully greedy);
- optionally record episodes as `.mp4` (if `--record` is provided).

### TensorBoard

If TensorBoard logging is enabled in the config (default):

```bash
tensorboard --logdir artifacts/tensorboard
```

You will see:

- `train/episode_return`
- `train/loss`
- `train/epsilon`

and other scalars depending on the config.

---

## Reproducing the Experiments

Notebook `notebooks/01_cartpole_experiments.ipynb` contains:

- Baseline **DQN** run;
- **DDQN** with target network;
- Adding **Dueling** head;
- Switching replay to **PER + n-step**;
- Comparison of learning curves (moving average return).

The notebook uses the same CLI entrypoint (`train.train_cartpole`) and
updates YAML configs on the fly.

---

## Results (CartPole-v1)

- Environment considered **solved** when `MA100 ≥ 400`.
- Final configuration (DDQN + PER + n-step + dueling) achieves:

  - `best_moving_avg_100 ≈ 420–430`
  - stable convergence with target network and soft updates

(Exact numbers may vary due to different random seeds.)

---

## License

MIT License — free to use for learning, experiments and portfolio projects.
