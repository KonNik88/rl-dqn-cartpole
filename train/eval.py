# train/eval.py
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import torch
import imageio

from envs.cartpole_wrappers import make_cartpole
from models.dqn_mlp import MLPQ, DuelingMLP
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default="CartPole-v1", help='Gymnasium env id (default: CartPole-v1)')
    p.add_argument('--checkpoint', type=str, required=True, help='path to .pth with model weights')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--record', type=str, default=None, help='folder to save mp4 (optional)')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_dueling_flag(default: bool = True) -> bool:
    """Try to read dueling flag from configs/cartpole.yaml (fallback to default)."""
    cfg_path = ROOT / "configs" / "cartpole.yaml"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return bool(cfg.get("agent", {}).get("dueling", default))
        except Exception:
            return default
    return default


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Build env & network ---
    env = make_cartpole(seed=args.seed, render_mode='rgb_array')
    obs, _ = env.reset()
    act_dim = env.action_space.n
    obs_dim = int(np.prod(obs.shape))

    dueling = load_dueling_flag(default=True)
    if dueling:
        qnet = DuelingMLP(obs_dim=obs_dim, act_dim=act_dim).to(device)
        print("Using DuelingMLP")
    else:
        qnet = MLPQ(obs_dim=obs_dim, act_dim=act_dim).to(device)
        print("Using MLPQ")

    # --- Load checkpoint ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    qnet.load_state_dict(state)
    print(f"Loaded checkpoint: {ckpt_path.name}")

    # --- Prepare recorder ---
    rec_dir = Path(args.record) if args.record else None
    if rec_dir:
        rec_dir.mkdir(parents=True, exist_ok=True)

    # --- Evaluate episodes ---
    for ep in range(args.episodes):
        frames, ret = [], 0.0
        obs, _ = env.reset()
        done = False

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = qnet(obs_tensor)
                act = int(torch.argmax(q, dim=1).item())

            obs, r, terminated, truncated, _info = env.step(act)
            done = terminated or truncated
            ret += float(r)

            frame = env.render()  # rgb_array
            if frame is not None:
                frames.append(frame)

        print(f"Episode {ep+1}: return={ret:.2f}")

        if rec_dir and frames:
            out = rec_dir / f"{args.env}_ep{ep+1}.mp4"
            imageio.mimsave(out, frames, fps=30)
            print(f"Saved video to {out}")

    print("Evaluation finished.")


if __name__ == "__main__":
    main()
