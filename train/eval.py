# train/eval.py
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# --- make sure project root is on sys.path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import torch
import imageio

from envs.cartpole_wrappers import make_cartpole
from envs.atari_wrappers import make_pong
from models.dqn_mlp import MLPQ, DuelingMLP
from models.dqn_cnn import NatureCNN
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, required=True, help='CartPole-v1 or PongNoFrameskip-v4')
    p.add_argument('--checkpoint', type=str, required=False, help='path to .pth')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--record', type=str, default=None, help='folder to save mp4')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_dueling_flag(default: bool = True) -> bool:
    """Try to read dueling flag from configs/cartpole.yaml (fallback to default)."""
    cfg_path = ROOT / "configs" / "cartpole.yaml"
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
            return bool(cfg.get("agent", {}).get("dueling", default))
        except Exception:
            return default
    return default


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Build env & network ---
    if 'Pong' in args.env:
        env = make_pong(render_mode='rgb_array')  # returns uint8 (H,W,C) stacked frames
        obs, _ = env.reset()
        act_dim = env.action_space.n
        # for Pong we always use CNN; dueling head can be toggled if needed
        qnet = NatureCNN(in_channels=obs.shape[-1], act_dim=act_dim, dueling=True).to(device)
    else:
        env = make_cartpole(render_mode='rgb_array')
        obs, _ = env.reset()
        act_dim = env.action_space.n
        dueling = load_dueling_flag(default=True)
        qnet = (DuelingMLP(obs.shape[0], act_dim) if dueling else MLPQ(obs.shape[0], act_dim)).to(device)

    qnet.eval()

    # --- Load checkpoint (optional) ---
    if args.checkpoint:
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
            if 'Pong' in args.env:
                # obs: uint8 (H,W,C); network expects BCHW uint8
                with torch.no_grad():
                    x = torch.as_tensor(obs, dtype=torch.uint8, device=device).unsqueeze(0)  # B,H,W,C
                    x = x.permute(0, 3, 1, 2)  # B,C,H,W
                    q = qnet(x)
                    act = int(torch.argmax(q, dim=1).item())
            else:
                with torch.no_grad():
                    x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q = qnet(x)
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
