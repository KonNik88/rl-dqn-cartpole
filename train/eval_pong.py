# train/eval_pong.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from envs.atari_wrappers import make_atari_env
from models.dqn_cnn import NatureCNN


def load_ckpt(path: Path, device: torch.device):
    data = torch.load(str(path), map_location=device, weights_only=True)
    state_dict = data.get("model", data)
    meta = {k: v for k, v in data.items() if k != "model"}
    return state_dict, meta


@torch.no_grad()
def evaluate(
    ckpt: str,
    episodes: int = 5,
    epsilon: float = 0.01,
    frame_stack: int = 4,
    dueling: Optional[bool] = None,
    minimal_actions: bool = True,
    render: bool = False,
    video_dir: Optional[str] = None,
    seed: Optional[int] = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Для записи видео нужен rgb_array
    render_mode = "human" if (render and not video_dir) else ("rgb_array" if video_dir else None)

    env = make_atari_env(
        env_id="PongNoFrameskip-v4",
        frame_stack=frame_stack,
        minimal_actions=minimal_actions,
        seed=seed,
        render_mode=render_mode,
    )

    if video_dir:
        from gymnasium.wrappers import RecordVideo
        os.makedirs(video_dir, exist_ok=True)
        # Gymnasium API: video_folder=..., а не video_dir
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda e: True)

    n_actions = env.action_space.n
    state_dict, meta = load_ckpt(Path(ckpt), device)

    if dueling is None:
        dueling = bool(meta.get("dueling", True))
        print(f"[eval] dueling from ckpt: {dueling}")

    model = NatureCNN(in_channels=frame_stack, act_dim=n_actions, dueling=dueling).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[eval][warn] missing={missing}, unexpected={unexpected}")
    model.eval()

    def to_tensor(s):
        # Без copy=False — совместимо с NumPy 2.x и LazyFrames
        arr = np.asarray(s, dtype=np.float32) / 255.0  # [H,W,C] или [C,H,W]
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):  # NHWC → NCHW
            arr = np.transpose(arr, (2, 0, 1))
        t = torch.from_numpy(arr)[None]  # [1,C,H,W]
        return t.to(device)

    returns = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed)
        done = False
        ep_ret = 0.0
        while not done:
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                q = model(to_tensor(s))
                a = int(q.argmax(dim=1).item())
            s, r, term, trunc, _ = env.step(a)
            ep_ret += float(r)
            done = bool(term or trunc)

        returns.append(ep_ret)
        print(f"[eval] episode {ep+1}/{episodes} return={ep_ret:.2f}")

    env.close()
    mean_ret = float(np.mean(returns)) if returns else float("nan")
    print(f"[eval] mean_return over {episodes} episodes: {mean_ret:.2f}")
    return returns, mean_ret


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--frame_stack", type=int, default=4)
    p.add_argument("--dueling", type=str, default=None, help="true/false/None")
    p.add_argument("--minimal_actions", type=int, default=1)
    p.add_argument("--render", action="store_true")
    p.add_argument("--video_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.dueling is not None:
        s = args.dueling.lower()
        if s in ("true", "1", "yes", "y"):
            args.dueling = True
        elif s in ("false", "0", "no", "n"):
            args.dueling = False
        else:
            args.dueling = None
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        ckpt=args.ckpt,
        episodes=args.episodes,
        epsilon=args.epsilon,
        frame_stack=args.frame_stack,
        dueling=args.dueling,
        minimal_actions=bool(args.minimal_actions),
        render=args.render,
        video_dir=args.video_dir,
        seed=args.seed,
    )
