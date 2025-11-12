# train/train_pong.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Pong DQN/DDQN тренер с поддержкой:
- ReplayBuffer / PrioritizedReplayBuffer (PER) и n-step
- make_atari_env из envs.atari_wrappers
- DDQN target (онлайн argmax → target gather)
- линейные расписания epsilon и beta (PER)
- запись TB-логов, чекпоинтов и мета-полей (в т.ч. dueling)
"""

import time
import yaml
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

# === корректные импорты согласно структуре проекта ===
from envs.atari_wrappers import make_atari_env
from models.dqn_cnn import NatureCNN
from memory.replay_buffer import ReplayBuffer
from memory.per_buffer import PrioritizedReplayBuffer
from utils.schedule import LinearSchedule


def dqn_target(q_next_target, rewards, dones, discounts):
    # DQN target: max_a' Q_target(s', a')
    max_next = q_next_target.max(dim=1).values
    return rewards + discounts * (1.0 - dones) * max_next


def ddqn_target(q_next_online, q_next_target, rewards, dones, discounts):
    # DDQN target: a* = argmax_a Q_online(s', a); y = r + γ^n (1-done) * Q_target(s', a*)
    a_star = q_next_online.argmax(dim=1)
    q_tgt  = q_next_target.gather(1, a_star.view(-1, 1)).squeeze(1)
    return rewards + discounts * (1.0 - dones) * q_tgt


def hard_update(target_net: torch.nn.Module, source_net: torch.nn.Module) -> None:
    target_net.load_state_dict(source_net.state_dict())


def train(config_path: str) -> None:
    # === cfg ===
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    # === env ===
    ecfg = cfg.get("env", {})
    frame_stack      = int(ecfg.get("frame_stack", 4))
    minimal_actions  = bool(ecfg.get("minimal_actions", True))
    seed             = int(ecfg.get("seed", 42))

    env = make_atari_env(
        env_id="PongNoFrameskip-v4",
        frame_stack=frame_stack,
        minimal_actions=minimal_actions,
        seed=seed,
        render_mode=None,
    )

    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === model ===
    mcfg = cfg.get("model", {})
    dueling = bool(mcfg.get("dueling", True))
    online = NatureCNN(in_channels=frame_stack, act_dim=n_actions, dueling=dueling).to(device)
    target = NatureCNN(in_channels=frame_stack, act_dim=n_actions, dueling=dueling).to(device)
    hard_update(target, online)

    # === replay ===
    rcfg = cfg.get("replay", {})
    rtype       = str(rcfg.get("type", "uniform")).lower()  # uniform | per
    capacity    = int(rcfg.get("capacity", 1_000_000))
    n_step      = int(cfg.get("algo", {}).get("n_step", 1))
    gamma       = float(cfg.get("algo", {}).get("gamma", 0.99))

    if rtype == "per":
        alpha       = float(rcfg.get("alpha", 0.5))
        beta_start  = float(rcfg.get("beta_start", 0.4))
        beta_end    = float(rcfg.get("beta_end", 1.0))
        buffer = PrioritizedReplayBuffer(
            capacity=capacity, alpha=alpha, beta_start=beta_start, beta_end=beta_end,
            n_step=n_step, gamma=gamma
        )
        beta_sched = LinearSchedule(beta_start, beta_end, int(cfg.get("train", {}).get("total_frames", 3_000_000)))
        print(f"[replay] PER(cap={capacity}, alpha={alpha}, beta {beta_start}->{beta_end}, n={n_step})")
    else:
        buffer = ReplayBuffer(capacity=capacity, n_step=n_step, gamma=gamma)
        beta_sched = None
        print(f"[replay] Uniform(cap={capacity}, n={n_step})")

    # === epsilon schedule ===
    epscfg = cfg.get("epsilon", {})
    eps_sched = LinearSchedule(
        start=float(epscfg.get("start", 1.0)),
        end=float(epscfg.get("end", 0.1)),
        steps=int(epscfg.get("decay_steps", 1_000_000)),
    )
    print(f"[sched] epsilon {epscfg.get('start',1.0)} -> {epscfg.get('end',0.1)} over {epscfg.get('decay_steps',1_000_000)}")

    # === optim ===
    ocfg   = cfg.get("optim", {})
    opt_nm = str(ocfg.get("name", "adam")).lower()
    lr     = float(ocfg.get("lr", 2.5e-4))
    if opt_nm == "adam":
        opt = optim.Adam(online.parameters(), lr=lr)
    else:
        # классический RMSprop из DQN
        opt = optim.RMSprop(online.parameters(), lr=6.25e-5, alpha=0.95, eps=0.01)

    # === algo ===
    algo_name = str(cfg.get("algo", {}).get("name", "ddqn")).lower()  # dqn | ddqn

    # === train params ===
    tcfg          = cfg.get("train", {})
    total_frames  = int(tcfg.get("total_frames", 3_000_000))
    warmup_steps  = int(tcfg.get("warmup_steps", 20_000))
    batch_size    = int(tcfg.get("batch_size", 32))
    target_update = int(tcfg.get("target_update_freq", 10_000))
    save_every    = int(tcfg.get("save_every", 250_000))

    # === logging & ckpt ===
    lcfg = cfg.get("logging", {})
    tb_dir     = Path(lcfg.get("tb_dir", "artifacts_pong/tensorboard"))
    run_name   = str(lcfg.get("run_name", f"run_{int(time.time())}"))
    log_int    = int(lcfg.get("log_interval", 1000))
    ckpt_dir   = Path(lcfg.get("ckpt_dir", "artifacts_pong/checkpoints")) / run_name / time.strftime("%Y%m%d-%H%M%S")

    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir / run_name / time.strftime("%Y%m%d-%H%M%S")))

    def save_ckpt(tag: str):
        path = ckpt_dir / f"{tag}.pth"
        torch.save(
            {
                "model": online.state_dict(),
                "target": target.state_dict(),
                "opt": opt.state_dict(),
                "global_step": global_step,
                "cfg": cfg,
                "dueling": dueling,
            },
            str(path),
        )
        print(f"[ckpt] Saved: {path}")

    # === loop ===
    global_step = 0
    nzr_window = deque(maxlen=500)

    s, _ = env.reset(seed=seed)
    print(f"[train] start total_frames={total_frames} warmup={warmup_steps} algo={algo_name} dueling={dueling} actions={n_actions}")

    while global_step < total_frames:
        # epsilon / beta
        eps_val = float(eps_sched.value(global_step))
        if beta_sched is not None:
            buffer.beta = float(beta_sched.value(global_step))

        # действие (онлайн инференс) — гарантируем NCHW
        if np.random.rand() < eps_val:
            a = env.action_space.sample()
        else:
            arr = np.asarray(s, dtype=np.float32) / 255.0           # [H,W,C] или [C,H,W]
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):           # NHWC → NCHW
                arr = np.transpose(arr, (2, 0, 1))
            t = torch.from_numpy(arr)[None].to(device)               # [1,C,H,W]
            with torch.no_grad():
                q = online(t)
                a = int(q.argmax(dim=1).item())

        s2, r, terminated, truncated, _ = env.step(a)
        done = bool(terminated or truncated)
        nzr_window.append(1 if r != 0.0 else 0)
        buffer.push(s, a, float(r), s2, done)
        s = s2

        # обучение
        if global_step >= warmup_steps and len(buffer) >= batch_size:
            if rtype == "per":
                s_b, a_b, r_b, ns_b, d_b, disc_b, w_b, idxs = buffer.sample(batch_size)
                w_t = torch.as_tensor(w_b, dtype=torch.float32, device=device)
            else:
                s_b, a_b, r_b, ns_b, d_b, disc_b = buffer.sample(batch_size)
                w_t = None

            # NHWC -> NCHW пакетно, если надо
            if s_b.ndim == 4 and s_b.shape[-1] in (3, 4):
                s_b  = np.transpose(s_b,  (0, 3, 1, 2))
                ns_b = np.transpose(ns_b, (0, 3, 1, 2))

            s_t  = torch.as_tensor(s_b,  dtype=torch.float32, device=device) / 255.0
            ns_t = torch.as_tensor(ns_b, dtype=torch.float32, device=device) / 255.0
            a_t  = torch.as_tensor(a_b,  dtype=torch.int64, device=device).view(-1, 1)
            r_t  = torch.as_tensor(r_b,  dtype=torch.float32, device=device)
            d_t  = torch.as_tensor(d_b,  dtype=torch.float32, device=device)
            disc_t = torch.as_tensor(disc_b, dtype=torch.float32, device=device)

            q = online(s_t).gather(1, a_t).squeeze(1)

            with torch.no_grad():
                q_next_online = online(ns_t)
                q_next_target = target(ns_t)
                if algo_name == "ddqn":
                    y = ddqn_target(q_next_online, q_next_target, r_t, d_t, disc_t)
                else:
                    y = dqn_target(q_next_target, r_t, d_t, disc_t)

            td = y - q
            if w_t is not None:
                loss = (w_t * F.smooth_l1_loss(q, y, reduction="none")).mean()
            else:
                loss = F.smooth_l1_loss(q, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
            opt.step()

            # обновление приоритетов для PER
            if rtype == "per":
                buffer.update_priorities(idxs, td.detach().cpu().numpy())

            # hard update target-сети
            if global_step % target_update == 0:
                hard_update(target, online)

            # логи
            if global_step % log_int == 0 and global_step > 0:
                nzr = float(np.mean(nzr_window)) if nzr_window else 0.0
                writer.add_scalar("train/loss", float(loss.item()), global_step)
                writer.add_scalar("train/epsilon", eps_val, global_step)
                writer.add_scalar("train/nonzero_ratio", nzr, global_step)
                if beta_sched is not None:
                    writer.add_scalar("train/per_beta", float(buffer.beta), global_step)

        # конец эпизода
        if done:
            s, _ = env.reset()
            nzr_window.clear()

        # чекпоинт
        if (global_step > 0) and (global_step % save_every == 0):
            save_ckpt(f"pong_step_{global_step:07d}")

        global_step += 1

    save_ckpt("final")
    writer.close()
    env.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
