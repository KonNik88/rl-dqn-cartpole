from __future__ import annotations
import argparse, yaml, time, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import gymnasium as gym
import sys
from pathlib import Path
from tqdm import tqdm

from envs.cartpole_wrappers import make_cartpole
from models.dqn_mlp import MLPQ, DuelingMLP
from memory.replay_buffer import ReplayBuffer
from memory.per_buffer import PrioritizedReplayBuffer
from utils.seed import set_seed
from utils.schedule import LinearSchedule
from utils.logger import TBLogger

def parse_config():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    args = p.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def select_action(qnet, obs, eps, act_dim, device):
    if np.random.rand() < eps:
        return np.random.randint(act_dim)
    with torch.no_grad():
        q = qnet(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        return int(torch.argmax(q, dim=1).item())

def main():
    cfg = parse_config()
    seed = cfg.get('seed', 42)
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- env
    env = make_cartpole(seed=seed)
    obs, _ = env.reset()
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.n

    # --- cfg blocks with safe defaults
    agent_cfg = cfg.get('agent', {})
    net_cfg   = cfg.get('network', {})
    rep_cfg   = cfg.get('replay', {})
    expl_cfg  = cfg.get('exploration', {})
    log_cfg   = cfg.get('logging', {})
    ckpt_cfg  = cfg.get('checkpointing', {})

    # --- nets
    q_cls = DuelingMLP if agent_cfg.get('dueling', False) else MLPQ
    hidden = tuple(net_cfg.get('hidden_sizes', [256, 256]))
    layer_norm = bool(net_cfg.get('layer_norm', True))

    qnet = q_cls(obs_dim, act_dim, hidden, layer_norm).to(device)
    tgt  = q_cls(obs_dim, act_dim, hidden, layer_norm).to(device)
    tgt.load_state_dict(qnet.state_dict()); tgt.eval()

    # --- opt & alg params
    optim_q = optim.Adam(qnet.parameters(), lr=float(agent_cfg.get('lr', 5e-4)))
    gamma   = float(agent_cfg.get('gamma', 0.99))
    tau     = float(agent_cfg.get('tau', 0.005))
    n_step  = int(agent_cfg.get('n_step', 1))
    grad_clip = float(agent_cfg.get('grad_clip', 10.0))
    use_ddqn  = bool(agent_cfg.get('double_dqn', True))

    # --- buffer
    if rep_cfg.get('type', 'uniform') == 'per':
        buf = PrioritizedReplayBuffer(rep_cfg.get('capacity', 50000),
                                      float(rep_cfg.get('per_alpha', 0.6)),
                                      float(rep_cfg.get('per_beta_start', 0.4)),
                                      float(rep_cfg.get('per_beta_end', 1.0)))
    else:
        buf = ReplayBuffer(rep_cfg.get('capacity', 50000), n_step=n_step, gamma=gamma)

    # --- schedules & logging
    eps_sched = LinearSchedule(expl_cfg.get('eps_start', 1.0),
                               expl_cfg.get('eps_end', 0.05),
                               int(expl_cfg.get('eps_decay_steps', 20000)))
    tb_dir = Path(log_cfg.get('log_dir', 'artifacts')) / "tensorboard"
    logger = TBLogger(str(tb_dir))

    # --- training control
    total_steps = int(cfg.get('total_steps', 150000))
    warmup      = int(rep_cfg.get('warmup_steps', 1000))
    batch       = int(rep_cfg.get('batch_size', 64))

    # --- checkpointing
    CKPT_DIR = Path(ckpt_cfg.get('dir', 'artifacts/checkpoints'))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    save_every_ep  = int(ckpt_cfg.get('save_every_episodes', 0))  # 0 => off
    save_best_ma   = bool(ckpt_cfg.get('save_best_ma100', True))
    best_ma100     = -float("inf")

    step, episode, ep_return = 0, 0, 0.0
    returns_window = []
    pbar = tqdm(total=total_steps, desc="Training CartPole", ncols=0, file=sys.stdout)

    try:
        while step < total_steps:
            action = select_action(qnet, obs, eps_sched.value(), act_dim, device)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # push transition (PER/N-step сбоку совместимы)
            buf.push(obs, action, reward, next_obs, done)
            ep_return += reward
            obs = next_obs
            step += 1
            eps_sched.step()
            pbar.update(1)

            # learn after warmup
            if step >= warmup and len(buf) >= batch:
                if isinstance(buf, PrioritizedReplayBuffer):
                    s, a, r, ns, d, disc, w, idxs = buf.sample(batch)
                else:
                    s, a, r, ns, d, disc = buf.sample(batch)
                    w = np.ones_like(r, dtype=np.float32); idxs = None

                s    = torch.tensor(s,    dtype=torch.float32, device=device)
                a    = torch.tensor(a,    dtype=torch.long,    device=device)
                r    = torch.tensor(r,    dtype=torch.float32, device=device)
                ns   = torch.tensor(ns,   dtype=torch.float32, device=device)
                d    = torch.tensor(d,    dtype=torch.float32, device=device)
                disc = torch.tensor(disc, dtype=torch.float32, device=device)
                w_t  = torch.tensor(w,    dtype=torch.float32, device=device)

                with torch.no_grad():
                    if use_ddqn:
                        next_q   = qnet(ns)
                        next_act = torch.argmax(next_q, dim=1)
                        tgt_q    = tgt(ns).gather(1, next_act.unsqueeze(1)).squeeze(1)
                    else:
                        tgt_q    = tgt(ns).max(dim=1).values
                    y = r + disc * gamma * (1.0 - d) * tgt_q

                q = qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)
                td_errors = y - q
                loss = (w_t * (td_errors ** 2)).mean()

                optim_q.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(qnet.parameters(), grad_clip)
                optim_q.step()

                if isinstance(buf, PrioritizedReplayBuffer) and idxs is not None:
                    buf.update_priorities(idxs, td_errors.detach().cpu().numpy())

                # soft update
                with torch.no_grad():
                    for p, tp in zip(qnet.parameters(), tgt.parameters()):
                        tp.data.mul_(1.0 - tau).add_(tau * p.data)

                # logging
                if step % 500 == 0:
                    logger.log_scalar("train/loss", float(loss.item()), step)
                    logger.log_scalar("train/epsilon", float(eps_sched.value()), step)
                    pbar.set_postfix(loss=f"{loss.item():.3f}", eps=f"{eps_sched.value():.3f}", buf=len(buf))

            if done:
                episode += 1
                logger.log_scalar("train/episode_return", ep_return, episode)
                returns_window.append(ep_return)
                if len(returns_window) > 100:
                    returns_window.pop(0)
                ma100 = float(sum(returns_window) / len(returns_window))
                pbar.set_postfix(ep=episode, ret=f"{ep_return:.1f}", ma100=f"{ma100:.1f}",
                                 eps=f"{eps_sched.value():.3f}", buf=len(buf))
                # --- checkpointing ---
                if save_every_ep and (episode % save_every_ep == 0):
                    torch.save(qnet.state_dict(), CKPT_DIR / f"cartpole_ep{episode}.pth")
                if save_best_ma and ma100 > best_ma100 + 1e-9:
                    best_ma100 = ma100
                    torch.save(qnet.state_dict(), CKPT_DIR / "cartpole_best_ma100.pth")

                # reset episode
                obs, _ = env.reset()
                ep_return = 0.0
                if hasattr(buf, "finalize_episode"):
                    buf.finalize_episode()
    finally:
        pbar.close()
        logger.close()

if __name__ == "__main__":
    main()

