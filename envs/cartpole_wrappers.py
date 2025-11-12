from __future__ import annotations
import gymnasium as gym

def make_cartpole(seed: int = 42, render_mode: str | None = None):
    env = gym.make("CartPole-v1", render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
