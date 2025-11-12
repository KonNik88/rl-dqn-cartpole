# envs/atari_wrappers.py
# -*- coding: utf-8 -*-
"""
Атари-обёртки для Pong:
- NoopResetEnv: случайные NOOP кадры на старте эпизода
- FireResetEnv: принудительный FIRE перед началом игры (если требуется)
- MaxAndSkipEnv: пропуск кадров с max-пулингом
- ProcessFrame84: приведение к 84x84 (grayscale)
- FrameStack: стек из K последних кадров
- ActionMapWrapper: перенумерация набора допустимых действий (например, [2,5])
- make_atari_env/make_pong: фабрики окружения

FIRE должен применяться до сужения action space; action-map применяем последним и только ОДИН раз.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from collections import deque
from typing import Iterable, List, Optional, Tuple, Union


# ---------- базовые обёртки ----------

class NoopResetEnv(gym.Wrapper):
    """
    Делает от 1 до noop_max действий NOOP после reset() для стохастизации старта.
    """
    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        # В Atari NOOP = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        if "FIRE" not in meanings:
            # нет FIRE — обёртка не требуется
            self.needs_fire = False
        else:
            self.needs_fire = True
            self.fire_action = meanings.index("FIRE")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not self.needs_fire:
            return obs, info

        # делаем FIRE; если эпизод внезапно завершился — ресет
        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """
    Повторяет действие skip раз, возвращая max по последним 2 кадрам.
    """
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = deque(maxlen=2)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[-1]) if len(self._obs_buffer) == 2 else obs
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class ProcessFrame84(gym.ObservationWrapper):
    """
    Переводит RGB 210x160x3 в grayscale 84x84 (как в Nature DQN).
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

    def observation(self, obs):
        # Atari obs обычно (H=210, W=160, C=3)
        img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        img = img[13:13+84, :]  # кроп до 84x84
        return img.astype(np.uint8)


class LazyFrames:
    """
   Не дублируем кадры.
    """
    def __init__(self, frames: List[np.ndarray]):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.stack(self._frames, axis=0)
            # (T, 84, 84)
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._force()[i]


class FrameStack(gym.Wrapper):
    """
    Стек последних k grayscale-кадров -> (k,84,84).
    """
    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert len(shp) == 2 and shp == (84, 84), "ProcessFrame84 должен идти перед FrameStack"
        self.observation_space = spaces.Box(low=0, high=255, shape=(k, 84, 84), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return LazyFrames(list(self.frames))


class ActionMapWrapper(gym.ActionWrapper):
    """
    Перенумеровывает набор действий.
    """
    def __init__(self, env: gym.Env, action_map: Iterable[int]):
        super().__init__(env)
        self._map = list(action_map)
        self.action_space = spaces.Discrete(len(self._map))

    def action(self, act: int) -> int:
        return self._map[act]


# ---------- фабрики окружения ----------

def make_atari_env(
    env_id: str = "PongNoFrameskip-v4",
    frame_stack: int = 4,
    noop_max: int = 30,
    skip: int = 4,
    minimal_actions: bool = True,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
):
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        # seed для воспроизводимости
        env.reset(seed=seed)

    # Стохастический старт
    env = NoopResetEnv(env, noop_max=noop_max)
    # FIRE (если он нужен конкретной игре)
    env = FireResetEnv(env)
    # Пропуск кадров + max
    env = MaxAndSkipEnv(env, skip=skip)
    # 84x84 gray
    env = ProcessFrame84(env)
    # (k,84,84)
    env = FrameStack(env, k=frame_stack)

    if minimal_actions:
        env = ActionMapWrapper(env, [2, 5])

    return env


def make_pong(
    frame_stack: int = 4,
    noop_max: int = 30,
    skip: int = 4,
    minimal_actions: bool = True,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
):
    return make_atari_env(
        env_id="PongNoFrameskip-v4",
        frame_stack=frame_stack,
        noop_max=noop_max,
        skip=skip,
        minimal_actions=minimal_actions,
        render_mode=render_mode,
        seed=seed,
    )
