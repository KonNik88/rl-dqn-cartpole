# memory/replay_buffer.py
from __future__ import annotations
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


class NStepHelper:
    """
    Аккумулирует n-step возврат.
    Подаём по одному переходу (s, a, r, s', done).
    Когда накопилось >= n шагов или пришёл done — возвращаем свёрнутый n-step переход.
    Иначе возвращаем None.
    На done дополнительно нужно «слить хвост» — вызвать finalize_episode(),
    который отдаст оставшиеся (n-1) свёрнутых переходов.
    """

    def __init__(self, n: int = 1, gamma: float = 0.99):
        assert n >= 1
        self.n = n
        self.gamma = gamma
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque()

    def _compute_return(self) -> Tuple[np.ndarray, int, float, np.ndarray, bool, float]:
        """
        Сворачивает текущий буфер на первых n шагов (или меньше, если буфер короче при finalize).
        Возвращает (s0, a0, R_n, s_k, done_k, discount),
        где discount = gamma^k, k = фактическое число сложенных шагов.
        """
        R, discount = 0.0, 1.0
        k = 0
        for (_, _, r, _, _done) in list(self.buf)[: self.n]:
            R += discount * r
            discount *= self.gamma
            k += 1
            if _done:
                break

        s0, a0, _, _, _ = self.buf[0]
        s_k, done_k = self.buf[k - 1][3], self.buf[k - 1][4]
        discount_k = discount  # gamma^k

        return s0, a0, R, s_k, done_k, discount_k

    def push(self, tr: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> Optional[Tuple]:
        """
        Добавляет один шаг. Если готов n-step переход — возвращает его, иначе None.
        При done — сразу вернёт один переход (если что-то было в буфере),
        а остальное сольётся через finalize_episode().
        """
        self.buf.append(tr)

        if len(self.buf) >= self.n:
            s0, a0, R, s_k, done_k, discount_k = self._compute_return()
            self.buf.popleft()
            return (s0, a0, R, s_k, done_k, discount_k)

        if self.buf and self.buf[-1][4]:  # последний был done
            s0, a0, R, s_k, done_k, discount_k = self._compute_return()
            return (s0, a0, R, s_k, done_k, discount_k)

        return None

    def finalize_episode(self):
        """
        Вызывать при done. Сливает оставшиеся (n-1) переходов, если они есть.
        Возвращает список n-step переходов (может быть пустым).
        """
        out = []
        while self.buf:
            s0, a0, R, s_k, done_k, discount_k = self._compute_return()
            self.buf.popleft()
            out.append((s0, a0, R, s_k, done_k, discount_k))
        return out


class ReplayBuffer:
    """
    Простой uniform replay с поддержкой n-step.
    Не делает предположений о форме наблюдений:
      - для CartPole это будет (4,)
    """

    def __init__(self, capacity: int, n_step: int = 1, gamma: float = 0.99):
        self.capacity = int(capacity)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.pos = 0
        self.full = False

        self.obs = None
        self.next_obs = None
        self.actions = np.empty(self.capacity, dtype=np.int32)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.dones = np.empty(self.capacity, dtype=np.bool_)
        self.discounts = np.empty(self.capacity, dtype=np.float32)

        self.nhelper = NStepHelper(n=self.n_step, gamma=self.gamma)
        self.size = 0

    def __len__(self):
        return self.size

    def _init_obs_arrays(self, obs: np.ndarray):
        """
        Инициализируем массивы под наблюдения по первой форме obs.
        Поддерживаем любую форму (вектор, картинка, стек).
        """
        arr = np.asarray(obs)
        if arr.ndim == 0:
            raise ValueError(f"obs must have at least 1 dim, got scalar {arr!r}")
        self.obs = np.empty((self.capacity,) + arr.shape, dtype=arr.dtype)
        self.next_obs = np.empty_like(self.obs)

    def push(self, obs, action: int, reward: float, next_obs, done: bool):
        """
        Кладёт переход с учётом n-step.
        Может записать 0 или 1 элемент сейчас и (если done) — ещё несколько из finalize_episode().
        """
        if self.obs is None:
            self._init_obs_arrays(obs)

        def to_arr(x):
            return np.asarray(x, dtype=self.obs.dtype)

        tr = (to_arr(obs), int(action), float(reward), to_arr(next_obs), bool(done))
        out = self.nhelper.push(tr)

        def _write(s0, a0, R, s_k, done_k, discount_k):
            idx = self.pos
            self.obs[idx] = s0
            self.actions[idx] = a0
            self.rewards[idx] = R
            self.next_obs[idx] = s_k
            self.dones[idx] = done_k
            self.discounts[idx] = discount_k
            self.pos = (self.pos + 1) % self.capacity
            self.full = self.full or (self.pos == 0)
            self.size = min(self.size + 1, self.capacity)

        if out is not None:
            _write(*out)

        if done:
            tails = self.nhelper.finalize_episode()
            for trn in tails:
                _write(*trn)

    def sample(self, batch_size: int):
        assert self.size > 0, "Buffer is empty"
        idxs = np.random.randint(0, self.size, size=batch_size)
        s = self.obs[idxs]
        a = self.actions[idxs]
        r = self.rewards[idxs]
        ns = self.next_obs[idxs]
        d = self.dones[idxs].astype(np.float32)
        disc = self.discounts[idxs]
        return s, a, r, ns, d, disc
