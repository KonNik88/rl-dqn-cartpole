# memory/per_buffer.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .replay_buffer import NStepHelper


class SumTree:
    """
    SumTree для PER: хранит приоритеты в виде сегментного дерева.
    Листья: приоритеты переходов, внутренние узлы: суммы поддеревьев.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.size = 0
        self.write = 0
        self.tree = np.zeros(2 * self.capacity, dtype=np.float32)

    def _update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        i = tree_idx // 2
        while i >= 1:
            self.tree[i] += change
            i //= 2

    def add(self, priority: float):
        leaf_idx = self.capacity + self.write
        self._update(leaf_idx, float(priority))
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float):
        self._update(leaf_idx, float(priority))

    def total(self) -> float:
        return float(self.tree[1])

    def min(self) -> float:
        if self.size == 0:
            return 0.0
        leaves = self.tree[self.capacity : self.capacity + self.size]
        m = float(leaves.min(initial=np.inf))
        return m if np.isfinite(m) else 0.0

    def get_leaf(self, value: float) -> Tuple[int, float]:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx, float(self.tree[idx])


class PrioritizedReplayBuffer:
    """
    PER с n-step и хранением наблюдений произвольной формы.
    Интерфейс совместим с train_cartpole:
      - sample(batch_size) -> s, a, r, ns, d, disc, w, idxs
      - update_priorities(idxs, new_priorities)
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        n_step: int = 1,
        gamma: float = 0.99,
        eps: float = 1e-6,
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta_end)  # можно по желанию сделать расписание
        self.eps = float(eps)
        self.nhelper = NStepHelper(n=n_step, gamma=gamma)

        self.pos = 0
        self.size = 0
        self.full = False
        self.obs = None
        self.next_obs = None
        self.actions = np.empty(self.capacity, dtype=np.int32)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.dones = np.empty(self.capacity, dtype=np.bool_)
        self.discounts = np.empty(self.capacity, dtype=np.float32)

        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0

    def __len__(self):
        return self.size

    def _init_obs_arrays(self, obs: np.ndarray):
        arr = np.asarray(obs)
        if arr.ndim == 0:
            raise ValueError(f"obs must have at least 1 dim, got scalar {arr!r}")
        self.obs = np.empty((self.capacity,) + arr.shape, dtype=arr.dtype)
        self.next_obs = np.empty_like(self.obs)

    def _write(self, s0, a0, R, s_k, done_k, discount_k, priority: Optional[float] = None):
        if self.obs is None:
            self._init_obs_arrays(s0)

        idx = self.pos
        self.obs[idx] = s0
        self.actions[idx] = a0
        self.rewards[idx] = R
        self.next_obs[idx] = s_k
        self.dones[idx] = done_k
        self.discounts[idx] = discount_k

        p = self.max_priority if priority is None else float(priority)
        self.tree.add(p ** self.alpha)

        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or (self.pos == 0)
        self.size = min(self.size + 1, self.capacity)

    def push(self, obs, action: int, reward: float, next_obs, done: bool):
        """
        Добавляет переход с учётом n-step. Может записать 0 или 1 (и хвост на finalize).
        """
        arr_obs = np.asarray(obs)
        arr_next = np.asarray(next_obs)
        out = self.nhelper.push((arr_obs, int(action), float(reward), arr_next, bool(done)))

        if out is not None:
            self._write(*out, priority=None)

        if done:
            tails = self.nhelper.finalize_episode()
            for trn in tails:
                self._write(*trn, priority=None)

    def sample(self, batch_size: int):
        assert self.size > 0, "Buffer is empty"
        batch_idx = np.empty(batch_size, dtype=np.int32)
        batch_p = np.empty(batch_size, dtype=np.float32)

        total = self.tree.total()
        total = max(total, 1e-8)
        segment = total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            leaf_idx, p = self.tree.get_leaf(s)
            data_idx = (leaf_idx - self.capacity) % self.capacity
            if data_idx >= self.size:
                data_idx = np.random.randint(0, self.size)

            batch_idx[i] = data_idx
            batch_p[i] = p

        s = self.obs[batch_idx]
        a = self.actions[batch_idx]
        r = self.rewards[batch_idx]
        ns = self.next_obs[batch_idx]
        d = self.dones[batch_idx].astype(np.float32)
        disc = self.discounts[batch_idx]

        probs = batch_p / total
        probs = np.clip(probs, 1e-12, 1.0)
        w = (self.size * probs) ** (-self.beta)
        w /= w.max() if w.max() > 0 else 1.0

        idxs = batch_idx.copy()
        return s, a, r, ns, d, disc, w.astype(np.float32), idxs

    def update_priorities(self, idxs: np.ndarray, new_priorities: np.ndarray):
        """
        Обновляет приоритеты по индексам элементов в буфере.
        Ожидает idxs — индексы данных (0..size-1), а не индексы листьев.
        """
        new_p = np.abs(new_priorities.astype(np.float32)) + self.eps
        self.max_priority = max(self.max_priority, float(new_p.max(initial=0.0)))
        for data_idx, p in zip(idxs, new_p):
            data_idx = int(data_idx) % self.capacity
            leaf_idx = self.capacity + data_idx
            self.tree.update(leaf_idx, float(p) ** self.alpha)
