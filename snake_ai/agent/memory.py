from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Transition(NamedTuple):
    state_grid: np.ndarray      # (3, H, W) float32
    state_scalars: np.ndarray   # (8,) float32
    action: int                 # 0 / 1 / 2
    reward: float
    next_grid: np.ndarray       # (3, H, W) float32
    next_scalars: np.ndarray    # (8,) float32
    done: bool


class PrioritizedReplayBuffer:
    """
    Proportional PER using a flat array (simple O(N) sampling for buffer sizes
    up to 100k - fast enough and avoids sum-tree complexity).

    For very large buffers (>500k) a sum-tree implementation should be substituted.

    alpha: priority exponent  (0 = uniform, 1 = full PER)
    """

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self._buffer: list[Transition | None] = [None] * capacity
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0
        self._max_priority = 1.0

    def push(self, transition: Transition) -> None:
        self._buffer[self._pos] = transition
        self._priorities[self._pos] = self._max_priority ** self.alpha
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, beta: float
    ) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """
        Returns:
            transitions: list of Transition
            indices:     np.ndarray (B,) for priority update
            is_weights:  np.ndarray (B,) float32 importance-sampling weights
        """
        priorities = self._priorities[: self._size]
        probs = priorities / priorities.sum()
        indices = np.random.choice(self._size, batch_size, p=probs, replace=False)

        # IS weights: w_i = (N * P(i))^{-beta} / max(w)
        weights = (self._size * probs[indices]) ** (-beta)
        weights /= weights.max()

        transitions = [self._buffer[i] for i in indices]  # type: ignore[index]
        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        for idx, p in zip(indices, priorities):
            self._priorities[idx] = p
        self._max_priority = max(float(self._max_priority), float(priorities.max()))

    def __len__(self) -> int:
        return self._size
