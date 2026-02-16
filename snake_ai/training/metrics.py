from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class EpisodeStats:
    score: int = 0
    total_reward: float = 0.0
    steps: int = 0
    loss_sum: float = 0.0
    loss_count: int = 0

    @property
    def mean_loss(self) -> float | None:
        return self.loss_sum / self.loss_count if self.loss_count else None


class MetricsTracker:
    """Tracks rolling statistics across training episodes."""

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self.scores: list[int] = []
        self._recent: deque[int] = deque(maxlen=window)
        self.best_score: int = 0
        self.total_steps: int = 0

    def record(self, stats: EpisodeStats) -> None:
        self.scores.append(stats.score)
        self._recent.append(stats.score)
        self.best_score = max(self.best_score, stats.score)
        self.total_steps += stats.steps

    @property
    def mean_score(self) -> float:
        return sum(self._recent) / len(self._recent) if self._recent else 0.0

    @property
    def n_games(self) -> int:
        return len(self.scores)
