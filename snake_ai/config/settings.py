from __future__ import annotations
from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # ── Game ──────────────────────────────────────────────────────────
    grid_w: int = 32
    grid_h: int = 24

    # ── Training ──────────────────────────────────────────────────────
    memory_size: int = 100_000
    batch_size: int = 256
    lr: float = 1e-4
    gamma: float = 0.99        # longer planning horizon than old 0.9

    # ── Epsilon-greedy ────────────────────────────────────────────────
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9995  # multiplicative per step - never negative

    # ── Prioritized Experience Replay ─────────────────────────────────
    per_alpha: float = 0.6          # priority exponent
    per_beta_start: float = 0.4     # IS weight annealing start
    per_beta_end: float = 1.0       # IS weight annealing end
    per_beta_anneal_steps: int = 500_000

    # ── Target network ────────────────────────────────────────────────
    target_update_freq: int = 1_000  # steps between hard target syncs

    # ── Checkpointing ─────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    save_every_n_games: int = 100

    # ── Visualization ─────────────────────────────────────────────────
    render_every_n_games: int = 0    # 0 = never render during training
    plot_update_freq: int = 10       # update terminal plot every N games

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
