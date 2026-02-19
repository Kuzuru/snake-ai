from __future__ import annotations
from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    grid_w: int = 20
    grid_h: int = 20

    memory_size: int = 100_000
    batch_size: int = 256
    lr: float = 1e-4
    gamma: float = 0.99

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9999

    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 500_000

    tau: float = 0.005

    checkpoint_dir: str = "checkpoints"
    save_every_n_games: int = 100

    render_every_n_games: int = 0
    plot_update_freq: int = 10

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
