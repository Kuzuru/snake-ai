from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn


class DuelingCNN(nn.Module):
    """
    Dueling Double DQN network with two input branches:
      - CNN branch:    processes the (3, H, W) grid
      - Scalar branch: processes the 8 hand-crafted features

    Output: Q-values tensor of shape (B, n_actions).

    Dueling formula: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        n_scalars: int = 8,
        n_actions: int = 3,
    ) -> None:
        super().__init__()

        # ── CNN branch ────────────────────────────────────────────────
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # grid-size agnostic
        )
        cnn_out = 64 * 4 * 4  # 1024

        # ── Scalar branch ─────────────────────────────────────────────
        self.scalar_net = nn.Sequential(
            nn.Linear(n_scalars, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        combined = cnn_out + 64  # 1088

        # ── Shared FC ─────────────────────────────────────────────────
        # LayerNorm (not BatchNorm) - behaves correctly at batch_size=1 during inference
        self.shared = nn.Sequential(
            nn.Linear(combined, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # ── Dueling streams ───────────────────────────────────────────
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions),
        )

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid:    (B, 3, H, W) float32
            scalars: (B, 8)       float32
        Returns:
            Q-values (B, n_actions)
        """
        cnn_feat = self.conv(grid).flatten(start_dim=1)       # (B, 1024)
        scalar_feat = self.scalar_net(scalars)                 # (B, 64)
        combined = torch.cat([cnn_feat, scalar_feat], dim=1)  # (B, 1088)
        shared = self.shared(combined)                         # (B, 512)

        value = self.value_stream(shared)                      # (B, 1)
        advantage = self.advantage_stream(shared)              # (B, n_actions)

        # Q = V + (A - mean(A))  - mean subtraction improves identifiability
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        grid_h: int,
        grid_w: int,
        device: torch.device,
        n_scalars: int = 8,
        n_actions: int = 3,
    ) -> "DuelingCNN":
        model = cls(grid_h, grid_w, n_scalars=n_scalars, n_actions=n_actions)
        model.load_state_dict(torch.load(path, map_location=device))
        return model.to(device)
