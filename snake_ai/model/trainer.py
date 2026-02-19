from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .network import DuelingCNN


class DDQNTrainer:

    def __init__(
        self,
        online_net: DuelingCNN,
        target_net: DuelingCNN,
        lr: float = 1e-4,
        gamma: float = 0.99,
        device: torch.device | None = None,
    ) -> None:
        self.online = online_net
        self.target = target_net
        self.gamma = gamma
        self.device = device or torch.device("cpu")
        self.optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)
        self.target.load_state_dict(online_net.state_dict())
        self.target.eval()
        self._steps = 0

    def train_step(
        self,
        grids: torch.Tensor,
        scalars: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_grids: torch.Tensor,
        next_scalars: torch.Tensor,
        dones: torch.Tensor,
        is_weights: torch.Tensor,
    ) -> tuple[float, np.ndarray]:
        self.online.train()

        with torch.no_grad():
            next_q_online = self.online(next_grids, next_scalars)
            best_next_actions = next_q_online.argmax(dim=1)

            next_q_target = self.target(next_grids, next_scalars)
            next_q_vals = next_q_target.gather(
                1, best_next_actions.unsqueeze(1)
            ).squeeze(1)

            target_q = rewards + self.gamma * next_q_vals * (~dones)

        current_q_all = self.online(grids, scalars)
        current_q = current_q_all.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        td_errors = (target_q - current_q).detach().cpu().numpy()

        element_loss = nn.functional.huber_loss(current_q, target_q, reduction="none")
        loss = (is_weights * element_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._steps += 1
        return loss.item(), td_errors

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def soft_update_target(self, tau: float = 0.005) -> None:
        for t_p, o_p in zip(self.target.parameters(), self.online.parameters()):
            t_p.data.copy_(tau * o_p.data + (1.0 - tau) * t_p.data)
