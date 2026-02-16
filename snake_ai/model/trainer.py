from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .network import DuelingCNN


class DDQNTrainer:
    """
    Double DQN training step with Prioritized Experience Replay support.

    Double DQN:
        - Online network selects the best next action
        - Target network evaluates that action's Q-value
        > Eliminates Q-value overestimation bias of vanilla DQN

    Loss: Huber loss (more stable than MSE for large TD errors)
    Gradient clipping: max_norm=10 (prevents exploding gradients)
    """

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
        # Initialise target weights identically to online
        self.target.load_state_dict(online_net.state_dict())
        self.target.eval()
        self._steps = 0

    def train_step(
        self,
        grids: torch.Tensor,        # (B, 3, H, W)
        scalars: torch.Tensor,      # (B, 8)
        actions: torch.Tensor,      # (B,) long
        rewards: torch.Tensor,      # (B,) float
        next_grids: torch.Tensor,   # (B, 3, H, W)
        next_scalars: torch.Tensor, # (B, 8)
        dones: torch.Tensor,        # (B,) bool
        is_weights: torch.Tensor,   # (B,) float  - PER importance-sampling weights
    ) -> tuple[float, np.ndarray]:
        """
        Returns:
            loss:      scalar float for logging
            td_errors: numpy array (B,) for PER priority update
        """
        self.online.train()

        with torch.no_grad():
            # Double DQN: online picks action, target evaluates it
            next_q_online = self.online(next_grids, next_scalars)      # (B, A)
            best_next_actions = next_q_online.argmax(dim=1)             # (B,)

            next_q_target = self.target(next_grids, next_scalars)      # (B, A)
            next_q_vals = next_q_target.gather(
                1, best_next_actions.unsqueeze(1)
            ).squeeze(1)                                                # (B,)

            # Bellman target - zero out next-state value for terminal transitions
            target_q = rewards + self.gamma * next_q_vals * (~dones)  # (B,)

        # Current Q estimates for the actions that were actually taken
        current_q_all = self.online(grids, scalars)                     # (B, A)
        current_q = current_q_all.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)                                                    # (B,)

        td_errors = (target_q - current_q).detach().cpu().numpy()

        # PER-weighted Huber loss
        element_loss = nn.functional.huber_loss(current_q, target_q, reduction="none")
        loss = (is_weights * element_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._steps += 1
        return loss.item(), td_errors

    def sync_target(self) -> None:
        """Hard-copy online > target weights."""
        self.target.load_state_dict(self.online.state_dict())

    def soft_update_target(self, tau: float = 0.005) -> None:
        """Soft update: target = tau*online + (1-tau)*target."""
        for t_p, o_p in zip(self.target.parameters(), self.online.parameters()):
            t_p.data.copy_(tau * o_p.data + (1.0 - tau) * t_p.data)
