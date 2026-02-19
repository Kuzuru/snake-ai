from __future__ import annotations

import numpy as np
import torch

from ..config.settings import Config
from ..game.engine import SnakeGame, StepResult
from ..model.network import DuelingCNN
from ..model.trainer import DDQNTrainer
from .memory import PrioritizedReplayBuffer, Transition
from .state import StateBuilder


def compute_reward(result: StepResult, prev_bfs: int = -1) -> float:
    if result.done:
        return -10.0
    if result.ate_food:
        return 10.0

    reward = -0.01

    if prev_bfs > 0 and result.bfs_to_food > 0:
        if result.bfs_to_food < prev_bfs:
            reward += 0.3
        elif result.bfs_to_food > prev_bfs:
            reward -= 0.3

    if result.bfs_to_food < 0:
        reward -= 1.5

    return reward


class DuelingDDQNAgent:

    def __init__(self, config: Config, device: torch.device) -> None:
        self.config = config
        self.device = device

        self.online_net = DuelingCNN(
            config.grid_h, config.grid_w, n_scalars=12, n_actions=3
        ).to(device)

        self.target_net = DuelingCNN(
            config.grid_h, config.grid_w, n_scalars=12, n_actions=3
        ).to(device)

        self.trainer = DDQNTrainer(
            self.online_net,
            self.target_net,
            lr=config.lr,
            gamma=config.gamma,
            device=device,
        )

        self.memory = PrioritizedReplayBuffer(
            capacity=config.memory_size,
            alpha=config.per_alpha,
        )

        self.epsilon = config.epsilon_start
        self.n_games = 0
        self._steps = 0
        self._beta = config.per_beta_start

    def attach_game(self, game: SnakeGame) -> None:
        self.game = game
        self.state_builder = StateBuilder(game)

    def get_action(self, grid: np.ndarray, scalars: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, 3))

        self.online_net.eval()
        with torch.no_grad():
            g = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(self.device)
            s = torch.tensor(scalars, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_vals = self.online_net(g, s)
        return int(q_vals.argmax().item())

    def store(self, transition: Transition) -> None:
        self.memory.push(transition)

    def learn(self) -> float | None:
        cfg = self.config
        if len(self.memory) < cfg.batch_size:
            return None

        self._beta = min(
            cfg.per_beta_end,
            cfg.per_beta_start
            + self._steps * (cfg.per_beta_end - cfg.per_beta_start) / cfg.per_beta_anneal_steps,
        )

        transitions, indices, is_weights = self.memory.sample(cfg.batch_size, self._beta)

        grids = torch.tensor(
            np.stack([t.state_grid for t in transitions]), dtype=torch.float32
        ).to(self.device)
        scalars = torch.tensor(
            np.stack([t.state_scalars for t in transitions]), dtype=torch.float32
        ).to(self.device)
        actions = torch.tensor(
            [t.action for t in transitions], dtype=torch.long
        ).to(self.device)
        rewards = torch.tensor(
            [t.reward for t in transitions], dtype=torch.float32
        ).to(self.device)
        next_grids = torch.tensor(
            np.stack([t.next_grid for t in transitions]), dtype=torch.float32
        ).to(self.device)
        next_scalars = torch.tensor(
            np.stack([t.next_scalars for t in transitions]), dtype=torch.float32
        ).to(self.device)
        dones = torch.tensor(
            [t.done for t in transitions], dtype=torch.bool
        ).to(self.device)
        weights = torch.tensor(is_weights, dtype=torch.float32).to(self.device)

        loss, td_errors = self.trainer.train_step(
            grids, scalars, actions, rewards,
            next_grids, next_scalars, dones, weights,
        )

        self.memory.update_priorities(indices, td_errors)

        self.epsilon = max(cfg.epsilon_end, self.epsilon * cfg.epsilon_decay)
        self._steps += 1
        self.trainer.soft_update_target(tau=cfg.tau)

        return loss
