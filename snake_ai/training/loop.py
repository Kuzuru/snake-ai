from __future__ import annotations

from pathlib import Path

import torch

from ..agent.agent import DuelingDDQNAgent, compute_reward
from ..agent.memory import Transition
from ..config.settings import Config
from ..game.engine import SnakeGame
from ..game.renderer import PygameRenderer
from ..game.entities import SPEED_AI
from .metrics import EpisodeStats, MetricsTracker
from .visualization import TerminalPlotter


def _action_int_to_list(action: int) -> list[int]:
    a = [0, 0, 0]
    a[action] = 1
    return a


class TrainingLoop:

    def __init__(self, config: Config, render: bool = False, speed: int = SPEED_AI) -> None:
        self.config = config
        self.device = config.device
        self.render = render

        if render:
            renderer = PygameRenderer(
                config.grid_w, config.grid_h,
                speed=speed,
                show_stats=True,
            )
        else:
            renderer = None

        self.game = SnakeGame(config.grid_w, config.grid_h, renderer=renderer)
        self.agent = DuelingDDQNAgent(config, self.device)
        self.agent.attach_game(self.game)

        self.metrics = MetricsTracker()
        self.plotter = TerminalPlotter() if not render else None

        self._checkpoint_dir = Path(config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Device: {self.device}")
        print(f"Grid:   {config.grid_w}x{config.grid_h}  Speed: {speed} fps")
        print(f"Memory: {config.memory_size:,}  Batch: {config.batch_size}")
        print("-" * 60)
        if not render:
            print("\n" * 12)

    def run(self, max_games: int = 10_000) -> None:
        try:
            while self.agent.n_games < max_games:
                self._run_episode()

                renderer = self.game.renderer
                if renderer is not None and renderer.quit_requested:
                    break
        finally:
            print("\nTraining complete.")
            self._save_checkpoint("final.pth")

    def load_checkpoint(self, path: str) -> None:
        p = Path(path)
        state = torch.load(p, map_location=self.device)
        self.agent.online_net.load_state_dict(state["online_net"])
        self.agent.target_net.load_state_dict(state["target_net"])
        self.agent.n_games = state.get("n_games", 0)
        self.agent.epsilon = state.get("epsilon", self.config.epsilon_end)
        self.metrics.best_score = state.get("best_score", 0)
        print(f"Resumed from {p}  (game {self.agent.n_games})")

    def _run_episode(self) -> None:
        self.game.reset()
        stats = EpisodeStats()

        while True:
            grid, scalars = self.agent.state_builder.build()
            action_int = self.agent.get_action(grid, scalars)
            action_list = _action_int_to_list(action_int)

            prev_bfs = self.game._bfs_to_food()
            result = self.game.play_step(action_list)

            reward = compute_reward(result, prev_bfs)

            grid_new, scalars_new = self.agent.state_builder.build()
            self.agent.store(
                Transition(
                    state_grid=grid,
                    state_scalars=scalars,
                    action=action_int,
                    reward=reward,
                    next_grid=grid_new,
                    next_scalars=scalars_new,
                    done=result.done,
                )
            )

            loss = self.agent.learn()
            stats.steps += 1
            stats.total_reward += reward
            if loss is not None:
                stats.loss_sum += loss
                stats.loss_count += 1

            renderer = self.game.renderer
            if result.done or (renderer is not None and renderer.quit_requested):
                stats.score = result.score
                break

        self.agent.n_games += 1
        self.metrics.record(stats)

        if stats.score >= self.metrics.best_score:
            self._save_checkpoint("best.pth", stats.score)

        if self.agent.n_games % self.config.save_every_n_games == 0:
            self._save_checkpoint(f"game_{self.agent.n_games:06d}.pth", stats.score)

        if self.agent.n_games % self.config.plot_update_freq == 0:
            renderer = self.game.renderer
            if renderer is not None:
                renderer.update_stats(
                    score=stats.score,
                    mean_score=self.metrics.mean_score,
                    n_games=self.agent.n_games,
                    epsilon=self.agent.epsilon,
                    best_score=self.metrics.best_score,
                    loss=stats.mean_loss,
                )
            elif self.plotter is not None:
                self.plotter.update(
                    score=stats.score,
                    mean_score=self.metrics.mean_score,
                    n_game=self.agent.n_games,
                    epsilon=self.agent.epsilon,
                    loss=stats.mean_loss,
                )

    def _save_checkpoint(self, filename: str, score: int | None = None) -> None:
        path = self._checkpoint_dir / filename
        torch.save(
            {
                "online_net": self.agent.online_net.state_dict(),
                "target_net": self.agent.target_net.state_dict(),
                "n_games": self.agent.n_games,
                "epsilon": self.agent.epsilon,
                "best_score": self.metrics.best_score,
                "score": score,
            },
            path,
        )
