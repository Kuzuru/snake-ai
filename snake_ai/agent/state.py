from __future__ import annotations

import numpy as np

from ..game.engine import SnakeGame
from ..game.entities import Direction, Point


class StateBuilder:
    """
    Converts the current game state into the hybrid (grid, scalars) representation
    consumed by DuelingCNN.

    grid:    (3, grid_h, grid_w) float32
      ch0 - body persistence:  segment_value = (len-1-i) / (len-1)
                                tail → ~0.0 (disappears soon)
                                neck → ~1.0 (persists longest)
      ch1 - head:   binary 1 at head position
      ch2 - food:   binary 1 at food position

    scalars: (8,) float32
      [manhattan/max_dist, bfs/max_cells, fill_ratio,
       danger_straight, danger_right, danger_left,
       sin(angle), cos(angle)]
    """

    _DIRS = list(Direction)  # RIGHT=0, DOWN=1, LEFT=2, UP=3 (clockwise)

    def __init__(self, game: SnakeGame) -> None:
        self.game = game

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        return self._build_grid(), self._build_scalars()

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def _build_grid(self) -> np.ndarray:
        g = self.game
        grid = np.zeros((3, g.grid_h, g.grid_w), dtype=np.float32)

        snake_list = list(g.snake)
        n = len(snake_list)

        # Channel 0: body persistence (tail=low, near-head=high)
        for i, pt in enumerate(snake_list):
            persistence = (n - 1 - i) / max(n - 1, 1)
            grid[0, pt.y, pt.x] = persistence

        # Channel 1: head
        grid[1, g.head.y, g.head.x] = 1.0

        # Channel 2: food
        if g.food is not None:
            grid[2, g.food.y, g.food.x] = 1.0

        return grid

    # ------------------------------------------------------------------
    # Scalars
    # ------------------------------------------------------------------

    def _build_scalars(self) -> np.ndarray:
        g = self.game
        max_dist = g.grid_w + g.grid_h
        max_cells = g.grid_w * g.grid_h

        # Lookahead points for immediate danger
        idx = int(g.direction)
        ahead_pt = self._step(g.direction, 1)
        right_pt = self._step(self._DIRS[(idx + 1) % 4], 1)
        left_pt = self._step(self._DIRS[(idx - 1) % 4], 1)

        danger_straight = float(g.is_collision(ahead_pt))
        danger_right = float(g.is_collision(right_pt))
        danger_left = float(g.is_collision(left_pt))

        bfs = g._bfs_to_food()
        bfs_norm = bfs / max_cells if bfs >= 0 else 1.0  # 1.0 signals unreachable

        # Encode direction as unit-circle (avoids one-hot discontinuity)
        _angle = {
            Direction.RIGHT: 0.0,
            Direction.DOWN:  1.5707963,   # pi/2
            Direction.LEFT:  3.1415927,   # pi
            Direction.UP:    4.7123890,   # 3*pi/2
        }
        angle = _angle[g.direction]

        return np.array([
            g._manhattan() / max(max_dist, 1),
            bfs_norm,
            len(g.snake) / max_cells,
            danger_straight,
            danger_right,
            danger_left,
            float(np.sin(angle)),
            float(np.cos(angle)),
        ], dtype=np.float32)

    def _step(self, direction: Direction, steps: int = 1) -> Point:
        x, y = self.game.head
        if direction == Direction.RIGHT:
            x += steps
        elif direction == Direction.LEFT:
            x -= steps
        elif direction == Direction.DOWN:
            y += steps
        elif direction == Direction.UP:
            y -= steps
        return Point(x, y)
