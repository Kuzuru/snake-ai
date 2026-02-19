from __future__ import annotations

import numpy as np

from ..game.engine import SnakeGame
from ..game.entities import Direction, Point


class StateBuilder:

    _DIRS = list(Direction)  # RIGHT=0, DOWN=1, LEFT=2, UP=3 (clockwise)

    def __init__(self, game: SnakeGame) -> None:
        self.game = game

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        return self._build_grid(), self._build_scalars()

    def _build_grid(self) -> np.ndarray:
        g = self.game
        grid = np.zeros((3, g.grid_h, g.grid_w), dtype=np.float32)

        snake_list = list(g.snake)
        n = len(snake_list)

        for i, pt in enumerate(snake_list):
            persistence = (n - 1 - i) / max(n - 1, 1)
            grid[0, pt.y, pt.x] = persistence

        grid[1, g.head.y, g.head.x] = 1.0

        if g.food is not None:
            grid[2, g.food.y, g.food.x] = 1.0

        return grid

    def _build_scalars(self) -> np.ndarray:
        g = self.game
        max_dist = g.grid_w + g.grid_h
        max_cells = g.grid_w * g.grid_h

        idx = int(g.direction)
        ahead_dir = g.direction
        right_dir = self._DIRS[(idx + 1) % 4]
        left_dir  = self._DIRS[(idx - 1) % 4]
        ahead_pt = self._step(ahead_dir)
        right_pt = self._step(right_dir)
        left_pt  = self._step(left_dir)

        danger_straight = float(g.is_collision(ahead_pt))
        danger_right    = float(g.is_collision(right_pt))
        danger_left     = float(g.is_collision(left_pt))

        bfs = g._bfs_to_food()
        bfs_norm = bfs / max_cells if bfs >= 0 else 1.0

        _angle = {
            Direction.RIGHT: 0.0,
            Direction.DOWN:  1.5707963,   # pi/2
            Direction.LEFT:  3.1415927,   # pi
            Direction.UP:    4.7123890,   # 3*pi/2
        }
        angle = _angle[g.direction]

        body_except_head = set(list(g.snake)[1:])
        flood_current = g.flood_fill(g.head, body_except_head) / max_cells

        lookahead_blocked = set(list(g.snake)[:-1])

        def _flood_after(direction: Direction) -> float:
            new_head = self._step(direction)
            if (
                not (0 <= new_head.x < g.grid_w and 0 <= new_head.y < g.grid_h)
                or new_head in lookahead_blocked
            ):
                return 0.0
            return g.flood_fill(new_head, lookahead_blocked) / max_cells

        flood_if_straight = _flood_after(ahead_dir)
        flood_if_right    = _flood_after(right_dir)
        flood_if_left     = _flood_after(left_dir)

        return np.array([
            g._manhattan() / max(max_dist, 1),
            bfs_norm,
            len(g.snake) / max_cells,
            danger_straight,
            danger_right,
            danger_left,
            float(np.sin(angle)),
            float(np.cos(angle)),
            flood_current,
            flood_if_straight,
            flood_if_right,
            flood_if_left,
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
