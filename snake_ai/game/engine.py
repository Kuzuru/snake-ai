import random
from collections import deque
from typing import Optional, NamedTuple

from .entities import Direction, Point, GRID_WIDTH, GRID_HEIGHT


class StepResult(NamedTuple):
    done: bool
    score: int
    ate_food: bool
    manhattan_to_food: int
    bfs_to_food: int
    steps_without_food: int


class SnakeGame:

    def __init__(
        self,
        grid_w: int = GRID_WIDTH,
        grid_h: int = GRID_HEIGHT,
        renderer=None,
        max_steps_no_food: Optional[int] = None,
    ) -> None:
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.renderer = renderer
        self._max_steps_no_food = max_steps_no_food
        self.reset()

    def reset(self) -> None:
        cx, cy = self.grid_w // 2, self.grid_h // 2
        self.direction = random.choice(list(Direction))
        if self.direction == Direction.RIGHT:
            body = [Point(cx, cy), Point(cx - 1, cy), Point(cx - 2, cy)]
        elif self.direction == Direction.LEFT:
            body = [Point(cx, cy), Point(cx + 1, cy), Point(cx + 2, cy)]
        elif self.direction == Direction.DOWN:
            body = [Point(cx, cy), Point(cx, cy - 1), Point(cx, cy - 2)]
        else:
            body = [Point(cx, cy), Point(cx, cy + 1), Point(cx, cy + 2)]
        self.snake: deque[Point] = deque(body)
        self.head: Point = self.snake[0]
        self.score: int = 0
        self.steps: int = 0
        self.steps_without_food: int = 0
        self._place_food()

    def play_step(self, action: Optional[list] = None) -> StepResult:
        self.steps += 1
        self.steps_without_food += 1

        if self.renderer is not None:
            if action is not None:
                self.renderer.handle_events()
                new_dir = self._action_to_direction(action)
            else:
                new_dir = self.renderer.get_human_direction(self.direction)
        else:
            if action is None:
                raise ValueError("action must be provided when renderer is None (headless mode)")
            new_dir = self._action_to_direction(action)

        self.direction = new_dir
        new_head = self._advance_head()

        max_steps = self._max_steps_no_food or (100 * len(self.snake))
        timeout = self.steps_without_food > max_steps

        if self._is_collision(new_head) or timeout:
            if self.renderer is not None:
                self.renderer.render(self)
            return StepResult(
                done=True,
                score=self.score,
                ate_food=False,
                manhattan_to_food=self._manhattan(),
                bfs_to_food=self._bfs_to_food(),
                steps_without_food=self.steps_without_food,
            )

        # Move snake forward
        self.snake.appendleft(new_head)
        self.head = new_head
        ate_food = (self.head == self.food)

        if ate_food:
            self.score += 1
            self.steps_without_food = 0
            self._place_food()
        else:
            self.snake.pop()

        if self.renderer is not None:
            self.renderer.render(self)

        return StepResult(
            done=False,
            score=self.score,
            ate_food=ate_food,
            manhattan_to_food=self._manhattan(),
            bfs_to_food=self._bfs_to_food(),
            steps_without_food=self.steps_without_food,
        )

    def is_collision(self, pt: Optional[Point] = None) -> bool:
        return self._is_collision(pt if pt is not None else self.head)

    def _place_food(self) -> None:
        occupied = set(self.snake)
        free = [
            Point(x, y)
            for x in range(self.grid_w)
            for y in range(self.grid_h)
            if Point(x, y) not in occupied
        ]
        if not free:
            self.food = None
        else:
            self.food = random.choice(free)

    def _action_to_direction(self, action: list) -> Direction:
        dirs = list(Direction)
        idx = int(self.direction)
        if action[0]:
            return self.direction
        elif action[1]:
            return dirs[(idx + 1) % 4]
        else:
            return dirs[(idx - 1) % 4]

    def _advance_head(self) -> Point:
        x, y = self.head
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1
        return Point(x, y)

    def _is_collision(self, pt: Point) -> bool:
        if not (0 <= pt.x < self.grid_w and 0 <= pt.y < self.grid_h):
            return True
        # tail moves away on next step so exclude it
        body_without_tail = set(list(self.snake)[:-1])
        return pt in body_without_tail

    def _manhattan(self) -> int:
        if self.food is None:
            return 0
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

    def _bfs_to_food(self) -> int:
        if self.food is None:
            return 0
        occupied = set(list(self.snake)[1:])
        queue: deque[tuple[Point, int]] = deque([(self.head, 0)])
        visited = {self.head}
        while queue:
            pos, dist = queue.popleft()
            if pos == self.food:
                return dist
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = Point(pos.x + dx, pos.y + dy)
                if (
                    nxt not in visited
                    and nxt not in occupied
                    and 0 <= nxt.x < self.grid_w
                    and 0 <= nxt.y < self.grid_h
                ):
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
        return -1

    def flood_fill(self, start: Point, blocked: set) -> int:
        if not (0 <= start.x < self.grid_w and 0 <= start.y < self.grid_h):
            return 0
        if start in blocked:
            return 0
        visited: set[Point] = {start}
        queue: deque[Point] = deque([start])
        while queue:
            pos = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = Point(pos.x + dx, pos.y + dy)
                if (
                    nxt not in visited
                    and nxt not in blocked
                    and 0 <= nxt.x < self.grid_w
                    and 0 <= nxt.y < self.grid_h
                ):
                    visited.add(nxt)
                    queue.append(nxt)
        return len(visited)
