from __future__ import annotations

from pathlib import Path
from typing import Optional

import pygame

from .entities import (
    Direction, Point, BLOCK_SIZE,
    WHITE, BLACK, RED, BLUE1, BLUE2, GRAY,
    SPEED_HUMAN, SPEED_AI,
)

_FONT_PATH = Path(__file__).parent.parent.parent / "assets" / "fonts" / "fira_code.ttf"


class PygameRenderer:
    """
    Optional rendering layer for SnakeGame.

    When show_stats=True a StatsPanel is attached and the window is widened
    to display a live training dashboard alongside the game.
    """

    # Visual refresh rate cap — display never exceeds this regardless of speed
    _VISUAL_FPS: int = 60

    def __init__(
        self,
        grid_w: int,
        grid_h: int,
        speed: int = SPEED_AI,
        show_stats: bool = False,
    ) -> None:
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.game_w = grid_w * BLOCK_SIZE
        self.game_h = grid_h * BLOCK_SIZE
        self._quit_requested = False
        self._human_dir_queue: list[Direction] = []
        self._step = 0

        # speed = target game steps/sec
        # render_every = how many steps between actual draws
        # visual_fps  = clock cap for rendered frames (always ≤ _VISUAL_FPS)
        self._set_speed(speed)

        pygame.init()

        # Import here to avoid circular import at module level
        self._stats_panel: Optional["StatsPanel"] = None
        if show_stats:
            from .stats_panel import StatsPanel
            self._stats_panel = StatsPanel(self.game_h)
            total_w = self.game_w + StatsPanel.WIDTH
        else:
            total_w = self.game_w

        self.display = pygame.display.set_mode((total_w, self.game_h))
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()

        font_size = 20
        if _FONT_PATH.exists():
            self.font = pygame.font.Font(str(_FONT_PATH), font_size)
        else:
            self.font = pygame.font.SysFont("monospace", font_size)

    # ------------------------------------------------------------------
    # Speed control
    # ------------------------------------------------------------------

    def _set_speed(self, speed: int) -> None:
        """Set target steps/sec, recompute render cadence."""
        self.speed = max(1, speed)
        # Render every N steps so the display stays at ≤ _VISUAL_FPS
        self._render_every = max(1, self.speed // self._VISUAL_FPS)
        # Cap visual refresh so we don't burn GPU on pure display overhead
        self._visual_fps = min(self.speed, self._VISUAL_FPS)

    # ------------------------------------------------------------------
    # Called by SnakeGame
    # ------------------------------------------------------------------

    def handle_events(self) -> None:
        """Process pygame events (quit, keyboard)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_requested = True
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self._human_dir_queue.append(Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    self._human_dir_queue.append(Direction.RIGHT)
                elif event.key == pygame.K_UP:
                    self._human_dir_queue.append(Direction.UP)
                elif event.key == pygame.K_DOWN:
                    self._human_dir_queue.append(Direction.DOWN)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self._set_speed(self.speed + max(10, self.speed // 5))
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._set_speed(max(10, self.speed - max(10, self.speed // 5)))

    def get_human_direction(self, current: Direction) -> Direction:
        """Return next queued direction for human play (ignores 180 reversal)."""
        opposites = {
            Direction.RIGHT: Direction.LEFT,
            Direction.LEFT:  Direction.RIGHT,
            Direction.UP:    Direction.DOWN,
            Direction.DOWN:  Direction.UP,
        }
        while self._human_dir_queue:
            candidate = self._human_dir_queue.pop(0)
            if candidate != opposites[current]:
                return candidate
        return current

    def render(self, game) -> None:
        """Draw the current game state (and stats panel if attached).

        Skips drawing on non-render steps — this is what actually makes
        higher speed values train faster (less GPU/CPU time on display).
        Events are still processed every step via handle_events().
        """
        if self._quit_requested:
            return

        self._step += 1
        if self._step % self._render_every != 0:
            return  # skip draw, no sleep — training runs at full speed

        # ── Game area ─────────────────────────────────────────────────
        game_surf = pygame.Surface((self.game_w, self.game_h))
        game_surf.fill(BLACK)

        # Subtle grid
        for x in range(0, self.game_w, BLOCK_SIZE):
            pygame.draw.line(game_surf, GRAY, (x, 0), (x, self.game_h))
        for y in range(0, self.game_h, BLOCK_SIZE):
            pygame.draw.line(game_surf, GRAY, (0, y), (self.game_w, y))

        # Snake body
        for i, pt in enumerate(game.snake):
            px, py = pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE
            color = BLUE1 if i == 0 else BLUE2
            pygame.draw.rect(game_surf, color, (px, py, BLOCK_SIZE, BLOCK_SIZE))
            if i == 0:
                pygame.draw.rect(
                    game_surf, BLUE2,
                    (px + 4, py + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8),
                )

        # Food
        if game.food is not None:
            fx, fy = game.food.x * BLOCK_SIZE, game.food.y * BLOCK_SIZE
            pygame.draw.rect(game_surf, RED, (fx, fy, BLOCK_SIZE, BLOCK_SIZE))

        # HUD — show target speed (steps/sec) and skip ratio
        skip_str = f"x{self._render_every}" if self._render_every > 1 else ""
        hud = self.font.render(f"Score: {game.score}   {self.speed} sps {skip_str}", True, WHITE)
        game_surf.blit(hud, (4, 4))

        self.display.blit(game_surf, (0, 0))

        # ── Stats panel ───────────────────────────────────────────────
        if self._stats_panel is not None:
            self.display.blit(self._stats_panel.surf, (self.game_w, 0))

        pygame.display.flip()
        self.clock.tick(self._visual_fps)  # cap rendered frames only

    # ------------------------------------------------------------------
    # Training dashboard
    # ------------------------------------------------------------------

    def update_stats(
        self,
        score: int,
        mean_score: float,
        n_games: int,
        epsilon: float,
        best_score: int,
        loss: float | None,
    ) -> None:
        """Update the stats panel data (no-op when show_stats=False)."""
        if self._stats_panel is not None:
            self._stats_panel.update(score, mean_score, n_games, epsilon, best_score, loss)

    @property
    def has_stats(self) -> bool:
        return self._stats_panel is not None

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    def close(self) -> None:
        pygame.quit()
