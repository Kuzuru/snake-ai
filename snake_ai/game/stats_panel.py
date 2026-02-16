from __future__ import annotations

from collections import deque
from pathlib import Path

import pygame

_FONT_PATH = Path(__file__).parent.parent.parent / "assets" / "fonts" / "fira_code.ttf"

# Panel palette
_BG        = (18,  18,  22)
_GRID      = (38,  38,  48)
_WHITE     = (230, 230, 230)
_DIM       = (120, 120, 130)
_GREEN     = (50,  220, 110)   # score line
_YELLOW    = (220, 200,  50)   # mean line
_ORANGE    = (220, 130,  40)   # loss line
_CYAN      = ( 60, 190, 220)   # epsilon / accent
_RED       = (220,  70,  70)   # best score highlight


class StatsPanel:
    """
    A pygame Surface that draws a live training dashboard:
      - Stat readouts (game #, score, mean, best, epsilon, loss)
      - Score history line chart (score + rolling mean)
      - Loss history line chart

    Call update() at episode end, then panel.surf is ready to blit.
    """

    WIDTH = 400   # panel pixel width
    PAD   = 14    # inner padding

    def __init__(self, height: int) -> None:
        self.height = height
        self.surf = pygame.Surface((self.WIDTH, height))

        font_size_lg = 17
        font_size_sm = 13
        if _FONT_PATH.exists():
            self._font_lg = pygame.font.Font(str(_FONT_PATH), font_size_lg)
            self._font_sm = pygame.font.Font(str(_FONT_PATH), font_size_sm)
        else:
            self._font_lg = pygame.font.SysFont("monospace", font_size_lg)
            self._font_sm = pygame.font.SysFont("monospace", font_size_sm)

        self._scores:  deque[int]   = deque(maxlen=200)
        self._means:   deque[float] = deque(maxlen=200)
        self._losses:  deque[float] = deque(maxlen=200)

        self.n_games    = 0
        self.epsilon    = 1.0
        self.best_score = 0
        self._last_score = 0
        self._last_mean  = 0.0
        self._last_loss: float | None = None

        self._redraw()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        score: int,
        mean_score: float,
        n_games: int,
        epsilon: float,
        best_score: int,
        loss: float | None,
    ) -> None:
        self._scores.append(score)
        self._means.append(mean_score)
        if loss is not None:
            self._losses.append(loss)
        self.n_games     = n_games
        self.epsilon     = epsilon
        self.best_score  = best_score
        self._last_score = score
        self._last_mean  = mean_score
        self._last_loss  = loss
        self._redraw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        s = self.surf
        w, h = self.WIDTH, self.height
        pad = self.PAD

        s.fill(_BG)

        # Vertical separator line (left edge)
        pygame.draw.line(s, _GRID, (0, 0), (0, h), 2)

        y = pad

        # ── Title ─────────────────────────────────────────────────────
        y = self._text(s, "TRAINING", pad, y, _CYAN, self._font_lg) + 2
        pygame.draw.line(s, _GRID, (pad, y), (w - pad, y))
        y += 8

        # ── Stats grid ────────────────────────────────────────────────
        col2 = w // 2
        stats = [
            ("Game",    f"{self.n_games:,}",       _WHITE,  _WHITE),
            ("Score",   f"{self._last_score}",      _WHITE,  _GREEN),
            ("Best",    f"{self.best_score}",        _WHITE,  _RED),
            ("Mean",    f"{self._last_mean:.1f}",    _WHITE,  _YELLOW),
            ("Epsilon", f"{self.epsilon:.4f}",       _WHITE,  _CYAN),
            ("Loss",    f"{self._last_loss:.5f}" if self._last_loss else "n/a", _WHITE, _ORANGE),
        ]
        for i, (label, value, lc, vc) in enumerate(stats):
            col = pad if i % 2 == 0 else col2
            row_y = y + (i // 2) * 26
            self._text(s, label + ":", col, row_y, lc, self._font_sm)
            self._text(s, value, col + 70, row_y, vc, self._font_sm)

        y += ((len(stats) + 1) // 2) * 26 + 8
        pygame.draw.line(s, _GRID, (pad, y), (w - pad, y))
        y += 10

        # ── Score chart ───────────────────────────────────────────────
        chart_h = (h - y - pad) // 2 - 20
        chart_w = w - pad * 2
        y = self._draw_chart(
            s,
            title="Score History",
            y=y, chart_w=chart_w, chart_h=chart_h,
            series=[
                (list(self._scores), _GREEN,  "score"),
                (list(self._means),  _YELLOW, "mean"),
            ],
        )
        y += 18

        # ── Loss chart ────────────────────────────────────────────────
        loss_h = h - y - pad - 2
        self._draw_chart(
            s,
            title="Loss",
            y=y, chart_w=chart_w, chart_h=loss_h,
            series=[(list(self._losses), _ORANGE, "loss")],
        )

    def _draw_chart(
        self,
        surf: pygame.Surface,
        title: str,
        y: int,
        chart_w: int,
        chart_h: int,
        series: list[tuple[list, tuple, str]],
    ) -> int:
        pad = self.PAD
        y = self._text(surf, title, pad, y, _DIM, self._font_sm) + 4

        box_x, box_y = pad, y
        box_w, box_h = chart_w, chart_h

        # Background + border
        pygame.draw.rect(surf, (25, 25, 32), (box_x, box_y, box_w, box_h))
        pygame.draw.rect(surf, _GRID,        (box_x, box_y, box_w, box_h), 1)

        # Horizontal guide lines (3)
        for frac in (0.25, 0.5, 0.75):
            gy = box_y + int(frac * box_h)
            pygame.draw.line(surf, _GRID, (box_x, gy), (box_x + box_w, gy))

        # Data
        all_vals = [v for data, _, _ in series for v in data]
        if len(all_vals) >= 2:
            lo = min(all_vals)
            hi = max(all_vals)
            span = max(hi - lo, 1e-6)

            for data, color, _ in series:
                if len(data) < 2:
                    continue
                pts = data[-box_w:]          # at most one point per pixel column
                n = len(pts)
                points = []
                for i, v in enumerate(pts):
                    px = box_x + int(i / max(n - 1, 1) * (box_w - 1))
                    py = box_y + box_h - 1 - int((v - lo) / span * (box_h - 2))
                    py = max(box_y, min(box_y + box_h - 1, py))
                    points.append((px, py))
                if len(points) >= 2:
                    pygame.draw.lines(surf, color, False, points, 2)

            # Max value label (top right)
            hi_surf = self._font_sm.render(f"{hi:.1f}", True, _DIM)
            surf.blit(hi_surf, (box_x + box_w - hi_surf.get_width() - 2, box_y + 2))

        return box_y + box_h

    @staticmethod
    def _text(
        surf: pygame.Surface,
        text: str,
        x: int,
        y: int,
        color: tuple,
        font: pygame.font.Font,
    ) -> int:
        """Blit text and return the bottom y coordinate."""
        rendered = font.render(text, True, color)
        surf.blit(rendered, (x, y))
        return y + rendered.get_height()
