from __future__ import annotations

import os
import shutil
import sys
from collections import deque

# Force UTF-8 output on Windows so box-drawing characters render correctly
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


class TerminalPlotter:
    """
    Live training plot rendered directly in the terminal using ANSI codes.
    No IPython / Jupyter dependency - works in any shell.
    """

    def __init__(self, history_len: int = 100) -> None:
        self.history_len = history_len
        self._scores: deque[int] = deque(maxlen=history_len)
        self._means: deque[float] = deque(maxlen=history_len)
        self._first_render = True
        self._plot_lines = 0  # number of lines written last time

    def update(
        self,
        score: int,
        mean_score: float,
        n_game: int,
        epsilon: float,
        loss: float | None,
    ) -> None:
        self._scores.append(score)
        self._means.append(mean_score)
        self._render(n_game, epsilon, loss)

    def _render(self, n_game: int, epsilon: float, loss: float | None) -> None:
        term_w, _ = shutil.get_terminal_size((80, 24))
        plot_w = min(term_w - 4, 80)
        plot_h = 8

        scores = list(self._scores)
        means = list(self._means)
        if not scores:
            return

        max_score = max(max(scores), 1)

        # Build ASCII canvas
        canvas = [[" "] * plot_w for _ in range(plot_h)]

        # Draw mean line (dots)
        for i, m in enumerate(means[-plot_w:]):
            col = i
            row = int((1.0 - m / max_score) * (plot_h - 1))
            row = max(0, min(plot_h - 1, row))
            if canvas[row][col] == " ":
                canvas[row][col] = "·"

        # Draw score line (asterisks, overwrites dots)
        for i, s in enumerate(scores[-plot_w:]):
            col = i
            row = int((1.0 - s / max_score) * (plot_h - 1))
            row = max(0, min(plot_h - 1, row))
            canvas[row][col] = "*"

        lines: list[str] = []
        loss_str = f"{loss:.5f}" if loss is not None else "  n/a  "
        lines.append(
            f"  Game: {n_game:6d} | Score: {scores[-1]:4d} | "
            f"Mean({self.history_len}): {mean_score_str(means):>6} | "
            f"Best: {max(scores):4d} | ε: {epsilon:.4f} | loss: {loss_str}"
        )
        lines.append("  ┌" + "─" * plot_w + "┐")
        for row in canvas:
            lines.append("  │" + "".join(row) + "│")
        lines.append("  └" + "─" * plot_w + "┘")
        lines.append(f"  * score   · mean   (last {self.history_len} games)")

        total_lines = len(lines)

        # Move cursor up to overwrite previous output
        if not self._first_render and self._plot_lines > 0:
            sys.stdout.write(f"\033[{self._plot_lines}A\033[J")
        self._first_render = False
        self._plot_lines = total_lines

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()


def mean_score_str(means: list[float]) -> str:
    if not means:
        return "  0.0"
    return f"{means[-1]:.1f}"
