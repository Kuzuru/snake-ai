from enum import IntEnum
from typing import NamedTuple

BLOCK_SIZE: int = 20
GRID_WIDTH: int = 32   # cells  (640px / 20px)
GRID_HEIGHT: int = 24  # cells  (480px / 20px)

SPEED_HUMAN: int = 10
SPEED_AI: int = 60


class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3
    # Clockwise order: (idx + 1) % 4 = turn right, (idx - 1) % 4 = turn left


class Point(NamedTuple):
    x: int  # grid cell coordinates (NOT pixels)
    y: int


# Colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GRAY = (40, 40, 40)
