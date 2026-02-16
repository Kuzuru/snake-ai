# Snake AI

A Snake game agent trained with **Dueling Double DQN + Prioritized Experience Replay**.

The agent uses a hybrid CNN + Scalar feature state representation that gives it full spatial
awareness of the board, enabling it to plan paths around its own body.

## Installation

**Requirements:** Python 3.10+, CUDA-capable GPU recommended

```bash
pip install -r requirements.txt
```

For CUDA support:

```bash
# CUDA 12.6 (most RTX 30xx/40xx cards)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Training

```
python scripts/train.py [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--render` | flag | off | Open game window with live stats panel |
| `--speed N` | int | `60` | Target game steps/sec **(render mode only)** |
| `--games N` | int | `10000` | Total number of training episodes |
| `--resume PATH` | str | - | Resume from a `.pth` checkpoint |

### --speed

The HUD shows the current target and skip ratio: `Score: 3600sps x10`

### Examples

```bash
# Headless. Max GPU throughput, no window
python scripts/train.py

# Headless, custom game count
python scripts/train.py --games 50000

# Visual with stats panel, default speed (60 sps)
python scripts/train.py --render

# Visual, 10x faster (draw 1 in 3 steps)
python scripts/train.py --render --speed 600

# Resume from best checkpoint, visual
python scripts/train.py --render --resume checkpoints/best.pth

# Resume and train 20000 more games at high speed (if your GPU can handle it lol)
python scripts/train.py --resume checkpoints/best.pth --games 20000 --render --speed 1200
```

### Keyboard controls (render mode)

| Key | Action |
|---|---|
| `+` | Increase speed |
| `-` | Decrease speed |
| Close window | Stop training gracefully (saves `final.pth`) |

### Output files

Checkpoints are saved to `checkpoints/`:

| File | Saved when |
|---|---|
| `best.pth` | New high score reached |
| `game_NNNNNN.pth` | Every 100 games (configurable) |
| `final.pth` | Training completes or window is closed |

## Evaluation

Run a saved model for N episodes and print performance statistics.

```
python scripts/evaluate.py --checkpoint PATH [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--checkpoint PATH` | str | **required** | Path to `.pth` checkpoint |
| `--episodes N` | int | `100` | Number of evaluation episodes |
| `--render` | flag | off | Show game window during evaluation |

## Human Play

Evaluate your own neurons. Play Snake yourself!

```
python scripts/play.py
```
