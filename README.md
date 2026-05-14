# PadelDet — Padel Match Analysis System

Computer vision system for analyzing padel matches from a fixed phone camera placed behind the glass wall at one end of the court.

## Overview

A phone records the match, the video is sent to a server for CV analysis. The pipeline detects and tracks players and the ball, maps positions to real-world court coordinates, and computes match metrics.

## Pipeline

```
Video → Player Tracking (YOLO + DeepOCSORT + warm-up lock)
      → Court Calibration (manual keypoints + homography)
      → Ball Tracking (TrackNet heatmaps + post-processing)
      → Metrics (distance, speed, zones, formations)
```

## Quick Start

### Install

padeldet is a pip-installable Python package with a `pyproject.toml`. Install it into your project:

```bash
# Editable (for active development alongside a host project):
pip install -e /path/to/padeldet

# From GitHub (for deployment environments):
pip install git+https://github.com/VictorVigara/padeldet.git@main

# For scripts in this repo:
pip install -r requirements.txt
```

Model weights go in `weights/`. TrackNet weights (`TrackNet_best.pt`, `InpaintNet_best.pt`) are from the original TrackNet repository.

---

## Ball Tracking — Heatmap Workflow

TrackNet inference is the slow step (~minutes for a full video). The workflow below separates inference from post-processing so you can iterate on the latter instantly.

### Step 1 — Generate and save heatmaps

Run TrackNet once and cache the raw heatmaps to a `.npy` file:

```bash
python scripts/evaluate_ball_tracking.py \
    --video data/rally/video.mp4 \
    --tracker kalman \
    --tracknet-weights /home/vvipu/padel_analytics/weights/ball_detection/TrackNet_best.pt \
    --save-heatmaps data/rally/heatmaps.npy \
    --output /tmp/unused.mp4
```

This writes `data/rally/heatmaps.npy` (uint8, memory-mapped) and `data/rally/heatmaps.meta.npz` (video metadata). File size is ~750 MB for a 5-minute video; loading is instant.

**One-time conversion** if you have an old `heatmaps.npz`:

```python
import numpy as np
old = np.load("data/rally/heatmaps.npz")["heatmaps"]   # float32
arr = (old.clip(0, 1) * 255).astype(np.uint8)
np.save("data/rally/heatmaps.npy", arr)
```

### Step 2 — Compare blob selection strategies

Run all four strategies on the saved heatmaps and produce a side-by-side comparison video with stats:

```bash
python scripts/compare_strategies.py \
    --heatmaps data/rally/heatmaps.npy \
    --video data/rally/video.mp4 \
    --player-json data/rally/player_detections.json \
    --output data/rally/strategy_comparison.mp4
```

Optional flags:
- `--start N` / `--end N` — process a frame sub-range
- `--player-json` — enables occlusion-aware strategy (skippable)

**Strategies compared:**

| Strategy | Colour | Description |
|---|---|---|
| Intensity | Blue | Highest peak-intensity blob each frame |
| Trajectory | Yellow | Weighted score: velocity prediction + intensity |
| Kalman | Green | Kalman filter with Mahalanobis gating |
| Occlusion | Magenta | Trajectory + player-bbox occlusion coasting |

### Step 3 — Run ball tracking without re-running the network

```bash
python scripts/evaluate_ball_tracking.py \
    --video data/rally/video.mp4 \
    --tracker kalman \
    --load-heatmaps data/rally/heatmaps.npy \
    --output data/rally/ball_output.mp4
```

---

## Player Tracking

YOLO detects players each frame; DeepOCSORT with OSNet ReID keeps IDs stable through occlusions. After a selection window (default 10 s), the 4 deepest in-court tracks are auto-selected as the real players.

**Device selection** — `PlayerTrackerConfig.device` controls the compute device:
- `None` (default) — auto-detects: `"cuda"` if a GPU is available, `"cpu"` otherwise
- `"cuda"` / `"cpu"` — explicit override, useful for testing or forcing CPU locally

```bash
python scripts/evaluate_player_tracking.py \
    --video data/rally/video.mp4 \
    --court-keypoints data/rally/court_keypoints.json \
    --output data/rally/player_output.mp4 \
    --save-json data/rally/player_detections.json
```

The warm-up summary table is always printed after tracking. Two optional debug flags:

```bash
# Bar chart: in-court ratio, mean depth, score per track
--warmup-plot

# Save overview image + one crop per track to a directory
--save-warmup-crops data/rally/warmup_crops/
```

## Court Calibration

```bash
python scripts/calibrate_court.py --video data/rally/video.mp4
```

Click the 12 court keypoints in order. Result saved to `court_keypoints.json`.

## Metrics

```bash
python scripts/evaluate_metrics.py \
    --player-json data/rally/player_detections.json \
    --court-keypoints data/rally/court_keypoints.json \
    --video data/rally/video.mp4
```

---

## Project Structure

```
padeldet/
├── padeldet/
│   ├── models.py                    # Shared data models (BoundingBox, Detection, …)
│   ├── base.py                      # Tracker ABC
│   ├── video.py                     # Video I/O
│   ├── player_tracker/              # YOLO + ByteTrack
│   ├── ball_tracker/
│   │   ├── ball_tracker.py          # TrackNet + post-processing (original)
│   │   ├── kalman_ball_tracker.py   # TrackNet + Kalman candidate selection
│   │   ├── kalman.py                # BallKalmanFilter (constant-velocity)
│   │   └── selectors.py             # Pluggable blob selection strategies
│   ├── court/                       # Homography, CourtKeypoints
│   └── metrics/                     # Distance, speed, zones, formations
├── scripts/
│   ├── evaluate_player_tracking.py
│   ├── evaluate_ball_tracking.py
│   ├── compare_strategies.py        # Multi-strategy heatmap comparison
│   ├── evaluate_metrics.py
│   └── calibrate_court.py
├── data/
│   ├── rally/                       # Rally test video + cached results
│   └── iphone/                      # iPhone test video + cached results
└── weights/                         # Model weights (not committed)
```

## Tech Stack

- Python 3.10+
- [ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8 player detection
- [supervision](https://github.com/roboflow/supervision) — ByteTrack, annotation
- OpenCV — video I/O, image processing
- PyTorch — TrackNet inference
- scipy — Savitzky-Golay smoothing
