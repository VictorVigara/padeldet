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

## GPU Evaluation on Modal

`scripts/eval_on_modal.py` runs any strategy on a cloud T4 GPU and downloads the annotated video + JSON results back to disk — no production job queue involved.

### Prerequisites (one-time)

```bash
pip install modal
modal setup   # authenticate with your Modal account
```

### Usage

```bash
# Player tracking at 5 fps detection rate
modal run scripts/eval_on_modal.py \
    --video data/rally/video.mp4 \
    --keypoints data/rally/court_keypoints.json \
    --fps 5

# All frames (omit --fps or pass 0)
modal run scripts/eval_on_modal.py \
    --video data/rally/video.mp4

# Ball tracking only
modal run scripts/eval_on_modal.py \
    --video data/rally/video.mp4 \
    --strategy ball

# Both trackers, first 500 frames only
modal run scripts/eval_on_modal.py \
    --video data/rally/video.mp4 \
    --strategy all --start 0 --end 500
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--video` | required | Input video path |
| `--keypoints` | auto-detected | `court_keypoints.json` (looks in same folder as video if omitted) |
| `--strategy` | `player` | `player` \| `ball` \| `all` |
| `--fps` | `0` (all frames) | Detection rate in fps — stride computed automatically from video fps |
| `--start` | `0` | First frame to process |
| `--end` | end of video | Last frame to process |
| `--tracker` | `bytetrack` | `bytetrack` \| `deepocsort` |
| `--no-video` | off | Skip annotated video output — tracking only, much faster for debugging |

### Outputs

All files saved next to the input video:

| File | Contents |
|---|---|
| `modal_player_output.mp4` | Annotated video — bboxes, ankle dots, mini-court overlay |
| `modal_player_detections.json` | Raw per-frame player detections |
| `modal_metrics.json` | Distance, speed, heatmap stats (requires keypoints) |
| `modal_ball_output.mp4` | Annotated video with ball trajectory trail |
| `modal_ball_detections.json` | Raw per-frame ball detections |

### Notes

- First run builds the Modal image (~3–5 min, cached after that).
- Videos < 50 MB are passed directly to the GPU function (no upload step).
- Videos ≥ 50 MB are uploaded to the `padel-eval-videos` Modal Volume.
- TrackNet weights are loaded from `padel_analytics/weights/ball_detection/`. Ball strategy is skipped if the files are missing.
- The report includes `tracking_duration_s` and `annotation_duration_s` separately so you can benchmark the tracker in isolation. Use `--no-video` to skip encoding entirely.

---

## Player Tracking

YOLO detects players each frame; ByteTrack/DeepOCSORT keeps IDs stable through occlusions. After a selection window (default 10 s), the 4 deepest in-court tracks are auto-selected as the real players.

**Logical player IDs** are assigned by court quadrant and remain stable for the entire video:

| ID | Side | Position |
|----|------|----------|
| P1 | Near | Left |
| P2 | Near | Right |
| P3 | Far  | Left |
| P4 | Far  | Right |

Near = camera side (Y < 10 m), Far = opposite baseline (Y ≥ 10 m). Left/right from the camera's perspective (X < 5 m / X ≥ 5 m). The mini-court overlay uses fixed colors matching the app: P1=green, P2=orange, P3=blue, P4=pink.

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
│   ├── eval_on_modal.py             # Plug-and-play GPU evaluation (T4, any strategy)
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
