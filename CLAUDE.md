# PadelDet

## What This Project Is

Computer vision system for analyzing padel matches. A phone camera is placed at one end of a padel court, recording through the glass wall. The video is processed to detect players, the ball, and extract match metrics.

## Camera Setup

- Single fixed phone camera behind the glass wall at one end of the court
- Challenges: glass reflections, distortion, fixed perspective with depth variation, small ball size, fast motion

## Core Goals

1. **Player detection & tracking** — Detect and track all 4 players across frames
2. **Ball detection & tracking** — Detect the padel ball despite small size and fast motion
3. **Court detection** — Identify court lines for coordinate mapping
4. **Metrics extraction** — Player positioning, movement, shot analysis, game flow

## Deployment Model

- **Phone app** records the match video
- **Video is sent to a server** for CV analysis
- Server details are premature now — focus on getting detection/tracking working first
- Keep data models clean (plain dataclasses, JSON-serializable) so they're ready for API use later

## Architecture

### Data Models (`padeldet/models.py`)
- Plain dataclasses, no library coupling in the data layer
- `BoundingBox` → `Detection` → `TrackedDetection` → `FrameDetections`
- `serialize()` / `from_dict()` on all models for JSON persistence
- `draw()` on `FrameDetections` uses supervision annotators but converts internally (data model stays library-free)
- `save_detections()` / `load_detections()` for JSON caching

### Tracker ABC (`padeldet/base.py`)
- Single ABC combining detection + tracking (they always go together)
- Key method: `detect_and_track(frames) -> list[FrameDetections]`
- `run()` = detect_and_track + store results
- `save_results()` / `load_results()` for JSON caching
- Each tracker type lives in its own subdirectory

### Concrete Trackers (each in its own folder)
- `padeldet/player_tracker/` — YOLO + DeepOCSORT (OSNet ReID) + warm-up player lock
- `padeldet/ball_tracker/` — MOG2 background subtraction + HSV color + linear motion prediction

### Court Module (`padeldet/court/`)
- **Not a Tracker** — court detection is static, not a tracking problem
- `models.py` — `CourtKeypoints` dataclass (12-point schema), court dimensions in meters, `COURT_KEYPOINTS_2D` reference coords
- `projection.py` — `CourtProjection` (homography via `cv2.findHomography`), point projection, court polygon check, `draw_mini_court()` overlay
- `draw_mini_court()` labels players as `P1`–`P4` (falls back to raw track ID pre-selection); colors are fixed per logical ID matching `AppColors` (P1=green, P2=orange, P3=blue, P4=pink)
- Supports partial keypoints (some can be None/skipped), needs ≥4 for homography
- Manual calibration via `scripts/calibrate_court.py`, auto-detection can be added later

### Ball Tracker (`padeldet/ball_tracker/`)
- `BallTrackerConfig` — TrackNet weights, InpaintNet weights, batch size, post-processing params
- `BallTracker` — implements Tracker ABC, TrackNet U-Net CNN with temporal ensemble
- Pipeline: median background → sliding 8-frame windows → TrackNet heatmaps → temporal ensemble → outlier rejection (adaptive MAD) → temporal isolation → InpaintNet gap-filling → Savitzky-Golay smoothing
- Ball represented as `TrackedDetection` with synthetic small bbox via `ball_detection()` helper in `models.py`
- `confidence=0.0` means ball not visible in that frame
- Debug mode (`config.debug=True`) shows interactive frame-by-frame visualization at each pipeline stage
- Pre-trained weights: `TrackNet_best.pt` and `InpaintNet_best.pt`

### Metrics Module (`padeldet/metrics/`)
- **Offline computation** from saved player detection JSON + court keypoints
- `compute_metrics(detections, projection, fps)` → `MatchMetrics`
- **Position pipeline**: groups by `logical_player_id` (not raw track ID) → project to court meters → interpolate short gaps → Savitzky-Golay smoothing
- **Team assignment**: median Y over first 30 frames, split near/far at net (10m)
- **Per-player metrics**: distance covered, speed (avg/max/distribution), zone time (net/transition/back), net approaches, position heatmap
- **Per-team metrics**: formation classification (both-at-net, both-at-back, one-up-one-back)
- Zone boundaries relative to team side: net=0-3m from baseline, transition=3-7m, back=7-10m
- Speed categories: stationary (<0.5 m/s), walking (0.5-2), jogging (2-4), sprinting (>4)
- JSON serialization via `MatchMetrics.save()`/`load()`

### Player Filtering & Identity Strategy
- **Warm-up window** (default 60 s, configurable via `warmup_duration_sec`): observe all tracks, accumulate `in_court_ratio` and `mean_depth_m` (average distance to nearest court line in meters)
- **Score** = `in_court_ratio × mean_depth_m` — ranks real players (deep inside) above boundary spectators (near glass wall, depth ≈ 0)
- After warm-up: lock the top-4 by score as `locked_player_ids`; log extras as `suspected_duplicate`
- Filter entire video to keep only those 4 IDs
- Without court projection: falls back to raw detection frequency
- `tracker.warmup_metrics` dict (tid → ratio/depth/score/locked) available after tracking for inspection
- `tracker.warmup_frames_used` stores the actual frame count used for the warm-up window
- **Logical player IDs** are assigned by court quadrant (not selection order): P1=near-left, P2=near-right, P3=far-left, P4=far-right — consistent across the whole video including pre-selection frames

### Video I/O (`padeldet/video.py`)
- `VideoInfo` dataclass + `frame_generator()` using OpenCV
- `VideoWriter` context manager for output videos

### Key Design Decisions
- **Single Tracker ABC** (not separate Detector/Tracker) — simpler, detection+tracking always needed together
- **Config as dataclasses** (not module-level constants) — `PlayerTrackerConfig`, etc.
- **Data models are plain dataclasses** — supervision only used inside `draw()`, not stored
- **Each tracker in its own folder** — room for tracker-specific helpers and models
- **Evaluation scripts in `scripts/`** — quick and disposable, use the architecture layer

## Project Structure

```
padeldet/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── padeldet/
│   ├── __init__.py
│   ├── models.py                    # Shared data models
│   ├── base.py                      # Tracker ABC
│   ├── video.py                     # Video I/O
│   ├── player_tracker/
│   │   ├── __init__.py
│   │   └── player_tracker.py        # YOLO + ByteTrack + court-aware filtering
│   ├── ball_tracker/
│   │   ├── __init__.py
│   │   ├── ball_tracker.py          # TrackNet + ensemble + post-processing
│   │   └── models.py                # TrackNet, InpaintNet architectures
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── models.py                # PlayerTrack, MatchMetrics, enums
│   │   ├── positions.py             # Projection, interpolation, smoothing, teams
│   │   ├── zones.py                 # Zone classification, formations, heatmaps
│   │   ├── movement.py              # Distance, speed, speed distribution
│   │   └── compute.py               # MetricsConfig, compute_metrics()
│   └── court/
│       ├── __init__.py
│       ├── models.py                # CourtKeypoints, court dimensions
│       └── projection.py           # Homography, projection, mini-court
└── scripts/
    ├── evaluate_player_tracking.py  # --court-keypoints support
    ├── evaluate_ball_tracking.py    # Ball detection eval + trail visualization
    ├── evaluate_metrics.py          # Player metrics computation + heatmap plots
    └─��� calibrate_court.py           # Manual keypoint calibration tool
```

## Tech Stack

- Python 3.10+
- ultralytics (YOLO) for object detection
- supervision for tracking (ByteTrack) and visualization
- OpenCV for video I/O
- PyTorch as ML backend
- scipy for signal processing (smoothing filters)

## Reference

- `/home/vvipu/padel_analytics/` — reference repo studied for architecture patterns
- Test video: `/home/vvipu/padel_analytics/examples/videos/rally.mp4`

## Development Notes

- Always discuss structural decisions with user before creating new modules
- Detection/tracking done; metrics module implemented for positioning and movement
- Server/API concerns are premature — build clean data models now, wrap in API later
- **Commit and push when a feature works** — keep the remote up to date with working increments
