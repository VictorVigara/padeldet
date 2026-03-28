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
- `padeldet/player_tracker/` — YOLO + ByteTrack (implemented)
- `padeldet/ball_tracker/` — future
- `padeldet/court_detector/` — future

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
│   └── player_tracker/
│       ├── __init__.py
│       └── player_tracker.py        # YOLO + ByteTrack
└── scripts/
    └── evaluate_player_tracking.py
```

## Tech Stack

- Python 3.10+
- ultralytics (YOLO) for object detection
- supervision for tracking (ByteTrack) and visualization
- OpenCV for video I/O
- PyTorch as ML backend

## Reference

- `/home/vvipu/padel_analytics/` — reference repo studied for architecture patterns
- Test video: `/home/vvipu/padel_analytics/examples/videos/rally.mp4`

## Development Notes

- Always discuss structural decisions with user before creating new modules
- Focus on detection/tracking first, metrics come later
- Server/API concerns are premature — build clean data models now, wrap in API later
- **Commit and push when a feature works** — keep the remote up to date with working increments
