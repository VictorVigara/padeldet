# PadelDet - Padel Match Analysis System

Computer vision system for analyzing padel matches from a fixed camera placed behind the glass wall at the end of a court.

## Overview

A mobile phone camera is positioned at one end of a padel court, recording through the glass wall. The recorded video is processed to extract match analytics through computer vision and tracking algorithms.

## Core Detection & Tracking

- **Player detection & tracking** — Detect and track all 4 players throughout the match using pose estimation and bounding boxes
- **Ball detection & tracking** — Detect the padel ball across frames, handling occlusions, fast motion blur, and small object size
- **Court detection** — Identify court lines and boundaries to establish a coordinate reference system (homography mapping to top-down view)

## Target Metrics

### Player Positioning
- Heatmaps per player showing court coverage
- Average position per rally/game/set
- Distance to net over time
- Team formation analysis (both players' relative positions)

### Player Movement
- Total distance covered per player
- Sprint count and speed estimates
- Movement direction and patterns
- Recovery positioning between shots

### Shot Analysis
- Shot count per player
- Shot type classification (smash, bandeja, vibora, volley, drive, lob, etc.)
- Shot placement mapping (where the ball lands)
- Rally length tracking

### Game Flow
- Rally detection (start/end)
- Point scoring events
- Serve detection
- Ball-in/ball-out estimation

## Technical Approach

### Camera Setup
- Single fixed camera behind the glass wall at one end of the court
- Phone mounted at approximately mid-height
- Challenges: glass reflections, distortion, fixed perspective with depth variation

### Pipeline

```
Video Input → Frame Extraction → Court Detection → Player Detection → Ball Detection
                                       ↓                  ↓               ↓
                                 Homography          Tracking &       Trajectory
                                  Matrix            Pose Estimation   Reconstruction
                                       ↓                  ↓               ↓
                                       └──────── Metric Computation ──────┘
                                                         ↓
                                                  Analytics Output
```

### Key Technologies
- **Object Detection**: YOLO (ultralytics) for player and ball detection
- **Pose Estimation**: For shot classification and player action recognition
- **Object Tracking**: ByteTrack / BoT-SORT for multi-object tracking across frames
- **Homography**: Court line detection + perspective transform to map pixel coordinates to real-world court positions
- **Ball Trajectory**: Specialized small-object detection + interpolation for fast-moving ball

## Project Structure

```
padeldet/
├── README.md
├── requirements.txt
├── src/
│   ├── detection/       # Player and ball detection models
│   ├── tracking/        # Multi-object tracking
│   ├── court/           # Court detection and homography
│   ├── metrics/         # Analytics computation
│   └── pipeline/        # End-to-end video processing
├── models/              # Trained model weights
├── data/                # Sample videos and annotations
├── notebooks/           # Experimentation and visualization
└── tests/
```

## Getting Started

> **Status**: Project bootstrapping — detection modules are the first implementation target.

### Prerequisites
- Python 3.10+
- PyTorch
- ultralytics (YOLOv8/v11)
- OpenCV

### Next Steps
1. Set up project skeleton and dependencies
2. Implement court line detection and homography
3. Player detection and tracking with YOLO + ByteTrack
4. Ball detection (fine-tuned small object model)
5. Metric computation from tracked positions
