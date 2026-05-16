"""
Plug-and-play CV evaluation on Modal GPU (T4).

Runs player tracking, ball tracking, or both on a local video + court keypoints,
and downloads the annotated output video + JSON results back to disk.

Prerequisites (one-time):
    pip install modal
    modal setup          # authenticate with your Modal account

Usage (from padeldet/):
    # Player tracking at 5 fps detection rate
    modal run scripts/eval_on_modal.py \\
        --video data/rally/video.mp4 \\
        --keypoints data/rally/court_keypoints.json \\
        --fps 5

    # Ball tracking, all frames
    modal run scripts/eval_on_modal.py \\
        --video data/rally/video.mp4 \\
        --strategy ball

    # Both trackers, frames 0–500 only
    modal run scripts/eval_on_modal.py \\
        --video data/rally/video.mp4 \\
        --strategy all --start 0 --end 500

Arguments:
    --video       Path to input video (required)
    --keypoints   Path to court_keypoints.json (optional; auto-detected if in same folder)
    --strategy    player | ball | all  (default: player)
    --fps         Detection rate in fps, e.g. 5 on a 25fps video → every 5th frame.
                  0 or omitted = all frames. Stride is computed automatically from
                  the video's native fps so you never need to know it.
    --start       First frame to process (default: 0)
    --end         Last frame to process (default: end of video)
    --tracker     bytetrack | deepocsort  (default: bytetrack)

Strategies:
    player  — YOLO + ByteTrack player tracking with court-aware auto-selection
    ball    — TrackNet ball tracking with trajectory trail
    all     — both sequentially

Outputs saved next to the input video:
    player_output.mp4        annotated video with bboxes + mini-court overlay
    player_detections.json   raw per-frame player detections
    metrics.json             distance, speed, heatmap stats (requires keypoints)
    ball_output.mp4          annotated video with ball trail
    ball_detections.json     raw per-frame ball detections

Notes:
    - First run builds the Modal image (~3-5 min). Subsequent runs reuse the cache.
    - Videos < 50 MB are passed directly to Modal (no volume upload needed).
    - Videos >= 50 MB are uploaded via `modal volume put` to the 'padel-eval-videos'
      volume before the GPU function starts.
    - TrackNet weights are loaded from:
        /home/vvipu/padel_analytics/weights/ball_detection/TrackNet_best.pt
      Ball tracking is silently skipped if those files are missing.
"""

from __future__ import annotations

import modal
from pathlib import Path

# ── Local weight paths ────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_WEIGHTS_DIR = _REPO_ROOT / "weights"
_TRACKNET_WEIGHTS = Path("/home/vvipu/padel_analytics/weights/ball_detection/TrackNet_best.pt")
_INPAINTNET_WEIGHTS = Path("/home/vvipu/padel_analytics/weights/ball_detection/InpaintNet_best.pt")

# ── Modal image ───────────────────────────────────────────────────────────────
_eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.0.0",
        "ultralytics>=8.0.0",
        "supervision>=0.18.0",
        "boxmot>=10.0.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pillow>=9.0.0",
    )
    .run_commands(
        # Pre-bake detection weights so cold-starts don't re-download.
        'python -c "from ultralytics import YOLO; YOLO(\'yolov8m.pt\')"',
    )
    .add_local_file(
        str(_WEIGHTS_DIR / "osnet_x0_25_msmt17.pt"),
        "/weights/osnet_x0_25_msmt17.pt",
    )
    .add_local_file(
        str(_WEIGHTS_DIR / "yolo26m-pose.pt"),
        "/weights/yolo26m-pose.pt",
    )
    .add_local_python_source(
        "padeldet",
        ignore=["**/__pycache__", "**/*.pyc", "**/.venv/**", "**/*.pt", "**/scripts/**", "**/data/**"],
    )
)

# Add ball tracker weights only if they exist locally.
if _TRACKNET_WEIGHTS.exists():
    _eval_image = _eval_image.add_local_file(str(_TRACKNET_WEIGHTS), "/weights/TrackNet_best.pt")
if _INPAINTNET_WEIGHTS.exists():
    _eval_image = _eval_image.add_local_file(str(_INPAINTNET_WEIGHTS), "/weights/InpaintNet_best.pt")

# ── Modal app + volume ────────────────────────────────────────────────────────
app = modal.App("padel-cv-eval")
eval_volume = modal.Volume.from_name("padel-eval-videos", create_if_missing=True)

_MOUNT = "/eval_data"

# ── GPU function ──────────────────────────────────────────────────────────────

@app.function(
    image=_eval_image,
    gpu="T4",
    cpu=2.0,
    memory=8192,
    timeout=3600,
    volumes={_MOUNT: eval_volume},
)
def run_eval(
    video_filename: str,
    keypoints_dict: dict | None,
    strategy: str,
    detection_fps: float,
    start: int,
    end: int | None,
    tracker: str = "bytetrack",
    video_bytes: bytes | None = None,
    annotate_video: bool = True,
) -> dict:
    import json
    import time
    from collections import deque
    from pathlib import Path

    import cv2

    from padeldet.court.models import CourtKeypoints
    from padeldet.court.projection import CourtProjection, draw_mini_court
    from padeldet.models import save_detections
    from padeldet.video import VideoInfo, VideoWriter, frame_generator

    # Video either passed directly as bytes (small files) or already in the volume.
    if video_bytes is not None:
        video_path = f"/tmp/{video_filename}"
        Path(video_path).write_bytes(video_bytes)
    else:
        video_path = f"{_MOUNT}/{video_filename}"

    video_info = VideoInfo.from_path(video_path)
    stride = max(1, round(video_info.fps / detection_fps)) if detection_fps > 0 else 1
    effective_fps = video_info.fps / stride
    print(f"Video: {video_filename}  {video_info.width}x{video_info.height} @ {video_info.fps:.1f} fps  ({video_info.total_frames} frames)")
    print(f"Detection: stride={stride} → {effective_fps:.1f} fps")

    court_projection = None
    if keypoints_dict is not None:
        keypoints = CourtKeypoints.from_dict(keypoints_dict)
        court_projection = CourtProjection(keypoints)
        print(f"Court projection loaded ({keypoints.num_available}/12 keypoints)")

    report: dict = {}
    t0 = time.time()

    if strategy in ("player", "all"):
        report["player"] = _run_player(
            video_path, video_info, court_projection, stride, start, end, tracker,
            annotate_video=annotate_video,
        )

    if strategy in ("ball", "all"):
        report["ball"] = _run_ball(
            video_path, video_info, court_projection, start, end,
            annotate_video=annotate_video,
        )

    eval_volume.commit()
    report["total_duration_s"] = round(time.time() - t0, 1)
    return report


def _run_player(video_path, video_info, court_projection, stride, start, end, tracker="bytetrack", annotate_video=True):
    import time
    import cv2
    from padeldet.player_tracker import PlayerTracker, PlayerTrackerConfig
    from padeldet.models import save_detections
    from padeldet.video import VideoWriter, frame_generator
    from padeldet.court.projection import draw_mini_court

    t0 = time.time()
    config = PlayerTrackerConfig(
        model_path="yolov8m.pt",
        pose_model_path="/weights/yolo26m-pose.pt",
        confidence=0.1,
        iou_threshold=0.1,
        imgsz=960,
        batch_size=8,
        tracker_type=tracker,
        auto_select_players=True,
        device="cuda",
    )
    tracker_obj = PlayerTracker(config=config, video_info=video_info, court_projection=court_projection)
    frames = frame_generator(video_path, start=start, end=end, stride=stride)
    player_results = tracker_obj.run(frames, start_frame=start, stride=stride)
    tracking_duration_s = round(time.time() - t0, 1)

    if annotate_video:
        t_ann = time.time()
        out_path = f"{_MOUNT}/player_output.mp4"
        frame_lookup = {fd.frame_index: fd for fd in player_results}
        last_fd = None
        abs_idx = start
        with VideoWriter(out_path, video_info) as writer:
            for frame in frame_generator(video_path, start=start, end=end):
                if abs_idx in frame_lookup:
                    last_fd = frame_lookup[abs_idx]
                if last_fd is not None:
                    annotated = last_fd.draw(frame)
                    for det in last_fd.detections:
                        _draw_skeleton(annotated, det)
                        if det.ankle_midpoint is not None:
                            ax, ay = int(det.ankle_midpoint[0]), int(det.ankle_midpoint[1])
                            cv2.circle(annotated, (ax, ay), 6, (0, 0, 255), -1)
                            cv2.circle(annotated, (ax, ay), 8, (255, 255, 255), 2)
                    if court_projection is not None:
                        annotated = draw_mini_court(annotated, court_projection, last_fd)
                else:
                    annotated = frame
                writer.write(annotated)
                abs_idx += 1
        annotation_duration_s = round(time.time() - t_ann, 1)
    else:
        annotation_duration_s = 0

    save_detections(player_results, f"{_MOUNT}/player_detections.json")

    # Per-frame stats.
    counts = [len(fd) for fd in player_results]
    all_ids = {
        d.track_id
        for fd in player_results
        for d in fd.detections
        if d.track_id is not None
    }
    report = {
        "frames_processed": len(player_results),
        "unique_track_ids": len(all_ids),
        "avg_detections_per_frame": round(sum(counts) / max(len(counts), 1), 2),
        "frames_with_4_players": sum(1 for c in counts if c == 4),
        "frames_with_lt4_players": sum(1 for c in counts if c < 4),
        "tracking_duration_s": tracking_duration_s,
        "annotation_duration_s": annotation_duration_s,
        "duration_s": tracking_duration_s + annotation_duration_s,
    }

    if court_projection is not None:
        try:
            from padeldet.metrics.compute import compute_metrics
            metrics = compute_metrics(player_results, court_projection, video_info.fps)
            metrics.save(f"{_MOUNT}/metrics.json")
            report["metrics_saved"] = True
        except Exception as e:
            report["metrics_error"] = str(e)

    return report


_COCO_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 11), (6, 12), (11, 12),                  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),      # legs
]


def _draw_skeleton(frame, det, min_conf: float = 0.3) -> None:
    import cv2
    if det.keypoints is None:
        return
    kps = det.keypoints
    for a, b in _COCO_SKELETON:
        if a >= len(kps) or b >= len(kps):
            continue
        xa, ya, ca = kps[a]
        xb, yb, cb = kps[b]
        if ca < min_conf or cb < min_conf:
            continue
        if (xa == 0.0 and ya == 0.0) or (xb == 0.0 and yb == 0.0):
            continue
        cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2, cv2.LINE_AA)
    for x, y, c in kps:
        if c < min_conf or (x == 0.0 and y == 0.0):
            continue
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1, cv2.LINE_AA)


def _run_ball(video_path, video_info, court_projection, start, end):
    import time
    from collections import deque
    import cv2
    from padeldet.ball_tracker import BallTracker, BallTrackerConfig
    from padeldet.models import save_detections
    from padeldet.video import VideoWriter, frame_generator
    from padeldet.court.projection import draw_mini_court

    t0 = time.time()
    config = BallTrackerConfig(
        tracknet_weights="/weights/TrackNet_best.pt",
        inpaintnet_weights="/weights/InpaintNet_best.pt",
        batch_size=8,
    )
    ball_tracker = BallTracker(config=config, video_info=video_info)
    frames = frame_generator(video_path, start=start, end=end)
    ball_results = ball_tracker.run(frames, start_frame=start)

    out_path = f"{_MOUNT}/ball_output.mp4"
    trail_length = 15
    with VideoWriter(out_path, video_info) as writer:
        for i, frame in enumerate(frame_generator(video_path, start=start, end=end)):
            frame = _draw_ball_trail(frame, ball_results, i, trail_length)
            writer.write(frame)

    save_detections(ball_results, f"{_MOUNT}/ball_detections.json")

    detected = sum(
        1 for fd in ball_results
        if fd.detections and fd.detections[0].confidence > 0
    )
    return {
        "frames_processed": len(ball_results),
        "ball_detected": detected,
        "detection_rate_pct": round(100 * detected / max(len(ball_results), 1), 1),
        "duration_s": round(time.time() - t0, 1),
    }


def _draw_ball_trail(frame, results, current_idx, trail_length):
    from collections import deque
    import cv2
    import numpy as np

    trail: deque = deque(maxlen=trail_length)
    start_i = max(0, current_idx - trail_length + 1)
    for i in range(start_i, current_idx + 1):
        fd = results[i]
        if fd.detections and fd.detections[0].confidence > 0:
            trail.append(fd.detections[0].bbox.center)
        else:
            trail.append(None)

    positions = list(trail)
    for i in range(1, len(positions)):
        if positions[i - 1] is not None and positions[i] is not None:
            pt1 = (int(positions[i - 1][0]), int(positions[i - 1][1]))
            pt2 = (int(positions[i][0]), int(positions[i][1]))
            alpha = (i + 1) / len(positions)
            color = (int(255 * alpha), int(255 * alpha), 0)
            cv2.line(frame, pt1, pt2, color, max(1, int(2 * alpha)), cv2.LINE_AA)

    current_fd = results[current_idx]
    if current_fd.detections and current_fd.detections[0].confidence > 0:
        pos = current_fd.detections[0].bbox.center
        center = (int(pos[0]), int(pos[1]))
        cv2.circle(frame, center, 8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, center, 3, (0, 255, 255), -1, cv2.LINE_AA)

    return frame


# ── Local entrypoint (CLI) ────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    video: str,
    keypoints: str = "",
    strategy: str = "player",
    fps: float = 0.0,
    start: int = 0,
    end: str = "",
    tracker: str = "bytetrack",
    no_video: bool = False,
):
    import json
    from pathlib import Path

    video_path = Path(video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    video_dir = video_path.parent
    video_filename = video_path.name
    end_frame = int(end) if end else None

    if strategy not in ("player", "ball", "all"):
        raise ValueError(f"--strategy must be player, ball, or all (got: {strategy!r})")

    # Load court keypoints.
    keypoints_dict = None
    kp_path = Path(keypoints) if keypoints else video_dir / "court_keypoints.json"
    if kp_path.exists():
        with open(kp_path) as f:
            keypoints_dict = json.load(f)
        print(f"Keypoints: {kp_path}")
    else:
        print("No court keypoints found — running without court projection")

    # Transfer video to Modal.
    # Small files (< 50 MB): pass as bytes directly — avoids blocking batch_upload.
    # Large files (>= 50 MB): upload via `modal volume put` subprocess before the
    #   app's heartbeat loop starts, then pass None so the function reads from volume.
    size_mb = video_path.stat().st_size / 1024 / 1024
    video_bytes: bytes | None = None

    if size_mb < 50:
        print(f"Video is {size_mb:.1f} MB — passing directly to Modal function.")
        video_bytes = video_path.read_bytes()
    else:
        print(f"Video is {size_mb:.1f} MB — uploading to Modal volume via CLI...")
        import subprocess
        subprocess.run(
            ["modal", "volume", "put", "--force", "padel-eval-videos", str(video_path), video_filename],
            check=True,
        )
        print("Upload done.")

    fps_label = f"{fps} fps" if fps > 0 else "full fps"
    print(f"Running '{strategy}' strategy on T4 GPU (tracker={tracker}, detection={fps_label}, start={start}, end={end_frame or 'EOF'})...")
    report = run_eval.remote(
        video_filename=video_filename,
        keypoints_dict=keypoints_dict,
        strategy=strategy,
        detection_fps=fps,
        start=start,
        end=end_frame,
        tracker=tracker,
        video_bytes=video_bytes,
        annotate_video=not no_video,
    )

    print("\n=== Report ===")
    print(json.dumps(report, indent=2))

    # Download output files from volume.
    output_files: list[str] = []
    if strategy in ("player", "all"):
        if not no_video:
            output_files.append("player_output.mp4")
        output_files.append("player_detections.json")
        if keypoints_dict is not None:
            output_files.append("metrics.json")
    if strategy in ("ball", "all"):
        if not no_video:
            output_files.append("ball_output.mp4")
        output_files.append("ball_detections.json")

    print("\nDownloading outputs...")
    for filename in output_files:
        local_path = video_dir / filename
        try:
            data = b"".join(eval_volume.read_file(filename))
            local_path.write_bytes(data)
            size_mb = len(data) / 1024 / 1024
            print(f"  {local_path}  ({size_mb:.1f} MB)")
        except Exception as exc:
            print(f"  Warning: could not download {filename}: {exc}")

    print("\nDone.")
