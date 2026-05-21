"""
Quick evaluation script for player detection and tracking.

Usage:
    python scripts/evaluate_player_tracking.py --video data/rally/video.mp4

Defaults (all saved next to the video):
    --output          <video_dir>/player_tracking_output.mp4
    --court-keypoints <video_dir>/court_keypoints.json  (used if it exists)
    --save-json       <video_dir>/player_detections.json
"""

import argparse
import logging
import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to path so we can import padeldet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from padeldet.court.models import CourtKeypoints
from padeldet.court.projection import CourtProjection, draw_mini_court
from padeldet.match_profile import MATCH_PROFILES
from padeldet.models import save_detections
from padeldet.player_tracker import PlayerTracker, PlayerTrackerConfig
from padeldet.player_tracker.debug_overlay import FrameLogCapture, draw_log_overlay
from padeldet.video import VideoInfo, VideoWriter, frame_generator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate player detection and tracking")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--model", default="weights/yolov8m.pt", help="YOLO model path")
    parser.add_argument("--output", default=None, help="Output video path (default: <video_dir>/player_tracking_output.mp4)")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-json", default=None, help="Save detections to JSON (default: <video_dir>/player_detections.json)")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, default=None, help="End frame")
    parser.add_argument("--detection-fps", type=float, default=None, help="Run detection at this fps (e.g. 10 on a 25fps video → every 2-3 frames)")
    parser.add_argument("--court-keypoints", default=None, help="Court keypoints JSON (default: <video_dir>/court_keypoints.json if it exists)")
    parser.add_argument("--tracker-type", default="deepocsort", choices=["bytetrack", "deepocsort"],
                        help="Tracking algorithm (default: deepocsort)")
    parser.add_argument("--pose-model",
                        default="yolo26m-pose.pt",
                        help="YOLO pose model for ankle midpoint extraction")
    parser.add_argument("--match-type", default="doubles", choices=["doubles", "singles"],
                        help="Match type — selects player count + court dimensions (default: doubles)")
    parser.add_argument("--debug-overlay", action="store_true",
                        help="Capture per-frame [frame N] log records and draw them on the output video")
    args = parser.parse_args()

    # Fill in defaults relative to the video folder
    video_dir = Path(args.video).parent
    if args.output is None:
        args.output = str(video_dir / "player_tracking_output.mp4")
    if args.save_json is None:
        args.save_json = str(video_dir / "player_detections.json")
    if args.court_keypoints is None:
        candidate = video_dir / "court_keypoints.json"
        if candidate.exists():
            args.court_keypoints = str(candidate)

    return args


def print_stats(results, court_projection=None, video_fps=None, stride=1, n_players=4):
    """Print detection statistics."""
    counts = [len(fd) for fd in results]
    confidences = [
        d.confidence
        for fd in results
        for d in fd.detections
    ]

    print("\n" + "=" * 50)
    print("DETECTION STATS")
    print("=" * 50)
    fps_note = f" (every {stride} frames ≈ {video_fps / stride:.1f} fps)" if stride > 1 and video_fps else ""
    print(f"Detection frames processed: {len(results)}{fps_note}")
    print(f"Detections/frame — min: {min(counts)}, avg: {sum(counts)/len(counts):.1f}, max: {max(counts)}")

    if confidences:
        print(f"Confidence — min: {min(confidences):.3f}, avg: {sum(confidences)/len(confidences):.3f}, max: {max(confidences):.3f}")

    # Frames with unexpected player count
    wrong_count = sum(1 for c in counts if c != n_players)
    print(f"Frames with != {n_players} players: {wrong_count}/{len(results)} ({100*wrong_count/len(results):.1f}%)")

    # Distribution of detection counts
    from collections import Counter
    dist = Counter(counts)
    print("\nDetection count distribution:")
    for count in sorted(dist.keys()):
        pct = 100 * dist[count] / len(results)
        print(f"  {count} players: {dist[count]} frames ({pct:.1f}%)")

    # Track ID info
    all_ids = set()
    for fd in results:
        for d in fd.detections:
            if d.track_id is not None:
                all_ids.add(d.track_id)
    print(f"\nUnique track IDs seen: {len(all_ids)} (ideal: {n_players})")
    if all_ids:
        print(f"  IDs: {sorted(all_ids)}")

    # Court projection stats
    if court_projection is not None:
        print("\nCourt-projected positions (meters):")
        for tid in sorted(all_ids):
            xs, ys = [], []
            for fd in results:
                for d in fd.detections:
                    if d.track_id == tid and d.court_position is not None:
                        xs.append(d.court_position[0])
                        ys.append(d.court_position[1])
            if xs:
                ankle_pct = sum(
                    1 for fd in results for d in fd.detections
                    if d.track_id == tid and d.ankle_midpoint is not None
                ) / max(len(xs), 1) * 100
                print(f"  Player #{tid}: x=[{min(xs):.1f}, {max(xs):.1f}] y=[{min(ys):.1f}, {max(ys):.1f}]  ankle_midpoint: {ankle_pct:.0f}%")

    print("=" * 50)


def main():
    args = parse_args()

    video_path = args.video
    video_info = VideoInfo.from_path(video_path)
    print(f"Video: {video_path}")
    print(f"  Resolution: {video_info.width}x{video_info.height}")
    print(f"  FPS: {video_info.fps}")
    print(f"  Frames: {video_info.total_frames}")

    profile = MATCH_PROFILES[args.match_type]
    print(
        f"Match type: {args.match_type} "
        f"({profile.n_players} players on {profile.court.width_m:.1f}×{profile.court.length_m:.1f} m)"
    )

    # Load court keypoints if provided
    court_projection = None
    if args.court_keypoints:
        keypoints = CourtKeypoints.load(args.court_keypoints)
        court_projection = CourtProjection(
            keypoints,
            width_m=profile.court.width_m,
            length_m=profile.court.length_m,
        )
        print(f"\nCourt keypoints loaded: {keypoints.num_available}/12 points")

    # Set up tracker
    config = PlayerTrackerConfig(
        match_type=args.match_type,
        model_path=args.model,
        confidence=args.confidence,
        batch_size=args.batch_size,
        tracker_type=args.tracker_type,
        pose_model_path=args.pose_model,
    )
    tracker = PlayerTracker(
        config=config,
        video_info=video_info,
        court_projection=court_projection,
    )
    tracker.to(tracker.device)
    print(f"Running on: {tracker.device}")

    # Compute stride from detection fps
    stride = 1
    if args.detection_fps:
        stride = max(1, round(video_info.fps / args.detection_fps))
        print(f"Detection stride: {stride} (≈{video_info.fps / stride:.1f} fps)")

    # Optionally capture per-frame log records for video overlay.
    log_cap = None
    if args.debug_overlay:
        log_cap = FrameLogCapture(level=logging.DEBUG)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(log_cap)

    # Run detection + tracking
    print("\n--- Detection + Tracking ---")
    frames = frame_generator(video_path, start=args.start, end=args.end, stride=stride)
    try:
        results = tracker.run(frames, start_frame=args.start, stride=stride)
    finally:
        if log_cap is not None:
            logging.getLogger().removeHandler(log_cap)

    # Print stats
    print_stats(
        results, court_projection,
        video_fps=video_info.fps, stride=stride, n_players=profile.n_players,
    )

    # Save annotated video
    print(f"\nWriting annotated video to: {args.output}")
    frame_lookup = {fd.frame_index: fd for fd in results}
    last_fd = None
    abs_idx = args.start
    with VideoWriter(args.output, video_info) as writer:
        for frame in frame_generator(video_path, start=args.start, end=args.end):
            if abs_idx in frame_lookup:
                last_fd = frame_lookup[abs_idx]
            if last_fd is not None:
                annotated = last_fd.draw(frame)
                for det in last_fd.detections:
                    if det.ankle_midpoint is not None:
                        ax, ay = int(det.ankle_midpoint[0]), int(det.ankle_midpoint[1])
                        cv2.circle(annotated, (ax, ay), 6, (0, 0, 255), -1)
                        cv2.circle(annotated, (ax, ay), 8, (255, 255, 255), 2)
                if court_projection is not None:
                    annotated = draw_mini_court(annotated, court_projection, last_fd)
                if log_cap is not None:
                    annotated = draw_log_overlay(annotated, log_cap.lines_for(abs_idx))
            else:
                annotated = frame
            writer.write(annotated)
            abs_idx += 1
    print("Done.")

    # Save detections JSON
    save_detections(results, args.save_json)
    print(f"Detections saved to: {args.save_json}")


if __name__ == "__main__":
    main()
