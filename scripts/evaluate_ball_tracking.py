"""
Quick evaluation script for ball detection and tracking.

Usage:
    python scripts/evaluate_ball_tracking.py --video data/rally/video.mp4 --tracknet-weights weights/TrackNet_best.pt
    python scripts/evaluate_ball_tracking.py --video data/rally/video.mp4 --tracknet-weights weights/TrackNet_best.pt --inpaintnet-weights weights/InpaintNet_best.pt
    python scripts/evaluate_ball_tracking.py --video data/rally/video.mp4 --tracknet-weights weights/TrackNet_best.pt --player-json data/rally/player_detections.json
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from padeldet.ball_tracker import BallTracker, BallTrackerConfig
from padeldet.ball_tracker import KalmanBallTracker, KalmanBallTrackerConfig, save_heatmaps, load_heatmaps
from padeldet.ball_tracker import Mog2BallTracker, Mog2BallTrackerConfig
from padeldet.court.projection import CourtProjection, draw_mini_court
from padeldet.court.models import CourtKeypoints
from padeldet.models import FrameDetections, load_detections, save_detections
from padeldet.video import VideoInfo, VideoWriter, frame_generator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ball detection and tracking")
    parser.add_argument("--video", default="/home/vvipu/padel_analytics/examples/videos/rally.mp4", help="Input video path")
    parser.add_argument("--tracknet-weights", default="/home/vvipu/padel_analytics/weights/ball_detection/TrackNet_best.pt", help="TrackNet model weights path")
    parser.add_argument("--inpaintnet-weights", default="/home/vvipu/padel_analytics/weights/ball_detection/InpaintNet_best.pt", help="InpaintNet model weights path (optional)")
    parser.add_argument("--output", default="ball_tracking_output_debug.mp4", help="Output video path")
    parser.add_argument("--save-json", default=None, help="Save detections to JSON")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, default=None, help="End frame")
    parser.add_argument("--trail-length", type=int, default=15, help="Trajectory trail length")
    parser.add_argument("--batch-size", type=int, default=8, help="TrackNet batch size")
    parser.add_argument("--player-json", default=None, help="Load player detections to overlay")
    parser.add_argument("--court-keypoints", default=None, help="Court keypoints JSON")
    parser.add_argument("--debug", default=False, action="store_true", help="Show each frame interactively (any key=next, q=quit)")
    parser.add_argument("--tracker", default="default", choices=["default", "kalman", "mog2"], help="Tracker variant to use")
    parser.add_argument("--heatmap-debug", default=False, action="store_true", help="Show raw heatmap + binary per frame during inference")
    parser.add_argument("--save-heatmaps", default=None, help="Save raw TrackNet heatmaps to .npz after inference")
    parser.add_argument("--load-heatmaps", default=None, help="Load precomputed heatmaps from .npz, skip network inference")
    parser.add_argument("--detect-shadows", default=False, action="store_true", help="MOG2: detect shadows (marked as 127 in mask)")
    parser.add_argument("--native-resolution", default=False, action="store_true", help="MOG2: run at native video resolution instead of 288×512")
    return parser.parse_args()


def print_stats(results: list[FrameDetections]):
    """Print ball detection statistics."""
    detected = [
        fd for fd in results
        if fd.detections and fd.detections[0].confidence > 0
    ]

    total = len(results)
    found = len(detected)
    rate = 100 * found / total if total > 0 else 0

    print("\n" + "=" * 50)
    print("BALL DETECTION STATS")
    print("=" * 50)
    print(f"Total frames: {total}")
    print(f"Ball detected: {found}/{total} ({rate:.1f}%)")

    # Longest continuous tracking streak
    max_streak = 0
    current_streak = 0
    for fd in results:
        if fd.detections and fd.detections[0].confidence > 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    print(f"Longest tracking streak: {max_streak} frames")

    # Longest gap
    max_gap = 0
    current_gap = 0
    for fd in results:
        if not fd.detections or fd.detections[0].confidence == 0:
            current_gap += 1
            max_gap = max(max_gap, current_gap)
        else:
            current_gap = 0
    print(f"Longest gap: {max_gap} frames")

    # Average ball speed (pixels/frame)
    speeds = []
    prev_pos = None
    for fd in results:
        if fd.detections and fd.detections[0].confidence > 0:
            pos = fd.detections[0].bbox.center
            if prev_pos is not None:
                dx = pos[0] - prev_pos[0]
                dy = pos[1] - prev_pos[1]
                speed = np.sqrt(dx**2 + dy**2)
                speeds.append(speed)
            prev_pos = pos
        else:
            prev_pos = None

    if speeds:
        print(f"Ball speed (px/frame) — avg: {np.mean(speeds):.1f}, max: {np.max(speeds):.1f}")

    print("=" * 50)


def draw_ball_trail(
    frame: np.ndarray,
    results: list[FrameDetections],
    current_idx: int,
    trail_length: int = 15,
) -> np.ndarray:
    """Draw ball position and trajectory trail on the frame."""
    trail = deque(maxlen=trail_length)

    # Collect recent visible positions
    start = max(0, current_idx - trail_length + 1)
    for i in range(start, current_idx + 1):
        fd = results[i]
        if fd.detections and fd.detections[0].confidence > 0:
            trail.append(fd.detections[0].bbox.center)
        else:
            trail.append(None)

    # Draw connecting lines between consecutive visible positions
    positions = list(trail)
    for i in range(1, len(positions)):
        if positions[i - 1] is not None and positions[i] is not None:
            pt1 = (int(positions[i - 1][0]), int(positions[i - 1][1]))
            pt2 = (int(positions[i][0]), int(positions[i][1]))
            alpha = (i + 1) / len(positions)
            color = (int(255 * alpha), int(255 * alpha), 0)
            thickness = max(1, int(2 * alpha))
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw trail dots with decreasing radius
    for i, pos in enumerate(positions):
        if pos is None:
            continue
        alpha = (i + 1) / len(positions)
        radius = max(2, int(6 * alpha))
        color = (int(255 * alpha), int(255 * alpha), 0)
        center = (int(pos[0]), int(pos[1]))
        cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)

    # Draw current position with highlight
    current_fd = results[current_idx]
    if current_fd.detections and current_fd.detections[0].confidence > 0:
        pos = current_fd.detections[0].bbox.center
        center = (int(pos[0]), int(pos[1]))
        cv2.circle(frame, center, 8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, center, 3, (0, 255, 255), -1, cv2.LINE_AA)

    return frame


def main():
    args = parse_args()

    video_path = args.video
    video_info = VideoInfo.from_path(video_path)
    print(f"Video: {video_path}")
    print(f"  Resolution: {video_info.width}x{video_info.height}")
    print(f"  FPS: {video_info.fps}")
    print(f"  Frames: {video_info.total_frames}")

    # Load court keypoints if provided
    court_projection = None
    if args.court_keypoints:
        keypoints = CourtKeypoints.load(args.court_keypoints)
        court_projection = CourtProjection(keypoints)
        print(f"\nCourt keypoints loaded: {keypoints.num_available}/12 points")

    # Load player detections if provided
    player_results = None
    if args.player_json:
        player_results = load_detections(args.player_json)
        print(f"Player detections loaded: {len(player_results)} frames")

    # Build tracker
    if args.tracker == "kalman":
        config = KalmanBallTrackerConfig(
            tracknet_weights="" if args.load_heatmaps else args.tracknet_weights,
            inpaintnet_weights=args.inpaintnet_weights,
            batch_size=args.batch_size,
            debug=args.debug,
            debug_heatmaps=args.heatmap_debug,
        )
        tracker = KalmanBallTracker(config=config, video_info=video_info)
        print(f"\nKalman tracker loaded, batch_size={config.batch_size}")
    elif args.tracker == "mog2":
        config = Mog2BallTrackerConfig(
            debug=args.debug,
            detect_shadows=args.detect_shadows,
            use_native_resolution=args.native_resolution,
        )
        tracker = Mog2BallTracker(config=config, video_info=video_info)
        print(f"\nMOG2 tracker (no neural network)")
    else:
        config = BallTrackerConfig(
            tracknet_weights=args.tracknet_weights,
            inpaintnet_weights=args.inpaintnet_weights,
            batch_size=args.batch_size,
            debug=args.debug,
            debug_heatmaps=args.heatmap_debug,
        )
        tracker = BallTracker(config=config, video_info=video_info)
        print(f"\nTrackNet loaded, batch_size={config.batch_size}")
    print(f"Running on: {tracker.device}")

    # Run detection (or load precomputed heatmaps to skip network)
    print("\n--- Ball Detection ---")
    frames = frame_generator(video_path, start=args.start, end=args.end)

    if args.load_heatmaps and args.tracker in ("kalman", "mog2"):
        precomputed, _ = load_heatmaps(args.load_heatmaps)
        results = tracker.detect_and_track(frames, precomputed_heatmaps=precomputed)
        tracker._results = results
    else:
        results = tracker.run(frames, start_frame=args.start)

    if args.save_heatmaps and args.tracker in ("kalman", "mog2") and tracker.last_heatmaps is not None:
        save_heatmaps(tracker.last_heatmaps, args.save_heatmaps, video_info)

    # Print stats
    print_stats(results)

    # Save annotated video
    print(f"\nWriting annotated video to: {args.output}")
    with VideoWriter(args.output, video_info) as writer:
        for i, frame in enumerate(
            frame_generator(video_path, start=args.start, end=args.end)
        ):
            # Draw player boxes if available
            if player_results is not None and i < len(player_results):
                frame = player_results[i].draw(frame)

            # Draw ball trail
            frame = draw_ball_trail(frame, results, i, args.trail_length)

            # Draw mini-court if available
            if court_projection is not None:
                if player_results is not None and i < len(player_results):
                    frame = draw_mini_court(frame, court_projection, player_results[i])

            writer.write(frame)
    print("Done.")

    # Optionally save JSON
    if args.save_json:
        save_detections(results, args.save_json)
        print(f"Detections saved to: {args.save_json}")


if __name__ == "__main__":
    main()
