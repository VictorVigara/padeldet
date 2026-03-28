"""
Quick evaluation script for player detection and tracking.

Usage:
    python scripts/evaluate_player_tracking.py --video path/to/video.mp4
    python scripts/evaluate_player_tracking.py --video path/to/video.mp4 --court-keypoints court_keypoints.json
    python scripts/evaluate_player_tracking.py --video path/to/video.mp4 --model yolov8m.pt --output results.mp4
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import padeldet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from padeldet.court.models import CourtKeypoints
from padeldet.court.projection import CourtProjection, draw_mini_court
from padeldet.models import save_detections
from padeldet.player_tracker import PlayerTracker, PlayerTrackerConfig
from padeldet.video import VideoInfo, VideoWriter, frame_generator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate player detection and tracking")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLO model path")
    parser.add_argument("--output", default="player_tracking_output.mp4", help="Output video path")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-json", default=None, help="Save detections to JSON")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, default=None, help="End frame")
    parser.add_argument("--court-keypoints", default=None, help="Court keypoints JSON (from calibrate_court.py)")
    return parser.parse_args()


def print_stats(results, court_projection=None):
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
    print(f"Total frames: {len(results)}")
    print(f"Detections/frame — min: {min(counts)}, avg: {sum(counts)/len(counts):.1f}, max: {max(counts)}")

    if confidences:
        print(f"Confidence — min: {min(confidences):.3f}, avg: {sum(confidences)/len(confidences):.3f}, max: {max(confidences):.3f}")

    # Frames with unexpected player count
    not_four = sum(1 for c in counts if c != 4)
    print(f"Frames with != 4 players: {not_four}/{len(results)} ({100*not_four/len(results):.1f}%)")

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
    print(f"\nUnique track IDs seen: {len(all_ids)} (ideal: 4)")
    if all_ids:
        print(f"  IDs: {sorted(all_ids)}")

    # Court projection stats
    if court_projection is not None:
        print("\nCourt-projected positions (meters):")
        for tid in sorted(all_ids):
            xs, ys = [], []
            for fd in results:
                for d in fd.detections:
                    if d.track_id == tid:
                        cx, cy = court_projection.project_detection(d)
                        xs.append(cx)
                        ys.append(cy)
            if xs:
                print(f"  Player #{tid}: x=[{min(xs):.1f}, {max(xs):.1f}] y=[{min(ys):.1f}, {max(ys):.1f}]")

    print("=" * 50)


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

    # Set up tracker
    config = PlayerTrackerConfig(
        model_path=args.model,
        confidence=args.confidence,
        batch_size=args.batch_size,
    )
    tracker = PlayerTracker(
        config=config,
        video_info=video_info,
        court_projection=court_projection,
    )
    tracker.to(tracker.device)
    print(f"Running on: {tracker.device}")

    # Run detection + tracking
    print("\n--- Detection + Tracking ---")
    frames = frame_generator(video_path, start=args.start, end=args.end)
    results = tracker.run(frames, start_frame=args.start)

    # Print stats
    print_stats(results, court_projection)

    # Save annotated video
    print(f"\nWriting annotated video to: {args.output}")
    with VideoWriter(args.output, video_info) as writer:
        for fd, frame in zip(
            results,
            frame_generator(video_path, start=args.start, end=args.end),
        ):
            annotated = fd.draw(frame)
            if court_projection is not None:
                annotated = draw_mini_court(annotated, court_projection, fd)
            writer.write(annotated)
    print("Done.")

    # Optionally save JSON
    if args.save_json:
        save_detections(results, args.save_json)
        print(f"Detections saved to: {args.save_json}")


if __name__ == "__main__":
    main()
