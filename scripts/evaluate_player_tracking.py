"""
Quick evaluation script for player detection and tracking.

Usage:
    python scripts/evaluate_player_tracking.py --video data/rally/video.mp4
    python scripts/evaluate_player_tracking.py --video data/rally/video.mp4 --court-keypoints data/rally/court_keypoints.json
    python scripts/evaluate_player_tracking.py --video data/rally/video.mp4 --model weights/yolov8m.pt --output data/rally/player_tracking_output.mp4
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

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
    parser.add_argument("--model", default="weights/yolov8m.pt", help="YOLO model path")
    parser.add_argument("--output", default="player_tracking_output.mp4", help="Output video path")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-json", default=None, help="Save detections to JSON")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, default=None, help="End frame")
    parser.add_argument("--detection-fps", type=float, default=None, help="Run detection at this fps (e.g. 10 on a 25fps video → every 2-3 frames)")
    parser.add_argument("--court-keypoints", default=None, help="Court keypoints JSON (from calibrate_court.py)")
    parser.add_argument("--tracker-type", default="deepocsort", choices=["bytetrack", "deepocsort"],
                        help="Tracking algorithm (default: deepocsort)")
    parser.add_argument("--warmup-plot", action="store_true",
                        help="Show bar chart of warm-up metrics (ratio, depth, score) per track")
    parser.add_argument("--save-warmup-crops", default=None, metavar="DIR",
                        help="Save overview image + one crop per track from the warm-up window")
    return parser.parse_args()


def plot_warmup_metrics(warmup_metrics: dict) -> None:
    """Bar chart of warm-up metrics per track (ratio, mean depth, score)."""
    import matplotlib.pyplot as plt

    tids = list(warmup_metrics.keys())
    labels = [f"#{t}\n{'LOCKED' if warmup_metrics[t]['locked'] else 'dup'}" for t in tids]
    colors = ["steelblue" if warmup_metrics[t]["locked"] else "salmon" for t in tids]
    has_depth = any(warmup_metrics[t]["mean_depth_m"] > 0 for t in tids)

    ncols = 3 if has_depth else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, max(4, len(tids) * 0.5 + 1)))
    fig.suptitle("Warm-up player lock metrics", fontsize=13)

    def hbar(ax, values, title, fmt=".2f", xlim=None):
        bars = ax.barh(labels, values, color=colors)
        ax.set_title(title)
        ax.invert_yaxis()
        if xlim:
            ax.set_xlim(0, xlim)
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + (xlim or max(values)) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:{fmt}}", va="center", fontsize=8)

    hbar(axes[0], [warmup_metrics[t]["ratio"] for t in tids], "In-court ratio", xlim=1.05)
    if has_depth:
        hbar(axes[1], [warmup_metrics[t]["mean_depth_m"] for t in tids], "Mean depth (m)", xlim=6.0)
        hbar(axes[2], [warmup_metrics[t]["score"] for t in tids], "Score (ratio × depth)")
    else:
        hbar(axes[1], [warmup_metrics[t]["score"] for t in tids], "Score (frequency)")

    plt.tight_layout()
    plt.show()


def save_warmup_crops(tracker, results: list, video_path: str, output_dir: str) -> None:
    """Save an overview frame and one crop per track from the warm-up window."""
    import os
    from collections import defaultdict

    

    metrics = getattr(tracker, "warmup_metrics", {})
    warmup_count = getattr(tracker, "warmup_frames_used", len(results))
    if not metrics:
        print("No warm-up metrics available — skipping crop save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    warmup_results = results[:warmup_count]

    # Overview: frame where the most tracked players are visible simultaneously
    best_fd = max(
        warmup_results,
        key=lambda fd: sum(1 for d in fd.detections if d.track_id in metrics),
    )

    # Per-track representative: middle frame of each track's appearances
    track_appearances: dict[int, list] = defaultdict(list)
    for fd in warmup_results:
        for det in fd.detections:
            if det.track_id in metrics:
                track_appearances[det.track_id].append((fd.frame_index, det))

    track_rep: dict[int, tuple] = {
        tid: apps[len(apps) // 2]
        for tid, apps in track_appearances.items()
    }

    # Load only the needed frames from video
    needed = {best_fd.frame_index} | {fi for fi, _ in track_rep.values()}
    cap = cv2.VideoCapture(video_path)
    loaded: dict[int, np.ndarray] = {}
    for idx in sorted(needed):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            loaded[idx] = frame
    cap.release()

    def draw_box(img: np.ndarray, det, color: tuple, label: str) -> None:
        x1, y1, x2, y2 = int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    has_depth = any(metrics[t]["mean_depth_m"] > 0 for t in metrics)

    # --- Overview image ---
    if best_fd.frame_index in loaded:
        overview = loaded[best_fd.frame_index].copy()
        for det in best_fd.detections:
            tid = det.track_id
            if tid not in metrics:
                continue
            m = metrics[tid]
            color = (0, 180, 0) if m["locked"] else (0, 60, 220)
            if has_depth:
                label = f"#{tid} s={m['score']:.1f} {'L' if m['locked'] else 'D'}"
            else:
                label = f"#{tid} {'L' if m['locked'] else 'D'}"
            draw_box(overview, det, color, label)
        path = os.path.join(output_dir, "warmup_overview.jpg")
        cv2.imwrite(path, overview)
        print(f"  warmup_overview.jpg  (frame {best_fd.frame_index})")

    # --- Per-track crops ---
    PAD = 50
    for tid, (fi, det) in track_rep.items():
        if fi not in loaded:
            continue
        m = metrics[tid]
        frame = loaded[fi].copy()
        color = (0, 180, 0) if m["locked"] else (0, 60, 220)
        if has_depth:
            label = (
                f"#{tid}  ratio={m['ratio']:.2f}  "
                f"depth={m['mean_depth_m']:.1f}m  score={m['score']:.2f}  "
                f"{'LOCKED' if m['locked'] else 'duplicate'}"
            )
        else:
            label = f"#{tid}  ratio={m['ratio']:.2f}  {'LOCKED' if m['locked'] else 'duplicate'}"
        draw_box(frame, det, color, label)

        h, w = frame.shape[:2]
        x1 = max(0, int(det.bbox.x1) - PAD)
        y1 = max(0, int(det.bbox.y1) - PAD)
        x2 = min(w, int(det.bbox.x2) + PAD)
        y2 = min(h, int(det.bbox.y2) + PAD)
        crop = frame[y1:y2, x1:x2]

        status = "locked" if m["locked"] else "dup"
        fname = f"track_{tid:03d}_{status}.jpg"
        cv2.imwrite(os.path.join(output_dir, fname), crop)
        print(f"  {fname}  (frame {fi})")

    print(f"Warm-up crops saved to: {output_dir}/")


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
        tracker_type=args.tracker_type,
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

    # Run detection + tracking
    print("\n--- Detection + Tracking ---")
    frames = frame_generator(video_path, start=args.start, end=args.end, stride=stride)
    results = tracker.run(frames, start_frame=args.start, stride=stride)

    # Print stats
    print_stats(results, court_projection)

    # Warm-up plot
    if args.warmup_plot and hasattr(tracker, "warmup_metrics"):
        plot_warmup_metrics(tracker.warmup_metrics)

    # Warm-up crops
    if args.save_warmup_crops:
        print(f"\n--- Warm-up crops ---")
        save_warmup_crops(tracker, results, video_path, args.save_warmup_crops)

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
                if court_projection is not None:
                    annotated = draw_mini_court(annotated, court_projection, last_fd)
            else:
                annotated = frame
            writer.write(annotated)
            abs_idx += 1
    print("Done.")

    # Optionally save JSON
    if args.save_json:
        save_detections(results, args.save_json)
        print(f"Detections saved to: {args.save_json}")


if __name__ == "__main__":
    main()
