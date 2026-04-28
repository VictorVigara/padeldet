"""
Task 1 investigation: confirm boxmot embedding access path on a real video.

Drives PlayerTracker frame-by-frame, snapshotting active_tracks after each
deep_tracker.update() call. At the end, prints:
  - embedding shape and dtype
  - pairwise cosine similarity matrix
  - sorted list of all pairwise similarities (to calibrate threshold)
  - active track count over time (for max_age=1800 performance sanity check)

Saves per-track crop images and a duplicate-pairs visual to debug_embeddings_out/.

Usage:
  conda run -n padeldet python scripts/debug_embeddings.py \
      /path/to/video.mp4 --frames 300
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).parent.parent))

from padeldet.models import BoundingBox, FrameDetections, TrackedDetection
from padeldet.player_tracker.player_tracker import PlayerTracker, PlayerTrackerConfig
from padeldet.video import VideoInfo, frame_generator

CROP_H = 160
CROP_W = 80
MARGIN = 6  # px between crops in grid


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_embedding(emb_sum: np.ndarray, count: int) -> np.ndarray:
    v = emb_sum / count
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _crop_bbox(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x1c, y1c = max(0, int(x1)), max(0, int(y1))
    x2c, y2c = min(w, int(x2)), min(h, int(y2))
    if x2c <= x1c or y2c <= y1c:
        return np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
    crop = frame[y1c:y2c, x1c:x2c]
    return cv2.resize(crop, (CROP_W, CROP_H))


def _make_track_tile(crop: np.ndarray, track_id: int, n_frames: int) -> np.ndarray:
    """Add a label bar at the bottom of a crop."""
    tile = crop.copy()
    bar = np.zeros((20, CROP_W, 3), dtype=np.uint8)
    cv2.putText(bar, f"t{track_id} ({n_frames}f)", (2, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([tile, bar])


def _make_pair_tile(
    cropA: np.ndarray, idA: int, nA: int,
    cropB: np.ndarray, idB: int, nB: int,
    similarity: float,
) -> np.ndarray:
    tileA = _make_track_tile(cropA, idA, nA)
    tileB = _make_track_tile(cropB, idB, nB)
    gap = np.zeros((tileA.shape[0], MARGIN, 3), dtype=np.uint8)
    pair = np.hstack([tileA, gap, tileB])
    label = np.zeros((22, pair.shape[1], 3), dtype=np.uint8)
    cv2.putText(label, f"sim={similarity:.4f}", (2, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([label, pair])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to process")
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--reid-weights", default="weights/osnet_x0_25_msmt17.pt")
    parser.add_argument("--conf-threshold", type=float, default=0.4,
                        help="Min confidence to accept an embedding")
    parser.add_argument("--out-dir", default="debug_embeddings_out",
                        help="Directory for output images")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_info = VideoInfo.from_path(args.video)
    print(f"Video: {args.video}")
    print(f"Resolution: {video_info.width}x{video_info.height} @ {video_info.fps:.1f} fps")
    print(f"Processing first {args.frames} frames\n")

    config = PlayerTrackerConfig(
        model_path=args.model,
        reid_weights=args.reid_weights,
        tracker_type="deepocsort",
        tracker_max_age_sec=60.0,
        fp16=True,
        confidence=0.1,
        batch_size=1,  # one frame at a time so we can snapshot after each update
    )
    tracker = PlayerTracker(config=config, video_info=video_info)

    # Gallery: running sum + count per track_id
    emb_sum: dict[int, np.ndarray] = {}
    emb_count: dict[int, int] = defaultdict(int)
    active_track_counts: list[int] = []
    # Representative crops: first good bbox seen per track
    track_crops: dict[int, np.ndarray] = {}
    track_bbox: dict[int, tuple] = {}  # (x1, y1, x2, y2) of best crop

    first_update_done = False
    min_hits = tracker.deep_tracker.min_hits

    frames = list(frame_generator(args.video, start=0, end=args.frames))
    print(f"Loaded {len(frames)} frames")

    for frame_idx, bgr_frame in enumerate(frames):
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # YOLO detection (single frame)
        results = tracker.model.predict(
            [rgb_frame],
            conf=config.confidence,
            iou=config.iou_threshold,
            imgsz=config.imgsz,
            device=tracker.device,
            classes=[config.person_class_id],
            verbose=False,
        )
        result = results[0]

        sv_dets = sv.Detections.from_ultralytics(result)
        person_mask = sv_dets.class_id == config.person_class_id
        sv_dets = sv_dets[person_mask]

        if len(sv_dets) == 0:
            dets_np = np.empty((0, 6))
        else:
            dets_np = np.column_stack(
                [sv_dets.xyxy, sv_dets.confidence, sv_dets.class_id]
            ).astype(np.float32)

        try:
            tracker.deep_tracker.update(dets_np, bgr_frame)
        except np.linalg.LinAlgError:
            continue

        # --- Snapshot active tracks ---
        try:
            active = tracker.deep_tracker.active_tracks

            if not first_update_done:
                first_update_done = True
                print("=== After first update() ===")
                print(f"  active_tracks length: {len(active)}")
                if active:
                    trk0 = active[0]
                    print(f"  First track attrs: id={trk0.id}, "
                          f"time_since_update={trk0.time_since_update}, "
                          f"hits={trk0.hits}, conf={trk0.conf:.3f}")
                    if trk0.emb is not None:
                        print(f"  trk.emb shape={trk0.emb.shape}, dtype={trk0.emb.dtype}")
                    else:
                        print("  trk.emb is None on first track")
                print()

            active_track_counts.append(len(active))

            for trk in active:
                if (
                    trk.time_since_update == 0
                    and trk.emb is not None
                    and trk.hits >= min_hits
                    and trk.conf >= args.conf_threshold
                ):
                    tid = trk.id
                    if tid not in emb_sum:
                        emb_sum[tid] = trk.emb.copy().astype(np.float64)
                    else:
                        emb_sum[tid] += trk.emb.astype(np.float64)
                    emb_count[tid] += 1

                    # Capture representative crop from the first good frame
                    if tid not in track_crops and trk.last_observation is not None:
                        x1, y1, x2, y2 = trk.last_observation[:4]
                        crop = _crop_bbox(bgr_frame, x1, y1, x2, y2)
                        if crop.shape[0] > 0 and crop.shape[1] > 0:
                            track_crops[tid] = crop
                            track_bbox[tid] = (x1, y1, x2, y2)

        except AttributeError as e:
            print(f"ERROR: Could not access active_tracks or trk.emb: {e}")
            print("boxmot attribute path is broken — check installed version")
            sys.exit(1)

    # --- Summary ---
    print(f"\n=== Embedding Gallery Summary ===")
    print(f"Tracks with embeddings accumulated: {len(emb_sum)}")
    if not emb_sum:
        print("No embeddings collected — check confidence threshold or video content")
        return

    for tid in sorted(emb_sum):
        print(f"  track {tid:3d}: {emb_count[tid]} frames accumulated")

    print(f"\nActive track count over time: "
          f"min={min(active_track_counts)}, "
          f"max={max(active_track_counts)}, "
          f"mean={np.mean(active_track_counts):.1f}")
    if max(active_track_counts) > 20:
        print("  WARNING: >20 active tracks at peak — consider lowering tracker_max_age_sec")

    # Compute mean embeddings
    track_ids = sorted(emb_sum.keys())
    means = {tid: mean_embedding(emb_sum[tid], emb_count[tid]) for tid in track_ids}

    emb_dim = next(iter(means.values())).shape[0]
    print(f"\nEmbedding dimensionality: {emb_dim}")

    # Pairwise cosine similarity matrix
    n = len(track_ids)
    sim_matrix = np.zeros((n, n))
    for i, ti in enumerate(track_ids):
        for j, tj in enumerate(track_ids):
            sim_matrix[i, j] = cosine_similarity(means[ti], means[tj])

    print(f"\n=== Cosine Similarity Matrix (track IDs: {track_ids}) ===")
    header = "     " + "".join(f"{tid:>7}" for tid in track_ids)
    print(header)
    for i, ti in enumerate(track_ids):
        row = f"{ti:>4} " + "".join(f"{sim_matrix[i, j]:>7.3f}" for j in range(n))
        print(row)

    # Sorted pairwise similarities (off-diagonal only)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sim_matrix[i, j], track_ids[i], track_ids[j]))
    pairs.sort(reverse=True)

    print(f"\n=== Sorted Pairwise Similarities (intra-player gap visible here) ===")
    print(f"{'similarity':>10}  pair")
    for sim, ti, tj in pairs:
        marker = " <<< possible duplicate" if sim > 0.85 else ""
        print(f"  {sim:>8.4f}  ({ti}, {tj}){marker}")

    print(f"\nThreshold calibration guide:")
    if pairs:
        top = pairs[0][0]
        bottom = pairs[-1][0]
        print(f"  Highest similarity: {top:.4f}  (most likely same player if > 0.70)")
        print(f"  Lowest similarity:  {bottom:.4f}  (most likely different players)")
        sims = [s for s, _, _ in pairs]
        if len(sims) >= 2:
            sorted_sims = sorted(sims, reverse=True)
            gaps = [(sorted_sims[k] - sorted_sims[k+1], sorted_sims[k], sorted_sims[k+1])
                    for k in range(len(sorted_sims)-1)]
            biggest_gap = max(gaps, key=lambda x: x[0])
            print(f"  Largest gap: {biggest_gap[1]:.4f} → {biggest_gap[2]:.4f} "
                  f"(gap={biggest_gap[0]:.4f}) — consider threshold between these values")

    # --- Save crop images ---
    if track_crops:
        # All-tracks grid
        tiles = []
        for tid in track_ids:
            crop = track_crops.get(tid, np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8))
            tiles.append(_make_track_tile(crop, tid, emb_count[tid]))
        tile_h = tiles[0].shape[0]
        gap = np.zeros((tile_h, MARGIN, 3), dtype=np.uint8)
        grid = tiles[0]
        for t in tiles[1:]:
            grid = np.hstack([grid, gap, t])
        grid_path = out_dir / "all_tracks.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"\nSaved all-tracks grid → {grid_path}")

        # Duplicate-pairs visual (pairs above 0.80)
        high_pairs = [(s, ti, tj) for s, ti, tj in pairs if s >= 0.80]
        if high_pairs:
            pair_tiles = []
            for sim, ti, tj in high_pairs:
                cropA = track_crops.get(ti, np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8))
                cropB = track_crops.get(tj, np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8))
                pair_tiles.append(_make_pair_tile(
                    cropA, ti, emb_count[ti],
                    cropB, tj, emb_count[tj],
                    sim,
                ))
            # Stack pairs vertically
            max_w = max(t.shape[1] for t in pair_tiles)
            padded = []
            for t in pair_tiles:
                if t.shape[1] < max_w:
                    pad = np.zeros((t.shape[0], max_w - t.shape[1], 3), dtype=np.uint8)
                    t = np.hstack([t, pad])
                padded.append(t)
            pairs_img = np.vstack(padded)
            pairs_path = out_dir / "duplicate_pairs.png"
            cv2.imwrite(str(pairs_path), pairs_img)
            print(f"Saved duplicate-pairs visual → {pairs_path}")
        else:
            print("No pairs above 0.80 similarity — no duplicate-pairs image saved")


if __name__ == "__main__":
    main()
