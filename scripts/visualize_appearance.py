"""Offline clothing-color visualization.

Loads saved player detections JSON + source video and renders:
  - Skeleton overlay + highlighted shirt / shorts / shoe regions
  - Per-player color swatch panel (progressive warm-up)
  - 4×4 similarity matrix shown from frame 0 (best match highlighted)
  - Match-accuracy metrics printed at the end

Usage:
  python scripts/visualize_appearance.py \\
    --video data/yt/1/1.mp4 \\
    --detections data/yt/1/player_detections.json \\
    --warmup-sec 10 \\
    --output /tmp/appearance.mp4
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from padeldet.appearance import AppearanceLibrary, extract_player_colors
from padeldet.models import TrackedDetection, load_detections
from padeldet.video import VideoInfo, VideoWriter, frame_generator

_COCO_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

_SHIRT_COLOUR  = (255, 200,   0)
_SHORTS_COLOUR = (  0, 140, 255)
_SHOE_COLOUR   = (180,   0, 255)

_PANEL_WIDTH = 220
_SWATCH_H    = 28
_SWATCH_W    = 50
_PANEL_PAD   = 8
_LABEL_W     = 30

_MATRIX_CELL_W     = 52
_MATRIX_CELL_H     = 38
_MATRIX_ROW_LABEL_W = 34
_MATRIX_COL_LABEL_H = 22
_MATRIX_PAD         = 6


# ---------------------------------------------------------------------------
# Skeleton + region drawing
# ---------------------------------------------------------------------------

def _draw_skeleton(frame: np.ndarray, det, min_conf: float = 0.3) -> None:
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


def _draw_region_outline(frame: np.ndarray, pts: list[tuple[float, float]], colour: tuple) -> None:
    if len(pts) < 2:
        return
    arr = np.array(pts, dtype=np.float32)
    hull = cv2.convexHull(arr)
    cv2.polylines(frame, [hull.astype(np.int32)], True, colour, 2, cv2.LINE_AA)


def _draw_rect_outline(frame, p1, p2, half_width, colour) -> None:
    x1, y1 = p1; x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = max((dx**2 + dy**2) ** 0.5, 1e-6)
    nx, ny = -dy / length, dx / length
    corners = np.array([
        [x1 + nx * half_width, y1 + ny * half_width],
        [x1 - nx * half_width, y1 - ny * half_width],
        [x2 - nx * half_width, y2 - ny * half_width],
        [x2 + nx * half_width, y2 + ny * half_width],
    ], dtype=np.int32)
    cv2.polylines(frame, [corners], True, colour, 2, cv2.LINE_AA)


def _draw_clothing_regions(frame: np.ndarray, det) -> None:
    from padeldet.appearance import (
        _kp, _L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP,
        _L_KNEE, _R_KNEE, _L_ANKLE, _R_ANKLE,
    )
    if det.keypoints is None:
        return
    kps = det.keypoints
    C = 0.3

    pts = [p for p in [
        _kp(kps, _L_SHOULDER, C), _kp(kps, _R_SHOULDER, C),
        _kp(kps, _R_HIP, C),      _kp(kps, _L_HIP, C),
    ] if p is not None]
    if len(pts) >= 3:
        _draw_region_outline(frame, pts, _SHIRT_COLOUR)

    lh = _kp(kps, _L_HIP, C); rh = _kp(kps, _R_HIP, C)
    lk = _kp(kps, _L_KNEE, C); rk = _kp(kps, _R_KNEE, C)
    half_w = max(8.0, abs(rh[0] - lh[0]) * 0.4) if (lh and rh) else 15.0
    def _interp(a, b, t): return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    if lh and lk:
        _draw_rect_outline(frame, lh, _interp(lh, lk, 0.8), half_w, _SHORTS_COLOUR)
    if rh and rk:
        _draw_rect_outline(frame, rh, _interp(rh, rk, 0.8), half_w, _SHORTS_COLOUR)

    for idx in (_L_ANKLE, _R_ANKLE):
        kp = _kp(kps, idx, C)
        if kp:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 12, _SHOE_COLOUR, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Swatch panel
# ---------------------------------------------------------------------------

def _build_swatch_panel(library: AppearanceLibrary, height: int) -> np.ndarray:
    pids = [1, 2, 3, 4]
    col_labels = ["Shirt", "Short", "Shoe"]
    col_x = [_PANEL_PAD + _LABEL_W + i * (_SWATCH_W + 4) for i in range(3)]

    panel_h = max(_PANEL_PAD * 2 + 20 + len(pids) * (_SWATCH_H + 4), height)
    panel = np.full((panel_h, _PANEL_WIDTH, 3), (30, 30, 30), dtype=np.uint8)

    cv2.putText(panel, "Player Colours", (_PANEL_PAD, _PANEL_PAD + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    for i, label in enumerate(col_labels):
        cv2.putText(panel, label, (col_x[i], _PANEL_PAD + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

    for row, pid in enumerate(pids):
        y = _PANEL_PAD + 40 + row * (_SWATCH_H + 4)
        cv2.putText(panel, f"P{pid}", (_PANEL_PAD, y + _SWATCH_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        profile = library.mean_colors(pid)
        means = [
            profile.shirt_mean  if profile else None,
            profile.shorts_mean if profile else None,
            profile.shoe_mean   if profile else None,
        ]
        for col_i, mean_bgr in enumerate(means):
            x = col_x[col_i]
            if mean_bgr is not None:
                colour = (int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2]))
                cv2.rectangle(panel, (x, y), (x + _SWATCH_W - 2, y + _SWATCH_H - 2), colour, -1)
                cv2.rectangle(panel, (x, y), (x + _SWATCH_W - 2, y + _SWATCH_H - 2), (200, 200, 200), 1)
            else:
                cv2.rectangle(panel, (x, y), (x + _SWATCH_W - 2, y + _SWATCH_H - 2), (60, 60, 60), 1)
                cv2.putText(panel, "?", (x + 18, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    return panel


# ---------------------------------------------------------------------------
# Similarity matrix
# ---------------------------------------------------------------------------

def _score_to_bgr(score: float) -> tuple[int, int, int]:
    """Map [0, 1] → BGR colour: red → yellow → green."""
    s = float(np.clip(score, 0.0, 1.0))
    if s < 0.5:
        return (0, int(s * 2 * 200), 200)
    return (0, 200, int((1.0 - s) * 2 * 200))


def _draw_similarity_matrix(
    frame: np.ndarray,
    library: AppearanceLibrary,
    det_by_pid: dict[int, TrackedDetection],
    source_frame: np.ndarray,
    x0: int,
    y0: int,
    warmup_done: bool,
) -> None:
    pids = [1, 2, 3, 4]
    total_w = _MATRIX_ROW_LABEL_W + len(pids) * _MATRIX_CELL_W + _MATRIX_PAD * 2
    total_h = _MATRIX_COL_LABEL_H + len(pids) * _MATRIX_CELL_H + _MATRIX_PAD * 2

    cv2.rectangle(frame, (x0, y0), (x0 + total_w, y0 + total_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + total_w, y0 + total_h), (80, 80, 80), 1)

    label = "Similarity (live)" if warmup_done else "Similarity (building...)"
    cv2.putText(frame, label, (x0 + _MATRIX_PAD, y0 + _MATRIX_PAD + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)

    for col, pid in enumerate(pids):
        cx = x0 + _MATRIX_PAD + _MATRIX_ROW_LABEL_W + col * _MATRIX_CELL_W + _MATRIX_CELL_W // 2 - 8
        cy = y0 + _MATRIX_PAD + _MATRIX_COL_LABEL_H - 4
        cv2.putText(frame, f"L{pid}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

    for row, row_pid in enumerate(pids):
        ry = y0 + _MATRIX_PAD + _MATRIX_COL_LABEL_H + row * _MATRIX_CELL_H
        cv2.putText(frame, f"D{row_pid}",
                    (x0 + _MATRIX_PAD, ry + _MATRIX_CELL_H // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

        det = det_by_pid.get(row_pid)
        if det is not None and det.keypoints is not None:
            colors = extract_player_colors(source_frame, det)
            sims = library.all_similarities(colors) if colors is not None else {}
        else:
            sims = {}

        # Find best column for this row
        best_col_pid = max(sims, key=sims.get) if sims else None

        for col, col_pid in enumerate(pids):
            cx = x0 + _MATRIX_PAD + _MATRIX_ROW_LABEL_W + col * _MATRIX_CELL_W
            cy = ry
            score = sims.get(col_pid)

            if score is not None:
                bgr = _score_to_bgr(score)
                cv2.rectangle(frame, (cx, cy),
                              (cx + _MATRIX_CELL_W - 2, cy + _MATRIX_CELL_H - 2), bgr, -1)
                text_col = (0, 0, 0) if score > 0.55 else (220, 220, 220)
                cv2.putText(frame, f"{score:.2f}",
                            (cx + 4, cy + _MATRIX_CELL_H // 2 + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_col, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (cx, cy),
                              (cx + _MATRIX_CELL_W - 2, cy + _MATRIX_CELL_H - 2), (50, 50, 50), -1)
                cv2.putText(frame, "-",
                            (cx + _MATRIX_CELL_W // 2 - 4, cy + _MATRIX_CELL_H // 2 + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1, cv2.LINE_AA)

            # White border on diagonal (expected match)
            if row_pid == col_pid:
                cv2.rectangle(frame, (cx, cy),
                              (cx + _MATRIX_CELL_W - 2, cy + _MATRIX_CELL_H - 2),
                              (255, 255, 255), 1)
            # Cyan thick border on best match per row
            if col_pid == best_col_pid:
                cv2.rectangle(frame, (cx + 1, cy + 1),
                              (cx + _MATRIX_CELL_W - 3, cy + _MATRIX_CELL_H - 3),
                              (255, 220, 0), 2)


# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------

def _annotate_frame(
    frame: np.ndarray,
    fd,
    library: AppearanceLibrary,
    warmup_done: bool,
) -> np.ndarray:
    annotated = frame.copy()
    for det in fd.detections:
        _draw_skeleton(annotated, det)
        _draw_clothing_regions(annotated, det)

    # Swatch panel overlaid on the right side
    panel = _build_swatch_panel(library, annotated.shape[0])
    panel_h = min(panel.shape[0], annotated.shape[0])
    annotated[:panel_h, -_PANEL_WIDTH:] = panel[:panel_h]

    # Similarity matrix — always shown, top-left
    det_by_pid = {
        det.logical_player_id: det
        for det in fd.detections
        if det.logical_player_id is not None
    }
    _draw_similarity_matrix(annotated, library, det_by_pid, frame, 10, 10, warmup_done)

    if not warmup_done:
        mat_h = _MATRIX_COL_LABEL_H + 4 * _MATRIX_CELL_H + _MATRIX_PAD * 2
        cv2.putText(annotated, "Warming up...", (10, 10 + mat_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    cv2.putText(annotated, f"frame {fd.frame_index}", (10, annotated.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    return annotated


# ---------------------------------------------------------------------------
# Match-accuracy metrics
# ---------------------------------------------------------------------------

def _print_metrics(match_counts: dict[int, dict[int, int]]) -> None:
    pids = [1, 2, 3, 4]
    print("\n── Match-accuracy (post warmup) ─────────────────────────")
    header = f"{'':6s}" + "".join(f"  L{p:1d}  " for p in pids) + "  | Best match"
    print(header)
    print("─" * len(header))
    for det_pid in pids:
        counts = match_counts.get(det_pid, {})
        total = sum(counts.values())
        if total == 0:
            print(f"  D{det_pid}  " + "   —  " * 4)
            continue
        best_pid = max(counts, key=counts.get)
        best_pct = 100 * counts.get(best_pid, 0) / total
        row = f"  D{det_pid}  "
        for lib_pid in pids:
            c = counts.get(lib_pid, 0)
            row += f"  {c:3d}  " if lib_pid != best_pid else f" [{c:3d}] "
        row += f"  | L{best_pid} ({best_pct:.0f}%)"
        print(row)
    print("─" * len(header))
    correct = sum(
        match_counts.get(p, {}).get(p, 0) for p in pids
    )
    total_all = sum(
        sum(match_counts.get(p, {}).values()) for p in pids
    )
    if total_all > 0:
        print(f"  Overall diagonal accuracy: {correct}/{total_all} = {100*correct/total_all:.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise player clothing colours from pose keypoints")
    parser.add_argument("--video", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    detections_path = Path(args.detections)

    print(f"Loading detections from {detections_path} ...")
    all_frames = load_detections(detections_path)
    det_by_idx = {fd.frame_index: fd for fd in all_frames}

    video_info = VideoInfo.from_path(video_path)
    warmup_frames = int(args.warmup_sec * video_info.fps)
    print(f"Video: {video_info.width}x{video_info.height} @ {video_info.fps:.1f} fps  |  "
          f"warm-up: {warmup_frames} frames ({args.warmup_sec:.0f} s)")

    library = AppearanceLibrary()
    # match_counts[detected_pid][best_lib_pid] → frame count
    match_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    output_path = Path(args.output) if args.output else None
    writer: VideoWriter | None = None
    vis_info = VideoInfo(
        width=video_info.width,
        height=video_info.height,
        fps=video_info.fps,
        total_frames=video_info.total_frames,
    )

    frame_start = args.frame_start
    frame_end = args.frame_end

    print("Processing...")
    for abs_idx, frame in enumerate(
        frame_generator(video_path, start=frame_start, end=frame_end)
    ):
        video_frame_idx = frame_start + abs_idx
        fd = det_by_idx.get(video_frame_idx)
        if fd is None:
            continue

        warmup_done = video_frame_idx >= warmup_frames

        # Progressive library update during warmup
        if not warmup_done:
            for det in fd.detections:
                if det.logical_player_id is None or det.keypoints is None:
                    continue
                colors = extract_player_colors(frame, det)
                if colors is not None:
                    library.add(det.logical_player_id, colors)
            library.build()

        # Accumulate match metrics after warmup
        if warmup_done:
            for det in fd.detections:
                if det.logical_player_id is None or det.keypoints is None:
                    continue
                colors = extract_player_colors(frame, det)
                if colors is None:
                    continue
                sims = library.all_similarities(colors)
                if not sims:
                    continue
                best_pid = max(sims, key=sims.get)
                match_counts[det.logical_player_id][best_pid] += 1

        annotated = _annotate_frame(frame, fd, library, warmup_done)

        if output_path:
            if writer is None:
                writer = VideoWriter(output_path, vis_info)
            writer.write(annotated)
        else:
            cv2.imshow("Appearance", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer:
        writer.release()
        print(f"Saved to {output_path}")
    else:
        cv2.destroyAllWindows()

    _print_metrics(match_counts)


if __name__ == "__main__":
    main()
