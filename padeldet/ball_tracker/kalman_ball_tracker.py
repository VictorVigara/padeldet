"""Ball tracking with Kalman filter candidate selection.

Same TrackNet inference as BallTracker, but instead of picking the largest
heatmap contour, extracts top-K candidates and uses a Kalman filter to
select the one most consistent with predicted ball trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..base import Tracker
from ..models import FrameDetections, ball_detection
from ..video import VideoInfo
from .ball_tracker import (
    _compute_median,
    _debug_show,
    _debug_show_heatmaps,
    _generate_inpaint_mask,
    _load_inpaintnet,
    _load_tracknet,
    _preprocess_window,
    _reject_isolated,
    _smooth_trajectory,
    _HEIGHT,
    _WIDTH,
)
from .kalman import BallKalmanFilter
from .models import InpaintNet


@dataclass
class KalmanBallTrackerConfig:
    """Configuration for Kalman-based ball tracking."""

    # Model weights
    tracknet_weights: str = ""
    inpaintnet_weights: str = ""
    batch_size: int = 8
    seq_len: int = 8
    median_sample_frames: int = 300
    heatmap_threshold: float = 0.5

    # Kalman + candidate selection
    max_candidates: int = 5
    kalman_q_pos: float = 1.0
    kalman_q_vel: float = 5.0
    kalman_r_obs: float = 5.0
    kalman_gate: float = 50.0  # Mahalanobis gating threshold (heatmap px)
    max_consecutive_misses: int = 15

    # Post-processing (no MAD rejection — Kalman replaces it)
    min_consecutive: int = 3
    smooth_window: int = 5
    smooth_polyorder: int = 2

    debug: bool = False
    debug_heatmaps: bool = False  # show raw heatmap + binary per frame during Kalman selection


def _heatmap_to_candidates(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    max_k: int = 5,
) -> list[tuple[float, float, float]]:
    """Extract top-K ball candidates from a heatmap.

    Returns list of (cx, cy, score) in heatmap space, sorted by score descending.
    Score = contour_bbox_area × peak_intensity.
    """
    if np.amax(heatmap) == 0:
        return []

    binary = (heatmap * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Ensure heatmap is float for peak lookup (before thresholding)
    heatmap_f = heatmap.astype(np.float32) if heatmap.dtype != np.float32 else heatmap

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area == 0:
            continue
        # Peak intensity within the contour's bounding rect
        roi = heatmap_f[y : y + h, x : x + w]
        peak = float(np.amax(roi)) if roi.size > 0 else 0.0
        cx = x + w / 2.0
        cy = y + h / 2.0
        score = area * peak
        candidates.append((cx, cy, score))

    candidates.sort(key=lambda c: c[2], reverse=True)
    return candidates[:max_k]


def save_heatmaps(
    heatmaps: list[np.ndarray],
    path: str | Path,
    video_info: VideoInfo | None = None,
) -> None:
    """Save per-frame TrackNet heatmaps to a .npy file (uint8, memory-mappable).

    Heatmaps are quantised float32 → uint8 (×255) to reduce disk/RAM usage 4×.
    Metadata is saved alongside as a separate .meta.npz file.
    """
    path = Path(path).with_suffix(".npy")  # always .npy for mmap support
    arr = (np.stack(heatmaps, axis=0).clip(0, 1) * 255).astype(np.uint8)
    np.save(str(path), arr)
    meta_kwargs: dict = {}
    if video_info is not None:
        meta_kwargs["video_width"] = np.array(video_info.width)
        meta_kwargs["video_height"] = np.array(video_info.height)
        meta_kwargs["video_fps"] = np.array(video_info.fps)
    if meta_kwargs:
        np.savez(str(path.with_suffix(".meta.npz")), **meta_kwargs)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Heatmaps saved: {arr.shape} uint8 → {path}  ({size_mb:.1f} MB)")


def load_heatmaps(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load heatmaps saved by save_heatmaps().

    Returns (heatmaps, metadata).
    heatmaps is a memory-mapped (N, H, W) uint8 array — frames are paged in
    on demand, so loading is instant regardless of file size.
    Values are uint8 [0..255]; divide by 255.0 when you need float [0..1].
    metadata may contain video_width, video_height, video_fps.
    """
    path = Path(path).with_suffix(".npy")  # normalise: accept .npz paths too
    arr = np.load(str(path), mmap_mode="r")  # lazy — no full decompression
    meta: dict = {}
    meta_path = path.with_suffix(".meta.npz")
    if meta_path.exists():
        data = np.load(str(meta_path))
        for key in ("video_width", "video_height", "video_fps"):
            if key in data:
                meta[key] = float(data[key])
    print(f"Heatmaps ready: {arr.shape[0]} frames  (mmap, no full load)")
    return arr, meta


class KalmanBallTracker(Tracker):
    """Ball tracker using TrackNet + Kalman filter candidate selection.

    Instead of picking the largest heatmap contour per frame, extracts
    top-K candidates and uses a constant-velocity Kalman filter to select
    the candidate most consistent with the predicted trajectory.
    """

    def __init__(
        self,
        config: KalmanBallTrackerConfig | None = None,
        video_info: VideoInfo | None = None,
    ):
        super().__init__(video_info=video_info)
        self.config = config or KalmanBallTrackerConfig()

        if self.config.tracknet_weights:
            self.tracknet, self._seq_len, self._bg_mode = _load_tracknet(
                self.config.tracknet_weights, self.device
            )
        else:
            self.tracknet = None
            self._seq_len = self.config.seq_len
            self._bg_mode = "concat"

        self.inpaintnet = None
        if self.config.inpaintnet_weights:
            self.inpaintnet = _load_inpaintnet(
                self.config.inpaintnet_weights, self.device
            )
        self.last_heatmaps: list[np.ndarray] | None = None

    @property
    def name(self) -> str:
        return "kalman_ball_tracker"

    def reset(self) -> None:
        self._results = []
        self.last_heatmaps: list[np.ndarray] | None = None

    def to(self, device: str) -> None:
        self.tracknet.to(device)
        if self.inpaintnet is not None:
            self.inpaintnet.to(device)

    def detect_and_track(
        self,
        frames: Iterator[np.ndarray],
        start_frame: int = 0,
        precomputed_heatmaps: np.ndarray | list[np.ndarray] | None = None,
    ) -> list[FrameDetections]:
        """Run TrackNet + Kalman filter ball tracking.

        If precomputed_heatmaps is provided the network inference is skipped
        entirely — useful for iterating on post-processing without re-running
        the expensive network.  Pass the result of save_heatmaps/load_heatmaps.
        """
        frames_list = list(frames)
        total = len(frames_list)
        if total == 0:
            return []

        w_scale = self.video_info.width / _WIDTH if self.video_info else 1.0
        h_scale = self.video_info.height / _HEIGHT if self.video_info else 1.0

        if precomputed_heatmaps is not None:
            print(f"Using {len(precomputed_heatmaps)} precomputed heatmaps — skipping network.")
            heatmaps = precomputed_heatmaps[:total]
        else:
            # Step 1: Compute median background
            print("Computing median background...")
            median = _compute_median(frames_list, self.config.median_sample_frames)

            # Step 2: Run TrackNet with temporal ensemble → per-frame heatmaps
            heatmaps = self._run_tracknet_ensemble(frames_list, median, total)

        self.last_heatmaps = heatmaps  # cache for save_heatmaps() after run

        # Step 3: Kalman-guided candidate selection
        xs, ys, vis = self._kalman_select(heatmaps, frames_list)

        # Scale to video coordinates
        for i in range(len(xs)):
            if vis[i] == 1:
                xs[i] = xs[i] * w_scale
                ys[i] = ys[i] * h_scale

        if self.config.debug:
            _debug_show(frames_list, xs, ys, vis, "1-Kalman Selection")

        # Step 4: Post-processing (no MAD rejection)
        if len(xs) > 0:
            raw_xs, raw_ys, raw_vis = list(xs), list(ys), list(vis)
            xs, ys, vis = _reject_isolated(xs, ys, vis, self.config.min_consecutive)

            if self.config.debug:
                _debug_show(
                    frames_list, xs, ys, vis, "2-After Isolation",
                    raw_xs, raw_ys, raw_vis,
                )

        # Step 5: Optional inpainting
        if self.inpaintnet is not None and len(xs) > 0:
            pre_xs, pre_ys, pre_vis = list(xs), list(ys), list(vis)
            xs, ys, vis = self._run_inpainting(xs, ys, vis, w_scale, h_scale)

            if self.config.debug:
                _debug_show(
                    frames_list, xs, ys, vis, "3-After InpaintNet",
                    pre_xs, pre_ys, pre_vis,
                )

        # Step 6: Savitzky-Golay smoothing
        if self.config.smooth_window > 0 and len(xs) > 0:
            pre_xs, pre_ys, pre_vis = list(xs), list(ys), list(vis)
            xs, ys, vis = _smooth_trajectory(
                xs, ys, vis, self.config.smooth_window, self.config.smooth_polyorder
            )

            if self.config.debug:
                _debug_show(
                    frames_list, xs, ys, vis, "4-After Smoothing",
                    pre_xs, pre_ys, pre_vis,
                )

        # Step 7: Build FrameDetections
        results: list[FrameDetections] = []
        for i in range(total):
            if i < len(xs) and vis[i] == 1:
                det = ball_detection(xs[i], ys[i], confidence=1.0)
            else:
                det = ball_detection(0.0, 0.0, confidence=0.0)
            results.append(
                FrameDetections(frame_index=start_frame + i, detections=[det])
            )

        return results

    def _run_tracknet_ensemble(
        self,
        frames_list: list[np.ndarray],
        median: np.ndarray,
        total: int,
    ) -> list[np.ndarray]:
        """Run TrackNet sliding window + temporal ensemble.

        Returns a list of ensembled heatmaps (one per frame), each shape (H, W).
        Heatmaps are raw float values (not thresholded) for candidate extraction.
        """
        seq_len = self._seq_len
        num_samples = max(1, total - seq_len + 1)

        # Ensemble weights: center frames weighted higher
        weight = torch.ones(seq_len)
        for i in range(math.ceil(seq_len / 2)):
            weight[i] = i + 1
            weight[seq_len - i - 1] = i + 1
        weight = weight / weight.sum()

        buffer_size = seq_len - 1
        pred_buffer = torch.zeros(
            (buffer_size, seq_len, _HEIGHT, _WIDTH), dtype=torch.float32
        )

        # We collect raw ensembled heatmaps (not thresholded)
        heatmaps: list[np.ndarray] = []

        # Build windows
        windows = []
        for i in range(num_samples):
            end = min(i + seq_len, total)
            window = frames_list[i:end]
            while len(window) < seq_len:
                window.append(window[-1])
            windows.append(window)

        sample_count = 0
        sample_indices = torch.arange(seq_len)
        frame_indices = torch.arange(seq_len - 1, -1, -1)

        for batch_start in tqdm(
            range(0, len(windows), self.config.batch_size), desc="TrackNet"
        ):
            batch_windows = windows[batch_start : batch_start + self.config.batch_size]
            batch_tensors = []
            for w in batch_windows:
                tensor = _preprocess_window(w, median, self._bg_mode)
                batch_tensors.append(tensor)

            batch = torch.from_numpy(np.stack(batch_tensors)).to(self.device)

            with torch.no_grad():
                y_pred = self.tracknet(batch).detach().cpu()

            pred_buffer = torch.cat((pred_buffer, y_pred), dim=0)

            for si in range(len(batch_windows)):
                if sample_count < buffer_size:
                    ensembled = pred_buffer[
                        sample_indices + si, frame_indices
                    ].sum(0) / (sample_count + 1)
                else:
                    ensembled = (
                        pred_buffer[sample_indices + si, frame_indices]
                        * weight[:, None, None]
                    ).sum(0)

                heatmaps.append(ensembled.numpy())
                sample_count += 1

                # Tail frames after last sample
                if sample_count == num_samples:
                    zero_pad = torch.zeros(
                        (buffer_size, seq_len, _HEIGHT, _WIDTH), dtype=torch.float32
                    )
                    pred_buffer = torch.cat((pred_buffer, zero_pad), dim=0)
                    for fi in range(1, seq_len):
                        if len(heatmaps) >= total:
                            break
                        tail = pred_buffer[
                            sample_indices + si + fi, frame_indices
                        ].sum(0) / (seq_len - fi)
                        heatmaps.append(tail.numpy())

            pred_buffer = pred_buffer[-buffer_size:]

        return heatmaps

    def _kalman_select(
        self,
        heatmaps: list[np.ndarray],
        frames: list[np.ndarray] | None = None,
    ) -> tuple[list[float], list[float], list[int]]:
        """Select ball position per frame using Kalman-gated top-K candidates.

        Returns (xs, ys, vis) in heatmap space.
        """
        cfg = self.config
        kf = BallKalmanFilter(
            q_pos=cfg.kalman_q_pos,
            q_vel=cfg.kalman_q_vel,
            r_obs=cfg.kalman_r_obs,
        )

        xs: list[float] = []
        ys: list[float] = []
        vis: list[int] = []
        _prev_xy_heatmap: tuple[float, float] | None = None

        for frame_idx, heatmap in enumerate(heatmaps):
            # Normalise: accept both float [0,1] and uint8 [0,255]
            if heatmap.dtype == np.uint8:
                heatmap = heatmap.astype(np.float32) / 255.0
            # Threshold then extract candidates
            thresholded = (heatmap > cfg.heatmap_threshold).astype(np.float32)
            candidates = _heatmap_to_candidates(
                thresholded, threshold=0.5, max_k=cfg.max_candidates
            )

            detected_xy: tuple[float, float] | None = None

            if not candidates:
                # No detection this frame
                if kf.is_initialized:
                    kf.predict()
                    kf.missed_update()
            elif not kf.is_initialized:
                # Bootstrap: pick highest-score candidate
                best = candidates[0]
                kf.initialize(best[0], best[1])
                detected_xy = (best[0], best[1])
            else:
                # Kalman predict + gate candidates
                kf.predict()
                gated = []
                for cx, cy, score in candidates:
                    d = kf.gating_distance(cx, cy)
                    if d < cfg.kalman_gate:
                        gated.append((cx, cy, score, d))

                if gated:
                    best = min(gated, key=lambda t: t[3])
                    kf.update(best[0], best[1])
                    detected_xy = (best[0], best[1])
                else:
                    # All candidates outside gate — likely reflections
                    kf.missed_update()
                    if kf.consecutive_misses > cfg.max_consecutive_misses:
                        kf.reset()

            # Heatmap debug visualization
            if cfg.debug_heatmaps and frames is not None and frame_idx < len(frames):
                heatmap_bin_u8 = (thresholded * 255).astype(np.uint8)
                key = _debug_show_heatmaps(
                    frames[frame_idx],
                    heatmap,
                    heatmap_bin_u8,
                    detected_xy,
                    _prev_xy_heatmap,
                    frame_idx,
                )
                if key == 'q':
                    import cv2 as _cv2
                    _cv2.destroyAllWindows()

            if detected_xy is not None:
                _prev_xy_heatmap = detected_xy
                xs.append(detected_xy[0])
                ys.append(detected_xy[1])
                vis.append(1)
            else:
                xs.append(0.0)
                ys.append(0.0)
                vis.append(0)

        detected = sum(vis)
        total = len(vis)
        print(
            f"Kalman selection: {detected}/{total} frames "
            f"({100 * detected / total:.1f}% detection rate)"
        )

        return xs, ys, vis

    def _run_inpainting(
        self,
        xs: list[float],
        ys: list[float],
        vis: list[int],
        w_scale: float,
        h_scale: float,
    ) -> tuple[list[float], list[float], list[int]]:
        """Run InpaintNet to fill gaps in predictions."""
        th_h = self.video_info.height * 0.05 if self.video_info else 15.0
        inpaint_mask = _generate_inpaint_mask(xs, ys, vis, th_h)

        norm_xs = [x / (self.video_info.width if self.video_info else 1920) for x in xs]
        norm_ys = [y / (self.video_info.height if self.video_info else 1080) for y in ys]

        ckpt = torch.load(
            self.config.inpaintnet_weights, map_location="cpu", weights_only=False
        )
        inpaint_seq_len = ckpt["param_dict"]["seq_len"]

        n = len(xs)
        refined_xs = list(xs)
        refined_ys = list(ys)
        refined_vis = list(vis)

        for start in range(0, n - inpaint_seq_len + 1):
            end = start + inpaint_seq_len
            seg_mask = inpaint_mask[start:end]

            if sum(seg_mask) == 0:
                continue

            coor = torch.tensor(
                [[norm_xs[i], norm_ys[i]] for i in range(start, end)],
                dtype=torch.float32,
            ).unsqueeze(0)
            mask = torch.tensor(
                [[seg_mask[i - start]] for i in range(start, end)],
                dtype=torch.float32,
            ).unsqueeze(0)

            with torch.no_grad():
                out = self.inpaintnet(coor, mask).squeeze(0)

            for j in range(inpaint_seq_len):
                idx = start + j
                if seg_mask[j] == 1:
                    rx = out[j, 0].item() * (
                        self.video_info.width if self.video_info else 1920
                    )
                    ry = out[j, 1].item() * (
                        self.video_info.height if self.video_info else 1080
                    )
                    if rx > 0 or ry > 0:
                        refined_xs[idx] = rx
                        refined_ys[idx] = ry
                        refined_vis[idx] = 1

        return refined_xs, refined_ys, refined_vis
