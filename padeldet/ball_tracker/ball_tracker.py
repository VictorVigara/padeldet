"""Ball detection and tracking using TrackNet.

TrackNet is a U-Net style CNN that takes sequences of frames and outputs
heatmaps predicting ball position. Pre-trained weights are available for
padel/tennis balls.

The pipeline:
  1. Compute median background image from initial frames
  2. Slide an 8-frame window over the video
  3. For each window: resize to 288x512, concat median, run TrackNet
  4. Temporal ensemble: average overlapping predictions for robustness
  5. Extract ball coordinates from heatmaps via contour detection
  6. Post-processing: outlier rejection + temporal isolation filtering
  7. Optionally run InpaintNet to fill gaps in predictions
  8. Savitzky-Golay smoothing for micro-jitter removal
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.signal import savgol_filter
from tqdm import tqdm

from ..base import Tracker
from ..models import FrameDetections, ball_detection
from ..video import VideoInfo
from .models import InpaintNet, TrackNet


# TrackNet operates at this fixed resolution
_HEIGHT = 288
_WIDTH = 512


@dataclass
class BallTrackerConfig:
    """Configuration for TrackNet ball tracking."""

    tracknet_weights: str = ""
    inpaintnet_weights: str = ""  # optional, leave empty to skip inpainting
    batch_size: int = 8
    seq_len: int = 8  # TrackNet input sequence length
    median_sample_frames: int = 300  # frames to sample for median background
    heatmap_threshold: float = 0.5
    # Post-processing
    outlier_mad_factor: float = 6.0  # reject displacements > median + factor * MAD
    min_consecutive: int = 3  # min consecutive detections to keep (temporal isolation)
    smooth_window: int = 5  # Savitzky-Golay window (odd, 0 to disable)
    smooth_polyorder: int = 2  # Savitzky-Golay polynomial order
    debug: bool = False  # show live frame-by-frame debug visualization


def _load_tracknet(weights_path: str, device: str) -> tuple[TrackNet, int, str]:
    """Load TrackNet from checkpoint. Returns (model, seq_len, bg_mode)."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    params = ckpt["param_dict"]
    seq_len = params["seq_len"]
    bg_mode = params["bg_mode"]

    # bg_mode='concat' means input is (seq_len+1)*3 channels (frames + median)
    if bg_mode == "concat":
        in_dim = (seq_len + 1) * 3
    elif bg_mode == "subtract":
        in_dim = seq_len
    elif bg_mode == "subtract_concat":
        in_dim = seq_len * 4
    else:
        in_dim = seq_len * 3

    model = TrackNet(in_dim=in_dim, out_dim=seq_len)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    return model, seq_len, bg_mode


def _load_inpaintnet(weights_path: str, device: str) -> InpaintNet:
    """Load InpaintNet from checkpoint."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = InpaintNet()
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    return model


def _compute_median(frames: list[np.ndarray], max_frames: int = 300) -> np.ndarray:
    """Compute median background image from a subset of frames (RGB)."""
    sample = frames[:max_frames]
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sample]
    return np.median(np.array(rgb_frames), axis=0).astype(np.uint8)


def _preprocess_window(
    window: list[np.ndarray],
    median: np.ndarray,
    bg_mode: str,
) -> np.ndarray:
    """Preprocess a sequence of BGR frames into TrackNet input tensor.

    Returns array of shape (C, H, W) where C depends on bg_mode.
    """
    channels = []

    # Resize median to model resolution
    median_resized = np.array(Image.fromarray(median).resize((_WIDTH, _HEIGHT)))
    median_channels = np.moveaxis(median_resized, -1, 0)  # (3, H, W)

    for frame_bgr in window:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = np.array(Image.fromarray(rgb).resize((_WIDTH, _HEIGHT)))

        if bg_mode == "concat":
            # RGB channels per frame
            channels.append(np.moveaxis(img, -1, 0))  # (3, H, W)
        elif bg_mode == "subtract":
            diff = np.sum(np.abs(img.astype(float) - median_resized.astype(float)), axis=2).astype(np.uint8)
            channels.append(diff[np.newaxis])  # (1, H, W)
        elif bg_mode == "subtract_concat":
            diff = np.sum(np.abs(img.astype(float) - median_resized.astype(float)), axis=2).astype(np.uint8)
            channels.append(np.moveaxis(img, -1, 0))  # (3, H, W)
            channels.append(diff[np.newaxis])  # (1, H, W)
        else:
            channels.append(np.moveaxis(img, -1, 0))  # (3, H, W)

    if bg_mode == "concat":
        # Prepend median
        channels.insert(0, median_channels)

    return np.concatenate(channels, axis=0).astype(np.float32) / 255.0


def _heatmap_to_xy(heatmap: np.ndarray) -> tuple[float, float] | None:
    """Extract ball coordinates from a single heatmap.

    Returns (x, y) in heatmap space or None if no detection.
    """
    if np.amax(heatmap) == 0:
        return None

    binary = (heatmap * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find largest contour
    rects = [cv2.boundingRect(c) for c in contours]
    best = max(rects, key=lambda r: r[2] * r[3])
    x, y, w, h = best
    cx = x + w / 2
    cy = y + h / 2
    return (cx, cy)


def _generate_inpaint_mask(
    xs: list[float], ys: list[float], vis: list[int], th_h: float = 30
) -> list[int]:
    """Generate inpaint mask indicating where to fill missing predictions."""
    y_arr = np.array(ys)
    vis_arr = np.array(vis)
    mask = np.zeros_like(y_arr, dtype=int)

    i = 0
    while i < len(vis_arr):
        # Find start of invisible stretch
        while i < len(vis_arr) - 1 and vis_arr[i] == 1:
            i += 1
        j = i
        # Find end of invisible stretch
        while j < len(vis_arr) - 1 and vis_arr[j] == 0:
            j += 1
        if j == i:
            break
        elif i == 0 and y_arr[j] > th_h:
            mask[:j] = 1
        elif (i > 1 and y_arr[i - 1] > th_h) and (j < len(vis_arr) and y_arr[j] > th_h):
            mask[i:j] = 1
        i = j

    return mask.tolist()


def _reject_outliers(
    xs: list[float],
    ys: list[float],
    vis: list[int],
    mad_factor: float = 6.0,
) -> tuple[list[float], list[float], list[int]]:
    """Reject detections with physically impossible displacements.

    Uses median + mad_factor * MAD of per-frame displacements as threshold.
    Accounts for gap length: if a ball was invisible for N frames, the allowed
    displacement is N * threshold.
    """
    xs, ys, vis = list(xs), list(ys), list(vis)
    n = len(xs)
    if n < 2:
        return xs, ys, vis

    # Compute per-frame displacements between consecutive visible frames
    displacements: list[float] = []
    prev_idx: int | None = None
    for i in range(n):
        if vis[i] == 1:
            if prev_idx is not None:
                gap = i - prev_idx
                dx = xs[i] - xs[prev_idx]
                dy = ys[i] - ys[prev_idx]
                dist_per_frame = math.sqrt(dx * dx + dy * dy) / gap
                displacements.append(dist_per_frame)
            prev_idx = i

    if len(displacements) < 5:
        return xs, ys, vis

    med = float(np.median(displacements))
    mad = float(np.median(np.abs(np.array(displacements) - med)))
    # Avoid zero MAD (all displacements identical) — use median as floor
    mad = max(mad, med * 0.1, 1.0)
    threshold = med + mad_factor * mad

    # Second pass: reject outliers
    rejected = 0
    prev_idx = None
    for i in range(n):
        if vis[i] == 1:
            if prev_idx is not None:
                gap = i - prev_idx
                dx = xs[i] - xs[prev_idx]
                dy = ys[i] - ys[prev_idx]
                dist_per_frame = math.sqrt(dx * dx + dy * dy) / gap
                if dist_per_frame > threshold:
                    # Mark current as outlier (keep previous, it was validated)
                    vis[i] = 0
                    xs[i] = 0.0
                    ys[i] = 0.0
                    rejected += 1
                    continue
            prev_idx = i

    if rejected > 0:
        print(f"Outlier rejection: removed {rejected} detections "
              f"(threshold: {threshold:.1f} px/frame, median: {med:.1f}, MAD: {mad:.1f})")

    return xs, ys, vis


def _reject_isolated(
    xs: list[float],
    ys: list[float],
    vis: list[int],
    min_consecutive: int = 3,
) -> tuple[list[float], list[float], list[int]]:
    """Reject isolated detections shorter than min_consecutive frames.

    A single bright pixel or glass reflection that triggers a detection for
    1-2 frames is almost certainly noise.
    """
    xs, ys, vis = list(xs), list(ys), list(vis)
    n = len(xs)
    rejected = 0

    i = 0
    while i < n:
        if vis[i] == 0:
            i += 1
            continue
        # Find end of consecutive visible stretch
        j = i
        while j < n and vis[j] == 1:
            j += 1
        length = j - i
        if length < min_consecutive:
            for k in range(i, j):
                vis[k] = 0
                xs[k] = 0.0
                ys[k] = 0.0
                rejected += 1
        i = j

    if rejected > 0:
        print(f"Temporal isolation: removed {rejected} detections "
              f"(min consecutive: {min_consecutive})")

    return xs, ys, vis


def _smooth_trajectory(
    xs: list[float],
    ys: list[float],
    vis: list[int],
    window: int = 5,
    polyorder: int = 2,
) -> tuple[list[float], list[float], list[int]]:
    """Apply Savitzky-Golay smoothing to continuous visible segments."""
    if window < 3 or window % 2 == 0:
        return xs, ys, vis

    xs, ys = list(xs), list(ys)
    n = len(xs)

    # Find continuous visible segments and smooth each independently
    i = 0
    while i < n:
        if vis[i] == 0:
            i += 1
            continue
        j = i
        while j < n and vis[j] == 1:
            j += 1
        seg_len = j - i
        if seg_len >= window:
            seg_x = np.array(xs[i:j])
            seg_y = np.array(ys[i:j])
            xs[i:j] = savgol_filter(seg_x, window, polyorder).tolist()
            ys[i:j] = savgol_filter(seg_y, window, polyorder).tolist()
        i = j

    return xs, ys, vis


def _debug_show(
    frames: list[np.ndarray],
    xs: list[float],
    ys: list[float],
    vis: list[int],
    stage_name: str,
    prev_xs: list[float] | None = None,
    prev_ys: list[float] | None = None,
    prev_vis: list[int] | None = None,
) -> None:
    """Interactive frame-by-frame debug viewer.

    Shows ball position for the current stage (yellow) and optionally the
    previous stage (red, dimmer) so you can see what changed.
    Keys: any=next, b=back, q=quit stage, space=play/pause.
    """
    idx = 0
    playing = False
    n = min(len(frames), len(xs))
    print(f"\n[DEBUG] {stage_name} — {n} frames (any=next, b=back, q=skip, space=play)")

    while 0 <= idx < n:
        frame = frames[idx].copy()

        # Previous stage position (red, smaller)
        if prev_xs is not None and prev_vis is not None and idx < len(prev_vis):
            if prev_vis[idx] == 1:
                px, py = int(prev_xs[idx]), int(prev_ys[idx])
                cv2.circle(frame, (px, py), 6, (0, 0, 200), 2, cv2.LINE_AA)

        # Current stage position (yellow)
        has_ball = idx < len(vis) and vis[idx] == 1
        if has_ball:
            cx, cy = int(xs[idx]), int(ys[idx])
            cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1, cv2.LINE_AA)
            status = f"BALL ({cx}, {cy})"
        else:
            status = "NO BALL"

        # Displacement info
        disp_text = ""
        if has_ball:
            for j in range(idx - 1, -1, -1):
                if vis[j] == 1:
                    gap = idx - j
                    dx = xs[idx] - xs[j]
                    dy = ys[idx] - ys[j]
                    dist = math.sqrt(dx * dx + dy * dy)
                    disp_text = f"disp: {dist:.0f}px (gap={gap}f, {dist / gap:.0f}px/f)"
                    break

        # Changed marker
        changed = ""
        if prev_vis is not None and idx < len(prev_vis):
            was_ball = prev_vis[idx] == 1
            if was_ball and not has_ball:
                changed = " [REMOVED]"
            elif not was_ball and has_ball:
                changed = " [ADDED]"
            elif was_ball and has_ball:
                dx = xs[idx] - prev_xs[idx]
                dy = ys[idx] - prev_ys[idx]
                if math.sqrt(dx * dx + dy * dy) > 1.0:
                    changed = " [MOVED]"

        cv2.putText(frame, f"[{stage_name}] Frame {idx} | {status}{changed}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if disp_text:
            cv2.putText(frame, disp_text,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("Ball Debug", frame)
        key = cv2.waitKey(30 if playing else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            idx = max(0, idx - 1)
        elif key == ord(' '):
            playing = not playing
        else:
            idx += 1

    cv2.destroyAllWindows()


class BallTracker(Tracker):
    """
    Detects the ball using TrackNet (heatmap-based CNN) with temporal
    ensemble and optional InpaintNet gap-filling.
    """

    def __init__(
        self,
        config: BallTrackerConfig | None = None,
        video_info: VideoInfo | None = None,
    ):
        super().__init__(video_info=video_info)
        self.config = config or BallTrackerConfig()

        # Load models
        self.tracknet, self._seq_len, self._bg_mode = _load_tracknet(
            self.config.tracknet_weights, self.device
        )
        self.inpaintnet = None
        if self.config.inpaintnet_weights:
            self.inpaintnet = _load_inpaintnet(
                self.config.inpaintnet_weights, self.device
            )

    @property
    def name(self) -> str:
        return "ball_tracker"

    def reset(self) -> None:
        self._results = []

    def to(self, device: str) -> None:
        self.tracknet.to(device)
        if self.inpaintnet is not None:
            self.inpaintnet.to(device)

    def detect_and_track(
        self,
        frames: Iterator[np.ndarray],
        start_frame: int = 0,
    ) -> list[FrameDetections]:
        """Run TrackNet ball detection on a sequence of frames."""
        frames_list = list(frames)
        total = len(frames_list)
        if total == 0:
            return []

        w_scale = self.video_info.width / _WIDTH if self.video_info else 1.0
        h_scale = self.video_info.height / _HEIGHT if self.video_info else 1.0

        # Step 1: Compute median background
        print("Computing median background...")
        median = _compute_median(frames_list, self.config.median_sample_frames)

        # Step 2: Run TrackNet with sliding window + temporal ensemble
        seq_len = self._seq_len
        num_samples = max(1, total - seq_len + 1)

        # Ensemble weight: positional (center frames weighted higher)
        weight = torch.ones(seq_len)
        for i in range(math.ceil(seq_len / 2)):
            weight[i] = i + 1
            weight[seq_len - i - 1] = i + 1
        weight = weight / weight.sum()

        # Prediction buffer for temporal ensemble
        buffer_size = seq_len - 1
        pred_buffer = torch.zeros(
            (buffer_size, seq_len, _HEIGHT, _WIDTH), dtype=torch.float32
        )

        # Collect all predictions
        xs: list[float] = []
        ys: list[float] = []
        vis: list[int] = []

        # Build all windows
        windows = []
        for i in range(num_samples):
            end = min(i + seq_len, total)
            window = frames_list[i:end]
            # Pad if last window is short
            while len(window) < seq_len:
                window.append(window[-1])
            windows.append(window)

        # Process in batches (matching reference temporal ensemble logic)
        sample_count = 0
        sample_indices = torch.arange(seq_len)
        frame_indices = torch.arange(seq_len - 1, -1, -1)

        for batch_start in tqdm(range(0, len(windows), self.config.batch_size), desc="TrackNet"):
            batch_windows = windows[batch_start:batch_start + self.config.batch_size]
            batch_tensors = []
            for w in batch_windows:
                tensor = _preprocess_window(w, median, self._bg_mode)
                batch_tensors.append(tensor)

            batch = torch.from_numpy(np.stack(batch_tensors)).to(self.device)

            with torch.no_grad():
                y_pred = self.tracknet(batch).detach().cpu()

            # Append new predictions to buffer
            pred_buffer = torch.cat((pred_buffer, y_pred), dim=0)

            for si in range(len(batch_windows)):
                # si indexes within the buffer relative to start of this batch's preds
                # Buffer layout: [old buffer_size entries] [new batch entries]
                # So si=0 corresponds to index si in the newly appended region
                if sample_count < buffer_size:
                    ensembled = pred_buffer[
                        sample_indices + si, frame_indices
                    ].sum(0) / (sample_count + 1)
                else:
                    ensembled = (
                        pred_buffer[
                            sample_indices + si, frame_indices
                        ]
                        * weight[:, None, None]
                    ).sum(0)

                # Threshold and extract coordinates
                heatmap = (ensembled > self.config.heatmap_threshold).float().numpy()
                xy = _heatmap_to_xy(heatmap)

                if xy is not None:
                    cx = int(xy[0] * w_scale)
                    cy = int(xy[1] * h_scale)
                    xs.append(float(cx))
                    ys.append(float(cy))
                    vis.append(1)
                else:
                    xs.append(0.0)
                    ys.append(0.0)
                    vis.append(0)

                sample_count += 1

                # Handle tail frames after last sample
                if sample_count == num_samples:
                    zero_pad = torch.zeros(
                        (buffer_size, seq_len, _HEIGHT, _WIDTH), dtype=torch.float32
                    )
                    pred_buffer = torch.cat((pred_buffer, zero_pad), dim=0)
                    for fi in range(1, seq_len):
                        if len(xs) >= total:
                            break
                        tail = pred_buffer[
                            sample_indices + si + fi, frame_indices
                        ].sum(0) / (seq_len - fi)
                        heatmap = (tail > self.config.heatmap_threshold).float().numpy()
                        xy = _heatmap_to_xy(heatmap)
                        if xy is not None:
                            cx = int(xy[0] * w_scale)
                            cy = int(xy[1] * h_scale)
                            xs.append(float(cx))
                            ys.append(float(cy))
                            vis.append(1)
                        else:
                            xs.append(0.0)
                            ys.append(0.0)
                            vis.append(0)

            # Trim buffer — keep last buffer_size entries for next batch's ensemble
            pred_buffer = pred_buffer[-buffer_size:]

        # Debug: show raw TrackNet output
        if self.config.debug:
            _debug_show(frames_list, xs, ys, vis, "1-RAW TrackNet")

        # Step 3: Post-processing — outlier rejection + temporal isolation
        if len(xs) > 0:
            raw_xs, raw_ys, raw_vis = list(xs), list(ys), list(vis)
            xs, ys, vis = _reject_outliers(xs, ys, vis, self.config.outlier_mad_factor)
            xs, ys, vis = _reject_isolated(xs, ys, vis, self.config.min_consecutive)

            if self.config.debug:
                _debug_show(frames_list, xs, ys, vis, "2-After Outlier+Isolation",
                            raw_xs, raw_ys, raw_vis)

        # Step 4: Optional inpainting (fills gaps including rejected outliers)
        if self.inpaintnet is not None and len(xs) > 0:
            pre_inpaint_xs, pre_inpaint_ys, pre_inpaint_vis = list(xs), list(ys), list(vis)
            xs, ys, vis = self._run_inpainting(xs, ys, vis, w_scale, h_scale)

            if self.config.debug:
                _debug_show(frames_list, xs, ys, vis, "3-After InpaintNet",
                            pre_inpaint_xs, pre_inpaint_ys, pre_inpaint_vis)

        # Step 5: Savitzky-Golay smoothing
        if self.config.smooth_window > 0 and len(xs) > 0:
            pre_smooth_xs, pre_smooth_ys, pre_smooth_vis = list(xs), list(ys), list(vis)
            xs, ys, vis = _smooth_trajectory(
                xs, ys, vis, self.config.smooth_window, self.config.smooth_polyorder
            )

            if self.config.debug:
                _debug_show(frames_list, xs, ys, vis, "4-After Smoothing",
                            pre_smooth_xs, pre_smooth_ys, pre_smooth_vis)

        # Step 6: Build FrameDetections
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

        # Normalize coordinates to [0, 1] in model space
        norm_xs = [x / (self.video_info.width if self.video_info else 1920) for x in xs]
        norm_ys = [y / (self.video_info.height if self.video_info else 1080) for y in ys]

        ckpt = torch.load(self.config.inpaintnet_weights, map_location="cpu", weights_only=False)
        inpaint_seq_len = ckpt["param_dict"]["seq_len"]

        # Build sliding window sequences
        n = len(xs)
        refined_xs = list(xs)
        refined_ys = list(ys)
        refined_vis = list(vis)

        for start in range(0, n - inpaint_seq_len + 1):
            end = start + inpaint_seq_len
            seg_mask = inpaint_mask[start:end]

            # Only run inpainting if there are gaps to fill
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

            # Apply inpainted values only where mask is 1
            for j in range(inpaint_seq_len):
                idx = start + j
                if seg_mask[j] == 1:
                    rx = out[j, 0].item() * (self.video_info.width if self.video_info else 1920)
                    ry = out[j, 1].item() * (self.video_info.height if self.video_info else 1080)
                    if rx > 0 or ry > 0:
                        refined_xs[idx] = rx
                        refined_ys[idx] = ry
                        refined_vis[idx] = 1

        return refined_xs, refined_ys, refined_vis
