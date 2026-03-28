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
  6. Optionally run InpaintNet to fill gaps in predictions
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np
import torch
from PIL import Image
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

        # Step 3: Optional inpainting
        if self.inpaintnet is not None and len(xs) > 0:
            xs, ys, vis = self._run_inpainting(xs, ys, vis, w_scale, h_scale)

        # Step 4: Build FrameDetections
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
