"""Ball detection using OpenCV MOG2 background subtractor.

Produces per-frame binary foreground masks in the same (N, 288, 512) heatmap
format as TrackNet, so all selector strategies and compare_strategies.py work
unchanged. No neural network — runs in real time.

Pipeline per frame:
  1. Resize to 288×512 (same resolution as TrackNet heatmaps)
  2. MOG2 background subtraction → binary foreground mask
  3. Morphological open (remove noise) + close (fill ball blob holes)
  4. Contour area filter (drop sub-pixel noise and large non-ball regions)
  5. Normalise to float32 [0..1] — blobs are 1.0, background is 0.0

The resulting per-frame heatmaps are then passed through any BallSelectorBase
strategy from selectors.py to pick the best candidate, followed by the same
post-processing (temporal isolation + Savitzky-Golay smoothing) used by
KalmanBallTracker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np
from tqdm import tqdm

from ..base import Tracker
from ..models import FrameDetections, ball_detection
from ..video import VideoInfo
from .ball_tracker import _reject_isolated, _smooth_trajectory
from .kalman_ball_tracker import _HEIGHT, _WIDTH, save_heatmaps, load_heatmaps
from .selectors import (
    IntensitySelector,
    TrajectorySelector,
    KalmanSelector,
    OcclusionAwareSelector,
    BallSelectorBase,
)

_SELECTOR_REGISTRY: dict[str, type[BallSelectorBase]] = {
    "intensity": IntensitySelector,
    "trajectory": TrajectorySelector,
    "kalman": KalmanSelector,
    "occlusion": OcclusionAwareSelector,
}


# ── Module-level debug helper ─────────────────────────────────────────────────

def _debug_show_mog2(
    frame: np.ndarray,
    stages: list[tuple[str, np.ndarray]],
    xy_heatmap: tuple[float, float] | None,
    frame_idx: int,
    delay: int = 0,
) -> str:
    """Show original frame + one window per pipeline stage.

    Windows: "frame", then one per entry in stages (label → uint8 mask).
    Controls: any key = next frame, space = toggle play, q = quit
    """
    _W = 512  # display width for each window

    hm_to_orig_x = frame.shape[1] / _WIDTH
    hm_to_orig_y = frame.shape[0] / _HEIGHT
    hm_h = int(_W * _HEIGHT / _WIDTH)

    # Original frame
    orig_h = int(_W * frame.shape[0] / frame.shape[1])
    img_orig = cv2.resize(frame, (_W, orig_h))
    if xy_heatmap is not None:
        sx = _W / frame.shape[1]
        sy = orig_h / frame.shape[0]
        cx = int(xy_heatmap[0] * hm_to_orig_x * sx)
        cy = int(xy_heatmap[1] * hm_to_orig_y * sy)
        cv2.circle(img_orig, (cx, cy), 8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img_orig, (cx, cy), 3, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.imshow("frame", img_orig)

    # One window per stage
    for label, mask in stages:
        bgr = mask if mask.ndim == 3 else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(bgr, (_W, hm_h), interpolation=cv2.INTER_NEAREST)
        status = f"{label}  f={frame_idx}"
        if xy_heatmap is not None and label == stages[-1][0]:
            status += f"  BALL ({xy_heatmap[0]:.0f},{xy_heatmap[1]:.0f})"
            cv2.circle(img, (int(xy_heatmap[0]), int(xy_heatmap[1])),
                       6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, status, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.imshow(label, img)

    key = cv2.waitKey(delay) & 0xFF
    return chr(key) if key < 128 else ""


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Mog2BallTrackerConfig:
    """Configuration for MOG2-based ball detector."""

    # MOG2 background model parameters
    history: int = 10               # frames used to build background model
    var_threshold: float = 100.0      # Mahalanobis distance threshold for foreground
    detect_shadows: bool = False     # shadows add noise and slow things down
    learning_rate: float = -1.0      # -1 = automatic

    # Morphological cleanup (in 288×512 heatmap space)
    morph_open_size: int = 1         # removes isolated noise pixels
    morph_close_size: int = 10        # fills holes inside ball blob

    # Blob pixel filter — counts actual filled pixels inside each contour
    min_blob_pixels: int = 2         # discard sub-pixel noise
    max_blob_pixels: int = 100       # discard large regions (players, shadows)

    # Warm-up: MOG2 needs frames before the background model stabilises
    warmup_frames: int = 10          # first N frames produce empty masks

    # Blob selection strategy (applied on top of MOG2 masks)
    selector: str = "kalman"         # "intensity" | "trajectory" | "kalman" | "occlusion"

    # Post-processing
    min_consecutive: int = 3
    smooth_window: int = 5
    smooth_polyorder: int = 2

    # Resolution: use native video resolution instead of 288×512
    # Larger blobs, easier area filtering, but incompatible with TrackNet heatmap caching
    use_native_resolution: bool = False

    debug: bool = False              # show per-frame debug window during mask generation


# ── Tracker ───────────────────────────────────────────────────────────────────

class Mog2BallTracker(Tracker):
    """Ball tracker using MOG2 background subtraction.

    Generates binary foreground masks (same 288×512 heatmap format as
    TrackNet) then applies a pluggable selector strategy to pick the ball
    candidate per frame.

    Use detect_and_track() to run end-to-end.  Access last_heatmaps after
    a run to save with save_heatmaps() and replay via precomputed_heatmaps.
    """

    @property
    def name(self) -> str:
        return "mog2"

    def __init__(
        self,
        config: Mog2BallTrackerConfig | None = None,
        video_info: VideoInfo | None = None,
    ):
        super().__init__(video_info=video_info)
        self.config = config or Mog2BallTrackerConfig()
        self.last_heatmaps: list[np.ndarray] | None = None
        self._subtractor: cv2.BackgroundSubtractorMOG2 | None = None
        self._selector: BallSelectorBase = self._build_selector()

    def _build_selector(self) -> BallSelectorBase:
        cls = _SELECTOR_REGISTRY.get(self.config.selector.lower())
        if cls is None:
            raise ValueError(
                f"Unknown selector '{self.config.selector}'. "
                f"Choose from: {list(_SELECTOR_REGISTRY)}"
            )
        return cls()

    def reset(self) -> None:
        """Recreate the background subtractor and reset selector state."""
        cfg = self.config
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg.history,
            varThreshold=cfg.var_threshold,
            detectShadows=cfg.detect_shadows,
        )
        self._selector.reset()

    # ── Mask generation ───────────────────────────────────────────────────────

    def _frame_to_mask(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
        """Apply MOG2 + cleanup to one BGR frame.

        Returns:
            (filtered_float, stages) where
            - filtered_float: float32 (288×512) in [0..1], final result
            - stages: list of (label, uint8_mask) for each intermediate step
        """
        cfg = self.config
        small = frame if cfg.use_native_resolution else cv2.resize(frame, (_WIDTH, _HEIGHT))
        mask = self._subtractor.apply(small, learningRate=cfg.learning_rate)

        stages: list[tuple[str, np.ndarray]] = [("raw MOG2", mask.copy())]

        if cfg.morph_open_size > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (cfg.morph_open_size, cfg.morph_open_size)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            stages.append((f"open {cfg.morph_open_size}px", mask.copy()))

        if cfg.morph_close_size > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (cfg.morph_close_size, cfg.morph_close_size)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            stages.append((f"close {cfg.morph_close_size}px", mask.copy()))

        filtered = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blob_buf = np.zeros(mask.shape, dtype=np.uint8)
        if cfg.debug and contours:
            annotated = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        for c in contours:
            blob_buf[:] = 0
            cv2.drawContours(blob_buf, [c], -1, 255, -1)
            px = int(np.count_nonzero(blob_buf))
            area = cv2.contourArea(c)
            kept = cfg.min_blob_pixels <= px <= cfg.max_blob_pixels
            if kept:
                cv2.drawContours(filtered, [c], -1, 255, -1)
            if cfg.debug and contours:
                color = (0, 255, 0) if kept else (0, 0, 255)
                cv2.drawContours(annotated, [c], -1, color, -1)  # fill
                cv2.drawContours(annotated, [c], -1, color, 1)   # outline
                x, y, _, _ = cv2.boundingRect(c)
                for i, txt in enumerate([f"px={px}", f"a={area:.1f}"]):
                    tx, ty = x, max(y - 2, 14) + i * 13
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(annotated, (tx, ty - th - 1), (tx + tw, ty + 1), (0, 0, 0), -1)
                    cv2.putText(annotated, txt, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        if cfg.debug and contours:
            stages.append((f"px [{cfg.min_blob_pixels}-{cfg.max_blob_pixels}] G=keep R=cut", annotated))

        stages.append((f"filtered [{cfg.min_blob_pixels}-{cfg.max_blob_pixels}px]", filtered.copy()))

        return filtered.astype(np.float32) / 255.0, stages

    def _generate_masks(self, frames_list: list[np.ndarray]) -> list[np.ndarray]:
        """Run MOG2 over all frames, return list of (288×512) float masks."""
        self.reset()
        cfg = self.config
        masks: list[np.ndarray] = []
        play_mode = False

        for i, frame in enumerate(tqdm(frames_list, desc="MOG2 masks")):
            if i < cfg.warmup_frames:
                small = frame if cfg.use_native_resolution else cv2.resize(frame, (_WIDTH, _HEIGHT))
                self._subtractor.apply(small, learningRate=cfg.learning_rate)
                empty = np.zeros((_HEIGHT, _WIDTH), dtype=np.float32)
                masks.append(empty)
                if cfg.debug:
                    empty_u8 = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
                    warmup_stages = [("raw MOG2", empty_u8)]
                    delay = 30 if play_mode else 0
                    key = _debug_show_mog2(frame, warmup_stages, None, i, delay)
                    if key == "q":
                        break
                    if key == " ":
                        play_mode = not play_mode
            else:
                filtered, stages = self._frame_to_mask(frame)
                masks.append(filtered)
                if cfg.debug:
                    delay = 30 if play_mode else 0
                    key = _debug_show_mog2(frame, stages, None, i, delay)
                    if key == "q":
                        break
                    if key == " ":
                        play_mode = not play_mode

        return masks

    # ── Main entry point ──────────────────────────────────────────────────────

    def detect_and_track(
        self,
        frames: Iterator[np.ndarray],
        start_frame: int = 0,
        precomputed_heatmaps: np.ndarray | list[np.ndarray] | None = None,
    ) -> list[FrameDetections]:
        """Run MOG2 ball detection and tracking.

        If precomputed_heatmaps is provided (e.g. from load_heatmaps()), mask
        generation is skipped entirely — useful for iterating on selector
        parameters without re-processing the video.
        """
        frames_list = list(frames)
        total = len(frames_list)
        if total == 0:
            return []

        # ── Step 1: get heatmaps (MOG2 masks or precomputed) ──────────────────
        if precomputed_heatmaps is not None:
            print(f"Using {len(precomputed_heatmaps)} precomputed heatmaps — skipping MOG2.")
            heatmaps_raw = precomputed_heatmaps[:total]
        else:
            heatmaps_raw = self._generate_masks(frames_list)

        # Scale from mask space → video space (mask may be 288×512 or native res)
        if heatmaps_raw:
            first = heatmaps_raw[0] if not isinstance(heatmaps_raw, np.ndarray) else heatmaps_raw[0]
            mask_h, mask_w = first.shape[:2]
        else:
            mask_h, mask_w = _HEIGHT, _WIDTH
        w_scale = self.video_info.width / mask_w if self.video_info else 1.0
        h_scale = self.video_info.height / mask_h if self.video_info else 1.0

        # Cache for save_heatmaps() — store as float list
        if isinstance(heatmaps_raw, np.ndarray):
            self.last_heatmaps = [
                heatmaps_raw[i].astype(np.float32) / 255.0
                if heatmaps_raw[i].dtype == np.uint8
                else heatmaps_raw[i].astype(np.float32)
                for i in range(len(heatmaps_raw))
            ]
        else:
            self.last_heatmaps = [
                h.astype(np.float32) / 255.0 if h.dtype == np.uint8 else h.astype(np.float32)
                for h in heatmaps_raw
            ]

        # ── Step 2: select ball candidate per frame ───────────────────────────
        self._selector.reset()
        xs: list[float] = []
        ys: list[float] = []
        vis: list[int] = []

        for heatmap in self.last_heatmaps:
            xy = self._selector.select(heatmap, None, w_scale, h_scale)
            if xy is not None:
                xs.append(xy[0])
                ys.append(xy[1])
                vis.append(1)
            else:
                xs.append(0.0)
                ys.append(0.0)
                vis.append(0)

        # ── Step 3: post-processing ───────────────────────────────────────────
        cfg = self.config
        xs, ys, vis = _reject_isolated(xs, ys, vis, cfg.min_consecutive)
        xs, ys, vis = _smooth_trajectory(xs, ys, vis, cfg.smooth_window, cfg.smooth_polyorder)

        # ── Step 4: build FrameDetections ─────────────────────────────────────
        results: list[FrameDetections] = []
        for i in range(total):
            if vis[i] == 1:
                vid_x = xs[i] * w_scale
                vid_y = ys[i] * h_scale
                det = ball_detection(vid_x, vid_y, confidence=1.0)
            else:
                det = ball_detection(0.0, 0.0, confidence=0.0)
            results.append(
                FrameDetections(
                    frame_index=start_frame + i,
                    detections=[det],
                )
            )

        detected = sum(vis)
        print(f"MOG2: {detected}/{total} frames with ball detected ({100*detected/total:.1f}%)")
        return results
