"""Heatmap blob selection strategies for ball detection.

Each strategy is a stateful class with the same interface:

    selector.select(heatmap, player_bboxes_video, w_scale, h_scale)
    -> (cx, cy) in heatmap space, or None

They operate on the raw TrackNet sigmoid heatmap (float 0-1) before any
thresholding, so intensity information is preserved for scoring.

Usage:
    from padeldet.ball_tracker.selectors import (
        IntensitySelector, TrajectorySelector,
        KalmanSelector, OcclusionAwareSelector,
    )
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from .kalman import BallKalmanFilter

# Heatmap resolution (must match TrackNet constants)
_H, _W = 288, 512


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper
# ──────────────────────────────────────────────────────────────────────────────

def extract_blob_candidates(
    heatmap: np.ndarray,
    threshold: float = 0.25,
    max_k: int = 10,
) -> list[tuple[float, float, float]]:
    """Extract blob candidates from a raw float heatmap.

    Returns list of (cx, cy, peak_intensity) sorted by peak descending.

    Key design choice: scores by peak intensity only (not area × peak).
    Area penalises the ball (small tight blob) and rewards glass reflections
    (large flat blob). In padel, peak intensity is a better discriminator.
    """
    # Accept uint8 [0..255] (from mmap'd .npy) or float [0..1]
    if heatmap.dtype == np.uint8:
        heatmap = heatmap.astype(np.float32) / 255.0

    if heatmap.max() <= threshold:
        return []

    binary = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        roi = heatmap[y: y + h, x: x + w]
        peak = float(roi.max())
        cx = x + w / 2.0
        cy = y + h / 2.0
        candidates.append((cx, cy, peak))

    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates[:max_k]


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class BallSelectorBase:
    """Interface for stateful heatmap ball selectors."""

    name: str = "base"
    color: tuple[int, int, int] = (200, 200, 200)  # BGR

    def reset(self) -> None:
        """Clear internal state (call between videos)."""
        pass

    def select(
        self,
        heatmap: np.ndarray,
        player_bboxes_video: list | None = None,
        w_scale: float = 1.0,
        h_scale: float = 1.0,
    ) -> tuple[float, float] | None:
        """Select best ball candidate from heatmap.

        Args:
            heatmap: raw float (288, 512), range 0-1
            player_bboxes_video: list of BoundingBox in video space (optional)
            w_scale: heatmap-to-video x scale factor
            h_scale: heatmap-to-video y scale factor

        Returns:
            (cx, cy) in heatmap space, or None if no detection.
        """
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 1: Intensity — pure peak intensity, no trajectory
# ──────────────────────────────────────────────────────────────────────────────

class IntensitySelector(BallSelectorBase):
    """Baseline: pick the blob with the highest peak intensity.

    No trajectory information. Represents pure TrackNet confidence.
    Good for frames with no reflections; falls apart when a reflection
    has a stronger response than the ball.
    """

    name = "Intensity"
    color = (200, 80, 0)  # blue-ish

    def __init__(self, threshold: float = 0.25, max_k: int = 10):
        self.threshold = threshold
        self.max_k = max_k

    def reset(self) -> None:
        pass

    def select(self, heatmap, player_bboxes_video=None, w_scale=1.0, h_scale=1.0):
        candidates = extract_blob_candidates(heatmap, self.threshold, self.max_k)
        if not candidates:
            return None
        return (candidates[0][0], candidates[0][1])  # highest peak


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 2: Trajectory — velocity prediction + intensity weighting
# ──────────────────────────────────────────────────────────────────────────────

class TrajectorySelector(BallSelectorBase):
    """Trajectory-weighted candidate scoring.

    Keeps a history of recent detections and predicts the next position
    using a velocity estimate. Scores candidates by:

        score = w_traj × proximity + (1 - w_traj) × peak_intensity

    where w_traj decays with consecutive misses and direction inconsistency.
    This means after a bounce or hit (direction flip), the selector temporarily
    relies on intensity until a new trajectory is established.
    """

    name = "Trajectory"
    color = (0, 220, 220)  # yellow

    def __init__(
        self,
        threshold: float = 0.25,
        max_k: int = 10,
        history_len: int = 5,
        miss_decay: float = 0.2,   # w_traj reduction per missed frame
        inconsistency_factor: float = 0.3,  # w_traj multiplier when direction flips
    ):
        self.threshold = threshold
        self.max_k = max_k
        self.history_len = history_len
        self.miss_decay = miss_decay
        self.inconsistency_factor = inconsistency_factor

        self._history: deque[tuple[float, float]] = deque(maxlen=history_len)
        self._miss_count: int = 0

    def reset(self) -> None:
        self._history.clear()
        self._miss_count = 0

    def _predict(self) -> tuple[tuple[float, float], float] | None:
        """Compute predicted next position and trajectory confidence.

        Returns (pred_xy, consistency_factor) or None if insufficient history.
        """
        if len(self._history) < 2:
            return None

        hist = list(self._history)
        vx = hist[-1][0] - hist[-2][0]
        vy = hist[-1][1] - hist[-2][1]

        consistency = 1.0
        if len(hist) >= 3:
            vx2 = hist[-2][0] - hist[-3][0]
            vy2 = hist[-2][1] - hist[-3][1]
            dot = vx * vx2 + vy * vy2
            if dot < 0:
                # Direction changed (bounce / hit) — reduce trajectory confidence
                consistency = self.inconsistency_factor
            else:
                # Blend velocities: weight recent more
                vx = 0.7 * vx + 0.3 * vx2
                vy = 0.7 * vy + 0.3 * vy2

        steps = 1 + self._miss_count
        pred_x = hist[-1][0] + vx * steps
        pred_y = hist[-1][1] + vy * steps
        return (pred_x, pred_y), consistency

    def select(self, heatmap, player_bboxes_video=None, w_scale=1.0, h_scale=1.0):
        candidates = extract_blob_candidates(heatmap, self.threshold, self.max_k)

        if not candidates:
            self._miss_count += 1
            return None

        prediction = self._predict()

        if prediction is None:
            # No history yet — fall back to highest intensity
            best = candidates[0]
        else:
            pred_xy, consistency = prediction
            w_traj = consistency * max(0.0, 1.0 - self._miss_count * self.miss_decay)

            best = max(
                candidates,
                key=lambda c: (
                    w_traj * (1.0 / (1.0 + math.hypot(c[0] - pred_xy[0], c[1] - pred_xy[1])))
                    + (1.0 - w_traj) * c[2]
                ),
            )

        self._history.append((best[0], best[1]))
        self._miss_count = 0
        return (best[0], best[1])


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 3: Kalman — covariance-based gating with improved threshold
# ──────────────────────────────────────────────────────────────────────────────

class KalmanSelector(BallSelectorBase):
    """Kalman filter candidate selection.

    Uses the existing BallKalmanFilter (constant-velocity model) for
    Mahalanobis gating. Key improvement over current KalmanBallTracker:
    uses threshold 0.25 (not 0.5) so weaker ball responses aren't discarded
    before Kalman gating gets a chance to evaluate them.
    """

    name = "Kalman"
    color = (0, 200, 0)  # green

    def __init__(
        self,
        threshold: float = 0.25,
        max_k: int = 10,
        q_pos: float = 1.0,
        q_vel: float = 5.0,
        r_obs: float = 5.0,
        gate: float = 50.0,
        max_misses: int = 15,
    ):
        self.threshold = threshold
        self.max_k = max_k
        self.gate = gate
        self.max_misses = max_misses
        self._kf = BallKalmanFilter(q_pos=q_pos, q_vel=q_vel, r_obs=r_obs)

    def reset(self) -> None:
        self._kf.reset()

    def select(self, heatmap, player_bboxes_video=None, w_scale=1.0, h_scale=1.0):
        candidates = extract_blob_candidates(heatmap, self.threshold, self.max_k)

        if not candidates:
            if self._kf.is_initialized:
                self._kf.predict()
                self._kf.missed_update()
            return None

        if not self._kf.is_initialized:
            best = candidates[0]
            self._kf.initialize(best[0], best[1])
            return (best[0], best[1])

        self._kf.predict()
        gated = [
            (c, self._kf.gating_distance(c[0], c[1]))
            for c in candidates
            if self._kf.gating_distance(c[0], c[1]) < self.gate
        ]

        if gated:
            best_c, _ = min(gated, key=lambda t: t[1])
            self._kf.update(best_c[0], best_c[1])
            return (best_c[0], best_c[1])
        else:
            self._kf.missed_update()
            if self._kf.consecutive_misses > self.max_misses:
                self._kf.reset()
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 4: Occlusion-aware — trajectory + player bbox coasting
# ──────────────────────────────────────────────────────────────────────────────

class OcclusionAwareSelector(BallSelectorBase):
    """Trajectory selector extended with player-occlusion coasting.

    When the predicted ball position overlaps a player bounding box, the
    ball is likely occluded. Instead of reporting a false detection or
    a gap, the selector coasts using the velocity prediction — reporting
    the predicted position as the detection until the ball re-emerges.

    Requires player_bboxes_video to be passed to select().
    Falls back to TrajectorySelector behaviour when no player data available.
    """

    name = "Occlusion"
    color = (200, 0, 200)  # magenta

    def __init__(
        self,
        threshold: float = 0.25,
        max_k: int = 10,
        history_len: int = 5,
        miss_decay: float = 0.2,
        inconsistency_factor: float = 0.3,
        occlusion_margin: float = 1.5,    # bbox expansion factor for occlusion check
        max_occlusion_frames: int = 20,   # safety limit before giving up
    ):
        self._traj = TrajectorySelector(
            threshold=threshold,
            max_k=max_k,
            history_len=history_len,
            miss_decay=miss_decay,
            inconsistency_factor=inconsistency_factor,
        )
        self.occlusion_margin = occlusion_margin
        self.max_occlusion_frames = max_occlusion_frames
        self._occlusion_count: int = 0
        self._last_pred: tuple[float, float] | None = None

    def reset(self) -> None:
        self._traj.reset()
        self._occlusion_count = 0
        self._last_pred = None

    def _in_occlusion(
        self,
        pred_hm: tuple[float, float],
        player_bboxes_video: list,
        w_scale: float,
        h_scale: float,
    ) -> bool:
        """Check if predicted position (heatmap space) overlaps any player bbox."""
        if not player_bboxes_video:
            return False
        px = pred_hm[0] * w_scale
        py = pred_hm[1] * h_scale
        m = self.occlusion_margin
        for bbox in player_bboxes_video:
            bw = (bbox.x2 - bbox.x1) * m
            bh = (bbox.y2 - bbox.y1) * m
            cx = (bbox.x1 + bbox.x2) / 2
            cy = (bbox.y1 + bbox.y2) / 2
            if abs(px - cx) < bw / 2 and abs(py - cy) < bh / 2:
                return True
        return False

    def select(self, heatmap, player_bboxes_video=None, w_scale=1.0, h_scale=1.0):
        # Get prediction from trajectory before it consumes the frame
        prediction = self._traj._predict()

        if prediction is not None:
            pred_xy, _ = prediction
            self._last_pred = pred_xy

            if (
                player_bboxes_video
                and self._occlusion_count < self.max_occlusion_frames
                and self._in_occlusion(pred_xy, player_bboxes_video, w_scale, h_scale)
            ):
                # Coast through occlusion using velocity prediction
                self._occlusion_count += 1
                # Feed predicted position into trajectory history so velocity stays fresh
                self._traj._history.append(pred_xy)
                self._traj._miss_count = 0
                return pred_xy

        # Not in occlusion — delegate to trajectory selector
        self._occlusion_count = 0
        return self._traj.select(heatmap, player_bboxes_video, w_scale, h_scale)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 4: Classifier — threshold blobs, pick max classifier score
# ──────────────────────────────────────────────────────────────────────────────

class ClassifierSelector(BallSelectorBase):
    """Threshold heatmap → extract blob candidates → pick blob with max classifier score.

    Requires the original video frame to be passed as the `frame` keyword arg
    to select(). Falls back to highest peak intensity if no frame is provided
    or the classifier is not loaded.
    """

    name = "Classifier"
    color = (0, 140, 255)  # orange

    def __init__(
        self,
        classifier,  # BlobClassifier instance
        threshold: float = 0.25,
        max_k: int = 10,
        crop_radius: int = 60,
    ):
        self.clf = classifier
        self.threshold = threshold
        self.max_k = max_k
        self.crop_radius = crop_radius

    def reset(self) -> None:
        pass

    def select(self, heatmap, player_bboxes_video=None, w_scale=1.0, h_scale=1.0,
               frame: np.ndarray | None = None):
        candidates = extract_blob_candidates(heatmap, self.threshold, self.max_k)
        if not candidates:
            return None

        if frame is None or self.clf is None:
            return (candidates[0][0], candidates[0][1])

        H, W = frame.shape[:2]
        crops = []
        for cx, cy, _ in candidates:
            ox, oy = cx * w_scale, cy * h_scale
            x1 = max(0, int(ox - self.crop_radius))
            y1 = max(0, int(oy - self.crop_radius))
            x2 = min(W, int(ox + self.crop_radius))
            y2 = min(H, int(oy + self.crop_radius))
            crop = frame[y1:y2, x1:x2]
            crops.append(crop if crop.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8))

        scores = self.clf.predict_batch(crops)
        best_idx = int(np.argmax(scores))
        return (candidates[best_idx][0], candidates[best_idx][1])


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 5: KalmanClassifier — Kalman gating + classifier scoring
# ──────────────────────────────────────────────────────────────────────────────

class KalmanClassifierSelector(BallSelectorBase):
    """Kalman gating combined with blob classifier scoring.

    Pipeline:
      1. Extract blob candidates from thresholded heatmap.
      2. Predict next position with Kalman filter.
      3. Gate: discard candidates beyond `gate` px of the Kalman prediction
         (eliminates impossible jumps). If Kalman is not yet initialised,
         all candidates pass.
      4. Score remaining candidates:
             score = w_clf × clf_score  +  (1 - w_clf) × 1/(1 + kf_dist)
         Higher classifier confidence and closer to Kalman prediction wins.
      5. Update Kalman with the selected candidate.

    Requires the original video frame as `frame` kwarg in select().
    Falls back to highest peak intensity when no frame / classifier is available.
    """

    name = "Kalman+Clf"
    color = (255, 100, 0)  # cyan-ish

    def __init__(
        self,
        classifier,           # BlobClassifier instance
        threshold: float = 0.25,
        max_k: int = 10,
        crop_radius: int = 60,
        q_pos: float = 1.0,
        q_vel: float = 5.0,
        r_obs: float = 5.0,
        gate: float = 15.0,   # max Kalman distance (heatmap px) to keep a candidate
        max_misses: int = 15,
        w_clf: float = 0.3,   # weight of classifier score vs Kalman proximity
    ):
        self.clf = classifier
        self.threshold = threshold
        self.max_k = max_k
        self.crop_radius = crop_radius
        self.gate = gate
        self.max_misses = max_misses
        self.w_clf = w_clf
        self._kf = BallKalmanFilter(q_pos=q_pos, q_vel=q_vel, r_obs=r_obs)

    def reset(self) -> None:
        self._kf.reset()

    def _crop_candidates(
        self,
        candidates: list[tuple[float, float, float]],
        frame: np.ndarray,
        w_scale: float,
        h_scale: float,
    ) -> list[float]:
        """Return classifier scores for each candidate."""
        H, W = frame.shape[:2]
        crops = []
        for cx, cy, _ in candidates:
            ox, oy = cx * w_scale, cy * h_scale
            x1 = max(0, int(ox - self.crop_radius))
            y1 = max(0, int(oy - self.crop_radius))
            x2 = min(W, int(ox + self.crop_radius))
            y2 = min(H, int(oy + self.crop_radius))
            crop = frame[y1:y2, x1:x2]
            crops.append(crop if crop.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8))
        return self.clf.predict_batch(crops)

    def select(self, heatmap, player_bboxes_video=None, w_scale=1.0, h_scale=1.0,
               frame: np.ndarray | None = None):
        candidates = extract_blob_candidates(heatmap, self.threshold, self.max_k)

        if not candidates:
            if self._kf.is_initialized:
                self._kf.predict()
                self._kf.missed_update()
            return None

        # Kalman predict
        if self._kf.is_initialized:
            self._kf.predict()
            gated = [c for c in candidates
                     if self._kf.gating_distance(c[0], c[1]) < self.gate]
        else:
            gated = candidates  # no filter until initialised

        if not gated:
            self._kf.missed_update()
            if self._kf.consecutive_misses > self.max_misses:
                self._kf.reset()
            return None

        # Score gated candidates
        if frame is not None and self.clf is not None:
            clf_scores = self._crop_candidates(gated, frame, w_scale, h_scale)
        else:
            clf_scores = [c[2] for c in gated]  # fall back to peak intensity

        if self._kf.is_initialized:
            kf_dists = [self._kf.gating_distance(c[0], c[1]) for c in gated]
            max_dist = max(kf_dists) if kf_dists else 1.0
            kf_prox = [1.0 - d / (max_dist + 1e-6) for d in kf_dists]
        else:
            kf_prox = [1.0] * len(gated)

        scores = [
            self.w_clf * cs + (1.0 - self.w_clf) * kp
            for cs, kp in zip(clf_scores, kf_prox)
        ]

        best_idx = int(np.argmax(scores))
        best = gated[best_idx]

        if self._kf.is_initialized:
            self._kf.update(best[0], best[1])
        else:
            self._kf.initialize(best[0], best[1])

        return (best[0], best[1])


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

ALL_SELECTORS: list[BallSelectorBase] = [
    IntensitySelector(),
    TrajectorySelector(),
    KalmanSelector(),
]
