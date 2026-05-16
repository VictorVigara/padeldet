"""Clothing color extraction and per-player appearance library using pose keypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

import cv2
import numpy as np

from .models import TrackedDetection

_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16

_MIN_REGION_AREA_PX = 50


@dataclass
class RegionColors:
    shirt_mean: tuple[float, float, float] | None = None   # BGR
    shorts_mean: tuple[float, float, float] | None = None
    shoe_mean: tuple[float, float, float] | None = None
    shirt_hist: np.ndarray | None = None   # 512-float normalised HSV histogram
    shorts_hist: np.ndarray | None = None
    shoe_hist: np.ndarray | None = None


def _kp(
    keypoints: list[tuple[float, float, float]],
    idx: int,
    min_conf: float = 0.3,
) -> tuple[float, float] | None:
    if idx >= len(keypoints):
        return None
    x, y, c = keypoints[idx]
    if c < min_conf or (x == 0.0 and y == 0.0):
        return None
    return (x, y)


def _crop_polygon(frame: np.ndarray, pts: list[tuple[float, float]]) -> np.ndarray | None:
    """Return pixels inside the convex hull of pts, or None if too small."""
    if len(pts) < 3:
        return None
    arr = np.array(pts, dtype=np.float32)
    hull = cv2.convexHull(arr)
    x, y, w, h = cv2.boundingRect(hull)
    if w * h < _MIN_REGION_AREA_PX:
        return None
    # Clamp to frame
    fh, fw = frame.shape[:2]
    x = max(0, x); y = max(0, y)
    x2 = min(fw, x + w); y2 = min(fh, y + h)
    if x2 <= x or y2 <= y:
        return None
    roi = frame[y:y2, x:x2]
    mask = np.zeros((y2 - y, x2 - x), dtype=np.uint8)
    shifted = hull.squeeze(1) - np.array([x, y], dtype=np.float32)
    cv2.fillConvexPoly(mask, shifted.astype(np.int32), 255)
    pixels = roi[mask == 255]
    return pixels if len(pixels) >= _MIN_REGION_AREA_PX else None


def _rect_strip_pixels(
    frame: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    half_width: float,
) -> np.ndarray | None:
    """Pixels in a rectangle of given half-width centred on the line p1→p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = max((dx**2 + dy**2) ** 0.5, 1e-6)
    nx, ny = -dy / length, dx / length   # perpendicular unit vector
    corners = [
        (x1 + nx * half_width, y1 + ny * half_width),
        (x1 - nx * half_width, y1 - ny * half_width),
        (x2 - nx * half_width, y2 - ny * half_width),
        (x2 + nx * half_width, y2 + ny * half_width),
    ]
    return _crop_polygon(frame, corners)


def extract_shirt_pixels(
    frame: np.ndarray,
    keypoints: list[tuple[float, float, float]],
    min_conf: float = 0.3,
) -> np.ndarray | None:
    ls = _kp(keypoints, _L_SHOULDER, min_conf)
    rs = _kp(keypoints, _R_SHOULDER, min_conf)
    lh = _kp(keypoints, _L_HIP, min_conf)
    rh = _kp(keypoints, _R_HIP, min_conf)
    pts = [p for p in [ls, rs, rh, lh] if p is not None]
    return _crop_polygon(frame, pts)


def extract_shorts_pixels(
    frame: np.ndarray,
    keypoints: list[tuple[float, float, float]],
    min_conf: float = 0.3,
) -> np.ndarray | None:
    lh = _kp(keypoints, _L_HIP, min_conf)
    rh = _kp(keypoints, _R_HIP, min_conf)
    lk = _kp(keypoints, _L_KNEE, min_conf)
    rk = _kp(keypoints, _R_KNEE, min_conf)

    if lh and rh:
        half_w = max(8.0, abs(rh[0] - lh[0]) * 0.4)
    else:
        half_w = 15.0

    def _interp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    parts: list[np.ndarray] = []
    if lh and lk:
        px = _rect_strip_pixels(frame, lh, _interp(lh, lk, 0.8), half_w)
        if px is not None:
            parts.append(px)
    if rh and rk:
        px = _rect_strip_pixels(frame, rh, _interp(rh, rk, 0.8), half_w)
        if px is not None:
            parts.append(px)

    if not parts:
        return None
    return np.concatenate(parts, axis=0)


def extract_shoe_pixels(
    frame: np.ndarray,
    keypoints: list[tuple[float, float, float]],
    patch_size: int = 20,
    min_conf: float = 0.3,
) -> np.ndarray | None:
    fh, fw = frame.shape[:2]
    half = patch_size // 2
    parts: list[np.ndarray] = []
    for idx in (_L_ANKLE, _R_ANKLE):
        kp = _kp(keypoints, idx, min_conf)
        if kp is None:
            continue
        x, y = int(kp[0]), int(kp[1])
        x1, y1 = max(0, x - half), max(0, y - half)
        x2, y2 = min(fw, x + half), min(fh, y + half)
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            parts.append(roi.reshape(-1, 3))
    if not parts:
        return None
    return np.concatenate(parts, axis=0)


def mean_color(pixels: np.ndarray) -> tuple[float, float, float]:
    m = pixels.mean(axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))


def hsv_histogram(pixels: np.ndarray, bins: int = 8) -> np.ndarray:
    # Keep (1, N, 3) shape — calcHist requires a proper image layout
    hsv = cv2.cvtColor(pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [bins, bins, bins],
        [0, 180, 0, 256, 0, 256],
    ).flatten()
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.astype(np.float32)


def histogram_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """1 − histogram intersection. Range [0, 1], lower = more similar."""
    return float(1.0 - np.minimum(h1, h2).sum())


def extract_player_colors(
    frame: np.ndarray,
    detection: TrackedDetection,
    min_conf: float = 0.3,
) -> RegionColors | None:
    if detection.keypoints is None:
        return None
    kps = detection.keypoints
    rc = RegionColors()

    shirt_px = extract_shirt_pixels(frame, kps, min_conf)
    if shirt_px is not None and len(shirt_px) >= _MIN_REGION_AREA_PX:
        rc.shirt_mean = mean_color(shirt_px)
        rc.shirt_hist = hsv_histogram(shirt_px)

    shorts_px = extract_shorts_pixels(frame, kps, min_conf)
    if shorts_px is not None and len(shorts_px) >= _MIN_REGION_AREA_PX:
        rc.shorts_mean = mean_color(shorts_px)
        rc.shorts_hist = hsv_histogram(shorts_px)

    shoe_px = extract_shoe_pixels(frame, kps, min_conf=min_conf)
    if shoe_px is not None and len(shoe_px) >= _MIN_REGION_AREA_PX:
        rc.shoe_mean = mean_color(shoe_px)
        rc.shoe_hist = hsv_histogram(shoe_px)

    return rc


class AppearanceLibrary:
    """Accumulates per-player color samples and builds averaged profiles."""

    _WEIGHTS = {"shirt": 0.5, "shorts": 0.3, "shoes": 0.2}

    def __init__(self) -> None:
        self._samples: dict[int, list[RegionColors]] = defaultdict(list)
        self._profiles: dict[int, RegionColors] = {}

    def add(self, logical_player_id: int, colors: RegionColors) -> None:
        self._samples[logical_player_id].append(colors)

    def build(self) -> None:
        for pid, samples in self._samples.items():
            self._profiles[pid] = self._average(samples)

    def _average(self, samples: list[RegionColors]) -> RegionColors:
        def avg_means(attr: str):
            vals = [getattr(s, attr) for s in samples if getattr(s, attr) is not None]
            if not vals:
                return None
            arr = np.array(vals)
            m = arr.mean(axis=0)
            return (float(m[0]), float(m[1]), float(m[2]))

        def avg_hists(attr: str):
            vals = [getattr(s, attr) for s in samples if getattr(s, attr) is not None]
            if not vals:
                return None
            return np.stack(vals).mean(axis=0).astype(np.float32)

        return RegionColors(
            shirt_mean=avg_means("shirt_mean"),
            shorts_mean=avg_means("shorts_mean"),
            shoe_mean=avg_means("shoe_mean"),
            shirt_hist=avg_hists("shirt_hist"),
            shorts_hist=avg_hists("shorts_hist"),
            shoe_hist=avg_hists("shoe_hist"),
        )

    def mean_colors(self, pid: int) -> RegionColors | None:
        return self._profiles.get(pid)

    def player_ids(self) -> list[int]:
        return sorted(self._profiles.keys())

    def match(self, colors: RegionColors) -> int | None:
        """Return the logical_player_id whose profile best matches colors, or None."""
        if not self._profiles:
            return None
        best_pid, best_dist = None, float("inf")
        for pid, profile in self._profiles.items():
            dist = self._weighted_distance(colors, profile)
            if dist < best_dist:
                best_dist = dist
                best_pid = pid
        return best_pid

    def all_similarities(self, colors: RegionColors) -> dict[int, float]:
        """Similarity score (0–1, higher = better match) against every library profile."""
        return {
            pid: max(0.0, 1.0 - self._weighted_distance(colors, profile))
            for pid, profile in self._profiles.items()
        }

    def _weighted_distance(self, a: RegionColors, b: RegionColors) -> float:
        total_w, total_d = 0.0, 0.0
        pairs = [
            ("shirt", a.shirt_hist, b.shirt_hist),
            ("shorts", a.shorts_hist, b.shorts_hist),
            ("shoes", a.shoe_hist, b.shoe_hist),
        ]
        for name, ha, hb in pairs:
            if ha is not None and hb is not None:
                w = self._WEIGHTS[name]
                total_d += w * histogram_distance(ha, hb)
                total_w += w
        if total_w == 0.0:
            return float("inf")
        return total_d / total_w
