"""Movement metrics: distance, speed, speed distribution."""

from __future__ import annotations

import numpy as np

from .models import PlayerTrack, SpeedCategory


def compute_distance(track: PlayerTrack) -> float:
    """Total distance covered in meters (sum of frame-to-frame displacements)."""
    if track.num_frames < 2:
        return 0.0
    dx = np.diff(track.x)
    dy = np.diff(track.y)
    return float(np.sum(np.sqrt(dx**2 + dy**2)))


def compute_speeds(track: PlayerTrack) -> np.ndarray:
    """Instantaneous speed in m/s for each frame transition."""
    if track.num_frames < 2:
        return np.array([])
    dx = np.diff(track.x)
    dy = np.diff(track.y)
    dt = 1.0 / track.fps
    displacements = np.sqrt(dx**2 + dy**2)
    return displacements / dt


def compute_speed_distribution(speeds: np.ndarray) -> dict[str, float]:
    """Fraction of time in each speed category."""
    if len(speeds) == 0:
        return {cat.value: 0.0 for cat in SpeedCategory}

    n = len(speeds)
    return {
        SpeedCategory.STATIONARY.value: float(np.sum(speeds < 0.5) / n),
        SpeedCategory.WALKING.value: float(np.sum((speeds >= 0.5) & (speeds < 2.0)) / n),
        SpeedCategory.JOGGING.value: float(np.sum((speeds >= 2.0) & (speeds < 4.0)) / n),
        SpeedCategory.SPRINTING.value: float(np.sum(speeds >= 4.0) / n),
    }
