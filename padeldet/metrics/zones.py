"""Zone classification, formation detection, net approaches, and heatmaps."""

from __future__ import annotations

import numpy as np

from ..court.models import COURT_LENGTH, COURT_WIDTH, SERVICE_BOX_DEPTH
from .models import Formation, PlayerTrack, Zone


# Zone boundaries in meters from the respective baseline
# Net zone: 0 to SERVICE_BOX_DEPTH (3m)
# Transition zone: 3m to 7m
# Back zone: 7m to 10m (half court)
_ZONE_NET_MAX = SERVICE_BOX_DEPTH  # 3m
_ZONE_TRANSITION_MAX = 7.0  # 7m


def classify_zone(y: float, team: str) -> Zone:
    """Classify a court Y position into a zone relative to the team's side.

    Near team plays on Y=0..10 (baseline at 0, net at 10).
    Far team plays on Y=10..20 (net at 10, baseline at 20).
    Distance from baseline determines the zone.
    """
    if team == "near":
        dist_from_baseline = y  # baseline is at Y=0
    else:
        dist_from_baseline = COURT_LENGTH - y  # baseline is at Y=20

    # Clamp to valid range
    dist_from_baseline = max(0.0, min(dist_from_baseline, COURT_LENGTH / 2))

    if dist_from_baseline >= _ZONE_TRANSITION_MAX:
        return Zone.NET
    elif dist_from_baseline >= _ZONE_NET_MAX:
        return Zone.TRANSITION
    else:
        return Zone.BACK


def compute_zone_distribution(track: PlayerTrack, team: str) -> dict[str, float]:
    """Fraction of time spent in each zone."""
    zones = [classify_zone(float(y), team) for y in track.y]
    n = len(zones)
    if n == 0:
        return {z.value: 0.0 for z in Zone}

    return {
        z.value: sum(1 for zz in zones if zz == z) / n
        for z in Zone
    }


def compute_formation_timeline(
    track_a: PlayerTrack,
    track_b: PlayerTrack,
    team: str,
) -> list[Formation]:
    """Classify team formation for each frame where both players are present.

    Returns a list of Formation values, one per common frame.
    """
    # Build frame -> position lookup for each track
    a_pos = dict(zip(track_a.frames.tolist(), track_a.y.tolist()))
    b_pos = dict(zip(track_b.frames.tolist(), track_b.y.tolist()))

    common_frames = sorted(set(a_pos.keys()) & set(b_pos.keys()))

    timeline: list[Formation] = []
    for f in common_frames:
        zone_a = classify_zone(a_pos[f], team)
        zone_b = classify_zone(b_pos[f], team)

        if zone_a == Zone.NET and zone_b == Zone.NET:
            timeline.append(Formation.BOTH_AT_NET)
        elif zone_a == Zone.BACK and zone_b == Zone.BACK:
            timeline.append(Formation.BOTH_AT_BACK)
        else:
            timeline.append(Formation.ONE_UP_ONE_BACK)

    return timeline


def compute_formation_distribution(timeline: list[Formation]) -> dict[str, float]:
    """Fraction of time in each formation."""
    n = len(timeline)
    if n == 0:
        return {f.value: 0.0 for f in Formation}

    return {
        f.value: sum(1 for ff in timeline if ff == f) / n
        for f in Formation
    }


def count_net_approaches(
    track: PlayerTrack,
    team: str,
    min_frames_in_zone: int = 5,
) -> int:
    """Count transitions from back/transition zone to net zone.

    Only counts if the player stays in the net zone for at least
    min_frames_in_zone consecutive frames (filters noise).
    """
    zones = [classify_zone(float(y), team) for y in track.y]
    count = 0
    i = 0
    n = len(zones)

    while i < n:
        # Find start of a net zone stretch
        if zones[i] == Zone.NET:
            j = i
            while j < n and zones[j] == Zone.NET:
                j += 1
            net_len = j - i

            # Check if preceded by non-net zone
            was_outside = i == 0 or zones[i - 1] != Zone.NET
            if was_outside and net_len >= min_frames_in_zone:
                count += 1
            i = j
        else:
            i += 1

    return count


def compute_position_heatmap(
    track: PlayerTrack,
    x_bins: int = 20,
    y_bins: int = 40,
) -> np.ndarray:
    """2D histogram of player positions on the court, normalized to sum to 1."""
    heatmap, _, _ = np.histogram2d(
        track.x,
        track.y,
        bins=[x_bins, y_bins],
        range=[[0, COURT_WIDTH], [0, COURT_LENGTH]],
    )
    total = heatmap.sum()
    if total > 0:
        heatmap = heatmap / total
    return heatmap
