"""Position extraction, interpolation, smoothing, and team assignment."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from ..court.models import NET_POSITION
from ..court.projection import CourtProjection
from ..models import FrameDetections
from .models import PlayerTrack, TeamAssignment


def build_player_tracks(
    detections: list[FrameDetections],
    projection: CourtProjection,
    fps: float,
    max_interpolation_gap: int = 10,
    smooth_window: int = 11,
    smooth_polyorder: int = 3,
) -> dict[int, PlayerTrack]:
    """Build smoothed player tracks from frame detections.

    Projects each detection to court coordinates, groups by track_id,
    interpolates short gaps, and applies Savitzky-Golay smoothing.
    """
    # Collect raw positions per track_id
    raw: dict[int, list[tuple[int, float, float]]] = {}

    for fd in detections:
        for det in fd.detections:
            if det.track_id is None:
                continue
            court_pos = projection.project_detection(det)
            if not projection.is_inside_court(court_pos):
                continue
            raw.setdefault(det.track_id, []).append(
                (fd.frame_index, court_pos[0], court_pos[1])
            )

    # Build PlayerTrack for each track_id
    tracks: dict[int, PlayerTrack] = {}
    for tid, positions in raw.items():
        positions.sort(key=lambda p: p[0])
        frames = np.array([p[0] for p in positions], dtype=int)
        x_raw = np.array([p[1] for p in positions])
        y_raw = np.array([p[2] for p in positions])

        track = PlayerTrack(
            track_id=tid,
            fps=fps,
            frames=frames,
            x_raw=x_raw,
            y_raw=y_raw,
        )

        # Interpolate missing frames then smooth
        track = _interpolate_missing(track, max_interpolation_gap)
        track = _smooth_positions(track, smooth_window, smooth_polyorder)
        tracks[tid] = track

    return tracks


def _interpolate_missing(
    track: PlayerTrack,
    max_gap: int = 10,
) -> PlayerTrack:
    """Fill short gaps in the track with linear interpolation.

    Creates a continuous frame sequence from first to last frame,
    interpolating positions for missing frames (gaps <= max_gap).
    Longer gaps are filled with NaN.
    """
    if track.num_frames < 2:
        return track

    first, last = int(track.frames[0]), int(track.frames[-1])
    all_frames = np.arange(first, last + 1, dtype=int)

    # Start with NaN, fill known positions
    x_full = np.full(len(all_frames), np.nan)
    y_full = np.full(len(all_frames), np.nan)

    for i, f in enumerate(track.frames):
        idx = int(f) - first
        x_full[idx] = track.x_raw[i]
        y_full[idx] = track.y_raw[i]

    # Find gaps and interpolate only short ones
    known_mask = ~np.isnan(x_full)
    known_indices = np.where(known_mask)[0]

    if len(known_indices) >= 2:
        # Check each gap between consecutive known points
        for k in range(len(known_indices) - 1):
            gap_start = known_indices[k]
            gap_end = known_indices[k + 1]
            gap_len = gap_end - gap_start - 1

            if 0 < gap_len <= max_gap:
                # Linear interpolation for this gap
                for idx in range(gap_start + 1, gap_end):
                    t = (idx - gap_start) / (gap_end - gap_start)
                    x_full[idx] = x_full[gap_start] + t * (x_full[gap_end] - x_full[gap_start])
                    y_full[idx] = y_full[gap_start] + t * (y_full[gap_end] - y_full[gap_start])

    # Keep only non-NaN frames
    valid = ~np.isnan(x_full)
    track.frames = all_frames[valid]
    track.x_raw = x_full[valid]
    track.y_raw = y_full[valid]

    return track


def _smooth_positions(
    track: PlayerTrack,
    window: int = 11,
    polyorder: int = 3,
) -> PlayerTrack:
    """Apply Savitzky-Golay smoothing to x and y positions."""
    if track.num_frames < window:
        track.x_smooth = track.x_raw.copy()
        track.y_smooth = track.y_raw.copy()
        return track

    track.x_smooth = savgol_filter(track.x_raw, window, polyorder)
    track.y_smooth = savgol_filter(track.y_raw, window, polyorder)
    return track


def assign_teams(
    tracks: dict[int, PlayerTrack],
    n_init_frames: int = 30,
) -> TeamAssignment:
    """Assign players to near/far teams based on initial court position.

    Players with median Y < NET_POSITION (10m) in first N frames are near team.
    """
    median_y: dict[int, float] = {}
    for tid, track in tracks.items():
        # Use first n_init_frames worth of positions
        mask = track.frames < track.frames[0] + n_init_frames
        y_init = track.y[mask] if mask.any() else track.y
        median_y[tid] = float(np.median(y_init))

    # Sort by median Y, split into near (lower Y) and far (higher Y)
    sorted_ids = sorted(median_y.keys(), key=lambda t: median_y[t])

    if len(sorted_ids) < 4:
        # Not enough players — best effort: split in half
        mid = len(sorted_ids) // 2
        near = tuple(sorted_ids[:mid]) if mid >= 2 else tuple(sorted_ids[:2])
        far = tuple(sorted_ids[mid:]) if len(sorted_ids) - mid >= 2 else tuple(sorted_ids[2:4])
    else:
        # Standard case: 2 nearest to baseline = near team
        near = (sorted_ids[0], sorted_ids[1])
        far = (sorted_ids[2], sorted_ids[3])

    return TeamAssignment(near_team=near, far_team=far)
