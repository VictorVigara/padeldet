"""Position extraction, interpolation, smoothing, and team assignment."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

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
    # Collect raw positions per logical_player_id (stable 1-4); skip unassigned detections.
    raw: dict[int, list[tuple[int, float, float]]] = {}

    for fd in detections:
        for det in fd.detections:
            pid = det.logical_player_id
            if pid is None:
                continue
            court_pos = projection.project_detection(det)
            if not projection.is_inside_court(court_pos):
                continue
            raw.setdefault(pid, []).append(
                (fd.frame_index, court_pos[0], court_pos[1])
            )

    # Build PlayerTrack for each logical_player_id
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
    net_y: float | None = None,
) -> TeamAssignment:
    """Assign players to near/far teams based on initial court position.

    For doubles (4 players) sorts by median Y and splits half/half.
    For singles (2 players) the lower-Y player is near, the higher-Y far.
    For a single tracked player, splits by the net line (`net_y`, defaulting
    to the midpoint of the player's own median if unknown) so the player
    lands on the correct side without being duplicated onto both teams.
    """
    median_y: dict[int, float] = {}
    for tid, track in tracks.items():
        mask = track.frames < track.frames[0] + n_init_frames
        y_init = track.y[mask] if mask.any() else track.y
        median_y[tid] = float(np.median(y_init))

    sorted_ids = sorted(median_y.keys(), key=lambda t: median_y[t])

    # 0 players → empty teams; team_for is never called in that path.
    if not sorted_ids:
        return TeamAssignment(near_team=(), far_team=())

    # 1 player → place on its actual side using the net line if provided,
    # otherwise default to near. Never duplicate onto both teams.
    if len(sorted_ids) == 1:
        tid = sorted_ids[0]
        if net_y is not None and median_y[tid] > net_y:
            return TeamAssignment(near_team=(), far_team=(tid,))
        return TeamAssignment(near_team=(tid,), far_team=())

    # ≥2 players → split half/half by median Y.
    half = len(sorted_ids) // 2
    near = tuple(sorted_ids[:half])
    far = tuple(sorted_ids[half:])
    return TeamAssignment(near_team=near, far_team=far)
