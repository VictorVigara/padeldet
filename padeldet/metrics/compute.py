"""Top-level orchestrator for computing player metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..court.projection import CourtProjection
from ..models import FrameDetections
from .models import MatchMetrics, PlayerMetricsResult, TeamMetricsResult
from .movement import compute_distance, compute_speed_distribution, compute_speeds
from .positions import assign_teams, build_player_tracks
from .zones import (
    compute_formation_distribution,
    compute_formation_timeline,
    compute_position_heatmap,
    compute_zone_distribution,
    count_net_approaches,
)


@dataclass
class MetricsConfig:
    """Configuration for metrics computation."""

    # Position smoothing
    smooth_window: int = 11
    smooth_polyorder: int = 3
    max_interpolation_gap: int = 10

    # Team assignment
    team_init_frames: int = 30

    # Zone metrics
    net_approach_min_frames: int = 5

    # Heatmap
    heatmap_x_bins: int = 20
    heatmap_y_bins: int = 40


def compute_metrics(
    detections: list[FrameDetections],
    projection: CourtProjection,
    fps: float,
    config: MetricsConfig | None = None,
) -> MatchMetrics:
    """Compute all player metrics from detection data.

    Args:
        detections: Per-frame player detections (from load_detections).
        projection: Court projection for coordinate mapping.
        fps: Video frame rate.
        config: Optional configuration overrides.

    Returns:
        MatchMetrics with per-player and per-team results.
    """
    config = config or MetricsConfig()

    # 1. Build smoothed position tracks
    tracks = build_player_tracks(
        detections,
        projection,
        fps,
        max_interpolation_gap=config.max_interpolation_gap,
        smooth_window=config.smooth_window,
        smooth_polyorder=config.smooth_polyorder,
    )

    print(f"Built tracks for {len(tracks)} players: {sorted(tracks.keys())}")
    for tid, track in sorted(tracks.items()):
        print(f"  Player {tid}: {track.num_frames} frames")

    # 2. Assign teams
    teams = assign_teams(tracks, config.team_init_frames)
    print(f"Teams — near: {teams.near_team}, far: {teams.far_team}")

    # 3. Per-player metrics
    player_results: list[PlayerMetricsResult] = []
    for tid, track in sorted(tracks.items()):
        team = teams.team_for(tid)
        speeds = compute_speeds(track)

        player_results.append(
            PlayerMetricsResult(
                track_id=tid,
                team=team,
                total_distance_m=compute_distance(track),
                avg_speed_ms=float(np.mean(speeds)) if len(speeds) > 0 else 0.0,
                max_speed_ms=float(np.max(speeds)) if len(speeds) > 0 else 0.0,
                speed_distribution=compute_speed_distribution(speeds),
                zone_distribution=compute_zone_distribution(track, team),
                net_approaches=count_net_approaches(
                    track, team, config.net_approach_min_frames
                ),
                position_heatmap=compute_position_heatmap(
                    track, config.heatmap_x_bins, config.heatmap_y_bins
                ).tolist(),
            )
        )

    # 4. Per-team metrics
    team_results: list[TeamMetricsResult] = []
    for team_name, player_ids in [
        ("near", teams.near_team),
        ("far", teams.far_team),
    ]:
        if player_ids[0] in tracks and player_ids[1] in tracks:
            timeline = compute_formation_timeline(
                tracks[player_ids[0]], tracks[player_ids[1]], team_name
            )
            team_results.append(
                TeamMetricsResult(
                    team=team_name,
                    player_ids=player_ids,
                    formation_distribution=compute_formation_distribution(timeline),
                )
            )

    # 5. Assemble
    total_frames = len(detections)
    return MatchMetrics(
        fps=fps,
        total_frames=total_frames,
        duration_seconds=total_frames / fps if fps > 0 else 0.0,
        players=player_results,
        teams=team_results,
    )
