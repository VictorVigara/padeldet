"""Data models for player metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ── Position data ────────────────────────────────────────────────────


@dataclass
class CourtPosition:
    """A single player position in court meters."""

    frame_index: int
    x: float  # 0-10m (court width)
    y: float  # 0-20m (court length, 0=near baseline)


@dataclass
class PlayerTrack:
    """Smoothed trajectory for one player across the video."""

    track_id: int
    fps: float
    frames: np.ndarray  # frame indices (int)
    x_raw: np.ndarray  # raw court x positions
    y_raw: np.ndarray  # raw court y positions
    x_smooth: np.ndarray = field(default_factory=lambda: np.array([]))
    y_smooth: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def x(self) -> np.ndarray:
        """Best available x positions (smoothed if available)."""
        return self.x_smooth if len(self.x_smooth) > 0 else self.x_raw

    @property
    def y(self) -> np.ndarray:
        """Best available y positions (smoothed if available)."""
        return self.y_smooth if len(self.y_smooth) > 0 else self.y_raw


# ── Team assignment ──────────────────────────────────────────────────


@dataclass
class TeamAssignment:
    """Which players are on which side of the court."""

    near_team: tuple[int, int]  # track_ids, Y < 10m
    far_team: tuple[int, int]  # track_ids, Y > 10m

    def team_for(self, track_id: int) -> str:
        """Return 'near' or 'far' for a given track_id."""
        if track_id in self.near_team:
            return "near"
        if track_id in self.far_team:
            return "far"
        raise ValueError(f"track_id {track_id} not in any team")


# ── Enums ────────────────────────────────────────────────────────────


class Zone(Enum):
    NET = "net"
    TRANSITION = "transition"
    BACK = "back"


class Formation(Enum):
    BOTH_AT_NET = "both_at_net"
    BOTH_AT_BACK = "both_at_back"
    ONE_UP_ONE_BACK = "one_up_one_back"


class SpeedCategory(Enum):
    STATIONARY = "stationary"  # < 0.5 m/s
    WALKING = "walking"  # 0.5 - 2.0 m/s
    JOGGING = "jogging"  # 2.0 - 4.0 m/s
    SPRINTING = "sprinting"  # > 4.0 m/s


# ── Metric results ───────────────────────────────────────────────────


@dataclass
class PlayerMetricsResult:
    """Metrics for a single player."""

    track_id: int
    team: str  # "near" or "far"
    total_distance_m: float
    avg_speed_ms: float
    max_speed_ms: float
    speed_distribution: dict[str, float]  # SpeedCategory.value -> fraction
    zone_distribution: dict[str, float]  # Zone.value -> fraction
    net_approaches: int
    position_heatmap: list[list[float]]  # 2D histogram (normalized)

    def serialize(self) -> dict:
        return {
            "track_id": self.track_id,
            "team": self.team,
            "total_distance_m": round(self.total_distance_m, 2),
            "avg_speed_ms": round(self.avg_speed_ms, 2),
            "max_speed_ms": round(self.max_speed_ms, 2),
            "speed_distribution": {
                k: round(v, 4) for k, v in self.speed_distribution.items()
            },
            "zone_distribution": {
                k: round(v, 4) for k, v in self.zone_distribution.items()
            },
            "net_approaches": self.net_approaches,
            "position_heatmap": self.position_heatmap,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PlayerMetricsResult:
        return cls(**d)


@dataclass
class TeamMetricsResult:
    """Metrics for a team (pair of players)."""

    team: str  # "near" or "far"
    player_ids: tuple[int, int]
    formation_distribution: dict[str, float]  # Formation.value -> fraction

    def serialize(self) -> dict:
        return {
            "team": self.team,
            "player_ids": list(self.player_ids),
            "formation_distribution": {
                k: round(v, 4) for k, v in self.formation_distribution.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> TeamMetricsResult:
        d = dict(d)
        d["player_ids"] = tuple(d["player_ids"])
        return cls(**d)


@dataclass
class MatchMetrics:
    """All metrics for a match (or video segment)."""

    fps: float
    total_frames: int
    duration_seconds: float
    players: list[PlayerMetricsResult]
    teams: list[TeamMetricsResult]

    def serialize(self) -> dict:
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration_seconds, 2),
            "players": [p.serialize() for p in self.players],
            "teams": [t.serialize() for t in self.teams],
        }

    @classmethod
    def from_dict(cls, d: dict) -> MatchMetrics:
        return cls(
            fps=d["fps"],
            total_frames=d["total_frames"],
            duration_seconds=d["duration_seconds"],
            players=[PlayerMetricsResult.from_dict(p) for p in d["players"]],
            teams=[TeamMetricsResult.from_dict(t) for t in d["teams"]],
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.serialize(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> MatchMetrics:
        with open(path) as f:
            return cls.from_dict(json.load(f))
