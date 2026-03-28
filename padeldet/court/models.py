"""Court keypoint data model and court dimensions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# ── Court real-world dimensions (meters) ─────────────────────────────
COURT_WIDTH = 10.0       # baseline width
COURT_LENGTH = 20.0      # sideline length
SERVICE_BOX_DEPTH = 3.0  # from baseline to service line
NET_POSITION = 10.0      # from near baseline to net (center of court)

# ── 12-point keypoint schema ─────────────────────────────────────────
#
# k11--------------------k12   (far baseline)
# |                       |
# k8-----------k9--------k10   (far service line)
# |            |          |
# k6----------------------k7   (net)
# |            |          |
# k3-----------k4---------k5   (near service line)
# |                       |
# k1----------------------k2   (near baseline)
#

COURT_KEYPOINT_NAMES: list[tuple[str, str]] = [
    ("k1",  "near baseline left"),
    ("k2",  "near baseline right"),
    ("k3",  "near service line left"),
    ("k4",  "near service line center"),
    ("k5",  "near service line right"),
    ("k6",  "net left"),
    ("k7",  "net right"),
    ("k8",  "far service line left"),
    ("k9",  "far service line center"),
    ("k10", "far service line right"),
    ("k11", "far baseline left"),
    ("k12", "far baseline right"),
]

# Standard 2D court coordinates in meters (origin at k1 = bottom-left).
# Used as homography destination points.
COURT_KEYPOINTS_2D: list[tuple[float, float]] = [
    # k1: near baseline left
    (0.0, 0.0),
    # k2: near baseline right
    (COURT_WIDTH, 0.0),
    # k3: near service line left
    (0.0, SERVICE_BOX_DEPTH),
    # k4: near service line center
    (COURT_WIDTH / 2, SERVICE_BOX_DEPTH),
    # k5: near service line right
    (COURT_WIDTH, SERVICE_BOX_DEPTH),
    # k6: net left
    (0.0, NET_POSITION),
    # k7: net right
    (COURT_WIDTH, NET_POSITION),
    # k8: far service line left
    (0.0, COURT_LENGTH - SERVICE_BOX_DEPTH),
    # k9: far service line center
    (COURT_WIDTH / 2, COURT_LENGTH - SERVICE_BOX_DEPTH),
    # k10: far service line right
    (COURT_WIDTH, COURT_LENGTH - SERVICE_BOX_DEPTH),
    # k11: far baseline left
    (0.0, COURT_LENGTH),
    # k12: far baseline right
    (COURT_WIDTH, COURT_LENGTH),
]

NUM_KEYPOINTS = len(COURT_KEYPOINT_NAMES)


@dataclass
class CourtKeypoints:
    """Court keypoints in pixel coordinates.

    Stores up to 12 keypoints. Missing keypoints are represented as None.
    """

    points: list[tuple[float, float] | None]

    def __post_init__(self):
        if len(self.points) != NUM_KEYPOINTS:
            raise ValueError(
                f"Expected {NUM_KEYPOINTS} keypoints, got {len(self.points)}"
            )

    @property
    def available_indices(self) -> list[int]:
        """Indices of keypoints that are not None."""
        return [i for i, p in enumerate(self.points) if p is not None]

    @property
    def num_available(self) -> int:
        return len(self.available_indices)

    def get_available_pairs(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Return matched pairs of (pixel_points, court_2d_points) for available keypoints."""
        pixel_pts = []
        court_pts = []
        for i in self.available_indices:
            pixel_pts.append(self.points[i])
            court_pts.append(COURT_KEYPOINTS_2D[i])
        return pixel_pts, court_pts

    def serialize(self) -> dict:
        return {
            "points": [list(p) if p is not None else None for p in self.points],
        }

    @classmethod
    def from_dict(cls, d: dict) -> CourtKeypoints:
        points = [tuple(p) if p is not None else None for p in d["points"]]
        return cls(points=points)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.serialize(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CourtKeypoints:
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
