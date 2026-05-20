"""Court keypoint data model and padel-rule constants.

Court dimensions (width, length, net position, door y-band) are NOT defined
here — they depend on the match type (doubles vs singles) and live in
`padeldet.match_profile`. They are carried at runtime by `CourtProjection`
instances (width_m, length_m, net_y, door_y_min, door_y_max).

This module only owns:
  * the 12-point keypoint schema (`COURT_KEYPOINT_NAMES`)
  * padel-rule constants that don't vary with court size
    (`SERVICE_BOX_DEPTH`, `DOOR_HALF_WIDTH_M`)
  * `build_keypoints_2d(width_m, length_m)` — the factory that produces the
    2D reference layout for a given court size
  * the `CourtKeypoints` dataclass (pixel keypoints only)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# ── Padel-rule constants (same for doubles and singles) ──────────────
SERVICE_BOX_DEPTH = 3.0       # meters from baseline to service line
# Official padel door: 0.72 m wide, centered on the net line on each side wall.
# ±2 m tolerance around the net absorbs detection noise during exit/entry.
DOOR_HALF_WIDTH_M = 2.0

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


def build_keypoints_2d(
    width_m: float,
    length_m: float,
    service_box_depth_m: float = SERVICE_BOX_DEPTH,
) -> list[tuple[float, float]]:
    """Build the 12-point 2D reference layout for a given court size.

    Layout matches `COURT_KEYPOINT_NAMES`: origin at k1 (near baseline left),
    +x to the right, +y towards the far baseline. The net sits at length_m/2.
    """
    net_y = length_m / 2
    return [
        (0.0, 0.0),                                       # k1
        (width_m, 0.0),                                   # k2
        (0.0, service_box_depth_m),                       # k3
        (width_m / 2, service_box_depth_m),               # k4
        (width_m, service_box_depth_m),                   # k5
        (0.0, net_y),                                     # k6
        (width_m, net_y),                                 # k7
        (0.0, length_m - service_box_depth_m),            # k8
        (width_m / 2, length_m - service_box_depth_m),    # k9
        (width_m, length_m - service_box_depth_m),        # k10
        (0.0, length_m),                                  # k11
        (width_m, length_m),                              # k12
    ]


NUM_KEYPOINTS = len(COURT_KEYPOINT_NAMES)


@dataclass
class CourtKeypoints:
    """Court keypoints in pixel coordinates.

    Stores up to 12 keypoints. Missing keypoints are represented as None.
    The 2D reference coordinates that pair with these pixel points depend on
    the match type and are produced via `build_keypoints_2d(width_m, length_m)`;
    use `pair_with_2d(reference_2d)` to align available pixel points with the
    matching subset of a 2D reference layout.
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

    def pair_with_2d(
        self, reference_2d: list[tuple[float, float]]
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Return matched (pixel_points, court_2d_points) for available indices.

        `reference_2d` must be the 12-point layout produced by
        `build_keypoints_2d` for the target court size.
        """
        if len(reference_2d) != NUM_KEYPOINTS:
            raise ValueError(
                f"reference_2d must have {NUM_KEYPOINTS} entries, got {len(reference_2d)}"
            )
        pixel_pts: list[tuple[float, float]] = []
        court_pts: list[tuple[float, float]] = []
        for i in self.available_indices:
            pixel_pts.append(self.points[i])
            court_pts.append(reference_2d[i])
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
