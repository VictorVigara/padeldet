"""Match-type profiles bundling player count and court dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MatchType = Literal["doubles", "singles"]


@dataclass(frozen=True)
class CourtDimensions:
    """Real-world court dimensions in meters."""

    width_m: float
    length_m: float


@dataclass(frozen=True)
class MatchProfile:
    """High-level configuration for a match type.

    Bundles the number of players to track and the court dimensions. Phase 2
    will extend this with a pluggable identity-reassignment strategy.
    """

    name: MatchType
    n_players: int
    court: CourtDimensions


DOUBLES_PROFILE = MatchProfile(
    name="doubles",
    n_players=4,
    court=CourtDimensions(width_m=10.0, length_m=20.0),
)

SINGLES_PROFILE = MatchProfile(
    name="singles",
    n_players=2,
    court=CourtDimensions(width_m=6.0, length_m=20.0),
)

MATCH_PROFILES: dict[MatchType, MatchProfile] = {
    "doubles": DOUBLES_PROFILE,
    "singles": SINGLES_PROFILE,
}


def get_profile(match_type: MatchType) -> MatchProfile:
    return MATCH_PROFILES[match_type]
