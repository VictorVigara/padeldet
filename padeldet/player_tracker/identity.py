"""Logical player identity management: maps tracker IDs to persistent player IDs 1..4."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..court.models import COURT_LENGTH, COURT_WIDTH
from ..models import BoundingBox, FrameDetections, TrackedDetection
from .config import PlayerTrackerConfig


# ---------------------------------------------------------------------------
# Court-structure helpers
# ---------------------------------------------------------------------------

def _team_side(court_position: np.ndarray | None) -> Literal["near", "far"] | None:
    if court_position is None:
        return None
    return "near" if court_position[1] < COURT_LENGTH / 2 else "far"


def _play_side(court_position: np.ndarray | None) -> Literal["left", "right"] | None:
    if court_position is None:
        return None
    return "left" if court_position[0] < COURT_WIDTH / 2 else "right"


def _detection_team_side(det: TrackedDetection) -> Literal["near", "far"] | None:
    if det.court_position is None:
        return None
    return _team_side(np.array(det.court_position, dtype=float))


def _detection_play_side(det: TrackedDetection) -> Literal["left", "right"] | None:
    if det.court_position is None:
        return None
    return _play_side(np.array(det.court_position, dtype=float))


def _same_role(state: LogicalPlayerState, det: TrackedDetection) -> bool:
    return (
        state.team_side is not None
        and state.play_side is not None
        and _detection_team_side(det) == state.team_side
        and _detection_play_side(det) == state.play_side
    )


def _same_team_side(state: LogicalPlayerState, det: TrackedDetection) -> bool:
    return state.team_side is not None and _detection_team_side(det) == state.team_side


def _court_depth(court_position: np.ndarray) -> float:
    """Minimum distance from the given point to any court edge (in meters)."""
    x, y = float(court_position[0]), float(court_position[1])
    return min(x, COURT_WIDTH - x, y, COURT_LENGTH - y)


def _position_sort_key(det: TrackedDetection) -> tuple[int, int]:
    """Sort key mapping a detection to a position bucket.

    Bucket order: (team_side, play_side) → near-left=0, near-right=1, far-left=2, far-right=3.
    When detections are sorted by this key and assigned IDs 1-4 in order the result is:
      ID 1 → near-left, ID 2 → near-right, ID 3 → far-left, ID 4 → far-right.
    """
    if det.court_position is None:
        return (1, 1)
    x, y = det.court_position
    team_bucket = 0 if y < COURT_LENGTH / 2 else 1
    side_bucket = 0 if x < COURT_WIDTH / 2 else 1
    return (team_bucket, side_bucket)


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class LogicalPlayerState:
    logical_id: int
    current_track_id: int | None
    last_bbox: BoundingBox | None
    last_court_position: np.ndarray | None  # shape (2,) in court meters
    team_side: Literal["near", "far"] | None
    play_side: Literal["left", "right"] | None
    last_seen_frame: int
    missing_frames: int = 0  # for logging/debugging only, not used for assignment gating


# ---------------------------------------------------------------------------
# Identity manager
# ---------------------------------------------------------------------------

class LogicalPlayerManager:
    """Maps temporary tracker IDs to persistent logical player IDs 1..4."""

    def __init__(self, config: PlayerTrackerConfig):
        self.config = config
        self.states: list[LogicalPlayerState] = []

    def initialize(
        self,
        selected_detections: list[TrackedDetection],
        frame_index: int,
    ) -> None:
        """Create logical players 1..4 from selected detections.

        IDs are assigned by court quadrant regardless of input order:
        1=near-left, 2=near-right, 3=far-left, 4=far-right.
        """
        selected_detections = sorted(selected_detections, key=_position_sort_key)
        self.states = []
        for i, det in enumerate(selected_detections):
            court_pos = (
                np.array(det.court_position, dtype=float)
                if det.court_position is not None
                else None
            )
            self.states.append(
                LogicalPlayerState(
                    logical_id=i + 1,
                    current_track_id=det.track_id,
                    last_bbox=det.bbox,
                    last_court_position=court_pos,
                    team_side=_team_side(court_pos),
                    play_side=_play_side(court_pos),
                    last_seen_frame=frame_index,
                )
            )
        mapping = ", ".join(
            f"{s.logical_id}→tid={s.current_track_id} "
            f"({s.team_side or '?'}/{s.play_side or '?'})"
            for s in self.states
        )
        logging.info(f"Logical player mapping: [{mapping}]")

    def update(self, fd: FrameDetections) -> FrameDetections:
        """Assign this frame's detections to logical players; return only assigned ones."""
        if not self.states:
            return FrameDetections(frame_index=fd.frame_index, detections=[])

        frame_index = fd.frame_index
        by_tid: dict[int, TrackedDetection] = {
            d.track_id: d for d in fd.detections if d.track_id is not None
        }
        assigned_tids: set[int] = set()
        result_dets: list[TrackedDetection] = []
        matched_lids: set[int] = set()

        # Step 1: direct track_id matching
        # Trust DeepOCSORT's ReID here unconditionally — if the same track_id comes
        # back, the tracker has visual evidence it's the same person. The court-edge
        # depth filter only applies in Steps 2-5, where we're blindly assigning a
        # new track_id to a logical player by position/role.
        min_depth = self.config.min_direct_match_court_depth_m
        for state in self.states:
            if state.current_track_id is not None and state.current_track_id in by_tid:
                det = by_tid[state.current_track_id]
                assigned_tids.add(state.current_track_id)
                matched_lids.add(state.logical_id)
                self._assign(state, det, frame_index, "direct_track_id", result_dets, assigned_tids)
                logging.debug(
                    f"[frame {frame_index}] direct_track_id: "
                    f"logical_id={state.logical_id} tid={det.track_id}"
                )

        # Collect missing states and unassigned detections; increment missing_frames for logging
        missing_states = []
        for state in self.states:
            if state.logical_id not in matched_lids:
                state.missing_frames += 1
                missing_states.append(state)

        if not missing_states:
            return FrameDetections(frame_index=fd.frame_index, detections=result_dets)

        unassigned = [
            d for d in fd.detections
            if d.track_id is not None
            and d.track_id not in assigned_tids
            and (
                d.court_position is None
                or _court_depth(np.array(d.court_position, dtype=float)) >= min_depth
            )
        ]

        # Step 2: role-based reassignment — symmetric uniqueness (same team_side + play_side)
        unassigned, missing_states = self._structural_assign(
            missing_states, unassigned, assigned_tids, result_dets, frame_index,
            match_fn=_same_role, reason="same_role",
        )

        # Step 3: team-side reassignment — symmetric uniqueness (same team_side only)
        unassigned, missing_states = self._structural_assign(
            missing_states, unassigned, assigned_tids, result_dets, frame_index,
            match_fn=_same_team_side, reason="same_team_side",
        )

        # Step 4: proximity fallback — nearest last-known court position
        unassigned, missing_states = self._proximity_assign(
            missing_states, unassigned, assigned_tids, result_dets, frame_index,
            max_distance_m=self.config.proximity_max_distance_m,
        )

        # Step 5: one-to-one fallback
        if len(missing_states) == 1 and len(unassigned) == 1:
            state = missing_states[0]
            det = unassigned[0]
            self._assign(state, det, frame_index, "one_to_one_fallback", result_dets, assigned_tids)

        return FrameDetections(frame_index=fd.frame_index, detections=result_dets)

    def _structural_assign(
        self,
        missing_states: list[LogicalPlayerState],
        unassigned: list[TrackedDetection],
        assigned_tids: set[int],
        result_dets: list[TrackedDetection],
        frame_index: int,
        match_fn: object,
        reason: str,
    ) -> tuple[list[TrackedDetection], list[LogicalPlayerState]]:
        """Assign pairs where both sides have exactly one structural match (symmetric uniqueness)."""
        to_assign: list[tuple[LogicalPlayerState, TrackedDetection]] = []
        for state in missing_states:
            candidates = [d for d in unassigned if match_fn(state, d)]
            if len(candidates) != 1:
                continue
            det = candidates[0]
            reverse = [s for s in missing_states if match_fn(s, det)]
            if len(reverse) == 1:
                to_assign.append((state, det))

        # Deduplicate in case two states both uniquely matched the same detection
        used_det_ids: set[int] = set()
        used_state_lids: set[int] = set()
        for state, det in to_assign:
            if state.logical_id in used_state_lids or id(det) in used_det_ids:
                continue
            self._assign(state, det, frame_index, reason, result_dets, assigned_tids)
            used_state_lids.add(state.logical_id)
            used_det_ids.add(id(det))

        remaining_states = [s for s in missing_states if s.logical_id not in used_state_lids]
        remaining_dets = [d for d in unassigned if d.track_id not in assigned_tids]
        return remaining_dets, remaining_states

    def _proximity_assign(
        self,
        missing_states: list[LogicalPlayerState],
        unassigned: list[TrackedDetection],
        assigned_tids: set[int],
        result_dets: list[TrackedDetection],
        frame_index: int,
        max_distance_m: float,
    ) -> tuple[list[TrackedDetection], list[LogicalPlayerState]]:
        """Greedy nearest-court-position matching for remaining unmatched states."""
        # Only consider states and detections that have court positions
        eligible_states = [s for s in missing_states if s.last_court_position is not None]
        eligible_dets = [d for d in unassigned if d.court_position is not None]

        if not eligible_states or not eligible_dets:
            return unassigned, missing_states

        # Build distance matrix (states × detections)
        state_pts = np.array([s.last_court_position for s in eligible_states])  # (S, 2)
        det_pts = np.array([d.court_position for d in eligible_dets], dtype=float)  # (D, 2)
        dists = np.linalg.norm(state_pts[:, None, :] - det_pts[None, :, :], axis=2)  # (S, D)

        used_state_lids: set[int] = set()
        used_det_ids: set[int] = set()

        # Greedy: repeatedly pick the globally closest remaining pair
        while True:
            # Mask already-used rows/cols
            available = np.ones(dists.shape, dtype=bool)
            for si, s in enumerate(eligible_states):
                if s.logical_id in used_state_lids:
                    available[si, :] = False
            for di, d in enumerate(eligible_dets):
                if id(d) in used_det_ids:
                    available[:, di] = False

            if not available.any():
                break

            masked = np.where(available, dists, np.inf)
            si, di = np.unravel_index(np.argmin(masked), masked.shape)
            if masked[si, di] > max_distance_m:
                break

            state = eligible_states[si]
            det = eligible_dets[di]
            self._assign(state, det, frame_index, "proximity", result_dets, assigned_tids)
            used_state_lids.add(state.logical_id)
            used_det_ids.add(id(det))
            logging.info(
                f"[frame {frame_index}] proximity: logical_id={state.logical_id} "
                f"tid→{det.track_id} dist={dists[si, di]:.2f}m"
            )

        remaining_states = [s for s in missing_states if s.logical_id not in used_state_lids]
        remaining_dets = [d for d in unassigned if d.track_id not in assigned_tids]
        return remaining_dets, remaining_states

    def _assign(
        self,
        state: LogicalPlayerState,
        det: TrackedDetection,
        frame_index: int,
        reason: str,
        result_dets: list[TrackedDetection],
        assigned_tids: set[int],
    ) -> None:
        old_tid = state.current_track_id
        state.current_track_id = det.track_id
        self._update_state_matched(state, det, frame_index)
        det.logical_player_id = state.logical_id
        det.visible = True
        assigned_tids.add(det.track_id)
        result_dets.append(det)
        if reason != "direct_track_id":
            logging.info(
                f"[frame {frame_index}] {reason}: logical_id={state.logical_id} "
                f"tid {old_tid}→{det.track_id}"
            )

    def _update_state_matched(
        self,
        state: LogicalPlayerState,
        det: TrackedDetection,
        frame_index: int,
    ) -> None:
        new_court = (
            np.array(det.court_position, dtype=float)
            if det.court_position is not None
            else None
        )
        if new_court is not None:
            state.last_court_position = new_court
            # TODO: adaptive role updates using a long temporal window or point-level context
            if state.team_side is None:
                state.team_side = _team_side(new_court)
            if state.play_side is None:
                state.play_side = _play_side(new_court)
        state.last_bbox = det.bbox
        state.last_seen_frame = frame_index
        state.missing_frames = 0
