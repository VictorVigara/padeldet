"""Logical player identity management: maps tracker IDs to persistent player IDs.

Court dimensions and the number of logical players come from the active
`MatchProfile` (doubles → 4 players on a 10 × 20 m court; singles → 2 players
on a 6 × 20 m court). Helpers in this module take court dimensions explicitly
so they can be reused across match types; `LogicalPlayerManager` wraps them
with the dimensions resolved from its profile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..court.models import DOOR_HALF_WIDTH_M
from ..match_profile import MatchProfile
from ..models import BoundingBox, FrameDetections, TrackedDetection
from .config import PlayerTrackerConfig


# ---------------------------------------------------------------------------
# Court-structure helpers (court dimensions passed explicitly)
# ---------------------------------------------------------------------------

def _team_side(
    court_position: np.ndarray | None,
    length_m: float,
) -> Literal["near", "far"] | None:
    if court_position is None:
        return None
    return "near" if court_position[1] < length_m / 2 else "far"


def _team_side_with_deadzone(
    court_position: np.ndarray | None,
    deadzone_m: float,
    length_m: float,
) -> Literal["near", "far"] | None:
    """Like `_team_side` but returns None inside a ±deadzone band around the net."""
    if court_position is None:
        return None
    y = float(court_position[1])
    if abs(y - length_m / 2) < deadzone_m:
        return None
    return "near" if y < length_m / 2 else "far"


def _play_side(
    court_position: np.ndarray | None,
    team_side: Literal["near", "far"] | None,
    width_m: float,
) -> Literal["left", "right"] | None:
    """Return the player's own left/right given their court position and team side.

    Near players face the camera, so their own-left = camera-left (low court x).
    Far players face away from the camera, so their own-left = camera-right (high x).
    """
    if court_position is None or team_side is None:
        return None
    cam_left = court_position[0] < width_m / 2
    own_left = cam_left if team_side == "near" else not cam_left
    return "left" if own_left else "right"


def _same_role(
    state: LogicalPlayerState,
    det: TrackedDetection,
    deadzone_m: float,
    width_m: float,
    length_m: float,
) -> bool:
    if det.court_position is None or state.team_side is None or state.play_side is None:
        return False
    pos = np.array(det.court_position, dtype=float)
    det_team = _team_side_with_deadzone(pos, deadzone_m, length_m)
    if det_team is None:
        return False
    det_play = _play_side(pos, det_team, width_m)
    return det_team == state.team_side and det_play == state.play_side


def _same_team_side(
    state: LogicalPlayerState,
    det: TrackedDetection,
    deadzone_m: float,
    length_m: float,
) -> bool:
    if det.court_position is None or state.team_side is None:
        return False
    det_team = _team_side_with_deadzone(
        np.array(det.court_position, dtype=float), deadzone_m, length_m
    )
    return det_team == state.team_side


def _within_recovery_radius(
    state: LogicalPlayerState,
    det: TrackedDetection,
    frame_index: int,
    effective_fps: float,
    window_sec: float,
    initial_radius_m: float,
    final_radius_m: float,
) -> bool:
    """Time-window distance gate for structural reassignment.

    Within the first `window_sec` seconds after a player was last seen, the
    candidate detection must lie within a growing radius of the state's last
    known court position (linear interpolation from `initial_radius_m` at
    t=0 to `final_radius_m` at t=window). Outside the window, gating is off.

    Falls open (returns True) when either side lacks a court position — the
    caller's structural check is then the only constraint.
    """
    if state.last_court_position is None or det.court_position is None:
        return True
    fps = max(effective_fps, 1e-6)
    t_sec = max(0, frame_index - state.last_seen_frame) / fps
    if window_sec <= 0 or t_sec >= window_sec:
        return True
    frac = min(max(t_sec / window_sec, 0.0), 1.0)
    radius = initial_radius_m + (final_radius_m - initial_radius_m) * frac
    dist = float(np.linalg.norm(
        np.asarray(det.court_position, dtype=float) - state.last_court_position
    ))
    return dist <= radius


def _court_depth(
    court_position: np.ndarray,
    width_m: float,
    length_m: float,
) -> float:
    """Minimum distance from the given point to any court edge (in meters)."""
    x, y = float(court_position[0]), float(court_position[1])
    return min(x, width_m - x, y, length_m - y)


def _is_valid_player_position(
    court_position: tuple[float, float],
    width_m: float,
    length_m: float,
    door_y_min: float,
    door_y_max: float,
) -> bool:
    """Return True if a court-meter position is acceptable for player assignment.

    Accepts positions inside the court or exiting through a side door.
    Rejects anything else (spectator behind far/near glass, positional noise).
    """
    x, y = float(court_position[0]), float(court_position[1])
    in_court = 0.0 <= x <= width_m and 0.0 <= y <= length_m
    door_exit = (x < 0.0 or x > width_m) and door_y_min <= y <= door_y_max
    return in_court or door_exit


def _position_sort_key(
    det: TrackedDetection,
    width_m: float,
    length_m: float,
    n_players: int,
) -> tuple[int, ...]:
    """Sort key mapping a detection to a logical-id slot.

    Doubles (n_players == 4): (team_bucket, side_bucket) with player-relative
        L/R, yielding IDs 1=near-own-left, 2=near-own-right, 3=far-own-left,
        4=far-own-right when detections are sorted and enumerated.
    Singles (n_players == 2): (team_bucket,) only — IDs 1=near, 2=far.
    """
    if det.court_position is None:
        return (1, 1) if n_players >= 4 else (1,)
    x, y = det.court_position
    if n_players <= 2:
        team_bucket = 0 if y < length_m / 2 else 1
        return (team_bucket,)
    if y < length_m / 2:
        team_bucket = 0
        side_bucket = 0 if x < width_m / 2 else 1
    else:
        team_bucket = 1
        side_bucket = 0 if x >= width_m / 2 else 1
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
    is_on_court: bool = True


# ---------------------------------------------------------------------------
# Identity manager
# ---------------------------------------------------------------------------

class LogicalPlayerManager:
    """Maps temporary tracker IDs to persistent logical player IDs 1..n_players.

    Court dimensions and player count come from the supplied `MatchProfile`:
    doubles → 4 players + L/R distinction (P1=near-left … P4=far-right);
    singles → 2 players, team-side only (P1=near, P2=far, `play_side=None`).
    """

    def __init__(
        self,
        config: PlayerTrackerConfig,
        profile: MatchProfile,
        effective_fps: float = 30.0,
    ):
        self.config = config
        self.profile = profile
        self.width_m = profile.court.width_m
        self.length_m = profile.court.length_m
        self.net_y = self.length_m / 2
        self.door_y_min = self.net_y - DOOR_HALF_WIDTH_M
        self.door_y_max = self.net_y + DOOR_HALF_WIDTH_M
        self.n_players = profile.n_players
        self.uses_play_side = self.n_players >= 4
        self.effective_fps = effective_fps
        self.states: list[LogicalPlayerState] = []
        self.team1_side: Literal["near", "far"] = "near"
        self._side_change_allowed: bool = False

    # ------------------------------------------------------------------
    # Helper wrappers — bind court dims from the profile
    # ------------------------------------------------------------------

    def _is_valid_position(self, court_position: tuple[float, float]) -> bool:
        return _is_valid_player_position(
            court_position, self.width_m, self.length_m, self.door_y_min, self.door_y_max
        )

    def _depth(self, court_position: np.ndarray) -> float:
        return _court_depth(court_position, self.width_m, self.length_m)

    def initialize(
        self,
        selected_detections: list[TrackedDetection],
        frame_index: int,
    ) -> None:
        """Create logical players 1..n_players from selected detections.

        IDs are assigned by court structure regardless of input order. For
        doubles (n=4) the layout is 1=near-own-left, 2=near-own-right,
        3=far-own-left, 4=far-own-right. For singles (n=2) the layout is
        1=near, 2=far and `play_side` stays `None`.
        """
        selected_detections = sorted(
            selected_detections,
            key=lambda d: _position_sort_key(d, self.width_m, self.length_m, self.n_players),
        )
        self.states = []
        for i, det in enumerate(selected_detections):
            court_pos = (
                np.array(det.court_position, dtype=float)
                if det.court_position is not None
                else None
            )
            team_side = _team_side(court_pos, self.length_m)
            play_side = (
                _play_side(court_pos, team_side, self.width_m)
                if self.uses_play_side
                else None
            )
            self.states.append(
                LogicalPlayerState(
                    logical_id=i + 1,
                    current_track_id=det.track_id,
                    last_bbox=det.bbox,
                    last_court_position=court_pos,
                    team_side=team_side,
                    play_side=play_side,
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
        # ReID gives strong evidence, but we still validate court position to reject
        # cases where the tracker incorrectly re-uses a player's track_id on a spectator.
        # Accept only if the detection is inside the court OR exiting through a door.
        # Any other out-of-bounds position (e.g. spectator behind far glass) is rejected
        # — the player stays missing and falls through to Steps 2-5.
        min_depth = self.config.min_direct_match_court_depth_m
        for state in self.states:
            if state.current_track_id is not None and state.current_track_id in by_tid:
                det = by_tid[state.current_track_id]
                if det.court_position is not None and not self._is_valid_position(det.court_position):
                    logging.debug(
                        f"[frame {frame_index}] direct_track_id rejected (invalid position "
                        f"{det.court_position}): logical_id={state.logical_id} tid={det.track_id}"
                    )
                    continue
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
            self._update_team_sides(result_dets)
            return self._make_frame_detections(fd.frame_index, result_dets)

        unassigned = [
            d for d in fd.detections
            if d.track_id is not None
            and d.track_id not in assigned_tids
            and (
                d.court_position is None
                or self._depth(np.array(d.court_position, dtype=float)) >= min_depth
            )
        ]

        deadzone = self.config.net_deadzone_m
        width_m = self.width_m
        length_m = self.length_m

        # Time-window distance gate (Steps 2, 3, 5): within the first
        # `recovery_gating_window_sec` after a state was last seen, reject
        # candidates farther than a growing radius from the last court position.
        gate_window = self.config.recovery_gating_window_sec
        gate_r0 = self.config.recovery_gating_initial_radius_m
        gate_r1 = self.config.recovery_gating_final_radius_m
        gate_fps = self.effective_fps

        def _gate(state: LogicalPlayerState, det: TrackedDetection) -> bool:
            return _within_recovery_radius(
                state, det, frame_index, gate_fps,
                gate_window, gate_r0, gate_r1,
            )

        # Step 2: role-based reassignment — symmetric uniqueness (same team_side + play_side)
        # Singles: states have play_side=None, so _same_role returns False — this step is a no-op.
        unassigned, missing_states = self._structural_assign(
            missing_states, unassigned, assigned_tids, result_dets, frame_index,
            match_fn=lambda s, d: _same_role(s, d, deadzone, width_m, length_m) and _gate(s, d),
            reason="same_role",
        )

        # Step 3: team-side reassignment — symmetric uniqueness (same team_side only)
        unassigned, missing_states = self._structural_assign(
            missing_states, unassigned, assigned_tids, result_dets, frame_index,
            match_fn=lambda s, d: _same_team_side(s, d, deadzone, length_m) and _gate(s, d),
            reason="same_team_side",
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
            if det.court_position is not None and not self._is_valid_position(det.court_position):
                logging.debug(
                    f"[frame {frame_index}] one_to_one_fallback rejected (invalid position "
                    f"{det.court_position}): logical_id={state.logical_id} tid={det.track_id}"
                )
            elif not _gate(state, det):
                logging.debug(
                    f"[frame {frame_index}] one_to_one_fallback rejected (recovery_gate): "
                    f"logical_id={state.logical_id} tid={det.track_id}"
                )
            else:
                self._assign(state, det, frame_index, "one_to_one_fallback", result_dets, assigned_tids)

        self._update_team_sides(result_dets)
        return self._make_frame_detections(fd.frame_index, result_dets)

    def _make_frame_detections(self, frame_index: int, result_dets: list[TrackedDetection]) -> FrameDetections:
        fd_out = FrameDetections(frame_index=frame_index, detections=result_dets)
        fd_out.team_sides = {
            1: self.team1_side,
            2: "far" if self.team1_side == "near" else "near",
        }
        # Per-frame state snapshot for debug overlay / log inspection.
        # Uses ASCII markers ('+' assigned this frame, 'x' missing) so it renders
        # in the OpenCV overlay font. Each slot shows its logical role in the
        # team/side naming the user reads by (T1-L, T1-R, T2-L, T2-R).
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            assigned_lids = {d.logical_player_id for d in result_dets}
            slot_parts = []
            for s in self.states:
                marker = "+" if s.logical_id in assigned_lids else "x"
                slot_parts.append(
                    f"{marker}{self._role_label(s)}=tid{s.current_track_id}"
                )
            gate = "open" if self._side_change_allowed else "closed"
            logging.debug(
                f"[frame {frame_index}] state[+seen x=missing]: "
                f"{' '.join(slot_parts)} gate={gate}"
            )
        return fd_out

    def _role_label(self, state: LogicalPlayerState) -> str:
        """Human-readable slot name: team + own left/right.

        Doubles: P1->T1-L, P2->T1-R, P3->T2-L, P4->T2-R (team 1 is the pair
        that started near the camera). Singles: P1->T1, P2->T2. The team number
        follows the logical id (fixed identity), while the near/far and L/R are
        the slot's *current* team_side/play_side so a stale/flipped role is
        visible at a glance (e.g. 'T1-L' sitting on the far side of the net).
        """
        if self.n_players >= 4:
            team_no = 1 if state.logical_id <= 2 else 2
            side = state.play_side[0].upper() if state.play_side else "?"
            near_far = (state.team_side or "?")[0]  # n / f / ?
            return f"P{state.logical_id}[T{team_no}-{side} {near_far}]"
        team_no = state.logical_id
        near_far = (state.team_side or "?")[0]
        return f"P{state.logical_id}[T{team_no} {near_far}]"

    def _team_id_sets(self) -> tuple[set[int], set[int]]:
        """Split logical IDs into the two team buckets.

        Doubles: {1,2} vs {3,4}. Singles: {1} vs {2}.
        """
        if self.n_players >= 4:
            return {1, 2}, {3, 4}
        return {1}, {2}

    def _update_team_sides(self, result_dets: list[TrackedDetection]) -> None:
        if not self._side_change_allowed:
            return

        in_court = [d for d in result_dets if d.is_on_court and d.court_position is not None]
        team1_ids, team2_ids = self._team_id_sets()
        net_y = self.length_m / 2

        team1_near = any(d.court_position[1] < net_y for d in in_court if d.logical_player_id in team1_ids)
        team1_far  = any(d.court_position[1] >= net_y for d in in_court if d.logical_player_id in team1_ids)
        team2_near = any(d.court_position[1] < net_y for d in in_court if d.logical_player_id in team2_ids)
        team2_far  = any(d.court_position[1] >= net_y for d in in_court if d.logical_player_id in team2_ids)

        # TODO: Here is not checked if team2 left the court to enable a side change. So probably the method mentioned in _update_state_matched TODO would be better, which is to check if on court of one of the team players is true in a different side, and the players of the team occupying that side are not on court. This would allow to change sides during the match, and also to change back to the original side if needed.
        if team1_far and not team2_far:
            if self.team1_side != "far":
                logging.info("Team side change detected: Team 1 → far, Team 2 → near")
                self._flip_state_team_sides()
            self.team1_side = "far"
            self._side_change_allowed = False
        elif team1_near and not team2_near:
            if self.team1_side != "near":
                logging.info("Team side change detected: Team 1 → near, Team 2 → far")
                self._flip_state_team_sides()
            self.team1_side = "near"
            self._side_change_allowed = False

    def _flip_state_team_sides(self) -> None:
        # play_side is player-relative (own left/right), so it does not change
        # when a team physically swaps court ends — only team_side flips.
        for state in self.states:
            if state.team_side == "near":
                state.team_side = "far"
            elif state.team_side == "far":
                state.team_side = "near"

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
            # _assign emits the step=proximity line; add just the distance detail.
            logging.info(
                f"[frame {frame_index}] step=proximity {self._role_label(state)} "
                f"dist={dists[si, di]:.2f}m"
            )
            self._assign(state, det, frame_index, "proximity", result_dets, assigned_tids)
            used_state_lids.add(state.logical_id)
            used_det_ids.add(id(det))

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
        # One consistent line per assignment naming the STEP that made it and the
        # slot's role, so the overlay shows how each id was resolved this frame.
        # direct_track_id is the common case (ReID continuity) -> DEBUG to avoid
        # spamming INFO; every recovery step (a real reassignment) -> INFO.
        role = self._role_label(state)
        if reason == "direct_track_id":
            logging.debug(
                f"[frame {frame_index}] step=direct_track_id {role} tid={det.track_id}"
            )
        else:
            logging.info(
                f"[frame {frame_index}] step={reason} {role} tid {old_tid}->{det.track_id}"
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
                state.team_side = _team_side(new_court, self.length_m)
            if self.uses_play_side and state.play_side is None:
                state.play_side = _play_side(new_court, state.team_side, self.width_m)
            x, y = float(new_court[0]), float(new_court[1])
            exited_side = x < 0.0 or x > self.width_m
            near_door = self.door_y_min <= y <= self.door_y_max
            on_court = not (exited_side and near_door)
            if state.is_on_court and not on_court:
                # TODO: This is the only condition to allow side change. And even if the player comes back to the same
                # court side, side change allowed is not set to False. Probably a player should have on_court. And then, allow
                # the change if on court of one of the team players is true in a different side, and the players of the team occupying that
                # side are not on court. This would allow to change sides during the match, and also to change back to the original side if needed.
                self._side_change_allowed = True
                logging.info(
                    f"[frame {frame_index}] side_change_gate=OPENED "
                    f"{self._role_label(state)} exited court"
                )
            state.is_on_court = on_court
            det.is_on_court = on_court
        state.last_bbox = det.bbox
        state.last_seen_frame = frame_index
        state.missing_frames = 0
