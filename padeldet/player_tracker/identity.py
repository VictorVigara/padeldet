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


def _median_positions(
    window_results: list[FrameDetections],
    track_ids: set[int],
) -> dict[int, np.ndarray]:
    """Median court position (x, y) per track id over the selection window.

    Only in-window detections that carry a court position count. Returns a
    {track_id: array([median_x, median_y])} map; track ids never seen with a
    court position are omitted. The median makes the seed robust to a transient
    stack/cross at the exact selection frame.
    """
    xs: dict[int, list[float]] = {tid: [] for tid in track_ids}
    ys: dict[int, list[float]] = {tid: [] for tid in track_ids}
    for fd in window_results:
        for d in fd.detections:
            if d.track_id in track_ids and d.court_position is not None:
                xs[d.track_id].append(float(d.court_position[0]))
                ys[d.track_id].append(float(d.court_position[1]))
    out: dict[int, np.ndarray] = {}
    for tid in track_ids:
        if xs[tid]:
            out[tid] = np.array([float(np.median(xs[tid])), float(np.median(ys[tid]))])
    return out


def _own_left_camera_x_smaller(team_side: Literal["near", "far"]) -> bool:
    """Whether a player's own-left corresponds to the smaller camera-x.

    Near players face the camera → own-left = camera-left = smaller x.
    Far players face away → own-left = camera-right = larger x.
    """
    return team_side == "near"


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
    # Per-logical-player EMA appearance embedding (ReID gallery). Transient, built
    # live from confident matches; used to re-attach identities after an empty-court
    # break. None until the player is first seen with an embedding.
    emb_ema: np.ndarray | None = None
    # Frozen ANCHOR embedding, seeded from the early (clean) frames and updated
    # only very slowly. The anchor cannot be poisoned by a transient mislabel, so
    # matching against it lets a wrong id self-correct the next frame instead of
    # locking in. None until enough clean samples accumulate.
    emb_anchor: np.ndarray | None = None
    _anchor_samples: int = 0  # count folded into the anchor (freezes after a cap)


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
        # Break state: suspend recovery and freeze identities while the court is
        # empty (changeover / injury / between points), then re-attach by ReID on
        # return since teams may have swapped ends while empty.
        self.in_break: bool = False
        self._empty_streak_frames: int = 0
        self._break_min_empty_frames: int = max(
            1, round(self.config.break_min_empty_sec * self.effective_fps)
        )

    # ------------------------------------------------------------------
    # Helper wrappers — bind court dims from the profile
    # ------------------------------------------------------------------

    def _is_valid_position(self, court_position: tuple[float, float]) -> bool:
        return _is_valid_player_position(
            court_position, self.width_m, self.length_m, self.door_y_min, self.door_y_max
        )

    def _depth(self, court_position: np.ndarray) -> float:
        return _court_depth(court_position, self.width_m, self.length_m)

    def _det_team_side(self, det: TrackedDetection) -> Literal["near", "far"] | None:
        """A detection's team side via the net dead-zone; None if unknown.

        Single source of truth for which half of the net a detection is on.
        Returns None when the detection has no court position or lies inside the
        ±net_deadzone_m band (side ambiguous — must not be teamed).
        """
        if det.court_position is None:
            return None
        return _team_side_with_deadzone(
            np.array(det.court_position, dtype=float),
            self.config.net_deadzone_m,
            self.length_m,
        )

    def _in_door_zone(self, det: TrackedDetection) -> bool:
        """True if a detection is in the door y-band OR near any court edge.

        Near the door/edges all players pile up at the same location, so position
        is meaningless there — assignment must be ReID-only. Geometric steps
        (proximity, one-to-one, direct-by-position) must never assign here.
        """
        if det.court_position is None:
            return True  # unknown position → treat as ambiguous, ReID-only
        x, y = float(det.court_position[0]), float(det.court_position[1])
        m = self.config.door_zone_edge_margin_m
        in_door_band = self.door_y_min <= y <= self.door_y_max
        near_edge = (x < m or x > self.width_m - m or y < m or y > self.length_m - m)
        return in_door_band or near_edge

    def _team_matches(self, state: LogicalPlayerState, det: TrackedDetection) -> bool:
        """Hard team invariant: det may fill this slot only on the slot's own side.

        A detection whose team side is unknown (no position / in dead-zone) or on
        the opposite side of the net is rejected — recovery never crosses the net.
        """
        ts = self._det_team_side(det)
        return ts is not None and ts == state.team_side

    # ------------------------------------------------------------------
    # Break detection + ReID gallery helpers
    # ------------------------------------------------------------------

    def _count_valid_in_court(self, dets: list[TrackedDetection]) -> int:
        """Number of detections at a valid in-court position (the break signal)."""
        n = 0
        for d in dets:
            if d.court_position is not None and self._is_valid_position(d.court_position):
                n += 1
        return n

    def _make_empty_frame(self, frame_index: int) -> FrameDetections:
        """Empty FrameDetections that still carries team_sides (display-only)."""
        fd_out = FrameDetections(frame_index=frame_index, detections=[])
        fd_out.team_sides = {
            1: self.team1_side,
            2: "far" if self.team1_side == "near" else "near",
        }
        return fd_out

    def _update_gallery(self, state: LogicalPlayerState, det: TrackedDetection) -> None:
        """Fold a detection's embedding into the slot's EMA gallery.

        Mirrors boxmot's update_emb: emb = a*emb + (1-a)*e, renormalized. No-ops
        on a missing / zero-norm embedding (e.g. bytetrack), so ReID stays optional.
        """
        e = det.embedding
        if e is None:
            return
        e = np.asarray(e, dtype=np.float32)
        n = float(np.linalg.norm(e))
        if n < 1e-6:
            return
        e = e / n
        # Anchor: average the first `anchor_cap` clean samples, then FREEZE. This
        # reference cannot be poisoned by a later mislabel, so appearance matching
        # against it self-corrects a wrong id instead of locking it in.
        cap = self.config.reid_anchor_samples
        if state._anchor_samples < cap:
            if state.emb_anchor is None:
                state.emb_anchor = e.copy()
            else:
                k = state._anchor_samples
                v = (state.emb_anchor * k + e) / (k + 1)
                vn = float(np.linalg.norm(v))
                state.emb_anchor = v / vn if vn > 1e-6 else state.emb_anchor
            state._anchor_samples += 1
        # EMA (fast, adapts to lighting/pose); used as a secondary signal.
        if state.emb_ema is None:
            state.emb_ema = e.copy()
            return
        a = self.config.reid_gallery_alpha
        v = a * state.emb_ema + (1.0 - a) * e
        vn = float(np.linalg.norm(v))
        if vn > 1e-6:
            state.emb_ema = v / vn

    def _match_greedy_or_hungarian(
        self, sim: np.ndarray, min_cosine: float
    ) -> list[tuple[int, int]]:
        """Assign rows→cols maximizing cosine similarity; drop pairs < min_cosine.

        Uses scipy's Hungarian solver when available, else a greedy argmax.
        `sim` is (n_rows, n_cols). Returns a list of (row, col) index pairs.
        """
        if sim.size == 0:
            return []
        pairs: list[tuple[int, int]] = []
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(-sim)
            for r, c in zip(rows, cols):
                if sim[r, c] >= min_cosine:
                    pairs.append((int(r), int(c)))
            return pairs
        except Exception:
            pass
        # Greedy fallback: repeatedly take the global best remaining pair.
        s = sim.copy()
        while True:
            r, c = np.unravel_index(int(np.argmax(s)), s.shape)
            if s[r, c] < min_cosine or not np.isfinite(s[r, c]):
                break
            pairs.append((int(r), int(c)))
            s[r, :] = -np.inf
            s[:, c] = -np.inf
            if not np.isfinite(s).any():
                break
        return pairs

    def _reattach_after_break(
        self, fd: FrameDetections, frame_index: int
    ) -> list[TrackedDetection]:
        """Appearance re-attach on break exit; [] while the court is still empty.

        Teams may have swapped ends during an empty-court break, so returning
        tracks are matched to logical players by cosine similarity to each slot's
        EMA gallery — NOT by position. Matched states have their `team_side`
        re-derived from the new court position (play_side stays frozen). Any
        unmatched candidates / states fall through to the normal cascade the same
        frame. Falls back to geometric team-constrained recovery (via the normal
        cascade) when galleries or candidate embeddings are unavailable.
        """
        cands = [
            d for d in fd.detections
            if d.track_id is not None
            and d.court_position is not None
            and self._is_valid_position(d.court_position)
        ]
        if not cands:
            return []  # court still empty → caller keeps the break frozen

        states_with_emb = [s for s in self.states if s.emb_ema is not None]
        cand_with_emb = [
            d for d in cands
            if d.embedding is not None and float(np.linalg.norm(d.embedding)) > 1e-6
        ]

        if states_with_emb and cand_with_emb:
            G = np.stack([s.emb_ema for s in states_with_emb]).astype(np.float32)  # (S,512)
            D = np.stack([
                np.asarray(d.embedding, dtype=np.float32)
                / float(np.linalg.norm(d.embedding))
                for d in cand_with_emb
            ])  # (C,512)
            sim = G @ D.T  # (S,C) cosine (both L2-normalized)
            pairs = self._match_greedy_or_hungarian(
                sim, self.config.reid_reattach_min_cosine
            )
            result_dets: list[TrackedDetection] = []
            assigned_tids: set[int] = set()
            for si, ci in pairs:
                state, det = states_with_emb[si], cand_with_emb[ci]
                # Teams may have swapped ends: re-derive team_side from the new
                # position (None → re-seeded inside _update_state_matched).
                state.team_side = self._det_team_side(det)
                logging.info(
                    f"[frame {frame_index}] step=reid_reattach {self._role_label(state)} "
                    f"tid->{det.track_id} cos={sim[si, ci]:.2f}"
                )
                self._assign(state, det, frame_index, "reid_reattach", result_dets, assigned_tids)
            return cands  # non-empty → signals break exit; leftovers handled by cascade
        else:
            logging.warning(
                f"[frame {frame_index}] break exit with no usable embeddings — "
                f"geometric re-attach fallback (teams may be mislabelled if they swapped ends)"
            )
            return cands

    def _appearance_primary_assign(
        self,
        fd: FrameDetections,
        frame_index: int,
        assigned_tids: set[int],
        result_dets: list[TrackedDetection],
        matched_lids: set[int],
    ) -> None:
        """Assign detections to logical ids by gallery cosine, ahead of geometry.

        Runs each frame once every logical player has a mature gallery. Builds the
        (players x detections) cosine matrix, Hungarian-matches, and accepts pairs
        above `reid_primary_min_cosine`. Position is IGNORED for the match, so a
        player who crossed to the other end still gets their own id (fixes the
        end-swap inversion). For each accepted pair the state's `team_side` is
        re-derived from the NEW position (atomic per-player), and the gallery is
        updated only on a high-confidence match. Unmatched ids/detections fall
        through to the direct-track-id + geometric cascade.
        """
        # TWO-TIER, TEAM-CONSTRAINED appearance assignment.
        # Team membership (P1/P2 vs P3/P4) is FIXED identity — a player can only
        # ever be confused with their TEAMMATE, never an opponent. Only the court
        # SIDE flips on an end-swap. So:
        #   Tier 1: assign each side's detections to a TEAM atomically, by which
        #           team's galleries collectively fit that side best (appearance,
        #           not position). This moves a whole team across the net as a unit.
        #   Tier 2: within a team+side, match its two detections to its two ids.
        # A detection is therefore never assigned to an opponent's id.
        if not self.uses_play_side:
            return self._appearance_primary_singles(
                fd, frame_index, assigned_tids, result_dets, matched_lids)

        if any(self._ref_gallery(s) is None for s in self.states):
            return  # galleries not mature yet → let geometry run
        cands = [
            d for d in fd.detections
            if d.track_id is not None and d.track_id not in assigned_tids
            and d.embedding is not None and d.court_position is not None
            and self._is_valid_position(d.court_position)
            and float(np.linalg.norm(d.embedding)) > 1e-6
        ]
        if not cands:
            return

        team1_ids, team2_ids = self._team_id_sets()      # {1,2}, {3,4}
        st = {s.logical_id: s for s in self.states}
        # unit-normalized candidate embeddings
        emb = {id(d): np.asarray(d.embedding, np.float32) / float(np.linalg.norm(d.embedding))
               for d in cands}

        def team_fit(det, ids) -> float:
            """Best cosine of this detection to either gallery of a team."""
            gs = [self._ref_gallery(st[i]) for i in ids]
            return max(float(g @ emb[id(det)]) for g in gs if g is not None)

        # Split detections by side (near/far). Dead-zone side is unknown → side=None.
        by_side: dict[str, list] = {"near": [], "far": [], None: []}
        for d in cands:
            by_side[self._det_team_side(d)].append(d)

        # Tier 1: decide which TEAM occupies each side by total gallery fit.
        near_g, far_g = by_side["near"], by_side["far"]
        def group_fit(group, ids):
            return sum(team_fit(d, ids) for d in group) if group else 0.0
        # option A: team1 near / team2 far ; option B: team1 far / team2 near
        fitA = group_fit(near_g, team1_ids) + group_fit(far_g, team2_ids)
        fitB = group_fit(far_g, team1_ids) + group_fit(near_g, team2_ids)
        if fitA >= fitB:
            side_team = {"near": team1_ids, "far": team2_ids}
        else:
            side_team = {"near": team2_ids, "far": team1_ids}

        margin = self.config.reid_primary_margin
        min_cos = self.config.reid_primary_min_cosine

        # Tier 2: within each side, match that side's detections to that team's ids.
        for side in ("near", "far"):
            group = by_side[side]
            ids = sorted(side_team[side])
            if not group:
                continue
            # cosine matrix (team ids x dets-on-this-side)
            gal = [self._ref_gallery(st[i]) for i in ids]
            sim = np.array([[float(g @ emb[id(d)]) for d in group] for g in gal])  # (2, G)
            pairs = self._match_greedy_or_hungarian(sim, min_cos)
            for ri, ci in pairs:
                lid = ids[ri]
                state, det = st[lid], group[ci]
                if state.logical_id in matched_lids or det.track_id in assigned_tids:
                    continue
                cos = float(sim[ri, ci])
                # within-team clarity: must beat the TEAMMATE for this det by margin
                col = sim[:, ci].copy(); col[ri] = -np.inf
                if col.size and cos - float(col.max()) < margin:
                    continue  # ambiguous between teammates → leave to geometry/direct
                new_side = self._det_team_side(det)
                swapped = new_side is not None and state.team_side is not None and new_side != state.team_side
                if new_side is not None:
                    state.team_side = new_side
                matched_lids.add(state.logical_id)
                self._assign(
                    state, det, frame_index, "reid_primary", result_dets, assigned_tids,
                    gallery_ok=cos >= self.config.reid_gallery_update_min_cosine,
                )
                if swapped:
                    logging.info(
                        f"[frame {frame_index}] step=reid_primary SIDE-FOLLOW "
                        f"{self._role_label(state)} tid={det.track_id} cos={cos:.2f}"
                    )

    def _appearance_primary_singles(
        self, fd, frame_index, assigned_tids, result_dets, matched_lids
    ) -> None:
        """Singles: 2 players, no teams/L-R. Global gallery match with margin."""
        states = [s for s in self.states if self._ref_gallery(s) is not None]
        if len(states) < self.n_players:
            return
        cands = [
            d for d in fd.detections
            if d.track_id is not None and d.track_id not in assigned_tids
            and d.embedding is not None and d.court_position is not None
            and self._is_valid_position(d.court_position)
            and float(np.linalg.norm(d.embedding)) > 1e-6
        ]
        if not cands:
            return
        G = np.stack([self._ref_gallery(s) for s in states]).astype(np.float32)
        D = np.stack([np.asarray(d.embedding, np.float32) / float(np.linalg.norm(d.embedding))
                      for d in cands])
        sim = G @ D.T
        pairs = self._match_greedy_or_hungarian(sim, self.config.reid_primary_min_cosine)
        margin = self.config.reid_primary_margin
        for si, ci in pairs:
            state, det = states[si], cands[ci]
            if state.logical_id in matched_lids or det.track_id in assigned_tids:
                continue
            cos = float(sim[si, ci])
            col = sim[:, ci].copy(); col[si] = -np.inf
            if col.size and cos - float(col.max()) < margin:
                continue
            ns = self._det_team_side(det)
            if ns is not None:
                state.team_side = ns
            matched_lids.add(state.logical_id)
            self._assign(state, det, frame_index, "reid_primary", result_dets, assigned_tids,
                         gallery_ok=cos >= self.config.reid_gallery_update_min_cosine)

    def _ref_gallery(self, state: LogicalPlayerState) -> np.ndarray | None:
        """Reference embedding for matching: the frozen anchor if built, else EMA."""
        return state.emb_anchor if state.emb_anchor is not None else state.emb_ema

    def _appearance_agrees(self, state: LogicalPlayerState, det: TrackedDetection) -> bool:
        """Does this detection's appearance confirm it is `state` (not another player)?

        Used to validate a door-zone match. True when the detection's embedding is
        closest to `state`'s reference gallery (by `reid_primary_margin`) and above
        `reid_primary_min_cosine`. When no galleries exist yet (cold start) or the
        detection has no embedding (bytetrack), returns True so ReID does not block
        a path it cannot inform.
        """
        ref = self._ref_gallery(state)
        any_gallery = any(self._ref_gallery(s) is not None for s in self.states)
        if not any_gallery:
            return True  # no ReID in play at all (e.g. bytetrack) → don't block
        others = [self._ref_gallery(s) for s in self.states if s.logical_id != state.logical_id]
        others = [g for g in others if g is not None]
        e = det.embedding
        if ref is None or e is None:
            # ReID IS in play but this detection can't be confirmed → don't trust
            # it in the door zone (leave unassigned rather than risk a mislabel).
            return False
        e = np.asarray(e, dtype=np.float32)
        n = float(np.linalg.norm(e))
        if n < 1e-6:
            return True
        e = e / n
        cos = float(ref @ e)
        if cos < self.config.reid_primary_min_cosine:
            return False
        if others:
            best_other = max(float(g @ e) for g in others)
            if cos - best_other < self.config.reid_primary_margin:
                return False  # ambiguous with another player → don't trust here
        return True

    def initialize(
        self,
        selected_detections: list[TrackedDetection],
        frame_index: int,
        window_results: list[FrameDetections] | None = None,
    ) -> None:
        """Create logical players 1..n_players from selected detections.

        Layout: doubles → 1=near-own-left, 2=near-own-right, 3=far-own-left,
        4=far-own-right; singles → 1=near, 2=far (`play_side` stays `None`).

        Team side comes from court-y; within a team the own-left/right is
        seeded **relative to the teammate** (not an absolute court-center
        threshold), so two players who are momentarily on the same half at the
        selection instant still get distinct sides. When `window_results` is
        given, each track's representative position is its **median** over the
        selection window (robust to a transient stack/cross); otherwise the
        per-detection position is used. The seeded L/R is an identity: it is
        frozen here and rides with the tracked player through in-game movement
        (a player briefly crossing sides is NOT relabelled).
        """
        # Representative position per selected track: median over the window if
        # available, else the single selection-frame court position.
        track_ids = {d.track_id for d in selected_detections if d.track_id is not None}
        rep_pos: dict[int, np.ndarray] = {}
        if window_results is not None:
            rep_pos = _median_positions(window_results, track_ids)
        for d in selected_detections:
            if d.track_id is not None and d.track_id not in rep_pos and d.court_position is not None:
                rep_pos[d.track_id] = np.array(d.court_position, dtype=float)

        by_tid = {d.track_id: d for d in selected_detections if d.track_id is not None}
        self.states = []

        if self.uses_play_side:
            self._init_doubles(by_tid, rep_pos, frame_index)
        else:
            self._init_singles(by_tid, rep_pos, frame_index)

        mapping = ", ".join(
            f"{s.logical_id}→tid={s.current_track_id} ({self._role_label(s)})"
            for s in self.states
        )
        logging.info(f"Logical player mapping: [{mapping}]")

    def _init_doubles(
        self,
        by_tid: dict[int, TrackedDetection],
        rep_pos: dict[int, np.ndarray],
        frame_index: int,
    ) -> None:
        """Seed P1..P4 = near-left, near-right, far-left, far-right.

        Team by median-y; within each team, own-left/right by ranking the two
        teammates' own-frame camera-x against each other. A team with only one
        present player gets that player plus an empty (unassigned) slot for the
        missing teammate on the opposite side.
        """
        net_y = self.length_m / 2
        near_tids = [t for t, p in rep_pos.items() if p[1] < net_y]
        far_tids = [t for t, p in rep_pos.items() if p[1] >= net_y]

        # logical_id base per (team, side): near-left=1, near-right=2, far-left=3, far-right=4
        for team_side, tids, base in (("near", near_tids, 1), ("far", far_tids, 3)):
            left_smaller_x = _own_left_camera_x_smaller(team_side)
            # Resolve which tid fills the own-left vs own-right slot.
            slot_tid: dict[int, int | None] = {0: None, 1: None}  # 0=left, 1=right
            if len(tids) >= 2:
                # Two teammates: rank against EACH OTHER so they get distinct
                # sides regardless of which half they cluster on.
                ordered = sorted(tids, key=lambda t: rep_pos[t][0], reverse=not left_smaller_x)
                slot_tid[0], slot_tid[1] = ordered[0], ordered[1]
            elif len(tids) == 1:
                # One teammate present: no peer to rank against, so fall back to
                # the absolute court-half as the best guess; the missing teammate
                # takes the opposite (empty) slot.
                only = tids[0]
                own_left = _play_side(rep_pos[only], team_side, self.width_m) == "left"
                slot_tid[0 if own_left else 1] = only
            for side_idx in (0, 1):  # 0=left slot, 1=right slot
                logical_id = base + side_idx
                play_side = "left" if side_idx == 0 else "right"
                tid = slot_tid[side_idx]
                det = by_tid.get(tid) if tid is not None else None
                court_pos = (
                    np.array(det.court_position, dtype=float)
                    if det is not None and det.court_position is not None
                    else None
                )
                self.states.append(
                    LogicalPlayerState(
                        logical_id=logical_id,
                        current_track_id=tid,
                        last_bbox=det.bbox if det is not None else None,
                        last_court_position=court_pos,
                        team_side=team_side if tid is not None else None,
                        play_side=play_side,  # frozen identity; kept even if slot empty at init
                        last_seen_frame=frame_index,
                    )
                )
        self.states.sort(key=lambda s: s.logical_id)

    def _init_singles(
        self,
        by_tid: dict[int, TrackedDetection],
        rep_pos: dict[int, np.ndarray],
        frame_index: int,
    ) -> None:
        """Seed P1=near, P2=far (no play_side)."""
        net_y = self.length_m / 2
        near_tids = [t for t, p in rep_pos.items() if p[1] < net_y]
        far_tids = [t for t, p in rep_pos.items() if p[1] >= net_y]
        for team_side, tids, logical_id in (("near", near_tids, 1), ("far", far_tids, 2)):
            tid = tids[0] if tids else None
            det = by_tid.get(tid) if tid is not None else None
            court_pos = (
                np.array(det.court_position, dtype=float)
                if det is not None and det.court_position is not None
                else None
            )
            self.states.append(
                LogicalPlayerState(
                    logical_id=logical_id,
                    current_track_id=tid,
                    last_bbox=det.bbox if det is not None else None,
                    last_court_position=court_pos,
                    team_side=team_side if tid is not None else None,
                    play_side=None,
                    last_seen_frame=frame_index,
                )
            )
        self.states.sort(key=lambda s: s.logical_id)

    def update(self, fd: FrameDetections) -> FrameDetections:
        """Assign this frame's detections to logical players; return only assigned ones."""
        if not self.states:
            return FrameDetections(frame_index=fd.frame_index, detections=[])

        frame_index = fd.frame_index
        assigned_tids: set[int] = set()
        result_dets: list[TrackedDetection] = []
        matched_lids: set[int] = set()

        # BREAK-HOLD short-circuit: while in a break, all logical state is frozen —
        # no recovery, no last_seen bump. Try to re-attach returning players by
        # appearance (teams may have swapped ends). If the court is still empty,
        # emit an empty frame and stay frozen.
        if self.in_break:
            returning = self._reattach_after_break(fd, frame_index)
            if not returning:
                return self._make_empty_frame(frame_index)
            self.in_break = False
            self._empty_streak_frames = 0
            logging.info(f"[frame {frame_index}] BREAK exited (players returned)")
            # reid_reattach already assigned matched slots; seed the tracking maps
            # and let the normal cascade below fill any remaining slots this frame.
            for d in fd.detections:
                if d.logical_player_id is not None and d.track_id is not None:
                    assigned_tids.add(d.track_id)
                    matched_lids.add(d.logical_player_id)
                    result_dets.append(d)

        # Step 0: appearance-primary assignment (authority over geometry).
        # When every logical player has a mature gallery, match this frame's valid
        # in-court detections to logical ids by gallery cosine, ignoring position.
        # This is what carries identities correctly through an end-swap, which the
        # geometric cascade inverts. Anything not confidently matched falls through
        # to the direct-track-id / geometric steps below.
        if self.config.reid_primary and not self.in_break:
            self._appearance_primary_assign(
                fd, frame_index, assigned_tids, result_dets, matched_lids
            )

        by_tid: dict[int, TrackedDetection] = {
            d.track_id: d for d in fd.detections if d.track_id is not None
        }

        # Step 1: direct track_id matching
        # ReID gives strong evidence, but we still validate court position to reject
        # cases where the tracker incorrectly re-uses a player's track_id on a spectator.
        # Accept only if the detection is inside the court OR exiting through a door.
        # Any other out-of-bounds position (e.g. spectator behind far glass) is rejected
        # — the player stays missing and falls through to Steps 2-5.
        min_depth = self.config.min_direct_match_court_depth_m
        for state in self.states:
            if state.logical_id in matched_lids:
                continue  # already re-attached this frame on break exit
            if state.current_track_id is not None and state.current_track_id in by_tid:
                det = by_tid[state.current_track_id]
                if det.track_id in assigned_tids:
                    continue  # this det was consumed by a break-exit re-attach
                if det.court_position is not None and not self._is_valid_position(det.court_position):
                    pos = f"({det.court_position[0]:.1f},{det.court_position[1]:.1f})"
                    logging.info(
                        f"[frame {frame_index}] step=direct_track_id REJECTED(invalid_position {pos}) "
                        f"{self._role_label(state)} tid={det.track_id}"
                    )
                    continue
                # In the door zone a tracker id alone is not trusted (that is how a
                # transitioning player inherits a wrong label). Require appearance to
                # AGREE: the detection must match this player's anchor gallery, and
                # not match another player's anchor better.
                if self._in_door_zone(det) and not self._appearance_agrees(state, det):
                    logging.info(
                        f"[frame {frame_index}] step=direct_track_id REJECTED(door_zone_reid) "
                        f"{self._role_label(state)} tid={det.track_id}"
                    )
                    continue
                assigned_tids.add(state.current_track_id)
                matched_lids.add(state.logical_id)
                # _assign emits the accepted 'step=direct_track_id {role} tid=...' line.
                self._assign(state, det, frame_index, "direct_track_id", result_dets, assigned_tids)

        # Collect missing states and unassigned detections; increment missing_frames for logging
        missing_states = []
        for state in self.states:
            if state.logical_id not in matched_lids:
                state.missing_frames += 1
                missing_states.append(state)

        if not missing_states:
            return self._finalize(frame_index, result_dets)

        # Door-zone detections are EXCLUDED from the geometric cascade: near the
        # door/edges everyone piles up at the same spot, so position-based
        # assignment (Steps 2-5) would mislabel a transitioning player (the P4->P1
        # bug). They can only be claimed by appearance-primary (already run above);
        # otherwise they stay unassigned this frame and re-acquire once clearly
        # inside the court.
        unassigned = [
            d for d in fd.detections
            if d.track_id is not None
            and d.track_id not in assigned_tids
            and not self._in_door_zone(d)
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
                pos = f"({det.court_position[0]:.1f},{det.court_position[1]:.1f})"
                logging.info(
                    f"[frame {frame_index}] step=one_to_one_fallback REJECTED(invalid_position {pos}) "
                    f"{self._role_label(state)} tid={det.track_id}"
                )
            elif not _gate(state, det):
                logging.info(
                    f"[frame {frame_index}] step=one_to_one_fallback REJECTED(recovery_gate) "
                    f"{self._role_label(state)} tid={det.track_id}"
                )
            elif not self._team_matches(state, det):
                # Hard team invariant — never let the last-resort fallback cross the net.
                logging.info(
                    f"[frame {frame_index}] step=one_to_one_fallback REJECTED(cross_team "
                    f"det_side={self._det_team_side(det) or '?'}) {self._role_label(state)} tid={det.track_id}"
                )
            else:
                self._assign(state, det, frame_index, "one_to_one_fallback", result_dets, assigned_tids)

        return self._finalize(frame_index, result_dets)

    def _finalize(
        self, frame_index: int, result_dets: list[TrackedDetection]
    ) -> FrameDetections:
        """Run the side-swap gate, update break bookkeeping, and emit.

        Break entry: when zero valid in-court players are present for
        `_break_min_empty_frames` consecutive frames, enter the break and emit an
        empty frame. Any frame with a valid in-court player resets the streak.
        """
        self._update_team_sides(result_dets)

        if self._count_valid_in_court(result_dets) == 0:
            self._empty_streak_frames += 1
            if self._empty_streak_frames >= self._break_min_empty_frames:
                if not self.in_break:
                    self.in_break = True
                    logging.info(
                        f"[frame {frame_index}] BREAK entered "
                        f"(court empty {self._empty_streak_frames}f)"
                    )
                return self._make_empty_frame(frame_index)
        else:
            self._empty_streak_frames = 0

        return self._make_frame_detections(frame_index, result_dets)

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

        # HARD TEAM INVARIANT: a state may only match a detection on its own team
        # side. Cross-net pairs are set to +inf so they can never be the argmin.
        # A detection with unknown team side (dead-zone / no position handled above)
        # matches no state; a state with team_side=None matches no detection.
        det_team = [self._det_team_side(d) for d in eligible_dets]
        team_ok = np.array(
            [[s.team_side is not None and s.team_side == det_team[di]
              for di in range(len(eligible_dets))]
             for s in eligible_states],
            dtype=bool,
        )  # (S, D)
        dists = np.where(team_ok, dists, np.inf)

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
        gallery_ok: bool = True,
    ) -> None:
        old_tid = state.current_track_id
        state.current_track_id = det.track_id
        self._update_state_matched(state, det, frame_index)
        det.logical_player_id = state.logical_id
        det.visible = True
        assigned_tids.add(det.track_id)
        result_dets.append(det)
        # Keep the ReID gallery fresh on confident matches, but only from a valid
        # in-court detection so near-glass/spectator crops don't leak in. When the
        # caller flags low confidence (gallery_ok=False, e.g. a weak appearance-
        # primary match), skip the update so a transient mislabel can't poison the
        # model. No-ops without an embedding (bytetrack).
        if gallery_ok and det.court_position is not None and self._is_valid_position(det.court_position):
            self._update_gallery(state, det)
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
