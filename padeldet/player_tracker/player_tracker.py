"""Player detection and tracking using YOLO + ByteTrack or DeepOCSORT."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Literal

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from ..base import Tracker
from ..court.models import COURT_LENGTH, COURT_WIDTH
from ..court.projection import CourtProjection
from ..models import BoundingBox, FrameDetections, TrackedDetection
from ..video import VideoInfo


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _box_iou(box_a: list[float], boxes_b: np.ndarray) -> np.ndarray:
    """IoU between one box [x1,y1,x2,y2] and array of boxes (N,4)."""
    ax1, ay1, ax2, ay2 = box_a
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ix1 = np.maximum(ax1, boxes_b[:, 0])
    iy1 = np.maximum(ay1, boxes_b[:, 1])
    ix2 = np.minimum(ax2, boxes_b[:, 2])
    iy2 = np.minimum(ay2, boxes_b[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_b = (
        np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0])
        * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    )
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# Court-structure helpers (used by LogicalPlayerManager)
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


def _same_role(state: "LogicalPlayerState", det: TrackedDetection) -> bool:
    return (
        state.team_side is not None
        and state.play_side is not None
        and _detection_team_side(det) == state.team_side
        and _detection_play_side(det) == state.play_side
    )


def _same_team_side(state: "LogicalPlayerState", det: TrackedDetection) -> bool:
    return state.team_side is not None and _detection_team_side(det) == state.team_side


def _court_depth(court_position: np.ndarray) -> float:
    """Minimum distance from the given point to any court edge (in meters)."""
    x, y = float(court_position[0]), float(court_position[1])
    return min(x, COURT_WIDTH - x, y, COURT_LENGTH - y)


@dataclass
class PlayerTrackerConfig:
    """Configuration for player detection and tracking."""

    model_path: Literal["yolov8m.pt", "yolo26m.pt"] = "yolo26m.pt"
    confidence: float = 0.1
    iou_threshold: float = 0.1
    imgsz: int = 960
    batch_size: int = 8
    person_class_id: int = 0  # COCO person class
    max_players: int = 10
    # Seconds of video to process before pausing for manual player selection
    selection_window_sec: float = 10.0
    # Tracker selection
    tracker_type: Literal["bytetrack", "deepocsort"] = "deepocsort"
    # DeepOCSORT-specific
    reid_weights: str = "weights/osnet_x0_25_msmt17.pt"
    fp16: bool = True
    # How long (seconds) to keep a track alive without a detection match.
    tracker_max_age_sec: float = 2.0
    tracker_min_hits: int = 1

    # Court / detection box filtering
    court_margin_m: float = 0.75
    min_box_height_px: float = 25.0
    min_box_area_px: float = 500.0
    max_box_aspect_ratio: float = 6.0
    min_box_aspect_ratio: float = 0.8

    # Pose model for ankle midpoint extraction (opt-in)
    pose_model_path: str | None = None
    pose_confidence: float = 0.0001
    left_foot_kp_idx: int = 15   # COCO left ankle
    right_foot_kp_idx: int = 16  # COCO right ankle
    min_keypoint_conf: float = 0.1
    pose_iou_threshold: float = 0.3

    # Logical identity / occlusion recovery
    # Not used for assignment gating — kept for backwards compatibility
    max_missing_frames: int = 90
    max_recovery_distance_m: float = 2.0
    use_simple_occlusion_recovery: bool = True
    # Minimum distance from any court edge (m) for a direct track_id match to be trusted.
    # Detections closer than this are likely spectators near the boundary and will not be
    # accepted as a direct match — they fall through to structural reassignment instead.
    min_direct_match_court_depth_m: float = 0.05
    proximity_max_distance_m: float = 3.0

    # Selection frame scoring
    selection_require_in_court: bool = True
    # When True and court_projection is available: use depth-based auto-selection (no UI).
    # When False: fall back to manual PlayerSelectorUI click.
    auto_select_players: bool = True


# ---------------------------------------------------------------------------
# Logical player identity
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
        """Create logical players 1..4 from selected detections (click order)."""
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


# ---------------------------------------------------------------------------
# Player selection UI
# ---------------------------------------------------------------------------

class PlayerSelectorUI:
    """Interactive OpenCV window for manually selecting 4 player track IDs."""

    _MAX_CLICK_DIST = 100  # pixels — ignore clicks far from any detection

    def __init__(self, frame: np.ndarray, detections: FrameDetections):
        self._frame = frame
        self._detections = detections
        self._selected: list[int] = []

    def select(self) -> list[int]:
        """Block until user selects exactly 4 players. Returns track_ids in click order."""
        win = "Select 4 players - click each player"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.setMouseCallback(win, self._mouse_callback)
        while True:
            cv2.imshow(win, self._draw())
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):  # ENTER
                if len(self._selected) == 4:
                    break
            elif key in (8, 127):  # BACKSPACE / DEL
                if self._selected:
                    self._selected.pop()
            elif key == 27:  # ESC
                cv2.destroyWindow(win)
                raise RuntimeError(
                    "Player selection cancelled. Select exactly 4 players and press ENTER."
                )
        cv2.destroyWindow(win)
        return list(self._selected)

    def _draw(self) -> np.ndarray:
        img = self._frame.copy()
        _, w = img.shape[:2]
        for det in self._detections.detections:
            if det.track_id is None:
                continue
            tid = det.track_id
            x1 = int(det.bbox.x1)
            y1 = int(det.bbox.y1)
            x2 = int(det.bbox.x2)
            y2 = int(det.bbox.y2)
            is_sel = tid in self._selected
            color = (0, 200, 0) if is_sel else (200, 200, 200)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if is_sel else 1)
            if is_sel:
                label = f"#{tid} [{self._selected.index(tid) + 1}]"
            else:
                label = f"#{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                img, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
            )
        bar = np.zeros((36, w, 3), dtype=np.uint8)
        msg = (
            f"Selected: {len(self._selected)}/4"
            "    BACKSPACE: undo    ENTER: confirm (need 4)"
        )
        cv2.putText(bar, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([img, bar])

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        tid = self._find_nearest_track(x, y)
        if tid is None:
            return
        if tid in self._selected:
            self._selected.remove(tid)
        elif len(self._selected) < 4:
            self._selected.append(tid)

    def _find_nearest_track(self, x: int, y: int) -> int | None:
        best_tid, best_dist = None, float("inf")
        for det in self._detections.detections:
            if det.track_id is None:
                continue
            cx = (det.bbox.x1 + det.bbox.x2) / 2
            cy = (det.bbox.y1 + det.bbox.y2) / 2
            dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_tid = det.track_id
        return best_tid if best_dist <= self._MAX_CLICK_DIST else None


# ---------------------------------------------------------------------------
# Player tracker
# ---------------------------------------------------------------------------

class PlayerTracker(Tracker):
    """
    Detects and tracks players using YOLO for detection and either
    ByteTrack (supervision) or DeepOCSORT (boxmot + OSNet ReID) for tracking.

    On first run, pauses after the selection window (default 10 s) and shows an
    interactive OpenCV UI so the user can click on each of the 4 players to lock
    their logical IDs 1–4. Logical IDs persist through occlusions via simple
    court-position-based recovery.
    """

    def __init__(
        self,
        config: PlayerTrackerConfig | None = None,
        video_info: VideoInfo | None = None,
        court_projection: CourtProjection | None = None,
    ):
        super().__init__(video_info=video_info)
        self.config = config or PlayerTrackerConfig()
        self.court_projection = court_projection
        self.model = YOLO(self.config.model_path)
        self.pose_model = YOLO(self.config.pose_model_path) if self.config.pose_model_path else None
        self._init_tracker()

    # ------------------------------------------------------------------
    # Tracker initialisation
    # ------------------------------------------------------------------

    def _init_tracker(self, effective_fps: float | None = None) -> None:
        fps = effective_fps if effective_fps is not None else (
            self.video_info.fps if self.video_info else 30.0
        )
        if self.config.tracker_type == "bytetrack":
            self._init_byte_track(fps)
        else:
            self._init_deep_oc_sort(fps)

    def _init_byte_track(self, effective_fps: float) -> None:
        self.byte_track = sv.ByteTrack(frame_rate=int(effective_fps))

    def _init_deep_oc_sort(self, effective_fps: float) -> None:
        from pathlib import Path

        from boxmot.trackers.deepocsort.deepocsort import DeepOcSort

        weights = Path(self.config.reid_weights)
        weights.parent.mkdir(parents=True, exist_ok=True)
        half = self.config.fp16 and self.device != "cpu"
        max_age = max(1, round(self.config.tracker_max_age_sec * effective_fps))
        try:
            self.deep_tracker = DeepOcSort(
                reid_weights=weights,
                device=self.device,
                half=half,
                max_age=max_age,
                min_hits=self.config.tracker_min_hits,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize DeepOcSort. "
                f"Ensure OSNet weights can be downloaded to '{weights}' "
                f"and boxmot is installed (pip install boxmot). "
                f"Original error: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Tracker ABC interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"player_tracker_{self.config.tracker_type}"

    def reset(self) -> None:
        self._results = []
        self._init_tracker()

    def to(self, device: str) -> None:
        self.model.to(device)

    # ------------------------------------------------------------------
    # Court-aware filtering helpers
    # ------------------------------------------------------------------

    def _xyxy_foot_point(self, xyxy: np.ndarray) -> np.ndarray:
        """Return pixel foot point [cx, y2] for a raw xyxy box."""
        return np.array([(xyxy[0] + xyxy[2]) / 2.0, xyxy[3]])

    def _project_foot_to_court(self, foot_img: np.ndarray) -> np.ndarray | None:
        """Project a pixel foot point to court meters via homography.

        Returns None if no court_projection or if projection raises.
        """
        if self.court_projection is None:
            return None
        try:
            cx, cy = self.court_projection.project_point(
                (float(foot_img[0]), float(foot_img[1]))
            )
            return np.array([cx, cy])
        except Exception:
            return None

    def _is_court_point_valid(
        self, court_pt: np.ndarray, margin_m: float | None = None
    ) -> bool:
        """Check if a court-meter point is within court bounds + margin.

        Uses config.court_margin_m by default. Bypasses CourtProjection.is_inside_court()
        so we control the margin independently of how CourtProjection was constructed.
        Court origin is at the near-baseline-left corner:
            x: 0..COURT_WIDTH (10 m)
            y: 0..COURT_LENGTH (20 m)
        """
        if self.court_projection is None:
            return True
        m = margin_m if margin_m is not None else self.config.court_margin_m
        x, y = float(court_pt[0]), float(court_pt[1])
        return (-m <= x <= COURT_WIDTH + m) and (-m <= y <= COURT_LENGTH + m)

    def _detection_is_valid_box_xyxy(
        self, xyxy: np.ndarray, frame_shape: tuple[int, ...]
    ) -> bool:
        """Sanity-check a raw xyxy box: finite, oriented, min size, aspect ratio, in-frame."""
        if not np.all(np.isfinite(xyxy)):
            return False
        x1, y1, x2, y2 = xyxy
        if x2 <= x1 or y2 <= y1:
            return False
        h = y2 - y1
        w = x2 - x1
        fh, fw = frame_shape[:2]
        if h < self.config.min_box_height_px:
            return False
        if h * w < self.config.min_box_area_px:
            return False
        ratio = h / max(w, 1e-6)
        if not (self.config.min_box_aspect_ratio <= ratio <= self.config.max_box_aspect_ratio):
            return False
        # Require at least partial overlap with the image
        if x2 < 0 or y2 < 0 or x1 > fw or y1 > fh:
            return False
        return True

    def _filter_sv_detections_for_tracking(
        self, sv_dets: sv.Detections, frame_shape: tuple[int, ...]
    ) -> sv.Detections:
        """Apply box quality and court-ROI filters. Returns filtered sv.Detections."""
        if len(sv_dets) == 0:
            return sv_dets

        mask = np.zeros(len(sv_dets), dtype=bool)
        for i in range(len(sv_dets)):
            xyxy = sv_dets.xyxy[i]
            if not self._detection_is_valid_box_xyxy(xyxy, frame_shape):
                continue
            if self.court_projection is not None:
                foot = self._xyxy_foot_point(xyxy)
                court_pt = self._project_foot_to_court(foot)
                if court_pt is None or not self._is_court_point_valid(court_pt):
                    continue
            mask[i] = True

        logging.debug(
            f"detection filter: {len(sv_dets)} raw → {int(mask.sum())} kept"
        )
        return sv_dets[mask]

    def _enrich_court_positions(self, fd: FrameDetections) -> None:
        """Set court_position on each detection in place after tracking."""
        if self.court_projection is None:
            return
        for det in fd.detections:
            if det.ankle_midpoint is not None:
                foot = np.array(det.ankle_midpoint)
            else:
                foot = np.array([det.bbox.bottom_center[0], det.bbox.bottom_center[1]])
            court_pt = self._project_foot_to_court(foot)
            if court_pt is not None:
                det.court_position = (float(court_pt[0]), float(court_pt[1]))

    def _enrich_ankle_midpoints(self, fd: FrameDetections, pose_result) -> None:
        """Set ankle_midpoint on each detection using pose keypoints (in place)."""
        kps = pose_result.keypoints
        if kps is None or kps.xy is None:
            return
        pose_boxes = pose_result.boxes.xyxy.cpu().numpy()  # (N, 4)
        kp_xy = kps.xy.cpu().numpy()                       # (N, K, 2)
        kp_conf = kps.conf.cpu().numpy() if kps.conf is not None else None  # (N, K) or None

        lf_idx = self.config.left_foot_kp_idx
        rf_idx = self.config.right_foot_kp_idx

        for det in fd.detections:
            if len(pose_boxes) == 0:
                break
            ious = _box_iou(det.bbox.as_xyxy(), pose_boxes)
            best = int(np.argmax(ious))
            if ious[best] < self.config.pose_iou_threshold:
                continue

            lf = kp_xy[best, lf_idx]
            rf = kp_xy[best, rf_idx]

            lf_valid = not (lf[0] == 0.0 and lf[1] == 0.0)
            rf_valid = not (rf[0] == 0.0 and rf[1] == 0.0)

            if kp_conf is not None:
                if kp_conf[best, lf_idx] < self.config.min_keypoint_conf:
                    lf_valid = False
                if kp_conf[best, rf_idx] < self.config.min_keypoint_conf:
                    rf_valid = False

            if not lf_valid and not rf_valid:
                continue

            if lf_valid and rf_valid:
                mx = (lf[0] + rf[0]) / 2.0
                my = (lf[1] + rf[1]) / 2.0
            elif lf_valid:
                mx, my = float(lf[0]), float(lf[1])
            else:
                mx, my = float(rf[0]), float(rf[1])

            det.ankle_midpoint = (mx, my)

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_result(
        self, result, frame_index: int, bgr_frame: np.ndarray, pose_result=None
    ) -> FrameDetections:
        sv_dets = sv.Detections.from_ultralytics(result)
        person_mask = sv_dets.class_id == self.config.person_class_id
        sv_dets = sv_dets[person_mask]
        sv_dets = self._filter_sv_detections_for_tracking(sv_dets, bgr_frame.shape)
        if self.config.tracker_type == "bytetrack":
            fd = self._track_bytetrack(sv_dets, frame_index)
        else:
            fd = self._track_deepocsort(sv_dets, frame_index, bgr_frame)
        if pose_result is not None:
            self._enrich_ankle_midpoints(fd, pose_result)
        self._enrich_court_positions(fd)
        return fd

    def _track_bytetrack(
        self, sv_dets: sv.Detections, frame_index: int
    ) -> FrameDetections:
        sv_dets = self.byte_track.update_with_detections(sv_dets)
        tracked = []
        for i in range(len(sv_dets)):
            bbox = BoundingBox(*sv_dets.xyxy[i].tolist())
            det = TrackedDetection(
                bbox=bbox,
                confidence=float(sv_dets.confidence[i]),
                class_id=int(sv_dets.class_id[i]),
                label="player",
                track_id=(
                    int(sv_dets.tracker_id[i])
                    if sv_dets.tracker_id is not None
                    else None
                ),
            )
            tracked.append(det)
        return FrameDetections(frame_index=frame_index, detections=tracked)

    def _track_deepocsort(
        self, sv_dets: sv.Detections, frame_index: int, bgr_frame: np.ndarray
    ) -> FrameDetections:
        if len(sv_dets) == 0:
            dets_np = np.empty((0, 6))
        else:
            dets_np = np.column_stack(
                [sv_dets.xyxy, sv_dets.confidence, sv_dets.class_id]
            ).astype(np.float32)
        try:
            tracks = self.deep_tracker.update(dets_np, bgr_frame)
        except np.linalg.LinAlgError:
            tracks = np.empty((0, 8))
        tracked = []
        for row in tracks:
            x1, y1, x2, y2, tid, conf, cls, _ = row
            det = TrackedDetection(
                bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(conf),
                class_id=int(cls),
                label="player",
                track_id=int(tid),
            )
            tracked.append(det)
        return FrameDetections(frame_index=frame_index, detections=tracked)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_frames(
        self,
        frames_bgr: list[np.ndarray],
        frame_indices: list[int],
        desc: str = "tracking",
    ) -> list[FrameDetections]:
        """Run YOLO + tracker on a list of BGR frames, batched internally."""
        results: list[FrameDetections] = []
        pending: list[np.ndarray] = []
        pending_idx: list[int] = []

        def flush() -> None:
            if not pending:
                return
            batch = list(pending)
            indices = list(pending_idx)
            yolo_out = self.model.predict(
                batch,
                conf=self.config.confidence,
                iou=self.config.iou_threshold,
                imgsz=self.config.imgsz,
                device=self.device,
                classes=[self.config.person_class_id],
                verbose=False,
            )
            pose_out = (
                self.pose_model.predict(
                    batch,
                    conf=self.config.pose_confidence,
                    imgsz=self.config.imgsz,
                    device=self.device,
                    verbose=False,
                )
                if self.pose_model is not None
                else [None] * len(batch)
            )
            for fi, res, bgr, pose_res in zip(indices, yolo_out, batch, pose_out):
                results.append(self._process_result(res, fi, bgr, pose_result=pose_res))
            pending.clear()
            pending_idx.clear()

        for bgr, fi in tqdm(zip(frames_bgr, frame_indices), total=len(frames_bgr), desc=desc):
            pending.append(bgr)
            pending_idx.append(fi)
            if len(pending) >= self.config.batch_size:
                flush()
        flush()
        return results

    def _best_selection_frame(
        self, results: list[FrameDetections], frames_bgr: list[np.ndarray]
    ) -> tuple[np.ndarray, FrameDetections]:
        """Return frame with the most valid in-court detections for manual selection.

        For initial selection, it is better to show a frame where all 4 players are
        visible, even if there are extra detections. The user will manually click
        the correct 4 players.
        """
        best_score, best_i = float("-inf"), 0

        for i, fd in enumerate(results):
            valid_dets = []

            for d in fd.detections:
                if d.track_id is None:
                    continue

                if self.config.selection_require_in_court and self.court_projection is not None:
                    if d.court_position is None:
                        continue

                    court_pt = np.array(d.court_position, dtype=float)

                    # For selection, use strict court bounds, not the tracking margin.
                    x, y = float(court_pt[0]), float(court_pt[1])
                    inside_strict = 0.0 <= x <= COURT_WIDTH and 0.0 <= y <= COURT_LENGTH
                    if not inside_strict:
                        continue

                valid_dets.append(d)

            inside_count = len({d.track_id for d in valid_dets})

            # Main goal: maximize number of in-court detections.
            score = inside_count * 20.0

            # Strongly avoid frames with fewer than 4 visible in-court tracks.
            score -= max(0, 4 - inside_count) * 50.0

            # Mildly penalize many extra detections, but still allow 5 if all players are visible.
            score -= max(0, inside_count - 5) * 5.0

            if valid_dets:
                avg_conf = sum(d.confidence for d in valid_dets) / len(valid_dets)
                score += avg_conf * 2.0

                # Tie-breaker: prefer frames where detections are deeper inside the court.
                avg_depth = sum(
                    _court_depth(np.array(d.court_position))
                    for d in valid_dets
                    if d.court_position is not None
                ) / len(valid_dets)
                score += avg_depth * 2.0

            if score > best_score:
                best_score = score
                best_i = i

        return frames_bgr[best_i], results[best_i]

    @staticmethod
    def _score_and_select(
        fd: FrameDetections,
    ) -> "tuple[float, list[TrackedDetection]] | None":
        """Score a frame for auto-selection.

        Returns (score, top-4-candidates) or None if fewer than 4 valid in-court
        detections exist. Score = minimum court depth among the 4 chosen detections,
        so maximising it ensures even the shallowest selected player is well inside.
        """
        valid = [
            d for d in fd.detections
            if d.track_id is not None
            and d.court_position is not None
            and 0.0 <= d.court_position[0] <= COURT_WIDTH
            and 0.0 <= d.court_position[1] <= COURT_LENGTH
        ]
        if len(valid) < 4:
            return None
        valid.sort(key=lambda d: _court_depth(np.array(d.court_position)), reverse=True)
        candidates = valid[:4]
        score = min(_court_depth(np.array(d.court_position)) for d in candidates)
        return score, candidates

    def _auto_select_players(
        self, results: list[FrameDetections]
    ) -> "tuple[list[TrackedDetection], FrameDetections]":
        """Pick 4 player detections from the selection-window frames using court depth.

        Scans all frames, scores each via _score_and_select, and returns the candidate
        set and source frame from the frame with the highest score (best min court depth
        among chosen 4).
        """
        best_score: float = float("-inf")
        best_dets: list[TrackedDetection] = []
        best_fd: FrameDetections = results[-1]
        for fd in results:
            result = self._score_and_select(fd)
            if result is None:
                continue
            score, candidates = result
            if score > best_score:
                best_score, best_dets, best_fd = score, candidates, fd
        if len(best_dets) < 4:
            logging.warning(
                f"Auto-selection found only {len(best_dets)} player(s) in selection window."
            )
        return best_dets, best_fd

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def detect_and_track(
        self,
        frames: Iterator[np.ndarray],
        start_frame: int = 0,
        stride: int = 1,
    ) -> list[FrameDetections]:
        """Run YOLO + tracker, pause for manual player selection, return logical player tracks."""
        frames_list = list(frames)
        if not frames_list:
            return []

        fps = self.video_info.fps if self.video_info else 30.0
        effective_fps = fps / stride
        n_sel = int(self.config.selection_window_sec * effective_fps)
        n_sel = max(30, min(n_sel, len(frames_list)))

        # Re-init tracker with effective analysis FPS so all time-based params
        # (max_age, lost-track buffer) are calibrated to real seconds.
        self._init_tracker(effective_fps)

        init_bgr = frames_list[:n_sel]
        rest_bgr = frames_list[n_sel:]
        init_indices = [start_frame + i * stride for i in range(n_sel)]
        rest_indices = [start_frame + (n_sel + i) * stride for i in range(len(rest_bgr))]

        # Phase 1: process selection window
        print(f"\nProcessing first {n_sel} frames for player selection...")
        init_results = self._run_frames(init_bgr, init_indices, desc="selection window")

        # Debug: show raw detection counts per frame in selection window
        for fd in init_results:
            in_court = [
                d for d in fd.detections
                if d.court_position is not None
                and 0.0 <= d.court_position[0] <= COURT_WIDTH
                and 0.0 <= d.court_position[1] <= COURT_LENGTH
            ]
            print(
                f"  frame {fd.frame_index}: {len(fd.detections)} tracked dets, "
                f"{len(in_court)} strictly in court"
                + (
                    f", depths={[round(_court_depth(np.array(d.court_position)), 2) for d in in_court]}"
                    if in_court else ""
                )
            )

        # Phase 2: player selection (automatic or manual)
        if self.config.auto_select_players and self.court_projection is not None:
            selected_dets, best_fd = self._auto_select_players(init_results)
            selected_track_ids = [d.track_id for d in selected_dets if d.track_id is not None]
            print(f"Auto-selected player track IDs: {selected_track_ids}")
        else:
            best_frame, best_fd = self._best_selection_frame(init_results, init_bgr)
            n_tracks = len({d.track_id for d in best_fd.detections if d.track_id is not None})
            if n_tracks < 4:
                logging.warning(
                    f"Only {n_tracks} track(s) visible in best selection frame. "
                    "You may need to increase selection_window_sec."
                )
            print(
                f"\nOpening player selection UI — {n_tracks} track(s) visible."
                " Click 4 players then press ENTER."
            )
            selected_track_ids = PlayerSelectorUI(best_frame, best_fd).select()
            print(f"Locked player IDs (click order = logical order): {selected_track_ids}")
            by_tid = {d.track_id: d for d in best_fd.detections if d.track_id is not None}
            selected_dets = [by_tid[tid] for tid in selected_track_ids if tid in by_tid]
            if len(selected_dets) < 4:
                logging.warning(
                    f"Only {len(selected_dets)} of 4 selected IDs found in best frame. "
                    "Proceeding with partial selection."
                )

        # Phase 3: initialize identity manager from the selected frame
        identity_manager = LogicalPlayerManager(self.config)
        identity_manager.initialize(selected_dets, best_fd.frame_index)

        sel_frame_idx = best_fd.frame_index
        selected_tid_set = set(selected_track_ids)
        # Approximate logical ID by click order for frames before the selection frame
        tid_to_logical = {tid: (i + 1) for i, tid in enumerate(selected_track_ids)}

        # Phase 4: assign identities over init_results
        final_results: list[FrameDetections] = []
        for fd in init_results:
            if fd.frame_index < sel_frame_idx:
                # Before selection: keep detections with selected tracker IDs only.
                # Logical assignment is approximate (tracker ID has not changed yet).
                kept = []
                for d in fd.detections:
                    if d.track_id in selected_tid_set:
                        d.logical_player_id = tid_to_logical[d.track_id]
                        d.visible = True
                        kept.append(d)
                final_results.append(
                    FrameDetections(frame_index=fd.frame_index, detections=kept)
                )
            else:
                final_results.append(identity_manager.update(fd))

        # Phase 5: process remaining frames through identity manager
        if rest_bgr:
            print(f"\nTracking {len(rest_bgr)} remaining frames...")
            rest_results = self._run_frames(rest_bgr, rest_indices, desc=self.name)
            for fd in rest_results:
                final_results.append(identity_manager.update(fd))

        return final_results
