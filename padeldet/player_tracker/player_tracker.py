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

    model_path: str = "yolov8m.pt"
    confidence: float = 0.1
    iou_threshold: float = 0.1
    imgsz: int = 960
    batch_size: int = 8
    person_class_id: int = 0  # COCO person class
    max_players: int = 10
    # Seconds of video to process before pausing for manual player selection
    selection_window_sec: float = 3.0
    # Tracker selection
    tracker_type: Literal["bytetrack", "deepocsort"] = "deepocsort"
    # DeepOCSORT-specific
    reid_weights: str = "weights/osnet_x0_25_msmt17.pt"
    fp16: bool = True
    tracker_max_age: int = 60
    tracker_min_hits: int = 1

    # Court / detection box filtering
    court_margin_m: float = 0.75
    min_box_height_px: float = 25.0
    min_box_area_px: float = 500.0
    max_box_aspect_ratio: float = 6.0
    min_box_aspect_ratio: float = 0.8

    # Logical identity / occlusion recovery
    # Not used for assignment gating — kept for backwards compatibility
    max_missing_frames: int = 90
    max_recovery_distance_m: float = 2.0
    use_simple_occlusion_recovery: bool = True
    # Minimum distance from any court edge (m) for a direct track_id match to be trusted.
    # Detections closer than this are likely spectators near the boundary and will not be
    # accepted as a direct match — they fall through to structural reassignment instead.
    min_direct_match_court_depth_m: float = 0.1

    # Selection frame scoring
    selection_require_in_court: bool = True


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
        # Skip if the detection is too close to a court edge — the tracker may have
        # reassigned a lost player's ID to a spectator near the court boundary.
        min_depth = self.config.min_direct_match_court_depth_m
        for state in self.states:
            if state.current_track_id is not None and state.current_track_id in by_tid:
                det = by_tid[state.current_track_id]
                if det.court_position is not None:
                    depth = _court_depth(np.array(det.court_position, dtype=float))
                    if depth < min_depth:
                        logging.debug(
                            f"[frame {frame_index}] direct_track_id skipped: "
                            f"logical_id={state.logical_id} tid={det.track_id} "
                            f"court_depth={depth:.2f}m < {min_depth}m"
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
                state.missing_frames += frame_index - state.last_seen_frame
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

        # Step 4: one-to-one fallback
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
        self._init_tracker()

    # ------------------------------------------------------------------
    # Tracker initialisation
    # ------------------------------------------------------------------

    def _init_tracker(self) -> None:
        if self.config.tracker_type == "bytetrack":
            self._init_byte_track()
        else:
            self._init_deep_oc_sort()

    def _init_byte_track(self) -> None:
        fps = self.video_info.fps if self.video_info else 30.0
        self.byte_track = sv.ByteTrack(frame_rate=int(fps))

    def _init_deep_oc_sort(self) -> None:
        from pathlib import Path

        from boxmot.trackers.deepocsort.deepocsort import DeepOcSort

        weights = Path(self.config.reid_weights)
        weights.parent.mkdir(parents=True, exist_ok=True)
        half = self.config.fp16 and self.device != "cpu"
        try:
            self.deep_tracker = DeepOcSort(
                reid_weights=weights,
                device=self.device,
                half=half,
                max_age=self.config.tracker_max_age,
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
            foot = np.array([det.bbox.bottom_center[0], det.bbox.bottom_center[1]])
            court_pt = self._project_foot_to_court(foot)
            if court_pt is not None:
                det.court_position = (float(court_pt[0]), float(court_pt[1]))

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_result(
        self, result, frame_index: int, bgr_frame: np.ndarray
    ) -> FrameDetections:
        sv_dets = sv.Detections.from_ultralytics(result)
        person_mask = sv_dets.class_id == self.config.person_class_id
        sv_dets = sv_dets[person_mask]
        sv_dets = self._filter_sv_detections_for_tracking(sv_dets, bgr_frame.shape)
        if self.config.tracker_type == "bytetrack":
            fd = self._track_bytetrack(sv_dets, frame_index)
        else:
            fd = self._track_deepocsort(sv_dets, frame_index, bgr_frame)
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
            for fi, res, bgr in zip(indices, yolo_out, batch):
                results.append(self._process_result(res, fi, bgr))
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

            # Minor tie-breaker: prefer spatially separated detections.
            foot_pts = []
            for d in valid_dets:
                if d.court_position is not None:
                    foot_pts.append(np.array(d.court_position, dtype=float))
                else:
                    bc = d.bbox.bottom_center
                    foot_pts.append(np.array([bc[0] / 100.0, bc[1] / 100.0]))

            if len(foot_pts) >= 2:
                min_dist = min(
                    float(np.linalg.norm(foot_pts[a] - foot_pts[b]))
                    for a in range(len(foot_pts))
                    for b in range(a + 1, len(foot_pts))
                )
                score += 0.1 * min_dist

            if score > best_score:
                best_score = score
                best_i = i

        return frames_bgr[best_i], results[best_i]

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
        n_sel = int(self.config.selection_window_sec * fps / stride)
        n_sel = max(30, min(n_sel, len(frames_list)))

        init_bgr = frames_list[:n_sel]
        rest_bgr = frames_list[n_sel:]
        init_indices = [start_frame + i * stride for i in range(n_sel)]
        rest_indices = [start_frame + (n_sel + i) * stride for i in range(len(rest_bgr))]

        # Phase 1: process selection window
        print(f"\nProcessing first {n_sel} frames for player selection...")
        init_results = self._run_frames(init_bgr, init_indices, desc="selection window")

        # Phase 2: manual player selection
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
        selected_track_ids: list[int] = PlayerSelectorUI(best_frame, best_fd).select()
        print(f"Locked player IDs (click order = logical order): {selected_track_ids}")

        # Find selected detections in best_fd preserving click order
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
