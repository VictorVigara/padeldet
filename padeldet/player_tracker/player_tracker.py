"""Player detection and tracking using YOLO + ByteTrack or DeepOCSORT."""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from ..base import Tracker
from ..court.projection import CourtProjection
from ..match_profile import MATCH_PROFILES
from ..models import BoundingBox, FrameDetections, TrackedDetection
from ..video import VideoInfo
from .config import PlayerTrackerConfig
from .filters import (
    enrich_ankle_midpoints,
    enrich_court_positions,
    filter_sv_detections_for_tracking,
)
from .identity import LogicalPlayerManager, _court_depth
from .selector import PlayerSelectorUI


def _resolve_rfdetr_person_id() -> int:
    """Resolve the COCO class id RF-DETR returns for 'person'.

    predict() uses COCO-91 sparse ids (person == 1). Prefer the dict-keyed
    mapping in the rfdetr package, which is the source of truth for that id
    space — `RFDETRBase.class_names` is a 0-indexed display list that does
    NOT match predict()'s class_id output.
    """
    for mod_path in ("rfdetr.assets.coco_classes", "rfdetr.util.coco_classes"):
        try:
            mod = __import__(mod_path, fromlist=["COCO_CLASSES"])
            mapping = getattr(mod, "COCO_CLASSES")
            if isinstance(mapping, dict):
                for k, v in mapping.items():
                    if isinstance(v, str) and v.lower() == "person":
                        return int(k)
        except Exception:
            continue
    return 1


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
        self.profile = MATCH_PROFILES[self.config.match_type]
        self.n_players = self.profile.n_players
        self.court_projection = court_projection
        self._rfdetr_person_id: int | None = None
        if self.config.detector_type == "yolo":
            self.model = YOLO(self.config.model_path)
        elif self.config.detector_type == "rfdetr":
            self.model = self._init_rfdetr()
            self._rfdetr_person_id = _resolve_rfdetr_person_id()
            print(f"[player_tracker] RF-DETR person class id: {self._rfdetr_person_id}")
        else:
            raise ValueError(f"unknown detector_type: {self.config.detector_type!r}")
        self.pose_model = YOLO(self.config.pose_model_path) if self.config.pose_model_path else None
        self._init_tracker()

    def _init_rfdetr(self):
        from rfdetr import RFDETRBase
        kwargs: dict = {"device": self.device}
        if self.config.rfdetr_resolution is not None:
            if self.config.rfdetr_resolution % 56 != 0:
                raise ValueError(
                    f"rfdetr_resolution must be divisible by 56 "
                    f"(got {self.config.rfdetr_resolution})"
                )
            kwargs["resolution"] = self.config.rfdetr_resolution
        model = RFDETRBase(**kwargs)
        # Fuses/JITs ops for predict. One-time cost at startup, big win over
        # the hundreds-to-thousands of predict() calls in a tracking run.
        # Must match config.batch_size — the compiled graph is batch-size-specific
        # and predict() raises if a differently-sized batch is passed later.
        try:
            model.optimize_for_inference(batch_size=self.config.batch_size)
        except Exception as e:
            print(f"[player_tracker] RF-DETR optimize_for_inference skipped: {e}")
        return model

    @property
    def device(self) -> str:
        if self.config.device is not None:
            return self.config.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

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
        # boxmot's select_device rejects the string "cuda" when CUDA_VISIBLE_DEVICES
        # is set to a non-numeric value (e.g. Modal sets it to "cuda"). Pass integer
        # device index 0 instead, which bypasses that validation.
        boxmot_device = 0 if self.device == "cuda" else self.device
        try:
            self.deep_tracker = DeepOcSort(
                reid_weights=weights,
                device=boxmot_device,
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
        # RF-DETR's wrapper takes its device at construction time and exposes
        # no .to(); only YOLO needs an explicit move call.
        if self.config.detector_type == "yolo":
            self.model.to(device)

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_result(
        self,
        sv_dets: sv.Detections,
        frame_index: int,
        bgr_frame: np.ndarray,
        pose_result=None,
    ) -> FrameDetections:
        # sv_dets arrives already person-filtered from _predict_persons.
        sv_dets = filter_sv_detections_for_tracking(
            sv_dets, bgr_frame.shape, self.config, self.court_projection
        )
        if self.config.tracker_type == "bytetrack":
            fd = self._track_bytetrack(sv_dets, frame_index)
        else:
            fd = self._track_deepocsort(sv_dets, frame_index, bgr_frame)
        if pose_result is not None:
            enrich_ankle_midpoints(fd, pose_result, self.config)
        enrich_court_positions(fd, self.court_projection)
        return fd

    def _predict_persons(self, batch_bgr: list[np.ndarray]) -> list[sv.Detections]:
        """Run the configured detector and return one person-only sv.Detections
        per input frame, with class_id renormalized to config.person_class_id."""
        if self.config.detector_type == "yolo":
            results = self.model.predict(
                batch_bgr,
                conf=self.config.confidence,
                iou=self.config.iou_threshold,
                imgsz=self.config.imgsz,
                device=self.device,
                classes=[self.config.person_class_id],
                verbose=False,
            )
            out: list[sv.Detections] = []
            for r in results:
                sv_dets = sv.Detections.from_ultralytics(r)
                mask = sv_dets.class_id == self.config.person_class_id
                out.append(sv_dets[mask])
            return out

        # RF-DETR
        import cv2
        rgb_batch = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_bgr]
        # optimize_for_inference() JIT-traces the model for a fixed batch size
        # (config.batch_size). The final batch of a video is often smaller —
        # pad with repeated frames to match, then drop the padding results.
        n_real = len(rgb_batch)
        pad = self.config.batch_size - n_real
        if pad > 0:
            rgb_batch = rgb_batch + [rgb_batch[-1]] * pad
        raw = self.model.predict(rgb_batch, threshold=self.config.rfdetr_threshold)
        if isinstance(raw, sv.Detections):
            raw = [raw]
        if pad > 0:
            raw = raw[:n_real]
        person_id = self._rfdetr_person_id if self._rfdetr_person_id is not None else 1
        target_id = self.config.person_class_id
        out = []
        for sv_dets in raw:
            mask = sv_dets.class_id == person_id
            sv_dets = sv_dets[mask]
            if len(sv_dets) > 0:
                sv_dets.class_id = np.full(len(sv_dets), target_id, dtype=int)
            out.append(sv_dets)
        return out

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
            det_out = self._predict_persons(batch)
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
            for fi, sv_dets, bgr, pose_res in zip(indices, det_out, batch, pose_out):
                results.append(self._process_result(sv_dets, fi, bgr, pose_result=pose_res))
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
                    width_m = self.court_projection.width_m
                    length_m = self.court_projection.length_m
                    inside_strict = 0.0 <= x <= width_m and 0.0 <= y <= length_m
                    if not inside_strict:
                        continue

                valid_dets.append(d)

            inside_count = len({d.track_id for d in valid_dets})
            n_players = self.n_players

            # Main goal: maximize number of in-court detections.
            score = inside_count * 20.0

            # Strongly avoid frames with fewer than n_players visible in-court tracks.
            score -= max(0, n_players - inside_count) * 50.0

            # Mildly penalize many extra detections, but still allow one extra above n_players.
            score -= max(0, inside_count - (n_players + 1)) * 5.0

            if valid_dets:
                avg_conf = sum(d.confidence for d in valid_dets) / len(valid_dets)
                score += avg_conf * 2.0

                # Tie-breaker: prefer frames where detections are deeper inside the court.
                width_m = self.court_projection.width_m if self.court_projection else 0.0
                length_m = self.court_projection.length_m if self.court_projection else 0.0
                avg_depth = sum(
                    _court_depth(np.array(d.court_position), width_m, length_m)
                    for d in valid_dets
                    if d.court_position is not None
                ) / len(valid_dets)
                score += avg_depth * 2.0

            if score > best_score:
                best_score = score
                best_i = i

        return frames_bgr[best_i], results[best_i]

    def _score_and_select(
        self,
        fd: FrameDetections,
    ) -> "tuple[float, list[TrackedDetection]] | None":
        """Score a frame for auto-selection.

        Returns (score, top-n_players-candidates) or None if fewer than n_players
        valid in-court detections exist. Score = minimum court depth among the
        chosen candidates, so maximising it ensures even the shallowest selected
        player is well inside.
        """
        if self.court_projection is None:
            return None
        width_m = self.court_projection.width_m
        length_m = self.court_projection.length_m
        n_players = self.n_players
        valid = [
            d for d in fd.detections
            if d.track_id is not None
            and d.court_position is not None
            and 0.0 <= d.court_position[0] <= width_m
            and 0.0 <= d.court_position[1] <= length_m
        ]
        if len(valid) < n_players:
            return None
        valid.sort(
            key=lambda d: _court_depth(np.array(d.court_position), width_m, length_m),
            reverse=True,
        )
        candidates = valid[:n_players]
        score = min(
            _court_depth(np.array(d.court_position), width_m, length_m)
            for d in candidates
        )
        return score, candidates

    def _auto_select_players(
        self, results: list[FrameDetections]
    ) -> "tuple[list[TrackedDetection], FrameDetections]":
        """Pick `n_players` detections from the selection-window frames using court depth.

        Scans all frames, scores each via _score_and_select, and returns the candidate
        set and source frame from the frame with the highest score (best min court depth
        among chosen candidates).
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
        if len(best_dets) < self.n_players:
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
        proj = self.court_projection
        width_m = proj.width_m if proj is not None else 0.0
        length_m = proj.length_m if proj is not None else 0.0
        for fd in init_results:
            if proj is not None:
                in_court = [
                    d for d in fd.detections
                    if d.court_position is not None
                    and 0.0 <= d.court_position[0] <= width_m
                    and 0.0 <= d.court_position[1] <= length_m
                ]
            else:
                in_court = []
            print(
                f"  frame {fd.frame_index}: {len(fd.detections)} tracked dets, "
                f"{len(in_court)} strictly in court"
                + (
                    f", depths={[round(_court_depth(np.array(d.court_position), width_m, length_m), 2) for d in in_court]}"
                    if in_court else ""
                )
            )

        # Phase 2: player selection (automatic or manual)
        n_players = self.n_players
        if self.config.auto_select_players and self.court_projection is not None:
            selected_dets, best_fd = self._auto_select_players(init_results)
            selected_track_ids = [d.track_id for d in selected_dets if d.track_id is not None]
            print(f"Auto-selected player track IDs: {selected_track_ids}")
        else:
            best_frame, best_fd = self._best_selection_frame(init_results, init_bgr)
            n_tracks = len({d.track_id for d in best_fd.detections if d.track_id is not None})
            if n_tracks < n_players:
                logging.warning(
                    f"Only {n_tracks} track(s) visible in best selection frame. "
                    "You may need to increase selection_window_sec."
                )
            print(
                f"\nOpening player selection UI — {n_tracks} track(s) visible."
                f" Click {n_players} players then press ENTER."
            )
            selected_track_ids = PlayerSelectorUI(best_frame, best_fd, n_players=n_players).select()
            print(f"Locked player IDs (click order = logical order): {selected_track_ids}")
            by_tid = {d.track_id: d for d in best_fd.detections if d.track_id is not None}
            selected_dets = [by_tid[tid] for tid in selected_track_ids if tid in by_tid]
            if len(selected_dets) < n_players:
                logging.warning(
                    f"Only {len(selected_dets)} of {n_players} selected IDs found in best frame. "
                    "Proceeding with partial selection."
                )

        # Phase 3: initialize identity manager from the selected frame
        identity_manager = LogicalPlayerManager(self.config, self.profile, effective_fps)
        identity_manager.initialize(selected_dets, best_fd.frame_index)

        sel_frame_idx = best_fd.frame_index
        selected_tid_set = set(selected_track_ids)
        # Build tid→logical map from the identity manager so pre-selection frames
        # use the same position-sorted assignment as post-selection frames.
        # For example, a current tracker id 13, can be mapped to logical player 2.
        tid_to_logical = {
            s.current_track_id: s.logical_id
            for s in identity_manager.states
            if s.current_track_id is not None
        }
        

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
