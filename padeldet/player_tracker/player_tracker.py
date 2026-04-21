"""Player detection and tracking using YOLO + ByteTrack or DeepOCSORT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from ..base import Tracker
from ..court.projection import CourtProjection
from ..models import BoundingBox, FrameDetections, TrackedDetection
from ..video import VideoInfo


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
    # Tracker selection
    tracker_type: Literal["bytetrack", "deepocsort"] = "deepocsort"
    # DeepOCSORT-specific
    reid_weights: str = "weights/osnet_x0_25_msmt17.pt"
    fp16: bool = True        # auto-forced False on CPU
    tracker_max_age: int = 30    # frames before a lost track is dropped (Task 3 handles drink breaks separately)
    tracker_min_hits: int = 1


class PlayerTracker(Tracker):
    """
    Detects and tracks players using YOLO for detection and either
    ByteTrack (supervision) or DeepOCSORT (boxmot + OSNet ReID) for tracking.

    Select via PlayerTrackerConfig.tracker_type = "bytetrack" | "deepocsort".
    DeepOCSORT is the default; it adds appearance embeddings so IDs survive
    occlusions at the net. ByteTrack is preserved as a fallback for A/B comparison.

    If a CourtProjection is provided, filters detections to only players inside
    the court polygon. Otherwise falls back to keeping the top-N by confidence.
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
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_result(
        self, result, frame_index: int, bgr_frame: np.ndarray
    ) -> FrameDetections:
        """Convert a single YOLO result to FrameDetections with tracking."""
        sv_dets = sv.Detections.from_ultralytics(result)
        person_mask = sv_dets.class_id == self.config.person_class_id
        sv_dets = sv_dets[person_mask]

        if self.config.tracker_type == "bytetrack":
            return self._track_bytetrack(sv_dets, frame_index)
        else:
            return self._track_deepocsort(sv_dets, frame_index, bgr_frame)

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

        # tracks: (M, 8) — [x1, y1, x2, y2, track_id, conf, cls, det_idx]
        try:
            tracks = self.deep_tracker.update(dets_np, bgr_frame)
        except np.linalg.LinAlgError:
            # Kalman filter covariance became non-positive-definite (can happen
            # when a track hasn't been seen for many frames). Skip this frame.
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
    # Player ID identification and filtering
    # ------------------------------------------------------------------

    def _identify_player_ids(
        self, results: list[FrameDetections], num_frames: int = 30
    ) -> set[int]:
        """Identify real player track IDs from the first N frames.

        Uses court polygon to find detections inside the court during the
        initial frames (when players are standing normally), then returns
        the most frequent track IDs.
        """
        from collections import Counter

        id_counts: Counter[int] = Counter()
        frames_to_check = min(num_frames, len(results))

        for fd in results[:frames_to_check]:
            for det in fd.detections:
                if det.track_id is None:
                    continue
                if self.court_projection is not None:
                    if self.court_projection.is_inside_court_px(det.bbox.bottom_center):
                        id_counts[det.track_id] += 1
                else:
                    id_counts[det.track_id] += 1

        player_ids = {
            tid for tid, _ in id_counts.most_common(self.config.max_players)
        }
        return player_ids

    def _filter_by_ids(
        self, results: list[FrameDetections], player_ids: set[int]
    ) -> list[FrameDetections]:
        """Keep only detections whose track_id is in the player set."""
        filtered = []
        for fd in results:
            kept = [d for d in fd.detections if d.track_id in player_ids]
            filtered.append(
                FrameDetections(frame_index=fd.frame_index, detections=kept)
            )
        return filtered

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def detect_and_track(
        self,
        frames: Iterator[np.ndarray],
        start_frame: int = 0,
        stride: int = 1,
    ) -> list[FrameDetections]:
        """Run YOLO detection + tracker on a sequence of frames."""
        all_results: list[FrameDetections] = []
        batch: list[np.ndarray] = []
        batch_bgr: list[np.ndarray] = []
        batch_indices: list[int] = []
        frame_index = start_frame

        frames_list = list(frames)

        for frame in tqdm(frames_list, desc=self.name):
            batch_bgr.append(frame)
            batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            batch_indices.append(frame_index)
            frame_index += stride

            if len(batch) >= self.config.batch_size:
                results = self.model.predict(
                    batch,
                    conf=self.config.confidence,
                    iou=self.config.iou_threshold,
                    imgsz=self.config.imgsz,
                    device=self.device,
                    classes=[self.config.person_class_id],
                    verbose=False,
                )
                for idx, result, bgr in zip(batch_indices, results, batch_bgr):
                    all_results.append(self._process_result(result, idx, bgr))
                batch = []
                batch_bgr = []
                batch_indices = []

        # Process remaining frames
        if batch:
            results = self.model.predict(
                batch,
                conf=self.config.confidence,
                iou=self.config.iou_threshold,
                imgsz=self.config.imgsz,
                device=self.device,
                classes=[self.config.person_class_id],
                verbose=False,
            )
            for idx, result, bgr in zip(batch_indices, results, batch_bgr):
                all_results.append(self._process_result(result, idx, bgr))

        # Identify real player IDs from initial frames, then filter
        if self.court_projection is not None or len(all_results) > 0:
            player_ids = self._identify_player_ids(all_results)
            if player_ids:
                all_results = self._filter_by_ids(all_results, player_ids)

        return all_results
