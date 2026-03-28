"""Player detection and tracking using YOLO + ByteTrack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

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
    confidence: float = 0.5
    iou_threshold: float = 0.7
    imgsz: int = 640
    batch_size: int = 8
    person_class_id: int = 0  # COCO person class
    max_players: int = 4


class PlayerTracker(Tracker):
    """
    Detects and tracks players using YOLO for detection
    and ByteTrack for multi-object tracking.

    If a CourtProjection is provided, filters detections to only
    players inside the court polygon. Otherwise falls back to
    keeping the top-N by confidence.
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
        self._init_byte_track()

    def _init_byte_track(self) -> None:
        fps = self.video_info.fps if self.video_info else 30.0
        self.byte_track = sv.ByteTrack(frame_rate=int(fps))

    @property
    def name(self) -> str:
        return "player_tracker"

    def reset(self) -> None:
        self._results = []
        self._init_byte_track()

    def to(self, device: str) -> None:
        self.model.to(device)

    def _process_result(self, result, frame_index: int) -> FrameDetections:
        """Convert a single YOLO result to FrameDetections with tracking."""
        sv_dets = sv.Detections.from_ultralytics(result)

        # Filter to person class only
        person_mask = sv_dets.class_id == self.config.person_class_id
        sv_dets = sv_dets[person_mask]

        # Update tracker
        sv_dets = self.byte_track.update_with_detections(sv_dets)

        # Convert to our data models
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

        # Take the top max_players most frequent IDs
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

    def detect_and_track(
        self,
        frames: Iterator[np.ndarray],
        start_frame: int = 0,
    ) -> list[FrameDetections]:
        """Run YOLO detection + ByteTrack on a sequence of frames."""
        all_results: list[FrameDetections] = []
        batch: list[np.ndarray] = []
        batch_indices: list[int] = []
        frame_index = start_frame

        frames_list = list(frames)
        total = len(frames_list)

        for frame in tqdm(frames_list, desc=self.name):
            # YOLO expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch.append(rgb)
            batch_indices.append(frame_index)
            frame_index += 1

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
                for idx, result in zip(batch_indices, results):
                    all_results.append(self._process_result(result, idx))
                batch = []
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
            for idx, result in zip(batch_indices, results):
                all_results.append(self._process_result(result, idx))

        # Identify real player IDs from initial frames, then filter
        if self.court_projection is not None or len(all_results) > 0:
            player_ids = self._identify_player_ids(all_results)
            if player_ids:
                all_results = self._filter_by_ids(all_results, player_ids)

        return all_results
