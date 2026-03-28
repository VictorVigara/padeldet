"""Shared detection data models — plain dataclasses, no library coupling."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import supervision as sv


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.width / 2, self.y1 + self.height / 2)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Approximate feet position for a person detection."""
        return (self.x1 + self.width / 2, self.y2)

    def as_xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Detection:
    """Single object detection in one frame."""

    bbox: BoundingBox
    confidence: float
    class_id: int
    label: str = ""

    def serialize(self) -> dict:
        return {
            "bbox": self.bbox.as_xyxy(),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Detection:
        return cls(
            bbox=BoundingBox(*d["bbox"]),
            confidence=d["confidence"],
            class_id=d["class_id"],
            label=d.get("label", ""),
        )


@dataclass
class TrackedDetection(Detection):
    """Detection with a persistent track ID across frames."""

    track_id: int | None = None

    def serialize(self) -> dict:
        data = super().serialize()
        data["track_id"] = self.track_id
        return data

    @classmethod
    def from_dict(cls, d: dict) -> TrackedDetection:
        return cls(
            bbox=BoundingBox(*d["bbox"]),
            confidence=d["confidence"],
            class_id=d["class_id"],
            label=d.get("label", ""),
            track_id=d.get("track_id"),
        )


@dataclass
class FrameDetections:
    """All detections in a single frame."""

    frame_index: int
    detections: list[TrackedDetection] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self):
        return iter(self.detections)

    def __getitem__(self, i: int) -> TrackedDetection:
        return self.detections[i]

    def serialize(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "detections": [d.serialize() for d in self.detections],
        }

    @classmethod
    def from_dict(cls, d: dict) -> FrameDetections:
        return cls(
            frame_index=d["frame_index"],
            detections=[TrackedDetection.from_dict(det) for det in d["detections"]],
        )

    def to_sv_detections(self) -> sv.Detections:
        """Convert to supervision Detections for visualization."""
        if not self.detections:
            return sv.Detections.empty()

        xyxy = np.array([d.bbox.as_xyxy() for d in self.detections])
        confidence = np.array([d.confidence for d in self.detections])
        class_id = np.array([d.class_id for d in self.detections])
        tracker_id = np.array([
            d.track_id if d.track_id is not None else -1
            for d in self.detections
        ])

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
        )

    def draw(
        self,
        frame: np.ndarray,
        show_confidence: bool = True,
        show_track_id: bool = True,
    ) -> np.ndarray:
        """Draw all detections on frame using supervision annotators."""
        if not self.detections:
            return frame

        sv_dets = self.to_sv_detections()
        h, w = frame.shape[:2]
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(w, h))

        box_annotator = sv.BoxAnnotator(
            thickness=thickness,
            color=sv.Color.from_hex("#00FF00"),
        )
        label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.TOP_CENTER,
            text_scale=text_scale,
            text_thickness=thickness,
        )

        frame = box_annotator.annotate(scene=frame, detections=sv_dets)

        labels = []
        for det in self.detections:
            parts = []
            if show_track_id and det.track_id is not None:
                parts.append(f"#{det.track_id}")
            if show_confidence:
                parts.append(f"{det.confidence:.2f}")
            labels.append(" ".join(parts))

        frame = label_annotator.annotate(
            scene=frame, detections=sv_dets, labels=labels,
        )
        return frame


BALL_CLASS_ID = 1
BALL_RADIUS_PX = 5


def ball_detection(
    x: float,
    y: float,
    confidence: float,
    radius: float = BALL_RADIUS_PX,
) -> TrackedDetection:
    """Create a TrackedDetection for a ball position."""
    return TrackedDetection(
        bbox=BoundingBox(x - radius, y - radius, x + radius, y + radius),
        confidence=confidence,
        class_id=BALL_CLASS_ID,
        label="ball",
        track_id=0,
    )


def save_detections(detections: list[FrameDetections], path: str | Path) -> None:
    """Save a list of FrameDetections to JSON."""
    data = [fd.serialize() for fd in detections]
    with open(path, "w") as f:
        json.dump(data, f)


def load_detections(path: str | Path) -> list[FrameDetections]:
    """Load FrameDetections from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [FrameDetections.from_dict(d) for d in data]
