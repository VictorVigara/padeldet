"""Abstract base class for trackers (detection + tracking combined)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .models import FrameDetections, load_detections, save_detections
from .video import VideoInfo


class Tracker(ABC):
    """
    Base class for all trackers. Combines detection and tracking
    in a single interface — subclasses wrap a detection model and
    a tracking algorithm together.

    Subclasses must implement:
        - detect_and_track: run detection + tracking on a batch of frames
        - reset: clear internal tracking state
        - name: unique string identifier for this tracker type
    """

    def __init__(self, video_info: VideoInfo | None = None):
        self.video_info = video_info
        self._results: list[FrameDetections] = []

    @property
    def results(self) -> list[FrameDetections]:
        return self._results

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tracker type (e.g., 'player_tracker')."""
        ...

    @abstractmethod
    def detect_and_track(
        self,
        frames: Iterator[np.ndarray],
        start_frame: int = 0,
    ) -> list[FrameDetections]:
        """
        Run detection and tracking on a sequence of frames.

        Args:
            frames: iterator of BGR frames
            start_frame: frame index offset for the first frame

        Returns:
            list of FrameDetections, one per input frame
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracking state (e.g., ByteTrack tracker)."""
        ...

    def run(
        self,
        frames: Iterator[np.ndarray],
        start_frame: int = 0,
    ) -> list[FrameDetections]:
        """Run detection+tracking and store results."""
        self._results = self.detect_and_track(frames, start_frame)
        return self._results

    def save_results(self, path: str | Path) -> None:
        """Save accumulated results to JSON."""
        save_detections(self._results, path)
        print(f"{self.name}: saved {len(self._results)} frames to {path}")

    def load_results(self, path: str | Path) -> list[FrameDetections]:
        """Load results from JSON, replacing current results."""
        self._results = load_detections(path)
        print(f"{self.name}: loaded {len(self._results)} frames from {path}")
        return self._results

    def to(self, device: str) -> None:
        """Move model(s) to device. Override in subclass if needed."""
        pass

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, i: int) -> FrameDetections:
        return self._results[i]

    def __str__(self) -> str:
        return self.name
