"""Minimal video I/O utilities using OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    """Basic video metadata."""

    width: int
    height: int
    fps: float
    total_frames: int

    @classmethod
    def from_path(cls, path: str | Path) -> VideoInfo:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        info = cls(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        cap.release()
        return info

    @property
    def resolution_wh(self) -> tuple[int, int]:
        return (self.width, self.height)


def frame_generator(
    path: str | Path,
    start: int = 0,
    end: int | None = None,
    stride: int = 1,
) -> Iterator[np.ndarray]:
    """
    Yield frames from a video file.

    Frames are yielded in BGR format (OpenCV convention).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frame_idx = start
    try:
        while True:
            if end is not None and frame_idx >= end:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start) % stride == 0:
                yield frame

            frame_idx += 1
    finally:
        cap.release()


class VideoWriter:
    """Simple wrapper around cv2.VideoWriter."""

    def __init__(self, path: str | Path, video_info: VideoInfo):
        self.path = str(path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.path,
            fourcc,
            video_info.fps,
            video_info.resolution_wh,
        )

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
