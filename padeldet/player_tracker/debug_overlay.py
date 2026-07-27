"""Per-frame log capture and on-video overlay for debugging the tracker.

The identity manager emits `[frame N] ...` log records for every reassignment
decision (direct_track_id, same_role, same_team_side, proximity, one-to-one
fallback, side-change gate, state snapshot). `FrameLogCapture` buckets those
records by frame so they can be drawn onto the matching annotated frame.

Usage in an annotation script:

    from padeldet.player_tracker.debug_overlay import (
        FrameLogCapture, draw_log_overlay,
    )

    cap = FrameLogCapture(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(cap)
    try:
        results = tracker.run(frames, start_frame=start, stride=stride)
    finally:
        logging.getLogger().removeHandler(cap)

    # ... in annotation loop
    annotated = draw_log_overlay(annotated, cap.lines_for(abs_idx))
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import cv2
import numpy as np


_FRAME_RE = re.compile(r"\[frame (\d+)\]")


class FrameLogCapture(logging.Handler):
    """Logging handler that buckets records by `[frame N]` message prefix.

    Records whose message does not start with `[frame N]` are ignored — this
    is the project's convention for per-frame events. Use `lines_for(N)` to
    pull the captured strings for a given frame.
    """

    def __init__(self, level: int = logging.INFO):
        super().__init__(level)
        self.by_frame: dict[int, list[str]] = defaultdict(list)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        m = _FRAME_RE.match(msg)
        if not m:
            return
        # Strip the [frame N] prefix for compactness on-screen.
        line = msg[m.end():].lstrip()
        self.by_frame[int(m.group(1))].append(line)

    def lines_for(self, frame_index: int) -> list[str]:
        return self.by_frame.get(frame_index, [])

    def clear(self) -> None:
        self.by_frame.clear()


def draw_log_overlay(
    frame: np.ndarray,
    lines: list[str],
    *,
    max_lines: int = 12,
    max_chars: int = 140,  # ~fits a 1920px-wide frame at font_scale 0.45
    font_scale: float = 0.45,
    color: tuple[int, int, int] = (200, 255, 200),
    background: tuple[int, int, int] = (0, 0, 0),
    x: int = 10,
    y_start: int = 30,
    line_spacing: int = 6,
) -> np.ndarray:
    """Draw captured log lines on the frame (top-left corner by default).

    Lines longer than `max_chars` are truncated with an ellipsis. A black
    rectangle is drawn behind each line so the text is legible over the
    underlying video. Mutates and returns `frame`.
    """
    if not lines:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    y = y_start
    for line in lines[:max_lines]:
        if len(line) > max_chars:
            line = line[: max_chars - 3] + "..."
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 4, y + 4), background, -1)
        cv2.putText(frame, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += th + line_spacing

    if len(lines) > max_lines:
        more = f"+{len(lines) - max_lines} more..."
        (tw, th), _ = cv2.getTextSize(more, font, font_scale, thickness)
        cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 4, y + 4), background, -1)
        cv2.putText(frame, more, (x, y), font, font_scale, (180, 180, 180), thickness, cv2.LINE_AA)

    return frame
