"""Interactive OpenCV window for manually selecting 4 player track IDs."""

from __future__ import annotations

import cv2
import numpy as np

from ..models import FrameDetections


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
