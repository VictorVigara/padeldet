"""
Manual court keypoint calibration tool.

Opens a video frame and lets you click 12 court keypoints in order.

Usage:
    python scripts/calibrate_court.py --video path/to/video.mp4 --output court_keypoints.json
    python scripts/calibrate_court.py --video path/to/video.mp4 --load court_keypoints.json  # verify existing
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from padeldet.court.models import (
    COURT_KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    CourtKeypoints,
)


class CalibrationTool:
    """Interactive OpenCV tool for clicking court keypoints."""

    WINDOW_NAME = "Court Calibration"
    POINT_RADIUS = 6
    POINT_COLOR = (0, 255, 0)       # green
    HIGHLIGHT_COLOR = (0, 255, 255)  # yellow for next expected
    TEXT_COLOR = (255, 255, 255)
    LINE_COLOR = (0, 200, 0)

    # Court line connectivity: pairs of keypoint indices to draw lines between
    COURT_LINES = [
        (0, 1),    # near baseline
        (2, 4),    # near service line
        (5, 6),    # net
        (7, 9),    # far service line
        (10, 11),  # far baseline
        (0, 10),   # left sideline
        (1, 11),   # right sideline
        (3, 8),    # center service line
    ]

    def __init__(self, frame: np.ndarray):
        self.original_frame = frame.copy()
        self.points: list[tuple[float, float] | None] = [None] * NUM_KEYPOINTS
        self.current_index = 0
        self.done = False
        self.zoom_center: tuple[int, int] | None = None
        self.zoom_factor = 3
        self.show_zoom = False

    def _draw(self) -> np.ndarray:
        """Draw current state on the frame."""
        frame = self.original_frame.copy()

        # Draw court lines between placed points
        for i, j in self.COURT_LINES:
            if self.points[i] is not None and self.points[j] is not None:
                pt1 = (int(self.points[i][0]), int(self.points[i][1]))
                pt2 = (int(self.points[j][0]), int(self.points[j][1]))
                cv2.line(frame, pt1, pt2, self.LINE_COLOR, 1, cv2.LINE_AA)

        # Draw placed points
        for i, pt in enumerate(self.points):
            if pt is None:
                continue
            center = (int(pt[0]), int(pt[1]))
            color = self.POINT_COLOR
            cv2.circle(frame, center, self.POINT_RADIUS, color, -1, cv2.LINE_AA)
            cv2.circle(frame, center, self.POINT_RADIUS, (0, 0, 0), 1, cv2.LINE_AA)
            label = COURT_KEYPOINT_NAMES[i][0]
            cv2.putText(
                frame, label, (center[0] + 10, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1, cv2.LINE_AA,
            )

        # Instructions
        if self.current_index < NUM_KEYPOINTS:
            name, desc = COURT_KEYPOINT_NAMES[self.current_index]
            instruction = f"Click {name}: {desc} ({self.current_index + 1}/{NUM_KEYPOINTS})"
        else:
            instruction = "All points placed. Press ENTER to save, or BACKSPACE to undo."

        # Draw instruction bar at top
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(
            frame, instruction, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.HIGHLIGHT_COLOR, 1, cv2.LINE_AA,
        )

        # Draw zoom inset if active
        if self.show_zoom and self.zoom_center is not None:
            frame = self._draw_zoom_inset(frame)

        return frame

    def _draw_zoom_inset(self, frame: np.ndarray) -> np.ndarray:
        """Draw a zoomed inset in the top-right corner."""
        h, w = frame.shape[:2]
        inset_size = min(200, w // 4)
        zx, zy = self.zoom_center
        half = inset_size // (2 * self.zoom_factor)

        # Clamp crop region
        y1 = max(0, zy - half)
        y2 = min(h, zy + half)
        x1 = max(0, zx - half)
        x2 = min(w, zx + half)

        crop = self.original_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return frame

        zoomed = cv2.resize(crop, (inset_size, inset_size), interpolation=cv2.INTER_LINEAR)

        # Draw reticle: circle + center dot
        mid = inset_size // 2
        cv2.circle(zoomed, (mid, mid), 15, self.HIGHLIGHT_COLOR, 1, cv2.LINE_AA)
        cv2.circle(zoomed, (mid, mid), 2, self.HIGHLIGHT_COLOR, -1, cv2.LINE_AA)

        # Place in top-right
        frame[40:40 + inset_size, w - inset_size - 10:w - 10] = zoomed
        cv2.rectangle(frame, (w - inset_size - 10, 40), (w - 10, 40 + inset_size), self.HIGHLIGHT_COLOR, 1)
        return frame

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.zoom_center = (x, y)
            self.show_zoom = True
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.current_index < NUM_KEYPOINTS:
                self.points[self.current_index] = (float(x), float(y))
                self.current_index += 1

    def run(self) -> CourtKeypoints | None:
        """Run the calibration UI. Returns CourtKeypoints or None if cancelled."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1280, 720)
        cv2.setMouseCallback(self.WINDOW_NAME, self.on_mouse)

        print("\nCourt Calibration Tool")
        print("=" * 40)
        print("  Left click  — place keypoint")
        print("  Backspace   — undo last point")
        print("  S           — skip current point (mark as not visible)")
        print("  Enter/Space — save and exit (when all points placed)")
        print("  Escape      — cancel")
        print()

        while True:
            display = self._draw()
            cv2.imshow(self.WINDOW_NAME, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # Escape
                cv2.destroyAllWindows()
                return None

            elif key == 8:  # Backspace — undo
                if self.current_index > 0:
                    self.current_index -= 1
                    # Skip back over None entries to find the last placed point
                    while self.current_index > 0 and self.points[self.current_index] is None:
                        self.current_index -= 1
                    self.points[self.current_index] = None

            elif key == ord("s") or key == ord("S"):  # Skip
                if self.current_index < NUM_KEYPOINTS:
                    name, desc = COURT_KEYPOINT_NAMES[self.current_index]
                    print(f"  Skipped {name} ({desc})")
                    self.points[self.current_index] = None
                    self.current_index += 1

            elif key in (13, 32):  # Enter or Space — save
                available = sum(1 for p in self.points if p is not None)
                if available < 4:
                    print(f"  Need at least 4 keypoints for homography (have {available})")
                    continue
                cv2.destroyAllWindows()
                return CourtKeypoints(points=self.points)

        cv2.destroyAllWindows()
        return None


def visualize_keypoints(frame: np.ndarray, keypoints: CourtKeypoints) -> None:
    """Display existing keypoints on a frame for verification."""
    display = frame.copy()

    # Draw lines
    for i, j in CalibrationTool.COURT_LINES:
        if keypoints.points[i] is not None and keypoints.points[j] is not None:
            pt1 = (int(keypoints.points[i][0]), int(keypoints.points[i][1]))
            pt2 = (int(keypoints.points[j][0]), int(keypoints.points[j][1]))
            cv2.line(display, pt1, pt2, CalibrationTool.LINE_COLOR, 2, cv2.LINE_AA)

    # Draw points
    for i, pt in enumerate(keypoints.points):
        if pt is None:
            continue
        center = (int(pt[0]), int(pt[1]))
        cv2.circle(display, center, 6, CalibrationTool.POINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(display, center, 6, (0, 0, 0), 1, cv2.LINE_AA)
        label = COURT_KEYPOINT_NAMES[i][0]
        cv2.putText(
            display, label, (center[0] + 10, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    info = f"Keypoints: {keypoints.num_available}/{NUM_KEYPOINTS} available"
    cv2.rectangle(display, (0, 0), (display.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(
        display, info, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA,
    )

    cv2.namedWindow("Keypoint Verification", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Keypoint Verification", 1280, 720)
    cv2.imshow("Keypoint Verification", display)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Manual court keypoint calibration")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to use")
    parser.add_argument("--output", default="court_keypoints.json", help="Output JSON path")
    parser.add_argument("--load", default=None, help="Load and visualize existing keypoints")
    return parser.parse_args()


def main():
    args = parse_args()

    # Read the target frame
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        sys.exit(1)

    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Cannot read frame {args.frame}")
        sys.exit(1)

    print(f"Video: {args.video}")
    print(f"Frame: {args.frame}")
    print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")

    # Visualize existing keypoints
    if args.load:
        keypoints = CourtKeypoints.load(args.load)
        print(f"\nLoaded {keypoints.num_available} keypoints from {args.load}")
        visualize_keypoints(frame, keypoints)
        return

    # Run calibration
    tool = CalibrationTool(frame)
    result = tool.run()

    if result is None:
        print("\nCalibration cancelled.")
        sys.exit(0)

    # Save
    result.save(args.output)
    available = result.num_available
    skipped = NUM_KEYPOINTS - available
    print(f"\nSaved {available} keypoints to {args.output}")
    if skipped > 0:
        print(f"  ({skipped} keypoints skipped)")


if __name__ == "__main__":
    main()
