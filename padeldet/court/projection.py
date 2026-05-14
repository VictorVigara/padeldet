"""Homography-based court projection and mini-court visualization."""

from __future__ import annotations

import cv2
import numpy as np

from ..models import FrameDetections, TrackedDetection
from .models import (
    COURT_KEYPOINTS_2D,
    COURT_LENGTH,
    COURT_WIDTH,
    NET_POSITION,
    SERVICE_BOX_DEPTH,
    CourtKeypoints,
)


class CourtProjection:
    """Projects pixel coordinates to real court coordinates via homography."""

    def __init__(self, keypoints: CourtKeypoints, margin_meters: float = 0.5):
        if keypoints.num_available < 4:
            raise ValueError(
                f"Need at least 4 keypoints for homography, got {keypoints.num_available}"
            )
        self.keypoints = keypoints
        self.margin = margin_meters
        self._H = self._compute_homography()
        self._court_polygon_px = self._compute_court_polygon()

    def _compute_homography(self) -> np.ndarray:
        """Compute homography from pixel keypoints to 2D court meters."""
        pixel_pts, court_pts = self.keypoints.get_available_pairs()
        src = np.array(pixel_pts, dtype=np.float64)
        dst = np.array(court_pts, dtype=np.float64)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            raise ValueError("Could not compute homography — check keypoint quality")
        return H

    def _compute_court_polygon(self) -> np.ndarray:
        """Get the 4 outer court corners as a pixel polygon (for inside-court checks)."""
        # k1, k2, k12, k11 = corners in order
        corners_idx = [0, 1, 11, 10]  # k1, k2, k12, k11
        corners = []
        for i in corners_idx:
            if self.keypoints.points[i] is not None:
                corners.append(self.keypoints.points[i])
            else:
                # If a corner is missing, project back from court coords
                court_pt = np.array(
                    [COURT_KEYPOINTS_2D[i][0], COURT_KEYPOINTS_2D[i][1], 1.0]
                )
                H_inv = np.linalg.inv(self._H)
                px = H_inv @ court_pt
                px = px[:2] / px[2]
                corners.append((float(px[0]), float(px[1])))
        return np.array(corners, dtype=np.float32)

    @classmethod
    def from_matrix(cls, H: np.ndarray, margin_meters: float = 0.5) -> "CourtProjection":
        """Construct from a pre-computed 3×3 homography matrix (skips keypoint check)."""
        obj = object.__new__(cls)
        obj.keypoints = None
        obj.margin = margin_meters
        obj._H = np.array(H, dtype=np.float64)
        # Back-project the 4 court corners from court-meter space to pixel space
        H_inv = np.linalg.inv(obj._H)
        corners_m = [(0.0, 0.0), (COURT_WIDTH, 0.0), (COURT_WIDTH, COURT_LENGTH), (0.0, COURT_LENGTH)]
        px_corners = []
        for cx, cy in corners_m:
            pt = H_inv @ np.array([cx, cy, 1.0])
            pt /= pt[2]
            px_corners.append((float(pt[0]), float(pt[1])))
        obj._court_polygon_px = np.array(px_corners, dtype=np.float32)
        return obj

    @property
    def homography(self) -> np.ndarray:
        return self._H

    def project_point(self, pixel_xy: tuple[float, float]) -> tuple[float, float]:
        """Project a pixel coordinate to court meters."""
        pt = np.array([pixel_xy[0], pixel_xy[1], 1.0], dtype=np.float64)
        result = self._H @ pt
        result = result / result[2]
        return (float(result[0]), float(result[1]))

    def project_detection(self, det: TrackedDetection) -> tuple[float, float]:
        """Project a detection's feet position (bottom_center) to court meters."""
        return self.project_point(det.bbox.bottom_center)

    def project_frame(
        self, fd: FrameDetections
    ) -> list[tuple[int | None, float, float]]:
        """Project all detections in a frame. Returns (track_id, x_m, y_m) list."""
        results = []
        for det in fd.detections:
            x, y = self.project_detection(det)
            results.append((det.track_id, x, y))
        return results

    def is_inside_court_px(
        self, pixel_xy: tuple[float, float], margin_px: float = 20.0
    ) -> bool:
        """Check if a pixel point is inside the court polygon (with margin)."""
        dist = cv2.pointPolygonTest(
            self._court_polygon_px, (float(pixel_xy[0]), float(pixel_xy[1])), True
        )
        return dist >= -margin_px

    def is_inside_court(self, court_xy: tuple[float, float]) -> bool:
        """Check if a court-meter point is inside the court (with margin)."""
        x, y = court_xy
        return (
            -self.margin <= x <= COURT_WIDTH + self.margin
            and -self.margin <= y <= COURT_LENGTH + self.margin
        )

    def court_polygon_pixels(self) -> np.ndarray:
        """Return the 4 outer court corners as pixel coordinates."""
        return self._court_polygon_px.copy()


# ── Mini-court visualization ────────────────────────────────────────

# Track ID → color mapping (BGR)
# Index 0–3 match AppColors.player1–4 (green, orange, blue, pink); extras for fallback.
_PLAYER_COLORS = [
    (0, 200, 0),     # P1 green
    (0, 165, 255),   # P2 orange
    (255, 0, 0),     # P3 blue
    (180, 105, 255), # P4 pink
    (255, 0, 255),   # fallback magenta
    (255, 255, 0),   # fallback cyan
]


def draw_mini_court(
    frame: np.ndarray,
    projection: CourtProjection,
    detections: FrameDetections,
    width_frac: float = 0.15,
    padding: int = 15,
) -> np.ndarray:
    """Draw a mini 2D court overlay with player positions.

    Args:
        frame: video frame to draw on (modified in place and returned)
        projection: court projection for coordinate mapping
        detections: player detections for this frame
        width_frac: mini-court width as fraction of frame width
        padding: pixel padding from frame edge
    """
    fh, fw = frame.shape[:2]
    court_w = int(fw * width_frac)
    court_h = int(court_w * (COURT_LENGTH / COURT_WIDTH))  # maintain 1:2 aspect

    # Position: bottom-right corner
    x_offset = fw - court_w - padding
    y_offset = fh - court_h - padding

    # Draw court background (semi-transparent)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x_offset - 5, y_offset - 5),
        (x_offset + court_w + 5, y_offset + court_h + 5),
        (40, 40, 40),
        -1,
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def court_to_mini(cx: float, cy: float) -> tuple[int, int]:
        """Convert court meters to mini-court pixel position."""
        # Flip Y so near baseline is at bottom of mini-court
        mx = x_offset + int(cx / COURT_WIDTH * court_w)
        my = y_offset + court_h - int(cy / COURT_LENGTH * court_h)
        return (mx, my)

    # Draw court lines (white)
    line_color = (200, 200, 200)
    line_thickness = 1

    # Outer boundary
    corners = [
        court_to_mini(0, 0),
        court_to_mini(COURT_WIDTH, 0),
        court_to_mini(COURT_WIDTH, COURT_LENGTH),
        court_to_mini(0, COURT_LENGTH),
    ]
    for i in range(4):
        cv2.line(frame, corners[i], corners[(i + 1) % 4], line_color, line_thickness)

    # Net
    cv2.line(
        frame,
        court_to_mini(0, NET_POSITION),
        court_to_mini(COURT_WIDTH, NET_POSITION),
        (0, 200, 255),
        2,
    )

    # Service lines
    cv2.line(
        frame,
        court_to_mini(0, SERVICE_BOX_DEPTH),
        court_to_mini(COURT_WIDTH, SERVICE_BOX_DEPTH),
        line_color,
        line_thickness,
    )
    cv2.line(
        frame,
        court_to_mini(0, COURT_LENGTH - SERVICE_BOX_DEPTH),
        court_to_mini(COURT_WIDTH, COURT_LENGTH - SERVICE_BOX_DEPTH),
        line_color,
        line_thickness,
    )

    # Center service lines
    cv2.line(
        frame,
        court_to_mini(COURT_WIDTH / 2, SERVICE_BOX_DEPTH),
        court_to_mini(COURT_WIDTH / 2, COURT_LENGTH - SERVICE_BOX_DEPTH),
        line_color,
        line_thickness,
    )

    # Draw player positions
    fallback_color_map: dict[int, tuple[int, int, int]] = {}
    fallback_color_idx = len(_PLAYER_COLORS) - 2  # start at the fallback entries

    for det in detections.detections:
        court_pos = projection.project_detection(det)

        # Use logical_player_id for a stable color (P1–P4 match AppColors); fall back to track_id.
        if det.logical_player_id is not None:
            color = _PLAYER_COLORS[(det.logical_player_id - 1) % len(_PLAYER_COLORS)]
        else:
            tid = det.track_id if det.track_id is not None else -1
            if tid not in fallback_color_map:
                fallback_color_map[tid] = _PLAYER_COLORS[fallback_color_idx % len(_PLAYER_COLORS)]
                fallback_color_idx += 1
            color = fallback_color_map[tid]

        # Only draw if inside court (with generous margin for visualization)
        if projection.is_inside_court(court_pos):
            mx, my = court_to_mini(court_pos[0], court_pos[1])
            cv2.circle(frame, (mx, my), 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (mx, my), 5, (255, 255, 255), 1, cv2.LINE_AA)
            label = (
                f"P{det.logical_player_id}"
                if det.logical_player_id is not None
                else str(det.track_id) if det.track_id is not None
                else None
            )
            if label is not None:
                cv2.putText(
                    frame,
                    label,
                    (mx + 6, my - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                )

    return frame
