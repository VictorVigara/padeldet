"""Detection filtering and enrichment functions for the player tracker."""

from __future__ import annotations

import logging

import numpy as np
import supervision as sv

from ..court.projection import CourtProjection
from ..models import FrameDetections, TrackedDetection
from .config import PlayerTrackerConfig


def box_iou(box_a: list[float], boxes_b: np.ndarray) -> np.ndarray:
    """IoU between one box [x1,y1,x2,y2] and array of boxes (N,4)."""
    ax1, ay1, ax2, ay2 = box_a
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ix1 = np.maximum(ax1, boxes_b[:, 0])
    iy1 = np.maximum(ay1, boxes_b[:, 1])
    ix2 = np.minimum(ax2, boxes_b[:, 2])
    iy2 = np.minimum(ay2, boxes_b[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_b = (
        np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0])
        * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    )
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def xyxy_foot_point(xyxy: np.ndarray) -> np.ndarray:
    """Return pixel foot point [cx, y2] for a raw xyxy box."""
    return np.array([(xyxy[0] + xyxy[2]) / 2.0, xyxy[3]])


def project_foot_to_court(
    foot_img: np.ndarray, court_projection: CourtProjection | None
) -> np.ndarray | None:
    """Project a pixel foot point to court meters via homography.

    Returns None if no court_projection or if projection raises.
    """
    if court_projection is None:
        return None
    try:
        cx, cy = court_projection.project_point(
            (float(foot_img[0]), float(foot_img[1]))
        )
        return np.array([cx, cy])
    except Exception:
        return None


def is_court_point_valid(
    court_pt: np.ndarray,
    court_projection: CourtProjection | None,
    margin_m: float,
) -> bool:
    """Check if a court-meter point is within court bounds + margin.

    Court origin is at the near-baseline-left corner; dimensions come from
    the projection (doubles 10 × 20 m, singles 6 × 20 m).
    """
    if court_projection is None:
        return True
    x, y = float(court_pt[0]), float(court_pt[1])
    width_m = court_projection.width_m
    length_m = court_projection.length_m
    return (-margin_m <= x <= width_m + margin_m) and (-margin_m <= y <= length_m + margin_m)


def detection_is_valid_box_xyxy(
    xyxy: np.ndarray,
    frame_shape: tuple[int, ...],
    config: PlayerTrackerConfig,
) -> bool:
    """Sanity-check a raw xyxy box: finite, oriented, min size, aspect ratio, in-frame."""
    if not np.all(np.isfinite(xyxy)):
        return False
    x1, y1, x2, y2 = xyxy
    if x2 <= x1 or y2 <= y1:
        return False
    h = y2 - y1
    w = x2 - x1
    fh, fw = frame_shape[:2]
    if h < config.min_box_height_px:
        return False
    if h * w < config.min_box_area_px:
        return False
    ratio = h / max(w, 1e-6)
    if not (config.min_box_aspect_ratio <= ratio <= config.max_box_aspect_ratio):
        return False
    # Require at least partial overlap with the image
    if x2 < 0 or y2 < 0 or x1 > fw or y1 > fh:
        return False
    return True


def filter_sv_detections_for_tracking(
    sv_dets: sv.Detections,
    frame_shape: tuple[int, ...],
    config: PlayerTrackerConfig,
    court_projection: CourtProjection | None,
) -> sv.Detections:
    """Apply box quality and court-ROI filters. Returns filtered sv.Detections."""
    if len(sv_dets) == 0:
        return sv_dets

    mask = np.zeros(len(sv_dets), dtype=bool)
    for i in range(len(sv_dets)):
        xyxy = sv_dets.xyxy[i]
        if not detection_is_valid_box_xyxy(xyxy, frame_shape, config):
            continue
        mask[i] = True

    logging.debug(
        f"detection filter: {len(sv_dets)} raw → {int(mask.sum())} kept"
    )
    return sv_dets[mask]


def enrich_court_positions(
    fd: FrameDetections,
    court_projection: CourtProjection | None,
) -> None:
    """Set court_position on each detection in place after tracking."""
    if court_projection is None:
        return
    for det in fd.detections:
        if det.ankle_midpoint is not None:
            foot = np.array(det.ankle_midpoint)
        else:
            foot = np.array([det.bbox.bottom_center[0], det.bbox.bottom_center[1]])
        court_pt = project_foot_to_court(foot, court_projection)
        if court_pt is not None:
            det.court_position = (float(court_pt[0]), float(court_pt[1]))


def enrich_ankle_midpoints(
    fd: FrameDetections,
    pose_result,
    config: PlayerTrackerConfig,
) -> None:
    """Set ankle_midpoint on each detection using pose keypoints (in place)."""
    kps = pose_result.keypoints
    if kps is None or kps.xy is None:
        return
    pose_boxes = pose_result.boxes.xyxy.cpu().numpy()  # (N, 4)
    kp_xy = kps.xy.cpu().numpy()                       # (N, K, 2)
    kp_conf = kps.conf.cpu().numpy() if kps.conf is not None else None  # (N, K) or None

    lf_idx = config.left_foot_kp_idx
    rf_idx = config.right_foot_kp_idx

    for det in fd.detections:
        if len(pose_boxes) == 0:
            break
        ious = box_iou(det.bbox.as_xyxy(), pose_boxes)
        best = int(np.argmax(ious))
        if ious[best] < config.pose_iou_threshold:
            continue

        det.keypoints = [
            (float(kp_xy[best, k, 0]), float(kp_xy[best, k, 1]),
             float(kp_conf[best, k]) if kp_conf is not None else 1.0)
            for k in range(kp_xy.shape[1])
        ]

        lf = kp_xy[best, lf_idx]
        rf = kp_xy[best, rf_idx]

        lf_valid = not (lf[0] == 0.0 and lf[1] == 0.0)
        rf_valid = not (rf[0] == 0.0 and rf[1] == 0.0)

        if kp_conf is not None:
            if kp_conf[best, lf_idx] < config.min_keypoint_conf:
                lf_valid = False
            if kp_conf[best, rf_idx] < config.min_keypoint_conf:
                rf_valid = False

        if not lf_valid and not rf_valid:
            continue

        if lf_valid and rf_valid:
            mx = (lf[0] + rf[0]) / 2.0
            my = (lf[1] + rf[1]) / 2.0
        elif lf_valid:
            mx, my = float(lf[0]), float(lf[1])
        else:
            mx, my = float(rf[0]), float(rf[1])

        det.ankle_midpoint = (mx, my)
