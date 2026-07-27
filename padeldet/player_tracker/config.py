"""Configuration for player detection and tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..match_profile import MatchType


@dataclass
class PlayerTrackerConfig:
    """Configuration for player detection and tracking."""

    # High-level match type. Resolves to a `MatchProfile` (court dimensions +
    # number of players to track) via `padeldet.match_profile.MATCH_PROFILES`.
    match_type: MatchType = "doubles" # [doubles, singles]

    # YOLO-only (ignored when detector_type == "rfdetr"); RFDETRBase() uses its own bundled checkpoint.
    model_path: Literal["yolov8n.pt", "yolov8m.pt", "yolo26m.pt"] = "yolov8n.pt"
    confidence: float = 0.1
    iou_threshold: float = 0.1
    imgsz: int = 960
    batch_size: int = 8
    person_class_id: int = 0  # COCO person class
    max_players: int = 10
    # Seconds of video to process before pausing for manual player selection
    selection_window_sec: float = 2.0
    # Detector backend. RF-DETR is the default (Apache-2.0, commercial-use safe);
    # YOLO (Ultralytics, AGPL/paid-license) is kept as a fallback only.
    detector_type: Literal["yolo", "rfdetr"] = "rfdetr"
    # RF-DETR-only knobs (ignored when detector_type == "yolo")
    rfdetr_threshold: float = 0.5
    rfdetr_resolution: int | None = None  # must be % 56 == 0 if set
    # Tracker selection
    tracker_type: Literal["bytetrack", "deepocsort"] = "deepocsort"
    # DeepOCSORT-specific
    reid_weights: str = "weights/osnet_x0_25_msmt17.pt"
    fp16: bool = True
    # How long (seconds) to keep a track alive without a detection match.
    tracker_max_age_sec: float = 2.0
    tracker_min_hits: int = 1

    # Court / detection box filtering
    court_margin_m: float = 0.75
    min_box_height_px: float = 25.0
    min_box_area_px: float = 500.0
    max_box_aspect_ratio: float = 6.0
    min_box_aspect_ratio: float = 0.8

    # Pose model for ankle midpoint extraction (opt-in)
    pose_model_path: str | None = None
    pose_confidence: float = 0.0001
    left_foot_kp_idx: int = 15   # COCO left ankle
    right_foot_kp_idx: int = 16  # COCO right ankle
    min_keypoint_conf: float = 0.1
    pose_iou_threshold: float = 0.3

    # Device override — "cuda", "cpu", or None for auto-detect (torch.cuda.is_available())
    device: str | None = None

    # Logical identity / occlusion recovery
    # Not used for assignment gating — kept for backwards compatibility
    max_missing_frames: int = 90
    max_recovery_distance_m: float = 2.0
    use_simple_occlusion_recovery: bool = True
    # Minimum distance from any court edge (m) for a direct track_id match to be trusted.
    # Detections closer than this are likely spectators near the boundary and will not be
    # accepted as a direct match — they fall through to structural reassignment instead.
    min_direct_match_court_depth_m: float = 0.05
    proximity_max_distance_m: float = 3.0
    # Width (meters) of a band around the net line within which a detection's
    # team side is treated as unknown for reassignment purposes only. Avoids
    # snapping a re-entering player to the wrong team slot while their court
    # projection is still noisy near the door (the y-band around the net line).
    net_deadzone_m: float = 0.5

    # Time-window distance gate for structural reassignment (Steps 2, 3, 5).
    # During the first `recovery_gating_window_sec` seconds after a player is
    # lost, a candidate detection is only accepted if its court position lies
    # within a growing radius of the state's last_court_position. The radius
    # interpolates linearly from `initial_radius_m` (t=0) to `final_radius_m`
    # (t=window). After the window the gate is off (current behaviour).
    recovery_gating_window_sec: float = 2.0
    recovery_gating_initial_radius_m: float = 1.0
    recovery_gating_final_radius_m: float = 3.0

    # Selection frame scoring
    selection_require_in_court: bool = True
    # When True and court_projection is available: use depth-based auto-selection (no UI).
    # When False: fall back to manual PlayerSelectorUI click.
    auto_select_players: bool = True
