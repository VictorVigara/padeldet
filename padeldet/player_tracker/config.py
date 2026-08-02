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
    # How long (seconds) to keep a boxmot track alive without a detection match.
    # This bridges SHORT/known off-court gaps (e.g. a player chasing a ball,
    # ~10 s) so the raw tracker ID survives and re-entry is a seamless direct
    # track-id match. It is NOT the mechanism for undefined-length exits — those
    # are handled by LogicalPlayerManager, which never drops a logical player
    # (P1–P4) and re-attaches a fresh tracker ID on re-entry via its recovery
    # cascade. Kept modest on purpose: a very large value makes boxmot predict a
    # drifting "ghost" box for the absent player that can latch onto a spectator
    # near the glass/door. Note: boxmot bumps its appearance buffer max_obs to
    # max_age+5, so larger values also cost memory. See CV-003 / re-entry notes.
    tracker_max_age_sec: float = 10.0
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

    # Break detection: when zero valid in-court players are matched for this many
    # consecutive seconds, the manager enters a "break" — it emits empty frames
    # and FREEZES all logical-player state (no recovery, no last_seen bump) until
    # players return. Handles changeovers / injuries / between-point gaps without
    # letting recovery snap slots onto near-glass noise while nobody is playing.
    break_min_empty_sec: float = 2.0

    # ReID appearance gallery (DeepOCSORT/OSNet only; no-op under bytetrack).
    # Per-logical-player EMA embedding: emb = alpha*emb + (1-alpha)*new (mirrors
    # boxmot's update_emb). Used to re-attach identities when the court refills
    # after a fully-empty break, since teams may have swapped ends while empty —
    # position alone can't tell which team is which, so we match by appearance.
    reid_gallery_alpha: float = 0.9
    # Minimum cosine similarity to accept a post-break appearance re-attach.
    reid_reattach_min_cosine: float = 0.3

    # Appearance-primary assignment: once every logical player has a mature
    # gallery, each frame's valid in-court detections are matched to logical ids
    # by gallery cosine FIRST (Hungarian), ignoring position — appearance is the
    # authority over geometry. This is what follows a player through an end-swap
    # (teams changing court sides), which pure geometry inverts. Empirically
    # OSNet separates the 4 players ~0.98 vs ~0.48 for position (see
    # scripts/analyze_player_separability.py).
    reid_primary: bool = True
    # Min cosine to accept an appearance-primary match (stricter than re-attach —
    # this runs every frame, so we only override geometry on a confident match).
    reid_primary_min_cosine: float = 0.5
    # A player must be the CLEAR best for a detection: its cosine must beat the
    # runner-up player's by at least this margin. Prevents the confusable pair
    # (P1/P4) from being confidently assigned to the wrong one on a close call.
    reid_primary_margin: float = 0.05
    # Only fold a detection into a player's gallery when its match cosine is at
    # least this high, so a transient mislabel can't poison the appearance model.
    reid_gallery_update_min_cosine: float = 0.6
    # Number of clean confident samples averaged into the FROZEN anchor gallery
    # before it stops updating. Appearance-primary matches against the anchor, so a
    # later mislabel can't poison it and a wrong id self-corrects next frame.
    reid_anchor_samples: int = 60

    # Door zone: a detection in the door y-band OR within this many meters of any
    # court edge is treated as "transitioning" — near the door all players pile up
    # at the same spot, so position is meaningless. In the door zone assignment is
    # ReID-ONLY (appearance-primary); the geometric steps (proximity, one-to-one,
    # direct-by-position) never assign there. A door-zone detection with no
    # confident gallery match is left unassigned rather than mislabelled.
    door_zone_edge_margin_m: float = 1.0

    # Selection frame scoring
    selection_require_in_court: bool = True
    # When True and court_projection is available: use depth-based auto-selection (no UI).
    # When False: fall back to manual PlayerSelectorUI click.
    auto_select_players: bool = True
