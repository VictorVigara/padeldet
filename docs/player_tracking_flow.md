# Player Tracking Flow — `padeldet/player_tracker/player_tracker.py`

A full walkthrough of the player-tracking pipeline from the moment the match video
begins streaming into the system until a final list of identity-stable
`FrameDetections` is emitted. The entry point is
`PlayerTracker.detect_and_track(frames, start_frame, stride)`.

The tracker collaborates with three other modules to do its job:

- `filters.py` — pre-tracker box filtering and post-tracker enrichment
  (court projection, ankle midpoint from pose).
- `identity.py` — `LogicalPlayerManager`, the state machine that maps temporary
  tracker IDs to persistent logical player IDs **1–4** via a five-step
  cascade.
- `selector.py` — interactive OpenCV UI used when automatic selection is
  disabled or impossible.

---

## 1. Construction & initialization

`PlayerTracker.__init__` wires up four things:

1. A `PlayerTrackerConfig` — every threshold (confidences, IoU, box geometry
   limits, court margin, tracker timing, identity recovery distances,
   selection-window length, pose keypoint indices…) lives here.
2. An optional `CourtProjection` (homography from image pixels to court meters).
   Without it the tracker still runs but loses every court-aware feature:
   ROI filtering, depth scoring, structural reassignment, side-flip detection.
3. The YOLO detector (default `yolo26m.pt`, COCO person class).
4. An optional YOLO pose model used solely to extract ankle midpoints — these
   give a much more accurate foot position than the bbox bottom-center.

A device is auto-detected (`cuda` or `cpu`) unless `config.device` overrides it.

Then `_init_tracker()` constructs either:

- **ByteTrack** (supervision) — pure IoU-based, no appearance model. Cheap.
- **DeepOCSORT** (boxmot + OSNet ReID) — appearance-aware Kalman tracker.
  `max_age` is set in *seconds* (`tracker_max_age_sec`) and converted to
  frames using the **effective** FPS so behavior stays consistent when the
  user runs with a stride > 1. A boxmot quirk is patched here: when
  `CUDA_VISIBLE_DEVICES` is set to a non-numeric string (Modal does this),
  boxmot's `select_device` rejects the string `"cuda"`, so the integer index
  `0` is passed instead.

If pose is configured, a second YOLO model is loaded.

---

## 2. The main entry: `detect_and_track`

The pipeline is structured into **five explicit phases**.

### Phase 0 — Bookkeeping

```
fps           = video_info.fps (or 30 fallback)
effective_fps = fps / stride
n_sel         = round(selection_window_sec * effective_fps), clamped to [30, len(frames)]
```

The tracker is re-initialized with `effective_fps` so any time-based parameter
(track `max_age`, etc.) is calibrated against *real* seconds — important when
analysis is sub-sampled.

The frame list is split into two contiguous segments:

- `init_bgr`   — the **selection window** (first `n_sel` frames).
- `rest_bgr`   — everything afterwards.

Their absolute frame indices in the source video are reconstructed from
`start_frame` and `stride` so that downstream consumers (metrics, overlays,
ball-tracker fusion) see the true timestamps.

### Phase 1 — Process the selection window

`_run_frames` streams `init_bgr` through YOLO (and pose, if loaded) in
batches of `config.batch_size`. For every batch:

1. YOLO predicts persons only (`classes=[person_class_id]`).
2. Pose model (if any) predicts keypoints in parallel on the same batch.
3. For each frame the helper `_process_result` does:
   - `supervision.Detections.from_ultralytics(result)` and filter to person.
   - `filter_sv_detections_for_tracking` rejects boxes that:
     - have non-finite coords
     - are inverted (`x2 <= x1` or `y2 <= y1`)
     - are smaller than `min_box_height_px` or `min_box_area_px`
     - have aspect ratio outside `[min_box_aspect_ratio, max_box_aspect_ratio]`
     - have zero overlap with the frame
   - Run the chosen tracker:
     - `_track_bytetrack`: `byte_track.update_with_detections(...)`, wrap as
       `TrackedDetection`s with their assigned `track_id`.
     - `_track_deepocsort`: stack `[xyxy, conf, cls]`, call
       `deep_tracker.update(...)`; wraps the same way. Catches
       `np.linalg.LinAlgError` from the Kalman filter and emits an empty
       result rather than crashing the pipeline on a degenerate frame.
   - `enrich_ankle_midpoints` (only if pose was run): for each detection,
     find the pose box with highest IoU (≥ `pose_iou_threshold`), and use its
     left/right ankle keypoints (COCO indices 15 & 16). Each ankle is rejected
     if its (x, y) is exactly (0, 0) or its confidence is below
     `min_keypoint_conf`. If both ankles are valid the midpoint is averaged;
     if only one, that one is used; if none, no ankle is set.
   - `enrich_court_positions`: project the foot (ankle midpoint if available,
     else `bbox.bottom_center`) through the homography to court meters.
     Stored as `det.court_position`.

After the loop, the selection window is summarized to stdout with per-frame
in-court counts and court depths for debugging.

### Phase 2 — Player selection

Two modes, decided by `config.auto_select_players` and the availability of a
`court_projection`.

#### 2a. Automatic selection (`_auto_select_players`)

Default. Iterates over every `FrameDetections` in the window:

- `_score_and_select(fd)` keeps detections that have a tracker ID **and** a
  court position **strictly** inside the court (no margin). If fewer than 4
  remain, that frame is skipped. The 4 deepest detections (largest
  `_court_depth`, i.e. distance to the nearest court edge) are chosen, and
  the frame is scored by the **minimum** depth among those 4 — i.e. how deep
  even the shallowest of the chosen players is.
- The frame that maximizes this min-depth wins, and its 4 detections become
  the seed for the identity manager.

If no frame yields ≥4 in-court detections the function still returns the
last frame's best subset, but logs a warning.

The intuition: depth-based selection prefers frames where real players are
clearly inside the court (and thus deep), not frames where a spectator
standing right outside the side-glass is closer to a player than to any line.

#### 2b. Manual selection (`_best_selection_frame` + `PlayerSelectorUI`)

Used when auto-select is off, or when there is no court projection.

`_best_selection_frame` finds the frame most useful for clicking:

- Count distinct tracker IDs that pass the "in court" check (if the config
  requires it). With no court projection, every track counts.
- Score = `20 × inside_count − 50 × max(0, 4 − inside_count) − 5 × max(0, inside_count − 5)`
  — a strong reward for hitting 4 visible players, a heavy penalty for
  fewer, a mild penalty for too many extras.
- Add `2 × average confidence` and `2 × average court depth` as tie-breakers.

`PlayerSelectorUI` then opens an OpenCV window:

- Draws every tracker box, dimmed if unselected, green & thick if selected.
- Left-click selects the nearest track within 100 px; clicking a selected
  track deselects it.
- BACKSPACE undoes; ENTER confirms (only when exactly 4 are selected); ESC
  raises `RuntimeError`.
- Returns track IDs in click order, but **click order is discarded** when
  building logical IDs — see Phase 3.

If fewer than 4 visible tracks exist a warning is logged but the process
continues with whatever the user could click.

### Phase 3 — Initialize the `LogicalPlayerManager`
TODO: I would say player position should be 1 near left, 2 near right. And then 3 far left (but real left, which in the image is opposed), and 4 far right (but real right, inverted in camera because of playing in far court side)

The selected `TrackedDetection`s are sorted by `_position_sort_key`, which
buckets them as:

| Court quadrant | Logical ID |
|---|---|
| near-left  (y < L/2, x < W/2) | **1** |
| near-right (y < L/2, x ≥ W/2) | **2** |
| far-left   (y ≥ L/2, x < W/2) | **3** |
| far-right  (y ≥ L/2, x ≥ W/2) | **4** |

Click order is therefore irrelevant — logical IDs are deterministic from
court geometry. Each ID becomes a `LogicalPlayerState` carrying:

- current tracker ID
- last bbox
- last court position (meters)
- team side (`near`/`far`)
- play side (`left`/`right`) # TODO: Does it take into account the camera perspective? If we are playing in the far side, the left player in the image is actually the right player on the court, and vice versa. So maybe we should flip the logic based on team_side?
- last-seen frame
- `is_on_court` flag # TODO: This is the only condition to allow side change. And even if the player comes back to the same court side, side change allowed is not set to False. Probably a player should have on_court. And then, allow the change if on court of one of the team players is true in a different side, and the players of the team occupying that side are not on court. This would allow to change sides during the match, and also to change back to the original side if needed.

`team1_side` is initialized to `"near"` (logical 1+2 = team 1) and a
`_side_change_allowed` gate is left **closed**. Sides flip later only after
the gate is opened by an explicit court-exit event (see §3.5).

### Phase 4 — Back-fill identities across the selection window

The selection-winning frame index is `sel_frame_idx`. For every frame in the
selection window:

- **Before `sel_frame_idx`**: identity manager has no prior state — instead,
  the code keeps detections whose tracker ID is in the originally selected
  set, and copies the logical ID directly from the initial mapping. This is
  approximate (the tracker IDs were assumed not to have changed yet) but
  good enough for the first few seconds of the video to feed metrics &
  visualization. Other detections are dropped.
- **At or after `sel_frame_idx`**: routed through `identity_manager.update()`
  using the same logic as the live phase below.

### Phase 5 — Process the remainder of the video

`_run_frames(rest_bgr, …)` produces a stream of raw `FrameDetections`, each
of which is fed to `identity_manager.update()`. The result list of fully
labeled `FrameDetections` (containing **only** detections that map to a
logical player) is returned.

---

## 3. Identity assignment — the 5-step cascade

`LogicalPlayerManager.update(fd)` is where occlusion recovery happens. For
every frame it runs five ordered steps until each of the 4 logical players
has either been assigned a detection or is left "missing" for that frame.

### Step 1 — Direct tracker-ID match

For each logical state whose `current_track_id` is present in this frame's
detections:

- If a court position exists and is *invalid* (not inside the court and not
  exiting through a side door, see `_is_valid_player_position`), the match
  is **rejected**. This guards against the tracker re-using a player's ID
  on a spectator behind the glass. The state is then treated as missing.
- Otherwise the assignment goes through (reason: `direct_track_id`).

The court-validity gate accepts:
- Strict interior: `0 ≤ x ≤ W` and `0 ≤ y ≤ L`.
- Side-door exits: `x < 0` or `x > W` while `DOOR_Y_MIN ≤ y ≤ DOOR_Y_MAX`.

Everything else (e.g. spectators behind the far baseline) is rejected.

If every state was matched directly, the frame is returned immediately
(after team-side maintenance).

### Step 2 — `same_role` structural reassignment

For each *still-missing* state, find unassigned detections matching both its
`team_side` and `play_side`. The pair is only committed if it is
**symmetrically unique** — exactly one detection matches that state's role,
**and** exactly one state matches that detection's role. This prevents two
states from both fighting over the same quadrant during ambiguous frames.

Reason logged: `same_role`.

Unassigned detections are pre-filtered to those whose court depth is at
least `min_direct_match_court_depth_m` (default 5 cm) — boundary noise is
ignored.

### Step 3 — `same_team_side` structural reassignment

Same symmetric-uniqueness pattern but now matching only on `team_side`
(near vs far). Catches cases where two players on the same team have
swapped left/right (e.g. one walked behind the other and the tracker
shuffled IDs). Reason: `same_team_side`.

### Step 4 — Proximity fallback

Build a distance matrix between every remaining state's last known court
position and every remaining detection's court position. Greedily assign
the globally closest pair, then the next, etc., as long as the distance is
≤ `proximity_max_distance_m` (default 3 m). Reason: `proximity`.

Only states/detections with court positions participate; detections without
court positions are silently passed through.

### Step 5 — One-to-one fallback

If exactly one state and exactly one detection remain after steps 1–4, pair
them — provided the detection's position passes `_is_valid_player_position`.
Reason: `one_to_one_fallback`.

### What happens to detections that nobody claims

They are simply **dropped** — `LogicalPlayerManager.update` returns
`FrameDetections` containing only the assigned ones. The identity manager
never invents detections for missing players; that frame just has fewer
than 4 detections.

`missing_frames` is incremented for any state that wasn't matched — but
note that this counter is **never used to gate assignment**. It exists for
logging only. The `max_missing_frames` config field is kept for backwards
compatibility but no longer participates in decisions.

### `_assign` side effects

For every successful assignment:

- `state.current_track_id` is updated to the new tracker ID (a track-ID
  switch is logged for everything except `direct_track_id`).
- `det.logical_player_id` is stamped and `det.visible = True`.
- The state's last bbox, last court position, and last-seen frame are
  updated. `missing_frames` resets to 0.
- If the state had a court position **and** the detection's new position
  crosses out of the valid region (side-door exit), `is_on_court` flips
  from True → False, opening the side-change gate (see §3.5).
- If `team_side`/`play_side` were `None` (e.g. the seed had no projection)
  they're filled in opportunistically on the first valid position.

### 3.5 — Team-side maintenance

After all assignments, `_update_team_sides` runs. It only acts if the
side-change gate is open (a player has been observed leaving the court
through a side door since the last side change). When the gate is open it
checks whether team 1 (logical 1+2) and team 2 (logical 3+4) are observed
exclusively in the far half vs the near half, and if so flips the global. # TODO: HEre is not checked if team2 left the court to enable a side change. So probably the method mentioned in _update_state_matched TODO would be better, which is to check if on court of one of the team players is true in a different side, and the players of the team occupying that side are not on court. This would allow to change sides during the match, and also to change back to the original side if needed.

`team1_side` (with `_flip_state_team_sides` swapping near↔far and
left↔right on every state). The gate then closes again.

The resulting `FrameDetections.team_sides` map (`{1: …, 2: …}`) ships with
every frame so downstream code (heatmaps, formations) can render team
identity correctly even after the players have swapped courts between
games.

---

## 4. Cases the pipeline explicitly handles

| Case | How it's handled |
|---|---|
| Empty input | `detect_and_track` returns `[]` immediately. |
| Stride > 1 (sub-sampled analysis) | Tracker re-initialized with `effective_fps` so `max_age` stays calibrated in seconds; absolute frame indices reconstructed via `start_frame + i*stride`. |
| Modal / weird `CUDA_VISIBLE_DEVICES` | DeepOCSORT is given device index `0` instead of `"cuda"` to bypass boxmot's string validation. |
| DeepOCSORT Kalman blow-up | `np.linalg.LinAlgError` from `deep_tracker.update` caught; empty tracks returned for that frame. |
| Box garbage (NaN, inverted, micro, weird aspect, fully off-screen) | Rejected pre-tracker by `filter_sv_detections_for_tracking`. |
| Pose model misaligned with detection box | Pose features are dropped if IoU < `pose_iou_threshold`; foot falls back to `bbox.bottom_center`. |
| One ankle keypoint missing or low-confidence | Use the other; if both missing, no ankle midpoint. |
| Missing court projection | Court positions left `None`, ROI filter is a no-op, automatic selection cannot run, structural and proximity steps fall back to direct-ID-only matching. |
| Auto-selection finds < 4 in-court players | Warning logged, partial seed used; missing logical IDs are simply absent until they appear later (and even then only direct track-ID matches will rescue them, since their `last_court_position` was never set). |
| Manual selection cancelled | `PlayerSelectorUI.select` raises `RuntimeError` and the pipeline aborts. |
| Manual selection short of 4 | Warning logged, manager initialized with however many were clicked. |
| Spectator inheriting a player's tracker ID | `_is_valid_player_position` rejects matches outside court + side-door region in Step 1 and Step 5; player falls through to structural/proximity steps. |
| Boundary noise (player straddling line at ~0 m depth) | Filtered out of the unassigned pool when `_court_depth < min_direct_match_court_depth_m`. |
| Two players cross & swap tracker IDs (same team) | Step 3 (`same_team_side`) typically fixes it; if still ambiguous, Step 4 (`proximity`) picks the nearest. |
| Two players cross & swap roles entirely (same role bucket) | Step 2 (`same_role`) fixes it. |
| All four players' tracker IDs explode at once | Step 4 (`proximity`) reassigns each to the closest detection within 3 m. If still ambiguous, Step 5 handles the very last 1↔1 pair. |
| Players swap court ends between games | A side-door exit opens the side-change gate; `_update_team_sides` then detects "team 1 is all far, team 2 is all near" (or vice versa) and flips `team1_side` plus every state's `team_side`/`play_side`. The gate closes again until the next exit. |
| Tracker ID briefly disappears (occlusion) | Step 1 simply doesn't match this frame; `missing_frames` is bumped. On reappearance the same tracker ID is matched again by Step 1 (if DeepOCSORT's ReID kept it) or recovered via Steps 2–5 (if ID changed). |
| Detections that don't correspond to any logical player | Silently dropped from output — only assigned detections survive. |
| Frame before the selection-winning frame | Logical ID copied directly from the initial tracker-ID→logical-ID map; detections with tracker IDs not in the seed set are dropped. |
| First few seconds before the seed frame have no projection | The fallback assigns logical IDs even with no court info, since it uses tracker IDs only. |

---

## 5. Outputs

`detect_and_track` returns a `list[FrameDetections]` covering every frame in
the input iterator. For each frame:

- `detections` contains **only** items that have been mapped to a logical
  player (1–4). `det.logical_player_id` is set, `det.visible = True`,
  `det.is_on_court` reflects the current on-court state.
- `det.court_position` is in meters (origin near-baseline-left) when a
  homography was available.
- `det.ankle_midpoint` and `det.keypoints` are populated when pose was run
  and a confident match existed.
- `fd.team_sides` describes which team (1 or 2) is currently on the near/far
  side. It updates automatically when a side change is detected.

Frames where fewer than 4 logical players were recovered simply contain
fewer detections — there is no padding or interpolation at this layer; the
metrics module is responsible for filling short gaps later.
