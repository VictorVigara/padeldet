# Player Tracking Flow — `padeldet/player_tracker/`

A full walkthrough of the player-tracking pipeline, from the moment the match
video begins streaming into the system until a final list of identity-stable
`FrameDetections` is emitted. The entry point is
`PlayerTracker.detect_and_track(frames, start_frame, stride)`.

The module is split across five files:

- `player_tracker.py` — `PlayerTracker`: detection, tracking, selection, and the
  five-phase pipeline.
- `config.py` — `PlayerTrackerConfig`: every threshold in one dataclass.
- `filters.py` — pre-tracker box filtering and post-tracker enrichment (court
  projection, ankle midpoint from pose).
- `identity.py` — `LogicalPlayerManager`, the state machine that maps temporary
  tracker IDs to persistent logical player IDs.
- `selector.py` — interactive OpenCV UI used when automatic selection is
  disabled or impossible.

**Match profiles.** The number of logical players and the court dimensions come
from `config.match_type` → `padeldet.match_profile.MATCH_PROFILES`:

| Profile | Players | Court | Logical layout |
|---|---|---|---|
| `doubles` (default) | 4 | 10 × 20 m | P1 near-left, P2 near-right, P3 far-left, P4 far-right (own left/right) |
| `singles` | 2 | 6 × 20 m | P1 near, P2 far (`play_side` stays `None`) |

Everything below describes doubles unless singles is called out explicitly.

---

## 1. Construction & initialization

`PlayerTracker.__init__` wires up:

1. A `PlayerTrackerConfig` — all thresholds (confidences, IoU, box geometry
   limits, court margin, tracker timing, identity recovery distances, ReID
   thresholds, selection-window length, pose keypoint indices…).
2. A `MatchProfile` resolved from `config.match_type`, giving `n_players` and
   the court dimensions.
3. An optional `CourtProjection` (homography from image pixels to court meters).
   Without it the tracker still runs but loses every court-aware feature: depth
   scoring, auto-selection, structural reassignment, side-flip detection.
4. The detector — **RF-DETR by default** (`detector_type="rfdetr"`, Apache-2.0,
   commercial-use safe). YOLO is kept as a fallback (`detector_type="yolo"`,
   AGPL/paid license).
5. An optional YOLO pose model used solely to extract ankle midpoints — these
   give a much more accurate foot position than the bbox bottom-center.

Device is auto-detected (`cuda` or `cpu`) unless `config.device` overrides it.

### RF-DETR specifics

- `RFDETRBase(device=…, resolution=…)`; `rfdetr_resolution` must be divisible
  by 56 if set.
- `optimize_for_inference(batch_size=config.batch_size)` is called once at
  startup (fuses/JITs ops). The compiled graph is **batch-size-specific**, so
  `_predict_persons` pads a short final batch with repeated frames and drops
  the padded results.
- The person class id is resolved by `_resolve_rfdetr_person_id()`, which reads
  the dict-keyed `COCO_CLASSES` mapping from the `rfdetr` package (COCO-91
  sparse ids, `person == 1`). `RFDETRBase.class_names` is a 0-indexed *display*
  list and does **not** match `predict()`'s `class_id` output — do not use it.
  Falls back to `1`. Class ids are then renormalized to
  `config.person_class_id`.

### Tracker initialization

`_init_tracker(effective_fps)` constructs either:

- **ByteTrack** (supervision) — pure IoU-based, no appearance model. Cheap, but
  produces **no embeddings**, so every ReID feature below silently no-ops.
- **DeepOCSORT** (boxmot + OSNet ReID) — default, appearance-aware Kalman
  tracker. Two things matter here:
  - `tracker_max_age_sec` (default 10 s) is converted to frames using the
    **effective** FPS so behavior is stable under stride > 1. It is
    deliberately modest: it only bridges *short* known gaps so the raw tracker
    id survives. Undefined-length exits are the `LogicalPlayerManager`'s job.
    A very large `max_age` makes boxmot predict a drifting "ghost" box that can
    latch onto a spectator near the glass.
  - boxmot ≥ 18 takes a **pre-built ReID model object** via `reid_model=`, not
    a weights path. Passing the old `reid_weights=`/`device=`/`half=` API
    silently falls into `**kwargs`, leaves `tracker.model = None`, and crashes
    on the first `update()`. So a `ReID` backend is built explicitly and its
    `.model` handed over, then asserted non-None (see CV-003 in `BUGS.md`).
  - A boxmot quirk is patched: when `CUDA_VISIBLE_DEVICES` is a non-numeric
    string (Modal sets it to `"cuda"`), boxmot's `select_device` rejects the
    string `"cuda"`, so the integer index `0` is passed instead.

---

## 2. The main entry: `detect_and_track`

### Phase 0 — Bookkeeping

```
fps           = video_info.fps (or 30 fallback)
effective_fps = fps / stride
n_sel         = int(selection_window_sec * effective_fps), clamped to [30, len(frames)]
```

The tracker is re-initialized with `effective_fps` so any time-based parameter
(`max_age`, break threshold, recovery gate) is calibrated against *real*
seconds — important when analysis is sub-sampled.

Frames are split into `init_bgr` (the selection window) and `rest_bgr`.
Absolute source-video indices are reconstructed from `start_frame` and `stride`
so downstream consumers (metrics, overlays, ball-tracker fusion) see true
timestamps.

### Phase 1 — Process the selection window

`_run_frames` streams `init_bgr` through the detector (and pose, if loaded) in
batches of `config.batch_size`. Per frame, `_process_result`:

1. `filter_sv_detections_for_tracking` rejects boxes that are non-finite,
   inverted (`x2 <= x1` / `y2 <= y1`), below `min_box_height_px` or
   `min_box_area_px`, outside the aspect-ratio band
   `[min_box_aspect_ratio, max_box_aspect_ratio]`, or fully off-screen.
2. Runs the chosen tracker:
   - `_track_bytetrack`: `byte_track.update_with_detections(...)`.
   - `_track_deepocsort`: stacks `[xyxy, conf, cls]`, calls
     `deep_tracker.update(...)`, catching `np.linalg.LinAlgError` from the
     Kalman filter and emitting an empty result rather than crashing on a
     degenerate frame. It then calls `_harvest_deepocsort_embeddings()`, which
     reads the 512-dim OSNet embedding off each live track (`trk.emb`) and
     joins by id — **the embeddings are not in the returned rows**. Returns
     `{}` when ReID is off or unavailable, so the identity manager degrades to
     geometry.
3. `enrich_ankle_midpoints` (pose only): picks the pose box with the highest
   IoU (≥ `pose_iou_threshold`) and uses COCO ankle keypoints 15 & 16. An ankle
   is rejected if its (x, y) is exactly (0, 0) or its confidence is below
   `min_keypoint_conf`. Both valid → midpoint; one valid → that one; none → no
   ankle.
4. `enrich_court_positions`: projects the foot (ankle midpoint if available,
   else `bbox.bottom_center`) through the homography into `det.court_position`
   (meters).

The window is then summarized to stdout with per-frame in-court counts and
court depths for debugging.

### Phase 2 — Player selection

#### 2a. Automatic selection (`_auto_select_players`) — default

`_score_and_select(fd)` keeps detections with a tracker id **and** a court
position **strictly** inside the court (no margin). If fewer than `n_players`
remain, the frame is skipped. The `n_players` deepest detections (largest
`_court_depth` — distance to the nearest court edge) are chosen, and the frame
is scored by the **minimum** depth among them. The frame maximizing that score
wins and its detections seed the identity manager.

The intuition: depth-based selection prefers frames where real players are
clearly inside the court, not frames where a spectator standing right outside
the side glass is closer to a player than to any line.

#### 2b. Manual selection (`_best_selection_frame` + `PlayerSelectorUI`)

Used when `auto_select_players` is off, or there is no court projection.

`_best_selection_frame` scores each frame:

```
score = 20 × inside_count
      − 50 × max(0, n_players − inside_count)
      −  5 × max(0, inside_count − (n_players + 1))
      +  2 × avg_confidence
      +  2 × avg_court_depth
```

— a strong reward for hitting `n_players` visible players, a heavy penalty for
fewer, a mild penalty for many extras (one extra is tolerated).

`PlayerSelectorUI` then opens an OpenCV window: every tracker box is drawn
(dimmed if unselected, green & thick if selected), left-click selects the
nearest track within 100 px (clicking a selected track deselects), BACKSPACE
undoes, ENTER confirms only at exactly `n_players`, ESC raises `RuntimeError`.
Click order is returned but **discarded** — logical ids come from court
geometry in Phase 3.

### Phase 3 — Initialize the `LogicalPlayerManager`

`identity_manager.initialize(selected_dets, best_fd.frame_index, window_results=init_results)`.

The whole selection window is passed so each track's representative position is
its **median** court position over the window (`_median_positions`) — robust to
a transient stack or cross at the exact selection frame. Falls back to the
single selection-frame position for tracks with no windowed samples.

**Doubles (`_init_doubles`)** — team by median y (near = `y < L/2`), then within
each team the own-left/right is seeded **relative to the teammate**, not against
an absolute court-center threshold. The two teammates' camera-x values are
ranked against each other, so two players momentarily on the same half still
get distinct sides. Slots: near-left = 1, near-right = 2, far-left = 3,
far-right = 4.

Camera perspective **is** accounted for (`_own_left_camera_x_smaller`): near
players face the camera, so own-left = camera-left = smaller x; far players face
away, so own-left = camera-**right** = larger x.

If a team has only one player present, that player is placed by the absolute
court-half and the missing teammate gets the opposite empty slot (`play_side`
still set, `current_track_id=None`, `team_side=None`).

**Singles (`_init_singles`)** — P1 = near, P2 = far, `play_side=None`.

The seeded `play_side` is an **identity**: it is frozen at init and rides with
the player through in-game movement. A player briefly crossing to the other
side of their own half is *not* relabelled.

Each slot becomes a `LogicalPlayerState`:

| Field | Meaning |
|---|---|
| `logical_id` | 1…n_players, fixed forever |
| `current_track_id` | the tracker id currently bound to this slot |
| `last_bbox`, `last_court_position` | last confirmed observation |
| `team_side` | `near` / `far` — the slot's *current* end of the court |
| `play_side` | `left` / `right` (own frame), frozen identity; `None` in singles |
| `last_seen_frame`, `missing_frames` | bookkeeping; `missing_frames` is **logging only** |
| `is_on_court` | False once the player exits through a side door |
| `emb_ema` | fast EMA appearance embedding (adapts to lighting/pose) |
| `emb_anchor` | **frozen** anchor embedding, averaged over the first `reid_anchor_samples` clean matches |

`team1_side` starts as `"near"` (logical 1+2 = team 1) and `_side_change_allowed`
starts **closed**.

### Phase 4 — Back-fill identities across the selection window

For frames **before** `sel_frame_idx`: detections whose tracker id is in the
selected set are kept and stamped from the initial `tid → logical_id` map. This
is approximate (tracker ids are assumed unchanged that early) but good enough to
feed metrics and visualization. Everything else is dropped.

For frames **at or after** `sel_frame_idx`: routed through
`identity_manager.update()` like the live phase.

### Phase 5 — Process the remainder

`_run_frames(rest_bgr, …)` produces raw `FrameDetections`, each fed to
`identity_manager.update()`. The returned list contains **only** detections
mapped to a logical player.

---

## 3. Identity assignment — `LogicalPlayerManager.update`

Per frame, in order: **break hold → appearance-primary → direct track id →
structural cascade (Steps 2–5) → finalize**.

### 3.0 Break hold (freeze while the court is empty)

When zero valid in-court players are matched for `break_min_empty_sec`
consecutive seconds (changeover, injury, long gap between points), the manager
**enters a break**: it emits empty frames and freezes all logical state — no
recovery, no `last_seen` bump. This stops the recovery cascade from snapping
slots onto near-glass noise while nobody is playing.

On break **exit**, `_reattach_after_break` runs. Teams may have swapped ends
while the court was empty, so returning tracks are matched to logical players by
**cosine similarity to each slot's EMA gallery — never by position**. Matching
is Hungarian (scipy) with a greedy argmax fallback, accepting pairs above
`reid_reattach_min_cosine` (0.3). Matched states have `team_side` re-derived
from the new court position; `play_side` stays frozen. Leftover states and
detections fall through to the normal cascade the same frame.

If no usable embeddings exist (bytetrack, or cold galleries), a warning is
logged and the geometric cascade handles re-attach — teams may be mislabelled
if they actually swapped ends.

### 3.1 Step 0 — Appearance-primary assignment (authority over geometry)

Runs every frame once **every** logical player has a mature gallery, gated by
`config.reid_primary` (default `True`). Detections are matched to logical ids by
gallery cosine **ignoring position**. This is what carries identities correctly
through an end-swap, which pure geometry inverts.

Empirically OSNet separates the four players at ~0.98 cosine vs ~0.48 for
position (see `scripts/analyze_player_separability.py`).

**Reference gallery** (`_ref_gallery`) is the frozen `emb_anchor` when built,
else `emb_ema`. The anchor matters: because it cannot be poisoned by a later
mislabel, a wrong id **self-corrects on the next frame** instead of locking in.

**Doubles — two-tier, team-constrained.** Team membership (P1/P2 vs P3/P4) is
fixed identity: a player can only ever be confused with their *teammate*, never
an opponent. Only the court **side** flips on an end-swap. So:

- **Tier 1** — assign each side's detections to a whole **team** atomically, by
  which team's galleries collectively fit that side best (appearance, not
  position). This moves a team across the net as a unit. Detections in the net
  dead-zone have unknown side and are excluded from the tier-1 vote.
- **Tier 2** — within a team+side, Hungarian-match its detections to its two
  ids, accepting only pairs above `reid_primary_min_cosine` (0.5) that **also**
  beat the teammate's cosine for that detection by `reid_primary_margin`
  (0.05). Ambiguous teammate calls are left to geometry/direct-id.

A detection is therefore never assigned to an opponent's id.

**Singles (`_appearance_primary_singles`)** — no teams or L/R, so it's a single
global Hungarian match with the same min-cosine and margin.

For each accepted pair, `team_side` is re-derived from the **new** position
(atomic per-player, logged as `SIDE-FOLLOW` when it flips), and the gallery is
updated only when `cos >= reid_gallery_update_min_cosine` (0.6).

### 3.2 Step 1 — Direct tracker-id match

For each unmatched state whose `current_track_id` appears this frame:

- Rejected if the court position is **invalid** — not inside the court and not
  exiting through a side door (`_is_valid_player_position`). This guards against
  the tracker re-using a player's id on a spectator behind the glass.
- Rejected if the detection is in the **door zone** and appearance does *not*
  agree (`_appearance_agrees`): the embedding must match this player's
  reference gallery above `reid_primary_min_cosine` and beat every other
  player's gallery by `reid_primary_margin`. A tracker id alone is not trusted
  there — that is exactly how a transitioning player inherits a wrong label (the
  P4→P1 bug). When no galleries exist at all (bytetrack / cold start),
  `_appearance_agrees` returns True so ReID never blocks a path it can't inform;
  but when ReID *is* in play and the detection has no embedding, it returns
  False.

### The door zone

`_in_door_zone(det)` is True when the detection is in the door y-band
(`net_y ± DOOR_HALF_WIDTH_M`, 2 m) **or** within `door_zone_edge_margin_m`
(1 m) of any court edge — and also when it has no court position at all.

Near the door and edges every player piles up at the same location, so position
is meaningless. **In the door zone, assignment is ReID-only.** Door-zone
detections are excluded from the entire geometric cascade (Steps 2–5); a
door-zone detection with no confident gallery match is left **unassigned** for
that frame rather than mislabelled, and re-acquires once clearly inside.

### The recovery gate

Steps 2, 3, and 5 apply `_within_recovery_radius`: during the first
`recovery_gating_window_sec` (2 s) after a player was lost, a candidate is
rejected unless it lies within a radius that grows linearly from
`recovery_gating_initial_radius_m` (1 m) at t=0 to
`recovery_gating_final_radius_m` (3 m) at t=window. After the window the gate is
off. It falls **open** when either side lacks a court position. This stops a
nearby out-of-court spectator from capturing a slot during a brief miss.

### The team invariant

`_det_team_side` is the single source of truth for which half of the net a
detection is on: it returns `None` when the detection has no court position or
lies inside the `net_deadzone_m` (0.5 m) band around the net line — side
ambiguous, must not be teamed. Recovery **never crosses the net**: cross-net
pairs are hard-rejected in the one-to-one fallback and set to `+inf` in the
proximity distance matrix.

### 3.3 Step 2 — `same_role` structural reassignment

For each still-missing state, find unassigned detections matching both its
`team_side` and `play_side` (plus the recovery gate). Committed only when
**symmetrically unique** — exactly one detection matches that state's role and
exactly one state matches that detection's role. This stops two states from
fighting over the same quadrant during ambiguous frames.

Singles have `play_side=None`, so `_same_role` always returns False and this
step is a no-op.

### 3.4 Step 3 — `same_team_side` structural reassignment

Same symmetric-uniqueness pattern, matching on `team_side` only. Catches two
teammates who swapped left/right (one walked behind the other and the tracker
shuffled ids).

### 3.5 Step 4 — Proximity fallback

Distance matrix between every remaining state's `last_court_position` and every
remaining detection's `court_position`, with cross-net pairs masked to `+inf`.
Greedily assign the globally closest pair, then the next, while distance is
≤ `proximity_max_distance_m` (3 m). Only states/detections that have court
positions participate.

### 3.6 Step 5 — One-to-one fallback

If exactly one state and one detection remain, pair them — provided the position
is valid, the recovery gate passes, **and** `_team_matches` holds. Each
rejection is logged with its specific reason (`invalid_position`,
`recovery_gate`, `cross_team`).

### 3.7 What happens to unclaimed detections

They are **dropped**. `update` returns `FrameDetections` containing only the
assigned ones; the manager never invents detections for missing players.
`missing_frames` is incremented for unmatched states but is **never used to gate
assignment** — logging only. `max_missing_frames` is kept for backwards
compatibility and no longer participates in decisions.

### 3.8 `_assign` side effects

- `state.current_track_id` ← the new tracker id (track-id switches are logged at
  INFO for every recovery step; `direct_track_id` logs at DEBUG to avoid spam).
- `det.logical_player_id` stamped, `det.visible = True`.
- Last bbox / court position / last-seen frame updated; `missing_frames` reset.
- `team_side` and `play_side` are filled in opportunistically if they were
  `None` (e.g. a seed with no projection).
- `is_on_court` flips True → False when the new position is a side-door exit
  (`x < 0` or `x > W` while in the door y-band), which **opens the side-change
  gate**.
- The ReID gallery is updated (`_update_gallery`), but only when `gallery_ok`
  (the caller's confidence flag) **and** the position is a valid in-court one —
  so near-glass and spectator crops never leak into the appearance model.

### 3.9 The ReID gallery (`_update_gallery`)

Two embeddings per logical player, both L2-normalized:

- **Anchor** (`emb_anchor`) — a running mean of the first `reid_anchor_samples`
  (60) clean confident samples, then **frozen**. This is the primary matching
  reference precisely because it can't drift.
- **EMA** (`emb_ema`) — `emb = α·emb + (1−α)·e` with `α = reid_gallery_alpha`
  (0.9), mirroring boxmot's `update_emb`. Fast, adapts to lighting/pose; used as
  the secondary signal and for post-break re-attach.

No-ops on a missing or zero-norm embedding, so ReID stays entirely optional.

### 3.10 Finalize: team-side maintenance & break bookkeeping

`_finalize` runs `_update_team_sides`, then updates the empty-court streak
(entering a break at `break_min_empty_sec`, resetting on any valid in-court
player) and emits.

`_update_team_sides` acts only when the side-change gate is open (a player has
been observed leaving through a side door since the last change). It checks
whether team 1 and team 2 are observed exclusively in opposite halves and, if
so, flips `team1_side` and calls `_flip_state_team_sides` — which swaps
`near ↔ far` on every state but leaves `play_side` untouched, since own
left/right does not change when a team physically swaps ends. The gate then
closes.

> **Known limitation (TODO in code).** The gate never checks whether the *other*
> team left the court, and it is not re-closed when a player returns to the same
> side. A better rule: allow the change when one team member is on court on a
> different side *and* the players of the team occupying that side are not on
> court — which would also permit swapping back.

`FrameDetections.team_sides` (`{1: …, 2: …}`) ships with every frame so
downstream code (heatmaps, formations) renders team identity correctly even
after players swap courts between games.

At DEBUG level a per-frame slot snapshot is logged, e.g.
`+P1[T1-L n]=tid7 xP2[T1-R n]=tid3 … gate=closed`, where `+` = assigned this
frame and `x` = missing.

---

## 4. Cases the pipeline explicitly handles

| Case | How it's handled |
|---|---|
| Empty input | `detect_and_track` returns `[]` immediately. |
| Stride > 1 (sub-sampled analysis) | Tracker re-initialized with `effective_fps`; absolute indices via `start_frame + i*stride`. |
| Singles match | `match_type="singles"` → 2 players on a 6 × 20 m court; `play_side` is `None`, Step 2 no-ops, appearance-primary uses the global-match variant. |
| RF-DETR short final batch | Padded with repeated frames to `config.batch_size` (the JIT graph is batch-size-specific); padding results dropped. |
| RF-DETR class-id confusion | Person id resolved from the dict-keyed `COCO_CLASSES` mapping, not `class_names`; then renormalized to `person_class_id`. |
| Modal / non-numeric `CUDA_VISIBLE_DEVICES` | DeepOCSORT is given device index `0` instead of `"cuda"`. |
| boxmot ≥ 18 API change | `ReID` backend built explicitly and `.model` passed as `reid_model=`; non-None asserted at init (CV-003). |
| DeepOCSORT Kalman blow-up | `np.linalg.LinAlgError` caught; empty tracks for that frame. |
| Box garbage (NaN, inverted, micro, weird aspect, off-screen) | Rejected pre-tracker by `filter_sv_detections_for_tracking`. |
| Pose model misaligned with detection box | Dropped if IoU < `pose_iou_threshold`; foot falls back to `bbox.bottom_center`. |
| One ankle keypoint missing / low-confidence | Use the other; if both missing, no ankle midpoint. |
| Missing court projection | Court positions `None`, auto-selection cannot run, structural/proximity steps fall back to direct-id-only matching. |
| ByteTrack (no embeddings) | Every ReID path no-ops: no galleries, appearance-primary skipped, `_appearance_agrees` returns True so it never blocks. Pure geometric cascade. |
| Auto-selection finds < n_players | Warning logged, partial seed used; empty slots keep their `play_side` and are filled once the player appears. |
| Manual selection cancelled / short | ESC raises `RuntimeError`; a short selection logs a warning and proceeds. |
| Spectator inheriting a player's tracker id | `_is_valid_player_position` rejects it in Steps 1 and 5; the recovery gate blocks distant captures within the 2 s window. |
| Boundary noise (player at ~0 m depth) | Excluded from the unassigned pool below `min_direct_match_court_depth_m`. |
| Player transitioning through the door | Door zone is ReID-only: geometric steps never assign there, and a direct track-id match must be confirmed by appearance. No confident match → left unassigned this frame. |
| Two teammates cross & swap tracker ids | Appearance-primary tier 2 usually fixes it; else Step 3 (`same_team_side`), else Step 4 (`proximity`). |
| Teams swap court ends mid-match | Appearance-primary tier 1 moves a whole team across the net as a unit (`SIDE-FOLLOW`), which pure geometry would invert. `_update_team_sides` maintains the reported `team_sides` map. |
| Court empties (changeover / injury / long break) | Break entered after `break_min_empty_sec`; state frozen, empty frames emitted; on return, `_reattach_after_break` re-attaches by appearance, not position. |
| A mislabel slips through | The frozen anchor gallery cannot be poisoned (gallery updates also require `cos ≥ reid_gallery_update_min_cosine`), so the next frame's appearance match self-corrects. |
| Tracker id briefly disappears (occlusion) | Step 1 doesn't match; on reappearance the same id matches again, or Steps 2–5 recover it under the growing radius gate. |
| Detections matching no logical player | Silently dropped — only assigned detections survive. |
| Frames before the selection-winning frame | Logical id copied from the initial tracker-id map; other detections dropped. |

---

## 5. Outputs

`detect_and_track` returns a `list[FrameDetections]` covering every frame in the
input iterator. Per frame:

- `detections` contains **only** items mapped to a logical player.
  `det.logical_player_id` is set, `det.visible = True`, `det.is_on_court`
  reflects the current on-court state.
- `det.court_position` is in meters (origin at the near-baseline-left corner)
  when a homography was available.
- `det.ankle_midpoint` and `det.keypoints` are populated when pose ran and a
  confident match existed.
- `det.embedding` carries the OSNet appearance vector under DeepOCSORT.
- `fd.team_sides` describes which team is currently near/far, updating on a
  detected side change.

Frames where fewer players were recovered simply contain fewer detections — and
break frames contain none. There is no padding or interpolation at this layer;
the metrics module fills short gaps later.

---

## 6. Key tunables (`PlayerTrackerConfig`)

| Knob | Default | Effect |
|---|---|---|
| `match_type` | `doubles` | Resolves player count + court dimensions. |
| `detector_type` | `rfdetr` | RF-DETR (Apache-2.0) vs YOLO (AGPL). |
| `tracker_type` | `deepocsort` | `bytetrack` disables all ReID paths. |
| `tracker_max_age_sec` | 10.0 | How long boxmot keeps a track alive unmatched. Keep modest — see §1. |
| `selection_window_sec` | 2.0 | Length of the seeding window (min 30 frames). |
| `net_deadzone_m` | 0.5 | Band around the net where team side is "unknown". |
| `door_zone_edge_margin_m` | 1.0 | Edge margin defining the ReID-only door zone. |
| `proximity_max_distance_m` | 3.0 | Max Step 4 assignment distance. |
| `recovery_gating_window_sec` / `_initial_radius_m` / `_final_radius_m` | 2.0 / 1.0 / 3.0 | Growing-radius gate for Steps 2, 3, 5. |
| `break_min_empty_sec` | 2.0 | Empty-court duration before freezing state. |
| `reid_primary` | `True` | Appearance overrides geometry each frame. |
| `reid_primary_min_cosine` / `reid_primary_margin` | 0.5 / 0.05 | Accept threshold and clarity margin over the runner-up. |
| `reid_gallery_update_min_cosine` | 0.6 | Confidence needed to fold a sample into a gallery. |
| `reid_anchor_samples` | 60 | Clean samples averaged before the anchor freezes. |
| `reid_gallery_alpha` | 0.9 | EMA decay for the fast gallery. |
| `reid_reattach_min_cosine` | 0.3 | Looser threshold for post-break re-attach. |
