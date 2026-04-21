## Sprint 3 — Player Tracking Correctness

**Goal:** Upgrade the tracker so player IDs survive occlusions, drink breaks, and same-team identical kits. This is prerequisite to trustworthy distance and heatmap stats.

**Context:** Current pipeline uses YOLO11 + ByteTrack. ByteTrack has no appearance model → IDs swap at net crossings, and players who leave for a drink return with new IDs. Court homography (H matrix) is assumed available as input (computed upstream from manual keypoints) — Sprint 3 leverages it, does not replace it.

**Stack changes:** BoxMOT (already in deps) swap `bytetrack` → `deepocsort`. Add OSNet ReID weights (`osnet_x0_25_msmt17.pt`, ~1 MB). No detector change.

**Test setup:** Local machine with GPU (or CPU for small clips). Input = local video files + a JSON/YAML with the H matrix. Output = per-frame track JSON for inspection.

---

### Task 1 — Swap ByteTrack for DeepOCSORT ✅

**Files:** `pipeline/player_tracker.py`

**Steps:**
- Replace `ByteTrack(...)` with `DeepOCSORT(model_weights=Path('weights/osnet_x0_25_msmt17.pt'), device='cuda:0', fp16=True, max_age=600, min_hits=1)`. Use `device='cpu'` and `fp16=False` for CPU testing.
- Download OSNet weights once into a local `weights/` directory (gitignored).
- Keep detector output format identical (BoxMOT accepts the same `dets` tensor).

**Provides:** ID persistence through player-on-player occlusions at the net. DeepOCSORT adds OSNet appearance embeddings, so when two players cross paths the tracker re-assigns based on appearance not just motion. Immediate correctness lift on distance per player.

**Testable output:** Process a 2-minute local clip containing at least one net crossover. Dump per-frame tracker IDs to a JSON file. Both players retain their original IDs throughout the crossover. Compare to the same clip processed with the old ByteTrack code — expect fewer ID swaps.

---

### Task 2 — 4-player hard prior + court-region gating

**Files:** `pipeline/player_tracker.py`

**Steps:**
- Load H matrix from a local JSON/YAML config file alongside the input video.
- After tracker update, project every track's bbox-bottom-center through H into court coords.
- Drop any track whose projected position falls outside `[-1, 21] × [-1, 11]` meters (court is 20×10, 1m margin).
- If `len(active_tracks) > 4`, keep the 4 with highest `confidence × track_length`; log excess tracks as `suspected_duplicate`.
- Do NOT delete tracks below 4 — let `max_age=600` keep dormant ones alive for re-entry (Task 3).

**Provides:** Eliminates spectators behind the glass, referees, and passersby from polluting the 4 player tracks. Kills the "split track" failure where one physical player becomes two IDs after a brief occlusion. Makes downstream distance and heatmap stats actually correspond to the 4 players on court.

**Testable output:** Process a clip with visible spectators behind the glass wall. Final track list has exactly 4 IDs. Zero tracks with projected positions outside the court bounds. Log shows any `suspected_duplicate` events for manual review.

---

### Task 3 — Drink-break / re-entry handling

**Files:** `pipeline/player_tracker.py`, possibly new `pipeline/session_gallery.py`

**Steps:**
- Detect "court-empty" state: `<4` active tracks for `>90` consecutive frames (3 s at 30 fps).
- On entering that state, snapshot the last-known OSNet embedding for each of the 4 IDs as a `session_gallery` (dict: `id → embedding EMA`).
- On re-entry (new detections projected inside court), run Hungarian assignment: cost = `1 - cosine_similarity(new_embedding, gallery[id])` with a small additional penalty for cross-team-side assignments (players return to their starting half ~95% of the time; compute starting side from first 60 s of footage).
- Restore original IDs for matched detections; only spawn new IDs if cost exceeds threshold (e.g. `0.5`).

**Provides:** The drink-break failure mode. Without this, a player who walks off for water comes back as Player 5 — their first-half distance is orphaned on a dead track, and second-half distance restarts from zero for the same physical person. This is the difference between a toy metric and one users trust.

**Testable output:** Use a clip where all 4 players walk off-camera for ~10 s then return. Tracker restores IDs 1–4 to the original physical players on re-entry. No new IDs (5, 6, 7, 8) spawned.

---

### Task 4 — Validate on 2–3 real clips; decide on CAMELTrack upgrade

**Files:** new `eval/tracker_validation.md` (documentation only)

**Steps:**
- Run the full pipeline on 2–3 real match clips (own recordings or PadelTracker100 samples).
- Spot-check 20 random frames per clip: do IDs correspond to the correct physical player? Compute approximate IDF1 by hand.
- Sanity-check distance: padel players typically cover 1.5–3 km per match; flag any player outside 0.8–4 km.
- Document remaining failure modes (same-team identical-kit swaps, far-baseline detection misses, etc.).

**Decision gate:**
- If IDF1 ≥ 80 and distance is plausible → ship Sprint 3.
- If same-team players with identical kits still swap frequently → Sprint 3.5: either fine-tune OSNet on 500–2,000 padel crops, or swap DeepOCSORT → CAMELTrack (CVPR 2025, ~82 HOTA vs DeepOCSORT's 74). CAMELTrack is heavier to integrate — only invest if the simpler fix is proven insufficient.

**Provides:** Evidence-based decision on whether to stop here or invest further. Prevents speculative upgrades that consume time without addressing a real failure mode.

**Testable output:** `tracker_validation.md` containing per-clip IDF1 estimate, distance sanity check, list of remaining failure modes, and a go/no-go recommendation on CAMELTrack.

---

**Done when:** All 4 tasks merged, validation doc shows IDF1 ≥ 80 on real clips, and distance metrics are plausible. Drink-break re-entry verified on at least one recorded clip.

**Out of scope for Sprint 3:** Ball tracking, auto court keypoint detection, pose-based foot keypoints, CAMELTrack upgrade (Sprint 3.5, conditional).