"""Dense ground-truth player labeler with online IoU tracking (tracker-independent).

Reads RF-DETR candidate boxes dumped by `test_modal_gpu.py::dump_boxes`
(raw_boxes.json) and lets you produce a DENSE, per-frame ground truth for the 4
logical players P1..P4, human-verified frame by frame.

How it works
------------
You seed the 4 players on the first frame, then step forward. On each new frame
the tool propagates every player's box to the best-overlapping detector box by
IoU (pure geometry on the detector's boxes — no tracker / identity logic, so the
result stays valid to score any tracker version against). When IoU can't hold a
player (occlusion / exit / fast motion → no box overlaps enough) the tool PAUSES
and highlights that player; you click their box or mark them absent, then continue.
Because most frames are clear, you mostly hold a step key and only intervene at
the hard moments. Output is a box (or "absent") for every player on every frame.

Usage
-----
    python scripts/label_players.py --video data/full_score/2/2.mp4

    # --jump is in SECONDS (review cadence), default 2.0s. IoU still tracks every
    # frame in between and records them; --jump only sets how often you stop to look.
    # defaults: --raw-boxes <video_dir>/raw_boxes.json,
    #           --out <video_dir>/ground_truth.json

Controls (in the OpenCV window)
-------------------------------
    LEFT-click a box     assign it to the current player id (then id auto-advances)
    RIGHT-click a box    un-label that box (clears whatever player it holds) —
                         fast fix when a spectator box grabbed a player
    1 / 2 / 3 / 4        pick which player id the next click assigns
    x                    mark the current player id ABSENT on this frame
    c                    clear ALL assignments on this frame (start it over)
    r                    re-run IoU from the previous frame (undo bad auto-track)
    d / SPACE            step forward --jump SECONDS, IoU-tracking every frame between
    .                    step forward exactly 1 frame
    a / LEFT-arrow       step back one recorded frame (to review / fix)
    ENTER                confirm this frame as-is and step forward
    q / ESC              save and quit

If IoU loses a player on a forward step, playback PAUSES on that frame with the
lost player highlighted — resolve it (click their box, or press its number then
`x` to mark absent) before stepping on. Autosaves after every frame; relaunching
resumes at the last labeled frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Player colours (BGR), matching AppColors P1=green P2=orange P3=blue P4=pink.
_COLORS = {1: (0, 200, 50), 2: (0, 140, 255), 3: (255, 80, 80), 4: (180, 0, 255)}
_UNASSIGNED = (150, 150, 150)
_LOST = (0, 0, 255)  # red highlight for a player IoU just lost
_MAX_CLICK_DIST = 140  # px — ignore clicks far from any box centre
_IOU_TRACK_MIN = 0.3   # min IoU to auto-propagate a player to a box on the next frame


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class LabelSession:
    def __init__(self, video: str, raw_boxes: dict, out_path: Path,
                 jump_sec: float = 2.0, n_players: int = 4):
        self.video = video
        self.raw = raw_boxes
        self.out_path = out_path
        self.n_players = n_players
        self.fps = float(raw_boxes.get("fps") or 30.0)
        # Review cadence in SECONDS -> how many frames a forward step covers. IoU
        # still tracks EVERY frame internally; this only controls how often you
        # stop to look (plus any lost-player pause in between). fps-agnostic.
        self.jump_sec = jump_sec
        self.jump = max(1, round(jump_sec * self.fps))

        # Available detector frames (absolute source indices), in order.
        self.frame_indices = sorted(int(k) for k in raw_boxes["frames"].keys())
        self._pos_of = {f: i for i, f in enumerate(self.frame_indices)}

        # labels: {frame_idx(int): {pid(int): [x1,y1,x2,y2] | "absent"}}
        self.labels: dict[int, dict] = {}
        self._load_existing()

        # Start at the frame after the last labeled one (resume). If everything is
        # already labeled, sit on the last labeled frame so re-opening shows where
        # you left off (rather than jumping back to frame 0).
        self.idx = 0
        if self.labels:
            last = max(self.labels)
            if last in self._pos_of:
                nxt = self._pos_of[last] + 1
                self.idx = nxt if nxt < len(self.frame_indices) else self._pos_of[last]

        self.selected_pid = 1
        self.lost_pids: set[int] = set()   # players IoU couldn't hold this frame
        self._cap = cv2.VideoCapture(video)
        self._decoder_pos: int | None = None  # next frame the decoder will return

        # working state for the CURRENT frame
        self._boxes: list[list[float]] = []     # detector boxes [x1,y1,x2,y2,conf]
        self._assign: dict[int, int] = {}        # pid -> box_index (or absent via _absent)
        self._absent: set[int] = set()           # pids explicitly absent this frame
        self._img: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _load_existing(self) -> None:
        if self.out_path.exists():
            try:
                data = json.loads(self.out_path.read_text())
                self.labels = {int(k): v for k, v in data.get("labels", {}).items()}
                if self.labels:
                    print(f"Resuming — {len(self.labels)} frames already labeled "
                          f"(last frame {max(self.labels)}).")
            except Exception as e:
                print(f"WARN: could not read existing {self.out_path}: {e}")

    def _save(self) -> None:
        data = {
            "video": Path(self.video).name,
            "fps": self.fps,
            "width": self.raw.get("width"),
            "height": self.raw.get("height"),
            "n_players": self.n_players,
            "dense": True,
            "iou_track_min": _IOU_TRACK_MIN,
            "source": "manual-label + online-IoU / tracker-independent",
            "labels": {str(k): v for k, v in sorted(self.labels.items())},
        }
        self.out_path.write_text(json.dumps(data))

    # ------------------------------------------------------------------
    @property
    def frame_idx(self) -> int:
        return self.frame_indices[self.idx]

    def _blank(self) -> np.ndarray:
        h, w = int(self.raw.get("height", 1080)), int(self.raw.get("width", 1920))
        return np.zeros((h, w, 3), np.uint8)

    def _read_frame(self, frame_idx: int) -> np.ndarray:
        """Decode a single frame, cheaply.

        Random seeking (cap.set(POS_FRAMES)) re-decodes from the nearest keyframe
        and is ~60x slower than sequential reads. So when the target is at or a
        short distance AHEAD of the decoder's current position, grab() forward
        sequentially instead of seeking. Only seek for backward or long jumps.
        """
        cur = self._decoder_pos
        if cur is not None and 0 <= frame_idx - cur <= 90:
            # grab (decode+discard) intervening frames, then retrieve the target
            for _ in range(frame_idx - cur):
                if not self._cap.grab():
                    break
            ok, img = self._cap.read()
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, img = self._cap.read()
        self._decoder_pos = frame_idx + 1 if ok else None
        return img if ok else self._blank()

    def _grab_to(self, frame_idx: int) -> None:
        """Advance the decoder to just past `frame_idx` WITHOUT decoding pixels.

        grab() alone (no retrieve()) is ~cheap; used for intermediate jump frames
        that are recorded from boxes only and never displayed, so the eventual
        landing-frame read can grab sequentially instead of doing a costly seek.
        """
        cur = self._decoder_pos
        if cur is not None and 0 <= frame_idx - cur <= 300:
            for _ in range(frame_idx - cur + 1):
                if not self._cap.grab():
                    self._decoder_pos = None
                    return
            self._decoder_pos = frame_idx + 1
        else:
            # too far / backward — just record intent; next read() will seek.
            self._decoder_pos = None

    def _boxes_at(self, frame_idx: int) -> list[list[float]]:
        return [list(b) for b in self.raw["frames"].get(str(frame_idx), [])]

    def _load_frame_state(self, propagate_from: dict | None, read_image: bool = True) -> None:
        """Build assignments for the current frame.

        If this frame already has stored labels, load them. Otherwise IoU-
        propagate each player from `propagate_from` (the previous frame's label
        dict: {pid_str: box|'absent'}); players with no overlapping box are LOST.

        `read_image=False` skips the (expensive) pixel decode — used for the
        intermediate frames of a jump that are auto-committed but never shown; the
        decoder is still advanced cheaply via grab() so sequential reads stay fast.
        """
        fi = self.frame_idx
        if read_image:
            self._img = self._read_frame(fi)
        else:
            self._grab_to(fi)  # keep decoder synced without decoding pixels
        self._boxes = self._boxes_at(fi)
        self._assign, self._absent, self.lost_pids = {}, set(), set()

        stored = self.labels.get(fi)
        if stored is not None:
            for pid_str, val in stored.items():
                pid = int(pid_str)
                if val == "absent" or val is None:
                    self._absent.add(pid)
                else:
                    bi = self._nearest_box_by_iou(val, min_iou=0.0)
                    if bi is not None:
                        self._assign[pid] = bi
                    else:
                        self._absent.add(pid)  # box vanished; keep as recorded-absent
            return

        if propagate_from is None:
            # First-ever frame: nothing to propagate; user seeds all players.
            self.lost_pids = set(range(1, self.n_players + 1))
            return

        used: set[int] = set()
        for pid in range(1, self.n_players + 1):
            prev = propagate_from.get(str(pid))
            if prev == "absent" or prev is None:
                self._absent.add(pid)
                continue
            bi = self._nearest_box_by_iou(prev, min_iou=_IOU_TRACK_MIN, exclude=used)
            if bi is None:
                self.lost_pids.add(pid)  # occlusion / exit / fast motion → ask user
            else:
                self._assign[pid] = bi
                used.add(bi)

    def _nearest_box_by_iou(self, box, min_iou=0.0, exclude: set | None = None) -> int | None:
        exclude = exclude or set()
        best_i, best = None, min_iou
        for i, b in enumerate(self._boxes):
            if i in exclude:
                continue
            v = _iou(b, box)
            if v > best:
                best, best_i = v, i
        return best_i

    # ------------------------------------------------------------------
    def _draw(self) -> np.ndarray:
        img = self._img.copy()
        assigned_boxes = set(self._assign.values())
        # draw unassigned boxes faint
        for i, b in enumerate(self._boxes):
            if i in assigned_boxes:
                continue
            x1, y1, x2, y2 = [int(v) for v in b[:4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), _UNASSIGNED, 1)
        # draw assigned player boxes
        for pid, bi in self._assign.items():
            b = self._boxes[bi]
            x1, y1, x2, y2 = [int(v) for v in b[:4]]
            color = _LOST if pid in self.lost_pids else _COLORS[pid]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            lbl = f"P{pid}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, lbl, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        w = img.shape[1]
        bar = np.zeros((72, w, 3), np.uint8)
        t = self.frame_idx / self.fps
        assigned = sorted(self._assign.keys())
        absent = sorted(self._absent)
        lost = sorted(self.lost_pids)
        done = len(self.labels)
        line1 = (f"frame {self.frame_idx}  t={t:5.1f}s  ({self.idx + 1}/{len(self.frame_indices)})"
                 f"   labeled {done}   SELECT P{self.selected_pid}")
        status = f"assigned {assigned or '-'}  absent {absent or '-'}"
        if lost:
            status += f"   LOST-> {lost}  (click their box or press <id> then x)"
        line3 = ("[1-4]pick Lclick=assign Rclick=unlabel  x=absent c=clear r=reIoU  "
                 "SPACE=step .=+1 <-=back  q=quit")
        cv2.putText(bar, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    _COLORS[self.selected_pid], 2, cv2.LINE_AA)
        cv2.putText(bar, status, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if lost else (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(bar, line3, (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        return np.vstack([img, bar])

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        bi = self._nearest_box_center(x, y)
        if bi is None:
            return
        # RIGHT-CLICK: un-label whatever box this is, regardless of selected id.
        # Fast fix when a spectator box grabbed a player.
        if event == cv2.EVENT_RBUTTONDOWN:
            for p, b in list(self._assign.items()):
                if b == bi:
                    del self._assign[p]
            return
        pid = self.selected_pid
        # toggle off if this box already holds the selected pid
        if self._assign.get(pid) == bi:
            del self._assign[pid]
            return
        # a box belongs to exactly one pid; a pid to exactly one box
        for p, b in list(self._assign.items()):
            if b == bi:
                del self._assign[p]
        self._assign[pid] = bi
        self._absent.discard(pid)
        self.lost_pids.discard(pid)
        self._advance_selected_pid()

    def _advance_selected_pid(self) -> None:
        for _ in range(self.n_players):
            self.selected_pid = self.selected_pid % self.n_players + 1
            if self.selected_pid not in self._assign:
                break

    def _nearest_box_center(self, x, y) -> int | None:
        best_i, best = None, float("inf")
        for i, b in enumerate(self._boxes):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            d = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            if d < best:
                best, best_i = d, i
        return best_i if best <= _MAX_CLICK_DIST else None

    def _current_entry(self) -> dict:
        entry: dict = {}
        for pid in range(1, self.n_players + 1):
            if pid in self._assign:
                b = self._boxes[self._assign[pid]]
                entry[str(pid)] = [round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1)]
            else:
                entry[str(pid)] = "absent"
        return entry

    def _commit_current(self) -> None:
        self.labels[self.frame_idx] = self._current_entry()
        self._save()

    # ------------------------------------------------------------------
    def run(self) -> None:
        win = "player labeler (dense IoU)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1440, 860)
        cv2.setMouseCallback(win, self._on_mouse)
        # seed / load the starting frame
        prev = self.labels.get(self._prev_labeled_frame_idx())
        self._load_frame_state(propagate_from=prev)
        while True:
            cv2.imshow(win, self._draw())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):
                self._commit_current()
                break
            elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                pid = key - ord('0')
                if pid <= self.n_players:
                    self.selected_pid = pid
            elif key == ord('x'):
                # mark the selected player absent this frame
                self._assign.pop(self.selected_pid, None)
                self._absent.add(self.selected_pid)
                self.lost_pids.discard(self.selected_pid)
                self._advance_selected_pid()
            elif key == ord('c'):
                # clear ALL assignments on this frame (start it over)
                self._assign, self._absent, self.lost_pids = {}, set(), set()
                self.selected_pid = 1
            elif key == ord('r'):
                # re-run IoU propagation from the previous frame, discarding this
                # frame's current guesses (undo a bad auto-track / spectator grab)
                prev = self.labels.get(self._prev_labeled_frame_idx())
                self._load_frame_state(propagate_from=prev, read_image=False)
                self.selected_pid = 1
            elif key in (13, 10, 32, ord('d'), 83, ord('.')):  # confirm/step forward
                if self.lost_pids:
                    continue  # must resolve lost players before stepping on
                self._commit_current()
                step = 1 if key == ord('.') else self.jump
                if not self._step(step):
                    print("\nReached end of frames.")
            elif key in (ord('a'), 81):  # step back
                self._step_back()
        cv2.destroyAllWindows()
        self._cap.release()
        print(f"\nSaved {self.out_path}  ({len(self.labels)} frames labeled)")

    def _prev_labeled_frame_idx(self) -> int | None:
        cur = self.frame_idx
        prev = [f for f in self.labels if f < cur]
        return max(prev) if prev else None

    def _step(self, step: int) -> bool:
        """Confirm any intermediate frames with the same (propagated) labels too.

        Stepping by `step` records EVERY intervening frame (dense output). Each
        intermediate frame is IoU-propagated from the one before and committed;
        if any intermediate frame loses a player we STOP there and hand control
        back so the user can fix it.
        """
        for k in range(step):
            if self.idx + 1 >= len(self.frame_indices):
                # decode the frame we actually landed on for display
                self._img = self._read_frame(self.frame_idx)
                return False
            prev_entry = self.labels[self.frame_idx]  # just committed
            self.idx += 1
            is_last = (k == step - 1)
            # Only decode pixels for the frame we will SHOW (the landing frame).
            # Intermediate frames are tracked/committed from boxes only.
            self._load_frame_state(propagate_from=prev_entry, read_image=is_last)
            if self.lost_pids:
                # a mid-jump problem — decode it so the user can see and fix
                self._img = self._read_frame(self.frame_idx)
                return True
            self._commit_current()  # clean frame → auto-record and continue
        return True

    def _step_back(self) -> None:
        if self.idx == 0:
            return
        self.idx -= 1
        # reload from stored labels (this frame was committed earlier)
        prev = self.labels.get(self._prev_labeled_frame_idx())
        self._load_frame_state(propagate_from=prev)


def main():
    ap = argparse.ArgumentParser(description="Dense ground-truth player labeler (online IoU)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--raw-boxes", default=None,
                    help="raw_boxes.json (default: <video_dir>/raw_boxes.json)")
    ap.add_argument("--out", default=None,
                    help="ground_truth.json output (default: <video_dir>/ground_truth.json)")
    ap.add_argument("--jump", type=float, default=2.0,
                    help="SECONDS to advance per forward step (default 2.0); "
                         "IoU still tracks every frame in between and records them")
    ap.add_argument("--n-players", type=int, default=4)
    ap.add_argument("--restart", action="store_true",
                    help="ignore any existing ground_truth.json and start at frame 0 "
                         "(the existing file is backed up to *.bak first)")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"ERROR: video not found: {video}")
        sys.exit(1)
    raw_path = Path(args.raw_boxes) if args.raw_boxes else video.parent / "raw_boxes.json"
    if not raw_path.exists():
        print(f"ERROR: raw boxes not found: {raw_path}\n"
              f"Generate with: modal run cv_backend/scripts/test_modal_gpu.py::dump_boxes "
              f"--video {video}")
        sys.exit(1)
    out_path = Path(args.out) if args.out else video.parent / "ground_truth.json"

    if args.restart and out_path.exists():
        import time
        bak = out_path.with_suffix(f".{int(time.time())}.bak")
        out_path.rename(bak)
        print(f"--restart: backed up existing labels to {bak.name}, starting fresh at frame 0.")

    raw = json.loads(raw_path.read_text())
    session = LabelSession(str(video), raw, out_path, args.jump, args.n_players)
    print(f"{len(session.frame_indices)} detector frames @ {session.fps:.0f}fps. "
          f"Review every {args.jump}s (= {session.jump} frames); IoU tracks the rest. "
          f"Seed P1-P4 on the first frame, then step. Window opening...")
    session.run()


if __name__ == "__main__":
    main()
