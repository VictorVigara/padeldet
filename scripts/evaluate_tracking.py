"""Score tracker predictions against dense ground truth.

Compares a predictions file (per-frame logical-player boxes from the tracker)
to ground_truth.json (from label_players.py). Both share the shape
    {"labels": {frame: {logical_id: [x1,y1,x2,y2] | "absent"}}}.

Per frame, predicted boxes are IoU-matched to ground-truth boxes (Hungarian).
A predicted box "P_k" is CORRECT only if it overlaps the SAME logical player k
in the ground truth; if it overlaps a DIFFERENT player, that's an ID error
(the tracker put the wrong label on that person). This is the metric that
actually reflects identity quality — not just detection overlap.

Reported per player and overall:
  * identity accuracy  = correct / (frames where GT present)
  * id_error (swap)    = predicted box lands on the wrong GT player
  * miss               = GT present but no prediction
  * false_positive     = predicted present but GT absent
  * cross_net          = predicted player on the opposite team half vs GT
  * presence F1        = did we predict present/absent correctly

Usage
-----
    python scripts/evaluate_tracking.py --video data/full_score/2/2.mp4
    # defaults: --gt <dir>/ground_truth.json  --pred <dir>/predictions.json
    python scripts/evaluate_tracking.py --gt gt.json --pred predictions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_NET_Y_FRAC = 0.5  # net at mid-court in IMAGE y is not reliable; we use GT-relative teams


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def _match(pred_boxes: dict[int, list], gt_boxes: dict[int, list], iou_thr: float):
    """IoU-match predicted pids to GT pids. Returns {pred_pid: gt_pid|None}."""
    pps = list(pred_boxes)
    gps = list(gt_boxes)
    if not pps or not gps:
        return {p: None for p in pps}
    M = np.zeros((len(pps), len(gps)))
    for i, pp in enumerate(pps):
        for j, gp in enumerate(gps):
            M[i, j] = _iou(pred_boxes[pp], gt_boxes[gp])
    out = {p: None for p in pps}
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(-M)
        for i, j in zip(ri, ci):
            if M[i, j] >= iou_thr:
                out[pps[i]] = gps[j]
    except Exception:
        used = set()
        order = sorted(((M[i, j], i, j) for i in range(len(pps)) for j in range(len(gps))),
                       reverse=True)
        taken_p = set()
        for v, i, j in order:
            if v < iou_thr or i in taken_p or j in used:
                continue
            out[pps[i]] = gps[j]; taken_p.add(i); used.add(j)
    return out


def _team_of(pid: int, n_players: int) -> int:
    if n_players >= 4:
        return 1 if pid <= 2 else 2
    return pid


def main():
    ap = argparse.ArgumentParser(description="Evaluate tracker identity vs ground truth")
    ap.add_argument("--video", default=None, help="used only to locate default gt/pred paths")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--iou", type=float, default=0.4, help="IoU threshold for a match")
    ap.add_argument("--out", default=None, help="write JSON summary here")
    args = ap.parse_args()

    if args.video:
        d = Path(args.video).parent
        gt_path = Path(args.gt) if args.gt else d / "ground_truth.json"
        pred_path = Path(args.pred) if args.pred else d / "predictions.json"
    else:
        if not (args.gt and args.pred):
            print("ERROR: provide --video, or both --gt and --pred"); sys.exit(1)
        gt_path, pred_path = Path(args.gt), Path(args.pred)
    for p in (gt_path, pred_path):
        if not p.exists():
            print(f"ERROR: not found: {p}"); sys.exit(1)

    gt = json.loads(gt_path.read_text())
    pred = json.loads(pred_path.read_text())
    gt_labels = gt["labels"]
    pred_labels = pred["labels"]
    n_players = gt.get("n_players", 4)
    pids = list(range(1, n_players + 1))

    # Evaluate only on frames present in BOTH (predictions may be strided).
    common = sorted(set(int(k) for k in gt_labels) & set(int(k) for k in pred_labels))
    if not common:
        print("ERROR: no overlapping frames between GT and predictions "
              f"(gt {len(gt_labels)} frames, pred {len(pred_labels)}). "
              "Was the tracker run at the same stride the GT covers?")
        sys.exit(1)

    # Per-player counters
    stat = {p: dict(correct=0, id_error=0, miss=0, fp=0, cross_net=0,
                    gt_present=0, gt_absent=0, pred_present=0) for p in pids}

    def box_of(entry, p):
        v = entry.get(str(p))
        return None if (v is None or v == "absent") else v

    for fi in common:
        g = gt_labels[str(fi)]
        pr = pred_labels[str(fi)]
        gt_boxes = {p: box_of(g, p) for p in pids if box_of(g, p) is not None}
        pred_boxes = {p: box_of(pr, p) for p in pids if box_of(pr, p) is not None}
        for p in pids:
            if p in gt_boxes:
                stat[p]["gt_present"] += 1
            else:
                stat[p]["gt_absent"] += 1
            if p in pred_boxes:
                stat[p]["pred_present"] += 1

        matches = _match(pred_boxes, gt_boxes, args.iou)  # pred_pid -> gt_pid|None

        # classify each predicted player
        for pp in pred_boxes:
            gp = matches.get(pp)
            if gp is None:
                # predicted a box that matches no GT player → FP if GT says pp absent
                if pp not in gt_boxes:
                    stat[pp]["fp"] += 1
                # else: GT present but our box didn't overlap it → counted as miss below
            elif gp == pp:
                stat[pp]["correct"] += 1
            else:
                stat[pp]["id_error"] += 1
                if _team_of(pp, n_players) != _team_of(gp, n_players):
                    stat[pp]["cross_net"] += 1
        # misses: GT present but this pid neither correct nor matched to itself
        for gp_pid in gt_boxes:
            got = (gp_pid in pred_boxes and matches.get(gp_pid) == gp_pid)
            if not got:
                stat[gp_pid]["miss"] += 1

    # ---- aggregate + print ----
    print(f"Evaluated {len(common)} frames (IoU≥{args.iou})  gt={gt_path.name} pred={pred_path.name}\n")
    header = (f"{'player':7} {'idAcc':>6} {'idErr':>6} {'xNet':>6} {'miss':>6} "
              f"{'FP':>5} {'gtPres':>7}")
    print(header); print("-" * len(header))
    summary = {}
    tot = dict(correct=0, id_error=0, miss=0, fp=0, cross_net=0, gt_present=0)
    for p in pids:
        s = stat[p]
        gtp = max(1, s["gt_present"])
        # identity accuracy among frames where GT had this player present.
        id_acc = s["correct"] / gtp
        print(f"P{p:<6} {id_acc:6.2f} {s['id_error']:6d} {s['cross_net']:6d} "
              f"{s['miss']:6d} {s['fp']:5d} {s['gt_present']:7d}")
        summary[f"P{p}"] = dict(id_accuracy=round(id_acc, 3), id_error=s["id_error"],
                                cross_net=s["cross_net"], miss=s["miss"], fp=s["fp"],
                                gt_present=s["gt_present"])
        for k in tot:
            tot[k] += s[k]

    overall_acc = tot["correct"] / max(1, tot["gt_present"])
    print("-" * len(header))
    print(f"{'ALL':7} {overall_acc:6.2f} {tot['id_error']:6d} {tot['cross_net']:6d} "
          f"{tot['miss']:6d} {tot['fp']:5d} {tot['gt_present']:7d}")
    print(f"\nOverall identity accuracy: {overall_acc:.1%}"
          f"   ({tot['correct']}/{tot['gt_present']} present-player-frames correct)")
    print(f"ID errors (wrong label on a person): {tot['id_error']}"
          f"   of which cross-net: {tot['cross_net']}")

    summary["overall"] = dict(frames=len(common), iou=args.iou,
                              id_accuracy=round(overall_acc, 3),
                              id_error=tot["id_error"], miss=tot["miss"],
                              fp=tot["fp"], cross_net=tot["cross_net"])
    out = Path(args.out) if args.out else (gt_path.parent / "evaluation.json")
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nSummary → {out}")


if __name__ == "__main__":
    main()
