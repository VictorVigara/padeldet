"""How separable are the 4 players by appearance? — a feasibility study for re-id.

Given dense ground-truth player boxes (ground_truth.json from label_players.py),
crop each player from the video and compare several appearance features to answer:
  * Can the 4 players be told apart at all, and by how much?
  * WHICH feature separates them best (jersey color vs deep OSNet vs geometry)?
  * WHICH player pairs are hard (e.g. same-team P3/P4 in similar kit)?

For each feature we report:
  * intra-player vs inter-player mean cosine/hist distance (good feature: intra << inter)
  * per-pair separation matrix
  * 1-NN leave-one-out classification accuracy (assign a held-out crop to a player)
  * a 2D PCA scatter (PNG) so clusters are visible

Usage
-----
    python scripts/analyze_player_separability.py --video data/full_score/2/2.mp4
    # defaults: --gt <video_dir>/ground_truth.json, --out <video_dir>/separability/
    #           --max-per-player 400  (subsample crops for speed)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_PLAYER_COLORS = {1: "green", 2: "orange", 3: "blue", 4: "pink"}
_PLOT_COLORS = {1: "#2ecc71", 2: "#e67e22", 3: "#3498db", 4: "#e84393"}


# ---------------------------------------------------------------------------
# Crop extraction
# ---------------------------------------------------------------------------
def _torso_crop(img: np.ndarray, box: list[float]) -> np.ndarray | None:
    """Upper-center region of a person box ≈ torso/jersey (robust to legs/bg)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 8:
        return None
    # torso band: 15%..55% of height, central 60% of width
    tx1 = int(x1 + 0.20 * bw); tx2 = int(x1 + 0.80 * bw)
    ty1 = int(y1 + 0.15 * bh); ty2 = int(y1 + 0.55 * bh)
    tx1, tx2 = max(0, tx1), min(w, tx2)
    ty1, ty2 = max(0, ty1), min(h, ty2)
    if tx2 - tx1 < 3 or ty2 - ty1 < 3:
        return None
    return img[ty1:ty2, tx1:tx2]


def _full_crop(img: np.ndarray, box: list[float]) -> np.ndarray | None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Feature extractors — each returns a 1-D float vector (L2-normalized where apt)
# ---------------------------------------------------------------------------
def feat_hsv_hist(torso: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 12], [0, 180, 0, 256])
    hist = hist.flatten()
    n = hist.sum()
    return hist / n if n > 0 else hist


def feat_lab_hist(torso: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [1, 2], None, [16, 16], [0, 256, 0, 256])  # a,b channels
    hist = hist.flatten()
    n = hist.sum()
    return hist / n if n > 0 else hist


def feat_mean_color(torso: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return lab.mean(axis=0) / 255.0  # 3-d


def feat_geometry(box: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    aspect = bw / max(bh, 1e-6)
    # normalized center + size (depth cue via size/position)
    return np.array([bh / 1080.0, bw / 1920.0, aspect,
                     (x1 + x2) / 2 / 1920.0, (y1 + y2) / 2 / 1080.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Separability metrics
# ---------------------------------------------------------------------------
def _pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return 1.0 - an @ bn.T  # cosine distance in [0,2]


def separability(feats: dict[int, np.ndarray]) -> dict:
    """feats: {pid: (N_pid, D)}. Returns intra/inter, per-pair, 1-NN accuracy."""
    pids = sorted(feats.keys())
    # mean intra-player distance (average cosine dist within a player's crops)
    intra = {}
    for p in pids:
        X = feats[p]
        if len(X) < 2:
            intra[p] = float("nan"); continue
        d = _pairwise_cosine(X, X)
        iu = np.triu_indices(len(X), k=1)
        intra[p] = float(d[iu].mean())
    # per-pair inter distance (mean cosine dist between two players' crops)
    inter = {}
    for i, p in enumerate(pids):
        for q in pids[i + 1:]:
            d = _pairwise_cosine(feats[p], feats[q])
            inter[(p, q)] = float(d.mean())
    mean_intra = float(np.nanmean(list(intra.values())))
    mean_inter = float(np.mean(list(inter.values())))
    # separation ratio: inter / intra (>1 good; higher = more separable)
    ratio = mean_inter / mean_intra if mean_intra > 0 else float("inf")

    # 1-NN leave-one-out accuracy using class-centroid nearest (fast, robust)
    centroids = {p: feats[p].mean(axis=0) for p in pids}
    C = np.stack([centroids[p] for p in pids])
    correct = total = 0
    per_class_acc = {p: [0, 0] for p in pids}
    for p in pids:
        X = feats[p]
        d = _pairwise_cosine(X, C)      # (N, n_players)
        pred = [pids[j] for j in d.argmin(axis=1)]
        for pr in pred:
            total += 1
            per_class_acc[p][1] += 1
            if pr == p:
                correct += 1
                per_class_acc[p][0] += 1
    acc = correct / total if total else 0.0
    return {
        "intra": intra, "inter": inter,
        "mean_intra": mean_intra, "mean_inter": mean_inter,
        "separation_ratio": ratio, "nn_accuracy": acc,
        "per_class_acc": {p: (c / max(1, n)) for p, (c, n) in per_class_acc.items()},
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_pca(feats: dict[int, np.ndarray], title: str, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    X = np.concatenate([feats[p] for p in sorted(feats)])
    y = np.concatenate([[p] * len(feats[p]) for p in sorted(feats)])
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    if Xn.shape[1] > 2:
        Xn = PCA(n_components=2).fit_transform(Xn)
    fig, ax = plt.subplots(figsize=(6, 6))
    for p in sorted(feats):
        m = y == p
        ax.scatter(Xn[m, 0], Xn[m, 1], s=6, alpha=0.4,
                   c=_PLOT_COLORS[p], label=f"P{p} ({_PLAYER_COLORS[p]})")
    ax.set_title(title); ax.legend(markerscale=2, fontsize=8)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Player appearance separability study")
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", default=None, help="ground_truth.json (default: <video_dir>/ground_truth.json)")
    ap.add_argument("--out", default=None, help="output dir (default: <video_dir>/separability)")
    ap.add_argument("--max-per-player", type=int, default=400,
                    help="subsample this many crops per player (speed)")
    ap.add_argument("--osnet", action="store_true", default=True,
                    help="also compute OSNet deep embeddings (CPU ok, slower)")
    ap.add_argument("--no-osnet", dest="osnet", action="store_false")
    args = ap.parse_args()

    video = Path(args.video)
    gt_path = Path(args.gt) if args.gt else video.parent / "ground_truth.json"
    out_dir = Path(args.out) if args.out else video.parent / "separability"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not gt_path.exists():
        print(f"ERROR: ground truth not found: {gt_path}"); sys.exit(1)

    gt = json.loads(gt_path.read_text())
    labels = gt["labels"]
    n_players = gt.get("n_players", 4)

    # Collect (frame_idx, pid, box) for present players, then subsample per player.
    per_player_frames: dict[int, list[tuple[int, list]]] = {p: [] for p in range(1, n_players + 1)}
    for f_str, entry in labels.items():
        fi = int(f_str)
        for pid_str, box in entry.items():
            if box == "absent" or box is None:
                continue
            per_player_frames[int(pid_str)].append((fi, box))
    rng = np.random.default_rng(0)
    for p in per_player_frames:
        lst = per_player_frames[p]
        if len(lst) > args.max_per_player:
            idx = rng.choice(len(lst), args.max_per_player, replace=False)
            per_player_frames[p] = [lst[i] for i in sorted(idx)]
    print("crops/player:", {p: len(v) for p, v in per_player_frames.items()})

    # Read crops (grouped read to limit seeks — sort by frame index).
    want = {}  # frame_idx -> list[(pid, box)]
    for p, lst in per_player_frames.items():
        for fi, box in lst:
            want.setdefault(fi, []).append((p, box))
    cap = cv2.VideoCapture(str(video))

    # Optional OSNet model — features computed inline per frame (full frame + box).
    reid = None
    osnet: dict[int, list[np.ndarray]] = {p: [] for p in range(1, n_players + 1)}
    if args.osnet:
        try:
            from boxmot.reid.core import ReID
            reid = ReID(weights="weights/osnet_x0_25_msmt17.pt", device="cpu", half=False)
            print("OSNet loaded (CPU).")
        except Exception as e:
            print(f"OSNet unavailable ({e}); continuing without it.")

    crops: dict[int, list[np.ndarray]] = {p: [] for p in range(1, n_players + 1)}
    boxes: dict[int, list[list]] = {p: [] for p in range(1, n_players + 1)}
    for fi in sorted(want):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        if not ok:
            continue
        present = want[fi]
        for p, box in present:
            c = _full_crop(img, box)
            if c is not None:
                crops[p].append(c)
                boxes[p].append(box)
        if reid is not None and present:
            xyxy = np.array([b[:4] for _, b in present], dtype=float)
            try:
                feats = np.asarray(reid.model.get_features(xyxy, img))
                for (p, _), fv in zip(present, feats):
                    osnet[p].append(fv.astype(np.float32))
            except Exception:
                pass
    cap.release()
    print("crops read:", {p: len(v) for p, v in crops.items()})

    # Build feature sets.
    feature_sets: dict[str, dict[int, np.ndarray]] = {}

    def build(name, fn_crop):
        fs = {}
        for p in range(1, n_players + 1):
            vecs = []
            for c in crops[p]:
                t = _torso_crop_from_full(c)
                v = fn_crop(t) if t is not None else None
                if v is not None and np.isfinite(v).all():
                    vecs.append(v)
            if vecs:
                fs[p] = np.stack(vecs)
        if len(fs) >= 2:
            feature_sets[name] = fs

    build("hsv_hist", feat_hsv_hist)
    build("lab_hist", feat_lab_hist)
    build("mean_color", feat_mean_color)

    # geometry from boxes (no crop needed)
    geo = {}
    for p in range(1, n_players + 1):
        if boxes[p]:
            geo[p] = np.stack([feat_geometry(b) for b in boxes[p]])
    if len(geo) >= 2:
        feature_sets["geometry"] = geo

    # OSNet deep embeddings (collected inline above)
    osnet_fs = {p: np.stack(osnet[p]) for p in range(1, n_players + 1) if osnet[p]}
    if len(osnet_fs) >= 2:
        feature_sets["osnet"] = osnet_fs

    # Score + plot each feature.
    report = {}
    print("\n=== SEPARABILITY (higher separation_ratio & nn_accuracy = better) ===")
    ranked = []
    for name, fs in feature_sets.items():
        s = separability(fs)
        report[name] = s
        plot_pca(fs, f"{name}  (ratio={s['separation_ratio']:.2f}, 1-NN acc={s['nn_accuracy']:.2f})",
                 out_dir / f"pca_{name}.png")
        ranked.append((name, s["separation_ratio"], s["nn_accuracy"]))
        print(f"\n[{name}]  separation_ratio={s['separation_ratio']:.2f}  "
              f"1-NN_acc={s['nn_accuracy']:.2f}")
        print(f"   per-player acc: " +
              "  ".join(f"P{p}={s['per_class_acc'][p]:.2f}" for p in sorted(s['per_class_acc'])))
        worst = sorted(s["inter"].items(), key=lambda kv: kv[1])[:3]
        print("   hardest pairs (smallest inter-dist): " +
              ", ".join(f"P{a}-P{b}:{d:.3f}" for (a, b), d in worst))

    ranked.sort(key=lambda r: (r[2], r[1]), reverse=True)
    print("\n=== RANKING (by 1-NN accuracy) ===")
    for name, ratio, acc in ranked:
        print(f"   {name:12s} acc={acc:.2f}  ratio={ratio:.2f}")

    (out_dir / "separability_report.json").write_text(
        json.dumps({k: _jsonable(v) for k, v in report.items()}, indent=1))
    print(f"\nPlots + report -> {out_dir}/")


def _torso_crop_from_full(full_crop: np.ndarray) -> np.ndarray | None:
    """Torso band of an already-cropped person image."""
    h, w = full_crop.shape[:2]
    if h < 8 or w < 4:
        return None
    ty1, ty2 = int(0.15 * h), int(0.55 * h)
    tx1, tx2 = int(0.20 * w), int(0.80 * w)
    t = full_crop[ty1:ty2, tx1:tx2]
    return t if t.size > 0 else None


def _jsonable(s: dict) -> dict:
    out = {}
    for k, v in s.items():
        if isinstance(v, dict):
            out[k] = {(f"{kk[0]}-{kk[1]}" if isinstance(kk, tuple) else str(kk)): vv
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
