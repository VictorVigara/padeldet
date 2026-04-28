"""
Compute and display player metrics from saved detection data.

Usage:
    python scripts/evaluate_metrics.py --data-dir data/rally
    python scripts/evaluate_metrics.py --data-dir data/rally --fps 30 --plot

    # Legacy explicit paths still work:
    python scripts/evaluate_metrics.py --player-json data/rally/player_detections.json --court-keypoints data/rally/court_keypoints.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from padeldet.court.models import CourtKeypoints
from padeldet.court.projection import CourtProjection
from padeldet.metrics import MetricsConfig, MatchMetrics, compute_metrics
from padeldet.models import load_detections


def parse_args():
    parser = argparse.ArgumentParser(description="Compute player metrics")
    parser.add_argument("--data-dir", default=None, help="Folder containing player_detections.json and court_keypoints.json; outputs written here")
    parser.add_argument("--player-json", default=None, help="Player detections JSON (overrides --data-dir)")
    parser.add_argument("--court-keypoints", default=None, help="Court keypoints JSON (overrides --data-dir)")
    parser.add_argument("--fps", type=float, default=25.0, help="Video FPS")
    parser.add_argument("--output", default=None, help="Save metrics to JSON (overrides --data-dir)")
    parser.add_argument(
        "--plot", action="store_true", help="Generate heatmap plots (requires matplotlib)"
    )
    args = parser.parse_args()

    if args.data_dir:
        d = Path(args.data_dir)
        if args.player_json is None:
            args.player_json = str(d / "player_detections.json")
        if args.court_keypoints is None:
            args.court_keypoints = str(d / "court_keypoints.json")
        if args.output is None:
            args.output = str(d / "metrics.json")
        args._plot_prefix = str(d / "metrics")
    else:
        args._plot_prefix = args.output.replace(".json", "") if args.output else "metrics"

    if not args.player_json or not args.court_keypoints:
        parser.error("Provide --data-dir or both --player-json and --court-keypoints")

    return args


def print_summary(metrics: MatchMetrics) -> None:
    """Print a human-readable metrics summary."""
    print("\n" + "=" * 60)
    print("PLAYER METRICS SUMMARY")
    print("=" * 60)
    print(f"Duration: {metrics.duration_seconds:.1f}s ({metrics.total_frames} frames @ {metrics.fps:.1f} fps)")

    for p in metrics.players:
        print(f"\n--- Player {p.track_id} ({p.team} team) ---")
        print(f"  Distance:  {p.total_distance_m:.1f} m")
        print(f"  Speed:     avg {p.avg_speed_ms:.2f} m/s, max {p.max_speed_ms:.2f} m/s")
        print(f"  Speed distribution:")
        for cat, frac in p.speed_distribution.items():
            print(f"    {cat:12s}: {frac * 100:5.1f}%")
        print(f"  Zone distribution:")
        for zone, frac in p.zone_distribution.items():
            print(f"    {zone:12s}: {frac * 100:5.1f}%")
        print(f"  Net approaches: {p.net_approaches}")

    for t in metrics.teams:
        print(f"\n--- {t.team.capitalize()} team (players {t.player_ids}) ---")
        print(f"  Formation distribution:")
        for form, frac in t.formation_distribution.items():
            print(f"    {form:20s}: {frac * 100:5.1f}%")

    print("\n" + "=" * 60)


def plot_heatmaps(metrics: MatchMetrics, output_prefix: str) -> None:
    """Generate heatmap plots for each player."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    for p in metrics.players:
        heatmap = np.array(p.position_heatmap)
        fig, ax = plt.subplots(1, 1, figsize=(5, 10))

        # Transpose so X is horizontal, Y is vertical; flip Y so baseline is at bottom
        ax.imshow(
            heatmap.T[::-1],
            extent=[0, 10, 0, 20],
            aspect="equal",
            cmap="hot",
            interpolation="bilinear",
        )

        # Court lines
        ax.axhline(y=10, color="white", linewidth=2, label="Net")
        ax.axhline(y=3, color="white", linewidth=0.5, linestyle="--")
        ax.axhline(y=17, color="white", linewidth=0.5, linestyle="--")
        ax.axvline(x=5, color="white", linewidth=0.5, linestyle="--")

        ax.set_xlabel("Court width (m)")
        ax.set_ylabel("Court length (m)")
        ax.set_title(f"Player {p.track_id} ({p.team} team) — Position Heatmap")

        path = f"{output_prefix}_player{p.track_id}_heatmap.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved heatmap: {path}")


def main():
    args = parse_args()

    # Load data
    print(f"Loading detections from: {args.player_json}")
    detections = load_detections(args.player_json)
    print(f"  {len(detections)} frames loaded")

    print(f"Loading court keypoints from: {args.court_keypoints}")
    keypoints = CourtKeypoints.load(args.court_keypoints)
    projection = CourtProjection(keypoints)
    print(f"  {keypoints.num_available}/12 keypoints available")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(detections, projection, args.fps)

    # Print summary
    print_summary(metrics)

    # Save JSON
    if args.output:
        metrics.save(args.output)
        print(f"\nMetrics saved to: {args.output}")

    # Optional plots
    if args.plot:
        plot_heatmaps(metrics, args._plot_prefix)


if __name__ == "__main__":
    main()
