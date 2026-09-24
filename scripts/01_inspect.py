#!/usr/bin/env python3
"""Inspect Creative Creatures pickles and verify stroke-5 conversion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sketchvae.data import (
    compute_scale,
    load_raw_split,
    preprocess_split,
    project_root,
    sample_to_stroke5,
)
from sketchvae.viz import grid_sketches, side_by_side_raw_vs_stroke5


def length_stats(samples: list[dict]) -> dict:
    n_strokes = []
    n_points = []
    for s in samples:
        strokes = s["strokes"]
        n_strokes.append(len(strokes))
        n_points.append(sum(len(st) for st in strokes))
    n_strokes = np.asarray(n_strokes)
    n_points = np.asarray(n_points)
    return {
        "n": int(len(samples)),
        "strokes_mean": float(n_strokes.mean()),
        "strokes_median": float(np.median(n_strokes)),
        "strokes_min": int(n_strokes.min()),
        "strokes_max": int(n_strokes.max()),
        "points_mean": float(n_points.mean()),
        "points_median": float(np.median(n_points)),
        "points_min": int(n_points.min()),
        "points_max": int(n_points.max()),
        "points_p90": float(np.percentile(n_points, 90)),
        "points_p95": float(np.percentile(n_points, 95)),
        "points_p99": float(np.percentile(n_points, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root())
    parser.add_argument("--out-dir", type=Path, default=project_root() / "outputs" / "inspect")
    parser.add_argument("--epsilon", type=float, default=2.0, help="RDP epsilon (0 disables)")
    parser.add_argument("--max-len", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    report: dict = {"epsilon": args.epsilon, "max_len": args.max_len}

    for split in ("train", "val"):
        samples = load_raw_split(split, args.data_dir)
        raw = length_stats(samples)
        report[f"{split}_raw"] = raw
        print(f"\n[{split}] raw n={raw['n']}")
        print(
            f"  strokes/sketch: mean={raw['strokes_mean']:.1f} "
            f"med={raw['strokes_median']:.0f} [{raw['strokes_min']}, {raw['strokes_max']}]"
        )
        print(
            f"  points/sketch:  mean={raw['points_mean']:.1f} "
            f"med={raw['points_median']:.0f} p95={raw['points_p95']:.0f} "
            f"[{raw['points_min']}, {raw['points_max']}]"
        )

        # Grid of raw sketches
        idx = rng.choice(len(samples), size=min(8, len(samples)), replace=False)
        grid = [samples[i] for i in idx]
        path = grid_sketches(grid, save_path=args.out_dir / f"{split}_raw_grid.png")
        print(f"  wrote {path}")

        # Preprocess
        sequences, kept_ids, stats = preprocess_split(
            samples, epsilon=args.epsilon, max_len=args.max_len
        )
        report[f"{split}_preprocessed"] = stats
        print(
            f"  stroke-5 kept {int(stats['n_kept'])}/{int(stats['n_in'])} "
            f"(short={int(stats['n_short'])}, long={int(stats['n_long'])}, empty={int(stats['n_empty'])})"
        )
        print(
            f"  T: mean={stats['len_mean']:.1f} med={stats['len_median']:.0f} "
            f"[{stats['len_min']:.0f}, {stats['len_max']:.0f}]"
        )

        # Side-by-side for a few samples (same indices when possible)
        for j, i in enumerate(idx[:4]):
            sample = samples[int(i)]
            s5 = sample_to_stroke5(sample, epsilon=args.epsilon)
            side_by_side_raw_vs_stroke5(
                sample,
                s5,
                save_path=args.out_dir / f"{split}_raw_vs_s5_{j}.png",
            )

        if split == "train" and sequences:
            scale = compute_scale(sequences)
            report["train_scale"] = scale
            print(f"  offset scale (std dx/dy) = {scale:.4f}")

    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
