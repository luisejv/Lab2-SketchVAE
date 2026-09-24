#!/usr/bin/env python3
"""Preprocess train/val into normalized stroke-5 arrays and save a cache."""

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
    normalize_stroke5,
    preprocess_split,
    project_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root())
    parser.add_argument("--out-dir", type=Path, default=project_root() / "outputs" / "preprocessed")
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--max-len", type=int, default=250)
    parser.add_argument("--min-len", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "epsilon": args.epsilon,
        "max_len": args.max_len,
        "min_len": args.min_len,
    }

    train_raw = load_raw_split("train", args.data_dir)
    train_seqs, train_ids, train_stats = preprocess_split(
        train_raw, epsilon=args.epsilon, max_len=args.max_len, min_len=args.min_len
    )
    scale = compute_scale(train_seqs)
    train_norm = [normalize_stroke5(s, scale) for s in train_seqs]

    val_raw = load_raw_split("val", args.data_dir)
    val_seqs, val_ids, val_stats = preprocess_split(
        val_raw, epsilon=args.epsilon, max_len=args.max_len, min_len=args.min_len
    )
    val_norm = [normalize_stroke5(s, scale) for s in val_seqs]

    meta.update(
        {
            "scale": scale,
            "train": train_stats,
            "val": val_stats,
            "n_train": len(train_norm),
            "n_val": len(val_norm),
        }
    )

    out_path = args.out_dir / "stroke5.npz"
    # store as object arrays of variable-length sequences
    np.savez_compressed(
        out_path,
        train=np.array(train_norm, dtype=object),
        val=np.array(val_norm, dtype=object),
        train_ids=np.asarray(train_ids, dtype=np.int64),
        val_ids=np.asarray(val_ids, dtype=np.int64),
        scale=np.asarray(scale, dtype=np.float64),
        max_len=np.asarray(args.max_len, dtype=np.int64),
        epsilon=np.asarray(args.epsilon, dtype=np.float64),
    )
    meta_path = args.out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"scale={scale:.4f}")
    print(f"train kept {len(train_norm)}/{len(train_raw)}")
    print(f"val   kept {len(val_norm)}/{len(val_raw)}")
    print(f"wrote {out_path}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
