"""Load Creative Creatures pickles and convert to Sketch-RNN stroke-5."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

import numpy as np

# Pen states in stroke-5: (dx, dy, p1, p2, p3)
# p1 = draw, p2 = lift (end of stroke), p3 = end of sketch
PEN_DRAW = 0
PEN_LIFT = 1
PEN_EOS = 2


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_raw_split(split: str = "train", data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load train.pkl.gz or val.pkl.gz as a list of sample dicts."""
    data_dir = data_dir or project_root()
    path = data_dir / f"{split}.pkl.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    with gzip.open(path, "rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"Expected list in {path}, got {type(samples)}")
    return samples


def rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker polyline simplification.

    Args:
        points: (N, 2) absolute coordinates.
        epsilon: distance threshold; larger => fewer points.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3 or epsilon <= 0:
        return points.astype(np.float32)

    keep = np.zeros(len(points), dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(points) - 1)]

    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        segment = points[start : end + 1]
        a, b = segment[0], segment[-1]
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-12:
            d = np.linalg.norm(segment[1:-1] - a, axis=1)
        else:
            t = np.clip(((segment[1:-1] - a) @ ab) / ab_len2, 0.0, 1.0)
            proj = a + t[:, None] * ab
            d = np.linalg.norm(segment[1:-1] - proj, axis=1)
        idx = int(np.argmax(d))
        if d[idx] > epsilon:
            mid = start + 1 + idx
            keep[mid] = True
            stack.append((start, mid))
            stack.append((mid, end))

    return points[keep].astype(np.float32)


def simplify_strokes(strokes: list[np.ndarray], epsilon: float) -> list[np.ndarray]:
    if epsilon <= 0:
        return [np.asarray(s, dtype=np.float32) for s in strokes]
    out = []
    for s in strokes:
        simp = rdp_simplify(s, epsilon)
        if len(simp) >= 2:
            out.append(simp)
        elif len(s) >= 1:
            # keep at least endpoints of original if RDP collapsed too hard
            s = np.asarray(s, dtype=np.float32)
            out.append(s if len(s) == 1 else s[[0, -1]])
    return out


def strokes_to_stroke5(
    strokes: list[np.ndarray],
    *,
    epsilon: float = 0.0,
) -> np.ndarray:
    """Convert list of absolute (N_i, 2) strokes into stroke-5 sequence.

    Returns:
        Array of shape (T, 5): [dx, dy, p1, p2, p3]
        The first point of the sketch is relative to the origin (0, 0).
    """
    strokes = simplify_strokes(strokes, epsilon)
    if not strokes:
        raise ValueError("empty stroke list")

    rows: list[np.ndarray] = []
    prev = np.zeros(2, dtype=np.float32)

    for stroke in strokes:
        pts = np.asarray(stroke, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"expected (N,2) stroke, got {pts.shape}")
        if len(pts) == 0:
            continue
        for i, pt in enumerate(pts):
            delta = pt - prev
            prev = pt
            pen = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # draw by default
            if i == len(pts) - 1:
                pen = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # lift after last point
            rows.append(np.concatenate([delta, pen]))

    if not rows:
        raise ValueError("no points after conversion")

    seq = np.stack(rows, axis=0)
    # Last point of the full sketch: end-of-sketch instead of lift
    seq[-1, 2:] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return seq


def stroke5_to_absolute(stroke5: np.ndarray) -> list[np.ndarray]:
    """Invert stroke-5 offsets into a list of absolute stroke polylines."""
    stroke5 = np.asarray(stroke5, dtype=np.float32)
    strokes: list[np.ndarray] = []
    current: list[np.ndarray] = []
    xy = np.zeros(2, dtype=np.float32)

    for row in stroke5:
        xy = xy + row[:2]
        current.append(xy.copy())
        p1, p2, p3 = row[2], row[3], row[4]
        if p2 > 0.5 or p3 > 0.5:
            if current:
                strokes.append(np.stack(current, axis=0))
            current = []
        if p3 > 0.5:
            break

    if current:
        strokes.append(np.stack(current, axis=0))
    return strokes


def normalize_stroke5(stroke5: np.ndarray, scale: float) -> np.ndarray:
    """Scale dx, dy by 1/scale (pen channels unchanged)."""
    out = stroke5.copy()
    out[:, :2] = out[:, :2] / scale
    return out


def denormalize_stroke5(stroke5: np.ndarray, scale: float) -> np.ndarray:
    out = stroke5.copy()
    out[:, :2] = out[:, :2] * scale
    return out


def compute_scale(sequences: list[np.ndarray]) -> float:
    """Standard deviation of all dx, dy offsets (Sketch-RNN style)."""
    if not sequences:
        raise ValueError("no sequences")
    deltas = np.concatenate([s[:, :2] for s in sequences], axis=0)
    scale = float(np.std(deltas))
    if scale < 1e-6:
        scale = 1.0
    return scale


def sample_to_stroke5(sample: dict[str, Any], epsilon: float = 0.0) -> np.ndarray:
    return strokes_to_stroke5(sample["strokes"], epsilon=epsilon)


def preprocess_split(
    samples: list[dict[str, Any]],
    *,
    epsilon: float = 2.0,
    max_len: int = 250,
    min_len: int = 10,
) -> tuple[list[np.ndarray], list[int], dict[str, float]]:
    """Convert raw samples to stroke-5 and filter by length.

    Returns:
        sequences, kept_ids, stats dict
    """
    sequences: list[np.ndarray] = []
    kept_ids: list[int] = []
    n_empty = 0
    n_short = 0
    n_long = 0

    for sample in samples:
        try:
            seq = strokes_to_stroke5(sample["strokes"], epsilon=epsilon)
        except ValueError:
            n_empty += 1
            continue
        t = len(seq)
        if t < min_len:
            n_short += 1
            continue
        if t > max_len:
            n_long += 1
            continue
        sequences.append(seq)
        kept_ids.append(int(sample["id"]))

    lengths = np.array([len(s) for s in sequences], dtype=np.int32) if sequences else np.array([])
    stats = {
        "n_in": float(len(samples)),
        "n_kept": float(len(sequences)),
        "n_empty": float(n_empty),
        "n_short": float(n_short),
        "n_long": float(n_long),
        "len_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "len_median": float(np.median(lengths)) if len(lengths) else 0.0,
        "len_min": float(lengths.min()) if len(lengths) else 0.0,
        "len_max": float(lengths.max()) if len(lengths) else 0.0,
    }
    return sequences, kept_ids, stats


def pad_batch(sequences: list[np.ndarray], max_len: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pad stroke-5 sequences to a common length.

    Returns:
        batch: (B, T, 5)
        lengths: (B,)
    """
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    t = int(max_len if max_len is not None else lengths.max())
    batch = np.zeros((len(sequences), t, 5), dtype=np.float32)
    # default pen state for padding: eos
    batch[:, :, 4] = 1.0
    for i, seq in enumerate(sequences):
        n = min(len(seq), t)
        batch[i, :n] = seq[:n]
    return batch, lengths
