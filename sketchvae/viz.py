"""Visualization helpers for absolute strokes and stroke-5 sequences."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import stroke5_to_absolute


def draw_sketch(
    strokes: list[np.ndarray],
    ax: plt.Axes | None = None,
    *,
    title: str | None = None,
    color: str = "black",
    linewidth: float = 1.5,
    invert_y: bool = True,
) -> plt.Axes:
    """Plot a list of absolute (N, 2) strokes."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(4, 4))

    for stroke in strokes:
        pts = np.asarray(stroke, dtype=np.float32)
        if len(pts) < 2:
            if len(pts) == 1:
                ax.scatter(pts[0, 0], pts[0, 1], c=color, s=8)
            continue
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth, solid_capstyle="round")

    ax.set_aspect("equal")
    ax.axis("off")
    if invert_y:
        ax.invert_yaxis()
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def draw_stroke5(
    stroke5: np.ndarray,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    return draw_sketch(stroke5_to_absolute(stroke5), ax=ax, **kwargs)


def grid_sketches(
    samples: list[dict],
    *,
    nrows: int = 2,
    ncols: int = 4,
    save_path: Path | None = None,
    title_key: str = "description",
) -> Path | None:
    """Draw a grid of raw samples (uses 'strokes' key)."""
    n = min(len(samples), nrows * ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            s = samples[i]
            title = str(s.get(title_key, s.get("id", i)))[:40]
            draw_sketch(s["strokes"], ax=ax, title=title)
        else:
            ax.axis("off")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return None


def side_by_side_raw_vs_stroke5(
    sample: dict,
    stroke5: np.ndarray,
    *,
    save_path: Path | None = None,
) -> Path | None:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    draw_sketch(sample["strokes"], ax=axes[0], title="raw strokes")
    draw_stroke5(stroke5, ax=axes[1], title=f"stroke-5 (T={len(stroke5)})")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return None
