#!/usr/bin/env python3
"""Generate figures: unconditional samples, reconstructions, completions."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sketchvae.data import denormalize_stroke5
from sketchvae.dataset import load_stroke5_cache
from sketchvae.device import get_device
from sketchvae.model import SketchRNN, SketchRNNConfig, complete, generate, reconstruct
from sketchvae.viz import draw_stroke5


def load_model(ckpt_path: Path, device: torch.device) -> tuple[SketchRNN, float, int]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = SketchRNNConfig(**ckpt["cfg"])
    model = SketchRNN(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, float(ckpt["scale"]), int(ckpt.get("max_len", 250))


def tensor_to_stroke5(t: torch.Tensor, scale: float) -> np.ndarray:
    arr = t.detach().cpu().numpy().astype(np.float32)
    eos = np.where(arr[:, 4] > 0.5)[0]
    if len(eos):
        arr = arr[: eos[0] + 1]
    return denormalize_stroke5(arr, scale)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=ROOT / "outputs" / "train_v2" / "best.pt")
    p.add_argument("--cache", type=Path, default=ROOT / "outputs" / "preprocessed" / "stroke5.npz")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "eval_v2")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, scale, max_len = load_model(args.ckpt, device)
    _, val_arr, _, _ = load_stroke5_cache(args.cache)

    # --- Unconditional generation ---
    samples = generate(
        model, n=16, max_len=min(200, max_len), temperature=args.temperature, device=device
    )
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, i in zip(axes.ravel(), range(16)):
        draw_stroke5(tensor_to_stroke5(samples[i], scale), ax=ax, title=f"z#{i}")
    fig.suptitle(f"Unconditional samples (τ={args.temperature})", y=1.01)
    fig.tight_layout()
    path = args.out_dir / "unconditional.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)

    # --- Reconstructions (deterministic z=μ + greedy MDN) ---
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(val_arr), size=8, replace=False)
    for i, idx in enumerate(idxs):
        seq = torch.from_numpy(np.asarray(val_arr[idx], dtype=np.float32))
        recon = reconstruct(model, seq, device=device, greedy=True, use_mean_z=True)
        r, c = divmod(i, 2)
        draw_stroke5(tensor_to_stroke5(seq, scale), ax=axes[r, c * 2], title=f"gt #{idx}")
        draw_stroke5(tensor_to_stroke5(recon, scale), ax=axes[r, c * 2 + 1], title="recon")
    fig.suptitle("Reconstructions (encode μ → greedy decode)", y=1.01)
    fig.tight_layout()
    path = args.out_dir / "reconstructions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)

    # --- Completions ---
    idx = int(rng.choice(len(val_arr)))
    full = np.asarray(val_arr[idx], dtype=np.float32)
    n_prefix = max(5, len(full) // 3)
    lift = np.where(full[:n_prefix, 3] > 0.5)[0]
    if len(lift):
        n_prefix = int(lift[-1]) + 1
    prefix = torch.from_numpy(full[:n_prefix])

    comps = complete(
        model,
        prefix,
        n_samples=4,
        max_len=min(max_len, len(full) + 80),
        temperature=args.temperature,
        device=device,
        from_prefix_posterior=True,
    )
    fig, axes = plt.subplots(1, 6, figsize=(16, 3))
    draw_stroke5(tensor_to_stroke5(torch.from_numpy(full), scale), ax=axes[0], title="ground truth")
    draw_stroke5(tensor_to_stroke5(prefix, scale), ax=axes[1], title=f"prefix T={n_prefix}", color="crimson")
    for i in range(4):
        draw_stroke5(
            tensor_to_stroke5(comps[i], scale),
            ax=axes[i + 2],
            title=f"completion z#{i}",
        )
    fig.suptitle(f"Sketch completion (q(z|prefix), τ={args.temperature})", y=1.05)
    fig.tight_layout()
    path = args.out_dir / "completion.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)

    log_csv = args.ckpt.parent / "train_log.csv"
    if log_csv.exists():
        rows = list(csv.DictReader(log_csv.open()))
        epochs = [int(r["epoch"]) for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(epochs, [float(r["train_recon"]) for r in rows], label="train")
        axes[0].plot(epochs, [float(r["val_recon"]) for r in rows], label="val")
        axes[0].set_title("Reconstruction NLL")
        axes[0].legend()
        axes[1].plot(epochs, [float(r["train_kl"]) for r in rows], label="train KL")
        axes[1].plot(epochs, [float(r["val_kl"]) for r in rows], label="val KL")
        axes[1].plot(epochs, [float(r["kl_weight"]) for r in rows], label="β", ls="--")
        axes[1].set_title("KL + β")
        axes[1].legend()
        fig.tight_layout()
        path = args.out_dir / "curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
