#!/usr/bin/env python3
"""Train Sketch-RNN VAE on preprocessed stroke-5 cache."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sketchvae.dataset import make_loaders
from sketchvae.device import get_device
from sketchvae.model import SketchRNN, SketchRNNConfig


def kl_weight_at(step: int, start: float, end: float, anneal_steps: int) -> float:
    if anneal_steps <= 0:
        return end
    t = min(1.0, step / float(anneal_steps))
    return start + t * (end - start)


@torch.no_grad()
def evaluate(model: SketchRNN, loader, device: torch.device, kl_weight: float, free_bits: float) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "stroke_loss": 0.0, "pen_loss": 0.0}
    n = 0
    for batch in loader:
        strokes = batch["strokes"].to(device)
        lengths = batch["lengths"].to(device)
        mask = batch["mask"].to(device)
        out = model(strokes, lengths, mask, free_bits=free_bits, kl_weight=kl_weight)
        b = strokes.size(0)
        for k in totals:
            totals[k] += float(out[k].item()) * b
        n += b
    return {k: v / max(n, 1) for k, v in totals.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=ROOT / "outputs" / "preprocessed" / "stroke5.npz")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "train")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-4, help="cosine anneal floor")
    p.add_argument("--enc-hidden", type=int, default=256)
    p.add_argument("--dec-hidden", type=int, default=512)
    p.add_argument("--z-size", type=int, default=64)
    p.add_argument("--n-mixtures", type=int, default=20)
    # Softer free-bits + milder β keep the latent informative for generation/completion
    p.add_argument("--kl-start", type=float, default=0.0)
    p.add_argument("--kl-end", type=float, default=0.2)
    p.add_argument("--kl-anneal-steps", type=int, default=4000)
    p.add_argument("--free-bits", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=5)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, scale, max_len = make_loaders(args.cache, batch_size=args.batch_size)
    cfg = SketchRNNConfig(
        enc_hidden=args.enc_hidden,
        dec_hidden=args.dec_hidden,
        z_size=args.z_size,
        n_mixtures=args.n_mixtures,
    )
    model = SketchRNN(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs, 1), eta_min=args.lr_min
    )

    meta = {
        **vars(args),
        "cache": str(args.cache),
        "out_dir": str(args.out_dir),
        "scale": scale,
        "max_len": max_len,
        "device_resolved": str(device),
        "n_train_batches": len(train_loader),
        "n_val_batches": len(val_loader),
        "n_params": sum(x.numel() for x in model.parameters()),
    }
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in meta.items()}
    (args.out_dir / "config.json").write_text(json.dumps(meta, indent=2))

    log_path = args.out_dir / "train_log.csv"
    fieldnames = [
        "epoch",
        "step",
        "kl_weight",
        "lr",
        "train_loss",
        "train_recon",
        "train_kl",
        "val_loss",
        "val_recon",
        "val_kl",
        "sec",
    ]
    with log_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    global_step = 0
    best_val = float("inf")
    print(f"device={device}  params={meta['n_params']:,}  scale={scale:.4f}")
    print(
        f"KL schedule: {args.kl_start} → {args.kl_end} over {args.kl_anneal_steps} steps, "
        f"free_bits={args.free_bits}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        n_seen = 0
        kw_last = args.kl_start
        lr_now = opt.param_groups[0]["lr"]

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            strokes = batch["strokes"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)
            kw_last = kl_weight_at(global_step, args.kl_start, args.kl_end, args.kl_anneal_steps)

            out = model(
                strokes,
                lengths,
                mask,
                free_bits=args.free_bits,
                kl_weight=kw_last,
            )
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            b = strokes.size(0)
            running["loss"] += float(out["loss"].item()) * b
            running["recon"] += float(out["recon"].item()) * b
            running["kl"] += float(out["kl"].item()) * b
            n_seen += b
            global_step += 1
            pbar.set_postfix(
                loss=f"{out['loss'].item():.3f}",
                recon=f"{out['recon'].item():.3f}",
                kl=f"{out['kl'].item():.3f}",
                w=f"{kw_last:.3f}",
            )

        scheduler.step()
        train_metrics = {k: v / max(n_seen, 1) for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, device, kw_last, args.free_bits)
        sec = time.time() - t0

        row = {
            "epoch": epoch,
            "step": global_step,
            "kl_weight": kw_last,
            "lr": lr_now,
            "train_loss": train_metrics["loss"],
            "train_recon": train_metrics["recon"],
            "train_kl": train_metrics["kl"],
            "val_loss": val_metrics["loss"],
            "val_recon": val_metrics["recon"],
            "val_kl": val_metrics["kl"],
            "sec": sec,
        }
        with log_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        print(
            f"epoch {epoch:03d}  "
            f"train loss={train_metrics['loss']:.3f} recon={train_metrics['recon']:.3f} kl={train_metrics['kl']:.3f}  "
            f"val loss={val_metrics['loss']:.3f} recon={val_metrics['recon']:.3f} kl={val_metrics['kl']:.3f}  "
            f"w_kl={kw_last:.3f} lr={lr_now:.2e}  ({sec:.1f}s)"
        )

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "scale": scale,
            "max_len": max_len,
            "val_loss": val_metrics["loss"],
            "val_recon": val_metrics["recon"],
            "args": meta,
        }
        torch.save(ckpt, args.out_dir / "last.pt")
        if val_metrics["recon"] < best_val:
            best_val = val_metrics["recon"]
            torch.save(ckpt, args.out_dir / "best.pt")
            print(f"  saved best.pt (val_recon={best_val:.3f})")
        if epoch % args.save_every == 0:
            torch.save(ckpt, args.out_dir / f"epoch_{epoch:03d}.pt")

    print(f"done. best val_recon={best_val:.3f}  artifacts in {args.out_dir}")


if __name__ == "__main__":
    main()
