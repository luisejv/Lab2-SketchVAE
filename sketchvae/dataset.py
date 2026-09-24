"""PyTorch Dataset over preprocessed stroke-5 cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Stroke5Dataset(Dataset):
    """Variable-length stroke-5 sequences with SOS and padding.

    Each item returns:
        data: (T+1, 5) — SOS + sequence (padded to max_len+1 in collate, or here)
        length: number of real stroke steps (without SOS)
    """

    def __init__(self, sequences: np.ndarray | list[np.ndarray], max_len: int = 250):
        self.sequences = [np.asarray(s, dtype=np.float32) for s in sequences]
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq = self.sequences[idx]
        if len(seq) > self.max_len:
            seq = seq[: self.max_len]
        return torch.from_numpy(seq.copy()), len(seq)


def collate_stroke5(batch: list[tuple[torch.Tensor, int]]) -> dict[str, torch.Tensor]:
    """Pad sequences and prepend SOS (0,0,1,0,0).

    Returns:
        strokes: (B, T+1, 5) where T = max length in batch
        lengths: (B,) true stroke lengths (excluding SOS)
        mask: (B, T) True for valid prediction targets (decoder outputs)
    """
    seqs, lengths = zip(*batch)
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    t_max = int(lengths_t.max().item())
    b = len(seqs)

    # SOS + padded strokes
    strokes = torch.zeros(b, t_max + 1, 5, dtype=torch.float32)
    strokes[:, 0, 2] = 1.0  # SOS pen-down
    strokes[:, 1:, 4] = 1.0  # default pad = EOS

    mask = torch.zeros(b, t_max, dtype=torch.bool)
    for i, (seq, n) in enumerate(zip(seqs, lengths)):
        strokes[i, 1 : n + 1] = seq[:n]
        mask[i, :n] = True

    return {"strokes": strokes, "lengths": lengths_t, "mask": mask}


def load_stroke5_cache(path: Path | str) -> tuple[np.ndarray, np.ndarray, float, int]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    train = data["train"]
    val = data["val"]
    scale = float(data["scale"])
    max_len = int(data["max_len"]) if "max_len" in data.files else 250
    return train, val, scale, max_len


def make_loaders(
    cache_path: Path | str,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, float, int]:
    train_arr, val_arr, scale, max_len = load_stroke5_cache(cache_path)
    train_ds = Stroke5Dataset(train_arr, max_len=max_len)
    val_ds = Stroke5Dataset(val_arr, max_len=max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_stroke5,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_stroke5,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, scale, max_len
