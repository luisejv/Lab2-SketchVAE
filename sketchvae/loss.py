"""Reconstruction (GMM + pen) and KL losses for Sketch-RNN."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def bivariate_normal_pdf(
    dx: torch.Tensor,
    dy: torch.Tensor,
    mu_x: torch.Tensor,
    mu_y: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """Log-prob of bivariate normal for each mixture component.

    All tensors broadcast to (..., M).
    """
    sigma_x = sigma_x.clamp_min(1e-5)
    sigma_y = sigma_y.clamp_min(1e-5)
    rho = rho.clamp(-0.999, 0.999)

    z_x = (dx - mu_x) / sigma_x
    z_y = (dy - mu_y) / sigma_y
    z = z_x**2 + z_y**2 - 2.0 * rho * z_x * z_y
    denom = 2.0 * (1.0 - rho**2)
    norm = 2.0 * math.pi * sigma_x * sigma_y * torch.sqrt((1.0 - rho**2).clamp_min(1e-8))
    return -z / denom - torch.log(norm)


def reconstruction_loss(
    targets: torch.Tensor,
    pi: torch.Tensor,
    mu_x: torch.Tensor,
    mu_y: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    rho: torch.Tensor,
    pen_logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked NLL for offsets + pen state.

    Args:
        targets: (B, T, 5)
        pi: (B, T, M) mixture weights (probabilities)
        mu_*/sigma_*/rho: (B, T, M)
        pen_logits: (B, T, 3)
        mask: (B, T) bool
    """
    dx = targets[..., 0:1]
    dy = targets[..., 1:2]
    pen = targets[..., 2:]

    log_components = bivariate_normal_pdf(dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho)
    log_mix = torch.log(pi.clamp_min(1e-8)) + log_components
    stroke_nll = -torch.logsumexp(log_mix, dim=-1)  # (B, T)

    pen_nll = F.cross_entropy(
        pen_logits.reshape(-1, 3),
        pen.reshape(-1, 3).argmax(dim=-1),
        reduction="none",
    ).reshape(pen_logits.shape[:2])

    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    stroke_loss = (stroke_nll * mask_f).sum() / denom
    pen_loss = (pen_nll * mask_f).sum() / denom
    return stroke_loss + pen_loss, stroke_loss, pen_loss


def kl_divergence(mu: torch.Tensor, sigma_hat: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0,I)) averaged over batch.

    sigma = exp(sigma_hat / 2) as in Sketch-RNN.
    """
    sigma = torch.exp(sigma_hat / 2.0)
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # log(sigma^2) = sigma_hat
    kl = -0.5 * (1.0 + sigma_hat - mu.pow(2) - sigma.pow(2)).sum(dim=1)
    return kl.mean()


def free_bits_kl(mu: torch.Tensor, sigma_hat: torch.Tensor, free_bits: float = 0.2) -> torch.Tensor:
    """Per-dimension free-bits KL, then sum and batch-mean (Sketch-RNN style)."""
    sigma = torch.exp(sigma_hat / 2.0)
    kl_dim = -0.5 * (1.0 + sigma_hat - mu.pow(2) - sigma.pow(2))  # (B, Nz)
    kl_dim = torch.clamp(kl_dim, min=free_bits)
    return kl_dim.sum(dim=1).mean()
