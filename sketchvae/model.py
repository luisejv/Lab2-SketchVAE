"""Sketch-RNN sequence-to-sequence VAE."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import free_bits_kl, reconstruction_loss


@dataclass
class SketchRNNConfig:
    enc_hidden: int = 256
    dec_hidden: int = 512
    z_size: int = 64
    n_mixtures: int = 20
    dropout: float = 0.0


class Encoder(nn.Module):
    def __init__(self, cfg: SketchRNNConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=cfg.enc_hidden,
            bidirectional=True,
            batch_first=True,
        )
        self.mu = nn.Linear(2 * cfg.enc_hidden, cfg.z_size)
        self.sigma_hat = nn.Linear(2 * cfg.enc_hidden, cfg.z_size)

    def forward(self, strokes: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode stroke sequence (without SOS).

        Args:
            strokes: (B, T, 5)
            lengths: (B,)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            strokes, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (2, B, H) -> (B, 2H)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        mu = self.mu(h)
        sigma_hat = self.sigma_hat(h)
        sigma = torch.exp(sigma_hat / 2.0)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return z, mu, sigma_hat


class Decoder(nn.Module):
    def __init__(self, cfg: SketchRNNConfig):
        super().__init__()
        self.cfg = cfg
        self.z_to_hidden = nn.Linear(cfg.z_size, cfg.dec_hidden * 2)
        self.lstm = nn.LSTM(
            input_size=5 + cfg.z_size,
            hidden_size=cfg.dec_hidden,
            batch_first=True,
        )
        # 6M + 3 outputs
        self.out = nn.Linear(cfg.dec_hidden, 6 * cfg.n_mixtures + 3)

    def init_state(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_c = torch.tanh(self.z_to_hidden(z))
        h_0, c_0 = torch.split(h_c, self.cfg.dec_hidden, dim=-1)
        # (1, B, H)
        return h_0.unsqueeze(0), c_0.unsqueeze(0)

    def forward(
        self,
        inputs: torch.Tensor,
        z: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            inputs: (B, T, 5) teacher-forced previous points
            z: (B, Nz)
        """
        if state is None:
            state = self.init_state(z)
        z_exp = z.unsqueeze(1).expand(-1, inputs.size(1), -1)
        x = torch.cat([inputs, z_exp], dim=-1)
        out, state = self.lstm(x, state)
        y = self.out(out)
        params = split_mdn_params(y, self.cfg.n_mixtures)
        return params, state


def split_mdn_params(y: torch.Tensor, n_mixtures: int) -> dict[str, torch.Tensor]:
    """Split decoder logits into MDN + pen parameters."""
    m = n_mixtures
    pen_logits = y[..., :3]
    rest = y[..., 3:]
    pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho = torch.split(rest, m, dim=-1)
    pi = F.softmax(pi_logits, dim=-1)
    sigma_x = torch.exp(sigma_x)
    sigma_y = torch.exp(sigma_y)
    rho = torch.tanh(rho)
    return {
        "pi": pi,
        "pi_logits": pi_logits,
        "mu_x": mu_x,
        "mu_y": mu_y,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "rho": rho,
        "pen_logits": pen_logits,
        "pen": F.softmax(pen_logits, dim=-1),
    }


class SketchRNN(nn.Module):
    def __init__(self, cfg: SketchRNNConfig | None = None):
        super().__init__()
        self.cfg = cfg or SketchRNNConfig()
        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg)

    def forward(
        self,
        strokes: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        *,
        free_bits: float = 0.2,
        kl_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            strokes: (B, T+1, 5) with SOS at index 0
            lengths: (B,) stroke lengths excluding SOS
            mask: (B, T) valid target steps
        """
        # Encode ground-truth strokes (no SOS)
        enc_in = strokes[:, 1:, :]
        z, mu, sigma_hat = self.encoder(enc_in, lengths)

        # Decode: input SOS..x_{T-1} -> predict x_1..x_T
        dec_in = strokes[:, :-1, :]
        params, _ = self.decoder(dec_in, z)
        targets = strokes[:, 1:, :]

        recon, stroke_loss, pen_loss = reconstruction_loss(
            targets,
            params["pi"],
            params["mu_x"],
            params["mu_y"],
            params["sigma_x"],
            params["sigma_y"],
            params["rho"],
            params["pen_logits"],
            mask,
        )
        kl = free_bits_kl(mu, sigma_hat, free_bits=free_bits)
        loss = recon + kl_weight * kl
        return {
            "loss": loss,
            "recon": recon,
            "stroke_loss": stroke_loss,
            "pen_loss": pen_loss,
            "kl": kl,
            "z": z,
            "mu": mu,
            "sigma_hat": sigma_hat,
        }


@torch.no_grad()
def sample_mdn(
    params: dict[str, torch.Tensor],
    *,
    temperature: float = 1.0,
    greedy: bool = False,
) -> torch.Tensor:
    """Sample one stroke-5 step from MDN params at a single time index.

    params values shaped (B, M) / (B, 3). Returns (B, 5).
    If greedy=True, take the most likely mixture mean and pen state (no noise).
    """
    pi_logits = params["pi_logits"] / max(temperature, 1e-6)
    pi = F.softmax(pi_logits, dim=-1)
    if greedy:
        idx = pi.argmax(dim=-1)
    else:
        mix = torch.distributions.Categorical(probs=pi)
        idx = mix.sample()  # (B,)

    def gather(t: torch.Tensor) -> torch.Tensor:
        return t.gather(1, idx.unsqueeze(1)).squeeze(1)

    mu_x = gather(params["mu_x"])
    mu_y = gather(params["mu_y"])

    if greedy:
        dx, dy = mu_x, mu_y
    else:
        sigma_x = gather(params["sigma_x"]) * math.sqrt(max(temperature, 1e-6))
        sigma_y = gather(params["sigma_y"]) * math.sqrt(max(temperature, 1e-6))
        rho = gather(params["rho"]).clamp(-0.999, 0.999)
        eps = torch.randn(mu_x.shape[0], 2, device=mu_x.device, dtype=mu_x.dtype)
        a = sigma_x
        b = rho * sigma_y
        c = sigma_y * torch.sqrt((1.0 - rho**2).clamp_min(1e-8))
        dx = mu_x + a * eps[:, 0]
        dy = mu_y + b * eps[:, 0] + c * eps[:, 1]

    pen_logits = params["pen_logits"] / max(temperature, 1e-6)
    if greedy:
        pen = pen_logits.argmax(dim=-1)
    else:
        pen = torch.distributions.Categorical(logits=pen_logits).sample()
    pen_oh = F.one_hot(pen, num_classes=3).float()
    return torch.cat([dx.unsqueeze(1), dy.unsqueeze(1), pen_oh], dim=-1)


@torch.no_grad()
def generate(
    model: SketchRNN,
    *,
    n: int = 1,
    max_len: int = 200,
    temperature: float = 0.6,
    z: torch.Tensor | None = None,
    device: torch.device | None = None,
    greedy: bool = False,
) -> torch.Tensor:
    """Unconditional generation from prior (or provided z). Returns (n, T, 5)."""
    model.eval()
    device = device or next(model.parameters()).device
    cfg = model.cfg
    if z is None:
        z = torch.randn(n, cfg.z_size, device=device)
    else:
        z = z.to(device)
        n = z.size(0)

    state = model.decoder.init_state(z)
    prev = torch.zeros(n, 1, 5, device=device)
    prev[:, 0, 2] = 1.0  # SOS
    steps: list[torch.Tensor] = []

    for _ in range(max_len):
        params, state = model.decoder(prev, z, state)
        step_params = {k: v[:, 0] for k, v in params.items()}
        stroke = sample_mdn(step_params, temperature=temperature, greedy=greedy)
        steps.append(stroke)
        prev = stroke.unsqueeze(1)
        if bool((stroke[:, 4] > 0.5).all()):
            break

    return torch.stack(steps, dim=1)


@torch.no_grad()
def reconstruct(
    model: SketchRNN,
    sequence: torch.Tensor,
    *,
    device: torch.device | None = None,
    greedy: bool = True,
    temperature: float = 0.3,
    use_mean_z: bool = True,
) -> torch.Tensor:
    """Encode a stroke-5 sequence and decode it (for reconstruction plots).

    Args:
        sequence: (T, 5) normalized stroke-5 without SOS.
    Returns:
        (T_out, 5) reconstructed sequence.
    """
    model.eval()
    device = device or next(model.parameters()).device
    sequence = sequence.to(device)
    if sequence.dim() != 2:
        raise ValueError(f"expected (T,5), got {tuple(sequence.shape)}")
    strokes = torch.zeros(1, sequence.size(0) + 1, 5, device=device)
    strokes[0, 0, 2] = 1.0
    strokes[0, 1:] = sequence
    lengths = torch.tensor([sequence.size(0)], device=device)
    z, mu, _ = model.encoder(strokes[:, 1:], lengths)
    if use_mean_z:
        z = mu
    out = generate(
        model,
        n=1,
        max_len=sequence.size(0) + 40,
        temperature=temperature,
        z=z,
        device=device,
        greedy=greedy,
    )
    return out[0]


@torch.no_grad()
def complete(
    model: SketchRNN,
    prefix: torch.Tensor,
    *,
    n_samples: int = 4,
    max_len: int = 200,
    temperature: float = 0.6,
    z: torch.Tensor | None = None,
    device: torch.device | None = None,
    from_prefix_posterior: bool = True,
) -> torch.Tensor:
    """Complete a prefix with different latent samples.

    Args:
        prefix: (T_p, 5) stroke-5 steps (no SOS), normalized like training data.
        n_samples: number of completions (different z)
        from_prefix_posterior: if True and z is None, sample z ~ q(z|prefix)
            instead of the prior (better conditioned completions).
    Returns:
        (n_samples, T_total, 5) including the prefix points.
    """
    model.eval()
    device = device or next(model.parameters()).device
    cfg = model.cfg
    prefix = prefix.to(device)
    if prefix.dim() != 2 or prefix.size(-1) != 5:
        raise ValueError(f"prefix must be (T,5), got {tuple(prefix.shape)}")

    if z is None:
        if from_prefix_posterior and prefix.size(0) > 0:
            # Encode prefix once, then draw n_samples from q(z|prefix)
            pref_b = prefix.unsqueeze(0)
            lengths = torch.tensor([prefix.size(0)], device=device)
            _, mu, sigma_hat = model.encoder(pref_b, lengths)
            sigma = torch.exp(sigma_hat / 2.0)
            eps = torch.randn(n_samples, cfg.z_size, device=device)
            z = mu + sigma * eps
        else:
            z = torch.randn(n_samples, cfg.z_size, device=device)
    else:
        z = z.to(device)
        n_samples = z.size(0)

    sos = torch.zeros(n_samples, 1, 5, device=device)
    sos[:, 0, 2] = 1.0
    pref = prefix.unsqueeze(0).expand(n_samples, -1, -1)
    dec_in = torch.cat([sos, pref[:, :-1, :]], dim=1) if prefix.size(0) > 0 else sos

    state = model.decoder.init_state(z)
    if prefix.size(0) > 0:
        _, state = model.decoder(dec_in, z, state)
        prev = pref[:, -1:, :]
        generated = [pref]
    else:
        prev = sos
        generated = []

    remaining = max(1, max_len - prefix.size(0))
    free_steps: list[torch.Tensor] = []
    for _ in range(remaining):
        params, state = model.decoder(prev, z, state)
        step_params = {k: v[:, 0] for k, v in params.items()}
        stroke = sample_mdn(step_params, temperature=temperature)
        free_steps.append(stroke.unsqueeze(1))
        prev = stroke.unsqueeze(1)
        if bool((stroke[:, 4] > 0.5).all()):
            break

    if free_steps:
        generated.append(torch.cat(free_steps, dim=1))
    return torch.cat(generated, dim=1) if generated else pref
