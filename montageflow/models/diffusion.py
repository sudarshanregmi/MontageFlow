from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import montage
from .backbone import Backbone
from .encoder import Encoder


@dataclass
class DiffusionConfig:
    channels: int = 16
    length: int = 1000
    num_classes: int = 2
    patch: int = 100
    hidden: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    cross_channel: bool = True
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    loss: str = "l1"                      # "l1" or "l2" on the predicted noise
    label_dropout: float = 0.1
    projection: bool = False              # project the noise prediction and initial noise onto ker A
    latent_dim: int = 0                   # 0 disables latent conditioning
    encoder_width: int = 512
    encoder_depth: int = 8
    encoder_heads: int = 8
    encoder_patch: Optional[int] = None   # defaults to `patch`
    latent_noise: float = 1.0             # h + sigma * eta with sigma ~ U(0, latent_noise), training only


class DiffusionModel(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.channels, self.length, self.patch = cfg.channels, cfg.length, cfg.patch
        self.num_patches = cfg.length // cfg.patch
        self.num_classes = cfg.num_classes
        self.net = Backbone(cfg.channels, cfg.length, cfg.patch, cfg.num_classes,
                            hidden=cfg.hidden, depth=cfg.depth, heads=cfg.heads, mlp_ratio=cfg.mlp_ratio,
                            cross_channel=cfg.cross_channel, latent_dim=cfg.latent_dim)
        self.encoder = None
        if cfg.latent_dim > 0:
            self.encoder = Encoder(cfg.channels, cfg.length, cfg.encoder_patch or cfg.patch,
                                   latent_dim=cfg.latent_dim, width=cfg.encoder_width,
                                   depth=cfg.encoder_depth, heads=cfg.encoder_heads)
        if cfg.projection:
            self.register_buffer("P", torch.from_numpy(montage.projector()).float(), persistent=False)
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - betas, dim=0), persistent=False)

    @property
    def has_latent(self) -> bool:
        return self.cfg.latent_dim > 0

    def _encoder(self):
        if self.encoder is None:
            raise RuntimeError("this model was loaded from a release directory, which has no encoder weights; "
                               "training and fitting a prior need the training checkpoint")
        return self.encoder

    def _patches(self, x):
        return x.reshape(x.shape[0], self.channels, self.num_patches, self.patch)

    def _project(self, x):
        return montage.project(x, self.P)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Latents of clean windows, (B, C, L) -> (B, latent_dim)."""
        return self._encoder()(x)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        h = None
        if self.has_latent:
            h = self._encoder()(x)
            if self.training and cfg.latent_noise > 0:
                sigma = float(torch.rand(1, device=x.device).item()) * cfg.latent_noise
                h = h + sigma * torch.randn_like(h)

        x = self._patches(x)
        B = x.shape[0]
        if self.training:
            drop = torch.rand(B, device=x.device) < cfg.label_dropout
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        t = torch.randint(0, cfg.timesteps, (B,), device=x.device)
        a = self.alpha_bar[t].view(B, 1, 1, 1)
        eps = torch.randn_like(x)
        x_t = a.sqrt() * x + (1 - a).sqrt() * eps
        eps_hat = self.net(x_t, t.float() / cfg.timesteps, y, h)
        if cfg.projection:
            eps_hat = self._project(eps_hat)
        return F.mse_loss(eps_hat, eps) if cfg.loss == "l2" else (eps_hat - eps).abs().mean()

    def forward(self, x, y):
        return self.loss(x, y)

    @torch.no_grad()
    def sample(self, labels: torch.Tensor, h: Optional[torch.Tensor] = None, steps: int = 50,
               project: Optional[bool] = None) -> torch.Tensor:
        """Generate windows for the given labels (and latents, if the model uses them).

        `project` overrides the configured projection at sampling time.
        """
        cfg = self.cfg
        if self.has_latent != (h is not None):
            raise ValueError("pass latents h exactly when the model uses them")
        project = cfg.projection if project is None else project
        if project and not hasattr(self, "P"):
            self.register_buffer("P", torch.from_numpy(montage.projector()).float().to(labels.device),
                                 persistent=False)
        B, device = labels.shape[0], labels.device
        ts = torch.linspace(cfg.timesteps - 1, 0, steps, device=device).long()
        x = torch.randn(B, self.channels, self.num_patches, self.patch, device=device)
        if project:
            x = self._project(x)
        for i, t in enumerate(ts):
            a_t = self.alpha_bar[t]
            a_prev = self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            eps = self.net(x, torch.full((B,), float(t) / cfg.timesteps, device=device), labels, h)
            if project:
                eps = self._project(eps)
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-8)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x.reshape(B, self.channels, self.length)
