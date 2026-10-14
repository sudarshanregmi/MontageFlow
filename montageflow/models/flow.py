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
class FlowConfig:
    channels: int = 16
    length: int = 1000
    num_classes: int = 2
    patch: int = 100
    hidden: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    cross_channel: bool = True
    prediction: str = "velocity"          # "velocity" or "x" (clean-signal prediction, as in JET)
    loss: str = "l1"                      # "l1" or "l2"
    projection: bool = True               # project network output and initial noise onto ker A
    project_train_noise: bool = False     # also project the noise used to build x_t in training
    latent_dim: int = 256                 # 0 disables latent conditioning
    encoder_width: int = 512
    encoder_depth: int = 8
    encoder_heads: int = 8
    encoder_patch: Optional[int] = None   # defaults to `patch`
    latent_noise: float = 1.0             # h + sigma * eta with sigma ~ U(0, latent_noise), training only
    label_dropout: float = 0.1
    time_mean: float = -0.8               # t = sigmoid(N(time_mean, time_std^2))
    time_std: float = 0.8
    t_eps: float = 0.05                   # clamp for 1 - t when converting x-predictions to velocity
    stat_weight: float = 0.0              # JET's auxiliary losses
    tv_weight: float = 0.0
    corr_weight: float = 0.0


class FlowModel(nn.Module):
    def __init__(self, cfg: FlowConfig):
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
        if self.training:
            drop = torch.rand(y.shape[0], device=y.device) < cfg.label_dropout
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        t = torch.sigmoid(torch.randn(x.shape[0], device=x.device) * cfg.time_std + cfg.time_mean)
        t = t.view(-1, 1, 1, 1)
        e = torch.randn_like(x)
        if cfg.projection and cfg.project_train_noise:
            e = self._project(e)
        x_t = t * x + (1 - t) * e

        out = self.net(x_t, t.flatten(), y, h)
        if cfg.projection:
            out = self._project(out)
        target = x - e if cfg.prediction == "velocity" else x
        return self._objective(target, out)

    def forward(self, x, y):
        return self.loss(x, y)

    def _objective(self, target, pred):
        cfg = self.cfg
        err = pred - target
        loss = err.pow(2).mean() if cfg.loss == "l2" else err.abs().mean()
        # JET's auxiliary terms; zero for everything except the JET baselines
        if cfg.stat_weight > 0:
            loss = loss + cfg.stat_weight * (F.l1_loss(pred.std(dim=-1), target.std(dim=-1))
                                             + F.l1_loss(pred.mean(dim=-1), target.mean(dim=-1)))
        if cfg.tv_weight > 0:
            loss = loss + cfg.tv_weight * (pred[..., 1:] - pred[..., :-1]).abs().mean()
        if cfg.corr_weight > 0:
            p = pred.reshape(pred.shape[0], -1)
            q = target.reshape(target.shape[0], -1)
            p, q = p - p.mean(dim=1, keepdim=True), q - q.mean(dim=1, keepdim=True)
            corr = (q * p).sum(1) / torch.sqrt((q.pow(2).sum(1) + 1e-8) * (p.pow(2).sum(1) + 1e-8))
            loss = loss + cfg.corr_weight * (1.0 - corr.clamp(-1.0, 1.0)).mean()
        return loss

    @torch.no_grad()
    def sample(self, labels: torch.Tensor, h: Optional[torch.Tensor] = None, steps: int = 50,
               project: Optional[bool] = None) -> torch.Tensor:
        """Generate windows for the given labels (and latents, if the model uses them).

        `project` overrides the configured projection at sampling time; it is how the
        "projection at inference only" ablation is run on a model trained without it.
        """
        cfg = self.cfg
        if self.has_latent != (h is not None):
            raise ValueError("pass latents h exactly when the model uses them")
        project = cfg.projection if project is None else project
        if project and not hasattr(self, "P"):
            self.register_buffer("P", torch.from_numpy(montage.projector()).float().to(labels.device),
                                 persistent=False)
        B, device = labels.shape[0], labels.device

        x = torch.randn((B, self.channels, self.num_patches, self.patch), device=device)
        if project:
            x = self._project(x)
        grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
        for i in range(steps):
            t, t_next = float(grid[i]), float(grid[i + 1])
            out = self.net(x, torch.full((B,), t, device=device), labels, h)
            if project:
                out = self._project(out)
            v = out if cfg.prediction == "velocity" else (out - x) / max(cfg.t_eps, 1.0 - t)
            x = x + (t_next - t) * v
        return x.reshape(B, self.channels, self.length)
