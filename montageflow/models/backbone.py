"""Adapted from JET (https://github.com/Y-Research-SBU/JET, MIT License,
Copyright (c) 2026 Y-Research @SBU).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def sincos_embedding(dim: int, n: int) -> np.ndarray:
    omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=np.float64) / (dim / 2.0))
    out = np.einsum("m,d->md", np.arange(n, dtype=np.float32).astype(np.float64), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def modulate(x, shift, scale):
    while shift.dim() < x.dim():
        shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.freq_dim = freq_dim

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class LabelEmbedder(nn.Module):
    """One row per class plus a null row, which label dropout trains."""

    def __init__(self, num_classes: int, hidden: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden)

    def forward(self, y):
        return self.embedding_table(y)


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        width = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden, width), nn.GELU(), nn.Dropout(0.0),
                                 nn.Linear(width, hidden), nn.Dropout(0.0))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x, cond):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(cond).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_a, scale_a)
        x = x + gate_a.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        return x + gate_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_m, scale_m))


class FinalLayer(nn.Module):
    def __init__(self, hidden: int, patch: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.proj = nn.Linear(hidden, patch)

    def forward(self, x, cond):
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


class Backbone(nn.Module):
    def __init__(self, channels: int, length: int, patch: int, num_classes: int,
                 hidden: int = 768, depth: int = 12, heads: int = 12, mlp_ratio: float = 4.0,
                 cross_channel: bool = False, latent_dim: int = 0):
        super().__init__()
        if length % patch:
            raise ValueError(f"window length {length} is not a multiple of patch {patch}")
        self.channels, self.patch, self.num_patches = channels, patch, length // patch
        self.hidden = hidden
        self.cross_channel = cross_channel

        self.t_embedder = TimestepEmbedder(hidden)
        self.y_embedder = LabelEmbedder(num_classes, hidden)
        self.patch_embed = nn.Linear(patch, hidden)
        pos = torch.from_numpy(sincos_embedding(hidden, self.num_patches)).float().unsqueeze(0)
        self.pos_embed = nn.Parameter(pos, requires_grad=False)
        self.channel_emb = nn.Embedding(channels, hidden) if cross_channel else None
        self.blocks = nn.ModuleList([Block(hidden, heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden, patch)

        self.apply(self._xavier)                    # JET's init: xavier on every Linear
        if self.channel_emb is not None:
            nn.init.normal_(self.channel_emb.weight, std=0.02)

        # the latent enters the same adaLN vector as time and label; built after the
        # xavier pass so it keeps PyTorch's default init
        if latent_dim > 0:
            self.latent_norm = nn.RMSNorm(latent_dim)
            self.latent_proj = nn.Linear(latent_dim, hidden)
        else:
            self.latent_norm = self.latent_proj = None

    @staticmethod
    def _xavier(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, t, y, h=None):
        """x: (B, C, N, P) patches, t: (B,), y: (B,), h: (B, latent_dim) or None."""
        B, C, N, P = x.shape
        cond = self.t_embedder(t) + self.y_embedder(y)
        if self.latent_proj is not None:
            cond = cond + self.latent_proj(self.latent_norm(h))

        if self.cross_channel:
            tokens = self.patch_embed(x) + self.pos_embed.unsqueeze(1)
            tokens = tokens + self.channel_emb(torch.arange(C, device=x.device)).view(1, C, 1, -1)
            tokens = tokens.reshape(B, C * N, self.hidden)
            for block in self.blocks:
                tokens = block(tokens, cond)
            return self.final_layer(tokens, cond).view(B, C, N, P)

        cond = cond.repeat_interleave(C, dim=0)
        tokens = self.patch_embed(x).view(B * C, N, self.hidden) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens, cond)
        return self.final_layer(tokens, cond).view(B, C, N, P)
