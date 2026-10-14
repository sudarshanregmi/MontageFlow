"""Latent prior p(h | y): a small rectified-flow model fitted to encoded training windows.

Its own convention is z_t = (1 - t) h + t e (t=0 data, t=1 noise); samples are drawn by
integrating from t=1 to t=0 with Euler steps.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0) * (2.0 * math.pi)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.cond = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, h, c):
        return h + self.fc2(F.silu(self.fc1(self.norm(h)) + self.cond(c)))


class VelocityNet(nn.Module):
    def __init__(self, dim: int, num_classes: int = 0, width: int = 1024, n_blocks: int = 5,
                 time_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        if num_classes > 0:
            self.y_embed = nn.Embedding(num_classes, width)
            nn.init.zeros_(self.y_embed.weight)
        self.t_embed = TimeEmbedding(time_dim)
        self.t_mlp = nn.Sequential(nn.Linear(time_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.inp = nn.Linear(dim, width)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, t, y=None):
        c = self.t_mlp(self.t_embed(t))
        if self.num_classes > 0:
            c = c + self.y_embed(y)
        h = self.inp(z)
        for block in self.blocks:
            h = block(h, c)
        return self.out(self.out_norm(h))


class LatentPrior(nn.Module):
    def __init__(self, dim: int, num_classes: int = 0, width: int = 1024, n_blocks: int = 5,
                 time_dim: int = 128, sample_steps: int = 100):
        super().__init__()
        self.config = dict(dim=dim, num_classes=num_classes, width=width, n_blocks=n_blocks,
                           time_dim=time_dim, sample_steps=sample_steps)
        self.net = VelocityNet(dim, num_classes, width, n_blocks, time_dim)
        self.sample_steps = sample_steps

    @property
    def label_conditioned(self) -> bool:
        return self.net.num_classes > 0

    def fit(self, h: torch.Tensor, y: Optional[torch.Tensor] = None, steps: int = 12000,
            batch_size: int = 1024, lr: float = 1e-3, weight_decay: float = 1e-4, seed: int = 0,
            log_every: int = 500) -> "LatentPrior":
        device = next(self.parameters()).device
        h = h.to(device, torch.float32)
        y = None if not self.label_conditioned else torch.as_tensor(y, dtype=torch.long, device=device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        gen = torch.Generator(device=device).manual_seed(seed + 1)
        n, dim = h.shape
        bs = min(batch_size, n)
        self.train()
        running = 0.0
        for step in range(steps):
            idx = torch.randint(0, n, (bs,), device=device, generator=gen)
            h0 = h[idx]
            e = torch.randn(bs, dim, device=device, generator=gen)
            t = torch.sigmoid(torch.randn(bs, device=device, generator=gen))
            z_t = (1.0 - t[:, None]) * h0 + t[:, None] * e
            loss = F.mse_loss(self.net(z_t, t, None if y is None else y[idx]), e - h0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
            sched.step()
            running = loss.item() if step == 0 else 0.98 * running + 0.02 * loss.item()
            if log_every and (step % log_every == 0 or step == steps - 1):
                print(f"prior step {step:6d}/{steps}  loss {running:.4f}", flush=True)
        self.eval()
        return self

    @torch.no_grad()
    def sample(self, n: int, y: Optional[torch.Tensor] = None,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        if self.label_conditioned:
            y = torch.as_tensor(y, dtype=torch.long, device=device)
        z = torch.randn(n, self.config["dim"], generator=generator).to(device)
        dt = 1.0 / self.sample_steps
        for i in range(self.sample_steps):
            t = torch.full((n,), 1.0 - i * dt, device=device)
            z = z - dt * self.net(z, t, y)
        return z
