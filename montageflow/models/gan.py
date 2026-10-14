from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GANConfig:
    channels: int = 16
    length: int = 1000
    num_classes: int = 2
    z_dim: int = 200
    width: int = 1024
    n_blocks: int = 4
    gp_weight: float = 10.0
    n_critic: int = 5
    critic_lr: float = 1e-4


class UpBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 9, padding=4)
        self.c2 = nn.Conv1d(cout, cout, 9, padding=4)
        self.n = nn.GroupNorm(8, cout)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        return F.leaky_relu(self.n(self.c2(F.leaky_relu(self.c1(x), 0.2))), 0.2)


class EEGGAN(nn.Module):
    def __init__(self, cfg: GANConfig):
        super().__init__()
        self.cfg = cfg
        self.channels, self.length = cfg.channels, cfg.length
        self.num_classes = cfg.num_classes
        self.base = max(4, cfg.length // (2 ** cfg.n_blocks))
        w = cfg.width
        # generator
        self.y_emb = nn.Embedding(cfg.num_classes + 1, cfg.z_dim)
        self.fc = nn.Linear(cfg.z_dim, w * self.base)
        self.blocks = nn.ModuleList([UpBlock(w, w) for _ in range(cfg.n_blocks)])
        self.out = nn.Conv1d(w, cfg.channels, 9, padding=4)
        # critic
        layers, cin = [], cfg.channels
        for _ in range(cfg.n_blocks):
            layers += [nn.Conv1d(cin, w, 9, stride=2, padding=4), nn.LeakyReLU(0.2)]
            cin = w
        self.disc_conv = nn.Sequential(*layers)
        self.disc_head = nn.Linear(w, 1)
        self.disc_y = nn.Embedding(cfg.num_classes + 1, w)
        self._critic_opt = None

    has_latent = False

    def generator(self, z, y):
        h = self.fc(z + self.y_emb(y)).view(z.shape[0], -1, self.base)
        for block in self.blocks:
            h = block(h)
        x = self.out(h)
        if x.shape[-1] != self.length:
            x = F.interpolate(x, size=self.length, mode="linear", align_corners=False)
        return x

    def critic(self, x, y):
        h = self.disc_conv(x).mean(-1)
        return self.disc_head(h).squeeze(-1) + (h * self.disc_y(y)).sum(-1)

    def critic_parameters(self):
        return [*self.disc_conv.parameters(), *self.disc_head.parameters(), *self.disc_y.parameters()]

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Runs n_critic critic updates with the critic's own optimizer, then returns the
        generator loss (the critic is frozen while it is computed, so the outer optimizer
        only moves the generator)."""
        cfg = self.cfg
        B, device = x.shape[0], x.device
        if self._critic_opt is None:
            self._critic_opt = torch.optim.Adam(self.critic_parameters(), lr=cfg.critic_lr, betas=(0.0, 0.9))

        for _ in range(cfg.n_critic):
            with torch.no_grad():
                fake = self.generator(torch.randn(B, cfg.z_dim, device=device), y)
            eps = torch.rand(B, 1, 1, device=device)
            mix = (eps * x + (1 - eps) * fake).requires_grad_(True)
            grad = torch.autograd.grad(self.critic(mix, y).sum(), mix, create_graph=True)[0]
            gp = ((grad.reshape(B, -1).norm(2, dim=1) - 1) ** 2).mean()
            d_loss = self.critic(fake, y).mean() - self.critic(x, y).mean() + cfg.gp_weight * gp
            self._critic_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            self._critic_opt.step()
        # the outer optimizer also holds the critic; without gradients it skips it
        self._critic_opt.zero_grad(set_to_none=True)

        for p in self.critic_parameters():
            p.requires_grad_(False)
        g_loss = -self.critic(self.generator(torch.randn(B, cfg.z_dim, device=device), y), y).mean()
        for p in self.critic_parameters():
            p.requires_grad_(True)
        return g_loss

    def forward(self, x, y):
        return self.loss(x, y)

    @torch.no_grad()
    def sample(self, labels: torch.Tensor, steps: int = 0) -> torch.Tensor:
        return self.generator(torch.randn(labels.shape[0], self.cfg.z_dim, device=labels.device), labels)
