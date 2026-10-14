from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float = 8.0 / 3.0):
        super().__init__()
        hidden = max(128, int(round(mult * dim / 128.0)) * 128)
        self.w_in = nn.Linear(dim, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        gate, value = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(gate) * value)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        q, k, v = self.qkv(x).reshape(B, S, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, S, D))


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1, self.attn = RMSNorm(dim), Attention(dim, heads)
        self.n2, self.mlp = RMSNorm(dim), SwiGLU(dim)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class Encoder(nn.Module):
    def __init__(self, channels: int, length: int, patch: int, latent_dim: int = 256,
                 width: int = 512, depth: int = 8, heads: int = 8):
        super().__init__()
        if length % patch:
            raise ValueError(f"window length {length} is not a multiple of encoder patch {patch}")
        self.patch, self.num_patches = patch, length // patch
        self.patch_embed = nn.Linear(patch, width)
        self.channel_emb = nn.Embedding(channels, width)
        self.time_emb = nn.Parameter(0.02 * torch.randn(1, self.num_patches, width))
        self.reg = nn.Parameter(0.02 * torch.randn(1, 1, width))
        self.blocks = nn.ModuleList([EncoderBlock(width, heads) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.out = nn.Linear(width, latent_dim, bias=False)
        self.out_norm = RMSNorm(latent_dim)

    def forward(self, x):
        """x: (B, C, L) -> h: (B, latent_dim)."""
        B, C, L = x.shape
        width = self.reg.shape[-1]
        tokens = self.patch_embed(x.reshape(B, C, self.num_patches, self.patch))
        tokens = tokens + self.time_emb.unsqueeze(1)
        tokens = tokens + self.channel_emb(torch.arange(C, device=x.device)).view(1, C, 1, width)
        tokens = torch.cat([tokens.reshape(B, C * self.num_patches, width), self.reg.expand(B, 1, width)], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.out_norm(self.out(self.norm(tokens[:, -1])))
