"""Batched generation for any of the models, with the latent prior when the model has one."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


@torch.no_grad()
def generate(model, labels, prior=None, steps: int = 50, batch_size: int = 1000,
             project: Optional[bool] = None, generator: Optional[torch.Generator] = None) -> np.ndarray:
    """Generate one window per label. Returns (N, C, L) float32 in model scale.

    `project` is passed to model.sample (None keeps the model's own setting).
    Halves the batch size on CUDA out-of-memory.
    """
    device = next(model.parameters()).device
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    if getattr(model, "has_latent", False) and prior is None:
        raise ValueError("this model needs a latent prior; run scripts/fit_prior.py first")
    out, i = [], 0
    while i < len(labels):
        y = labels[i:i + batch_size].to(device)
        try:
            if getattr(model, "has_latent", False):
                h = prior.sample(len(y), y, generator=generator)
                x = model.sample(y, h, steps=steps, project=project)
            elif project is not None:
                x = model.sample(y, steps=steps, project=project)
            else:
                x = model.sample(y, steps=steps)
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 16:
                raise
            torch.cuda.empty_cache()
            batch_size //= 2
            continue
        out.append(x.float().cpu().numpy())
        i += len(y)
    return np.concatenate(out)
