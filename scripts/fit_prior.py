"""Fit the latent prior of a trained MontageFlow or MontageDiff model.

Encodes every training window with the EMA encoder and fits p(h | y) to the latents.
The prior is saved as prior.pt next to the checkpoint.

    python scripts/fit_prior.py runs/montageflow_tuab/ckpt_0161999.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from montageflow.checkpoint import model_from_training, save_prior  # noqa: E402
from montageflow.config import merge, parse_override  # noqa: E402
from montageflow.data import Windows  # noqa: E402
from montageflow.prior import LatentPrior  # noqa: E402


@torch.no_grad()
def encode_split(model, data, batch_size=128, chunk=8192):
    device = next(model.parameters()).device
    latents = []
    for start in range(0, len(data), chunk):
        x, _ = data.get(np.arange(start, min(start + chunk, len(data))))
        for i in range(0, len(x), batch_size):
            latents.append(model.encode(x[i:i + batch_size].to(device)).float().cpu())
    return torch.cat(latents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None, help="default: prior.pt next to the checkpoint")
    ap.add_argument("overrides", nargs="*", help="e.g. prior.label_conditioned=false")
    args = ap.parse_intermixed_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = model_from_training(args.checkpoint, device=device)
    for item in args.overrides:
        cfg = merge(cfg, parse_override(item))
    pc = cfg["prior"]
    if not model.has_latent:
        raise SystemExit("this model has no latent, so it needs no prior")

    data = Windows(cfg["data"]["train"], scale=cfg["data"]["scale"])
    h = encode_split(model, data)
    print(f"encoded {len(h)} training windows")

    torch.manual_seed(pc["seed"])
    num_classes = data.num_classes if pc["label_conditioned"] else 0
    prior = LatentPrior(h.shape[1], num_classes=num_classes, sample_steps=pc["sample_steps"]).to(device)
    prior.fit(h, torch.from_numpy(data.y) if num_classes else None, steps=pc["steps"], seed=pc["seed"])
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "prior.pt")
    save_prior(out, prior)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
