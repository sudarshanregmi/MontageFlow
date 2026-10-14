"""Generate EEG windows and save them as .npy (model scale, i.e. microvolts / 100).

Labels are taken from n random windows of a split, so the generated set has the split's
class mix. With --real, the matching real windows are saved too (for figures).

    python scripts/sample.py sudarshanregmi/MontageFlow/tuab --split data/tuab/train --n 2000 \
        --out samples/tuab_montageflow.npy --real samples/tuab_real.npy
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from montageflow.checkpoint import load, load_prior  # noqa: E402
from montageflow.data import Windows  # noqa: E402
from montageflow.generate import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="release dir, Hugging Face path, or training checkpoint")
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real", default=None, help="also save the real windows whose labels were used")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--prior", default=None)
    ap.add_argument("--project", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, prior, cfg = load(args.model, device=device)
    if args.prior:
        prior = load_prior(args.prior, device)
    data = Windows(args.split, scale=cfg["data"]["scale"])
    idx = np.sort(np.random.RandomState(args.seed).choice(len(data), size=min(args.n, len(data)), replace=False))

    torch.manual_seed(args.seed)
    x = generate(model, data.y[idx], prior=prior, steps=args.steps or cfg["sampling"]["steps"],
                 project=True if args.project else None,
                 generator=torch.Generator().manual_seed(10_000 + args.seed))
    out = args.out if args.out.endswith(".npy") else args.out + ".npy"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.save(out, x.astype(np.float32))
    np.save(out[:-4] + "_labels.npy", data.y[idx])
    if args.real:
        np.save(args.real, data.get(idx)[0].numpy())
    print(f"saved {x.shape} to {out}")


if __name__ == "__main__":
    main()
