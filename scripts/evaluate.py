"""Evaluate a model against an evaluation split.

By default the whole split is used: one window is generated per real window, with that
window's label, and every metric compares the full generated set with the full split.
`--n` evaluates on a random subset of n real windows instead.

    python scripts/evaluate.py release/tuab --split data/tuab/test --out results/montageflow_tuab_test.json
    python scripts/evaluate.py runs/ablation_row5_tuab/ckpt_0161999.pt --split data/tuab/test --project ...

Reports TS-FID, montage index, shape correlation and the five physiological distances.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from montageflow import montage  # noqa: E402
from montageflow.checkpoint import load, load_prior  # noqa: E402
from montageflow.data import Windows  # noqa: E402
from montageflow.generate import generate  # noqa: E402
from montageflow.metrics import Moments, frechet, physiology, shape, ts_fid  # noqa: E402


class Accumulator:
    def __init__(self):
        self.moments = Moments()
        self.corr = 0.0
        self.physio, self.residuals = [], []
        self.n = 0

    def update(self, x):
        self.moments.update(ts_fid.features(x))
        self.corr = self.corr + shape.correlation_sum(x)
        self.physio.append(physiology.window_features(x))
        self.residuals.append(montage.montage_residuals(x))
        self.n += len(x)

    def finish(self):
        self.physio = np.concatenate(self.physio)
        self.residuals = np.concatenate(self.residuals)
        return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="release dir, Hugging Face path, or training checkpoint")
    ap.add_argument("--split", required=True, help="evaluation split, e.g. data/tuab/test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="random subset size (0 = whole split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="sampling steps (default: from config)")
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--prior", default=None, help="prior.pt for a training checkpoint")
    ap.add_argument("--project", action="store_true", help="project onto ker A at sampling time")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, prior, cfg = load(args.model, device=device)
    if args.prior:
        prior = load_prior(args.prior, device)
    steps = args.steps or cfg["sampling"]["steps"]
    data = Windows(args.split, scale=cfg["data"]["scale"])

    order = np.random.RandomState(args.seed).permutation(len(data))
    if args.n:
        order = order[:args.n]
    labels = data.y[order]

    torch.manual_seed(args.seed)
    prior_gen = torch.Generator().manual_seed(10_000 + args.seed)
    real, gen = Accumulator(), Accumulator()
    t0 = time.time()
    for start in range(0, len(order), args.chunk):
        idx = order[start:start + args.chunk]
        x, _ = data.get(idx)
        real.update(x.numpy())
        fake = generate(model, labels[start:start + args.chunk], prior=prior, steps=steps,
                        batch_size=args.batch_size, project=True if args.project else None,
                        generator=prior_gen)
        gen.update(fake)
        done = start + len(idx)
        rate = done / (time.time() - t0)
        print(f"{done}/{len(order)}  {rate:.1f} windows/s", flush=True)
    real.finish()
    gen.finish()

    result = {
        "model": args.model, "split": args.split, "n": int(len(order)), "seed": args.seed,
        "steps": steps, "project_at_sampling": args.project,
        "ts_fid": frechet(gen.moments, real.moments),
        "montage_index": float(gen.residuals.mean() / montage.INDEPENDENT_RESIDUAL),
        "shape_correlation": shape.shape_correlation(gen.corr / gen.n, real.corr / real.n),
        "physiology": physiology.distances(gen.physio, real.physio),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
