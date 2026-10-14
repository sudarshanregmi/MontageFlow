"""Measure the cost of a model: parameters, peak training memory, seconds per optimizer
step (32 x 8 = 256 windows), and generation latency per window (batch of 32, 50 steps,
including the latent prior for MontageFlow).

    python scripts/benchmark.py release/tusz --split data/tusz/train
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from montageflow.checkpoint import load, load_prior  # noqa: E402
from montageflow.data import Windows  # noqa: E402
from montageflow.generate import generate  # noqa: E402


def timed(fn, repeats=5, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--split", required=True)
    ap.add_argument("--prior", default=None)
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    model, prior, cfg = load(args.model, device=device)
    if args.prior:
        prior = load_prior(args.prior, device)
    data = Windows(args.split, scale=cfg["data"]["scale"])
    x, y = data.get(np.arange(args.micro_batch))
    x, y = x.to(device), y.to(device)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    torch.cuda.reset_peak_memory_stats()

    def train_step():
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            (model(x, y) / args.accum).backward()
        opt.step()

    step_s = timed(train_step)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    model.eval()

    labels = np.zeros(args.gen_batch, dtype=np.int64)
    gen_s = timed(lambda: generate(model, labels, prior=prior, steps=args.steps, batch_size=args.gen_batch),
                  repeats=3, warmup=1)
    result = {"device": torch.cuda.get_device_name(0),
              "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
              "peak_train_memory_gb": peak_gb,
              "seconds_per_step": step_s,
              "batch": args.micro_batch * args.accum,
              "ms_per_window": 1000 * gen_s / args.gen_batch}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
