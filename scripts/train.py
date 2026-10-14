"""Train a model from a YAML config.

    python scripts/train.py configs/montageflow.yaml --data configs/data/tuab.yaml --out runs/montageflow_tuab
    torchrun --nproc_per_node 4 scripts/train.py ...          # multi-GPU

Extra `key=value` arguments override the config, e.g. `train.steps=1000`.
Resumes from the newest checkpoint in --out if there is one.
"""
import argparse
import copy
import glob
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from montageflow.checkpoint import save_training  # noqa: E402
from montageflow.config import load_config  # noqa: E402
from montageflow.data import BatchSampler, Windows  # noqa: E402
from montageflow.models import build_model  # noqa: E402


def setup_distributed():
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local)
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def latest_checkpoint(out):
    paths = glob.glob(os.path.join(out, "ckpt_*.pt"))
    return max(paths, key=lambda p: int(re.search(r"ckpt_(\d+)", p).group(1))) if paths else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data", required=True, help="dataset config, e.g. configs/data/tuab.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_intermixed_args()

    cfg = load_config(args.config, args.data, args.overrides)
    rank, world, device = setup_distributed()
    main_process = rank == 0
    tc = cfg["train"]

    data = Windows(cfg["data"]["train"], scale=cfg["data"]["scale"])
    C, L = data.shape
    cfg["model"].update(channels=C, length=L, num_classes=data.num_classes)

    torch.manual_seed(tc["seed"])
    model = build_model(cfg["model"]).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=(0.9, 0.95), weight_decay=tc["weight_decay"])

    steps = tc["steps"]
    warmup = round(tc["warmup_epochs"] * math.ceil(len(data) / tc["batch_size"]))
    if tc["batch_size"] % (tc["micro_batch"] * world):
        raise SystemExit(f"batch_size {tc['batch_size']} is not a multiple of micro_batch x GPUs "
                         f"({tc['micro_batch']} x {world}); change train.micro_batch")
    accum = tc["batch_size"] // (tc["micro_batch"] * world)

    start = 0
    ckpt_path = latest_checkpoint(args.out)
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck["config"]["model"] != cfg["model"]:
            raise SystemExit(f"{ckpt_path} was trained with a different model config; use a new --out")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        if ck["optimizer"] is not None:
            opt.load_state_dict(ck["optimizer"])
        start = ck["step"] + 1
        if main_process:
            print(f"resuming from {ckpt_path}")

    # each GPU draws its own batches and noise, and a resumed run does not replay old ones
    stream = int(np.random.SeedSequence([tc["seed"], rank, start]).generate_state(1)[0])
    torch.manual_seed(stream)
    sampler = BatchSampler(data.y, tc["balance_classes"], seed=stream)

    if world > 1 and cfg["model"]["type"] == "gan":
        raise SystemExit("EEG-GAN updates its critic inside the loss; train it on a single GPU")
    net = DDP(model, device_ids=[device.index]) if world > 1 else model
    if main_process:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{cfg['model']['type']} model, {n_params:.1f}M parameters, {len(data)} windows, "
              f"batch {tc['micro_batch']} x {world} x {accum}", flush=True)

    model.train()
    running, t0 = None, time.time()
    for step in range(start, steps):
        lr = tc["lr"] * min(1.0, step / max(1, warmup))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for micro in range(accum):
            x, y = data.get(sampler(tc["micro_batch"]))
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            sync = net.no_sync() if (world > 1 and micro < accum - 1) else torch.enable_grad()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = net(x, y)
                (loss / accum).backward()
            total += loss.item() / accum
        if tc["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        with torch.no_grad():
            for p_ema, p in zip(ema.parameters(), model.parameters()):
                p_ema.lerp_(p, 1.0 - tc["ema"])

        running = total if running is None else 0.98 * running + 0.02 * total
        if main_process and step % tc["log_every"] == 0:
            print(f"step {step:7d}  loss {running:.5f}  lr {lr:.2e}  "
                  f"{(time.time() - t0) / (step - start + 1):.3f}s/step", flush=True)
        last = step == steps - 1
        if main_process and step > 0 and (step % tc["save_every"] == 0 or last):
            save_training(os.path.join(args.out, f"ckpt_{step:07d}.pt"), cfg, model, ema,
                          None if last else opt, step)
            for path in sorted(glob.glob(os.path.join(args.out, "ckpt_*.pt")))[:-tc["keep_last"]]:
                os.remove(path)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
