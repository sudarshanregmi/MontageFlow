"""Checkpoints.

Training checkpoints (scripts/train.py) are a single .pt file with the run config, the raw and
EMA weights, and the optimizer state. Released checkpoints are a directory:

    config.yaml          model and data settings
    model.safetensors    EMA weights, without the encoder (generation uses the prior instead)
    prior.safetensors    fitted latent prior (MontageFlow and MontageDiff only)

`load` accepts a released directory, a Hugging Face path "user/repo/subfolder", or a training
checkpoint file.
"""
from __future__ import annotations

import os

import torch
import yaml
from safetensors.torch import load_file, save_file

from .models import build_model
from .prior import LatentPrior


def save_training(path, config, model, ema, optimizer, step):
    tmp = path + ".tmp"
    torch.save({"config": config, "model": model.state_dict(), "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "step": step}, tmp)
    os.replace(tmp, path)


def model_from_training(path, weights="ema", device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["config"]["model"])
    model.load_state_dict(ck[weights])
    return model.to(device).eval(), ck["config"]


def export(out_dir, model, config, prior=None, encoder=False):
    """Write a released checkpoint directory. The encoder weights are left out unless `encoder`."""
    from .models import model_spec
    os.makedirs(out_dir, exist_ok=True)
    meta = {"model": model_spec(model),
            "data": {"scale": config["data"]["scale"], "num_classes": model.num_classes},
            "sampling": {"steps": config["sampling"]["steps"]}}
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()
               if encoder or not k.startswith("encoder.")}
    save_file(tensors, os.path.join(out_dir, "model.safetensors"))
    if prior is not None:
        meta["prior"] = prior.config
        save_file({k: v.contiguous() for k, v in prior.state_dict().items()},
                  os.path.join(out_dir, "prior.safetensors"))
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def _resolve(source):
    if os.path.exists(source):
        return source
    from huggingface_hub import snapshot_download
    parts = source.split("/")
    repo, sub = "/".join(parts[:2]), "/".join(parts[2:])
    root = snapshot_download(repo, allow_patterns=[f"{sub}/*"] if sub else None)
    return os.path.join(root, sub)


def load(source, device="cpu"):
    """Returns (model, prior or None, config)."""
    path = _resolve(source)
    if os.path.isfile(path):
        model, config = model_from_training(path, device=device)
        prior_path = os.path.join(os.path.dirname(path), "prior.pt")
        prior = load_prior(prior_path, device) if os.path.exists(prior_path) else None
        return model, prior, config

    with open(os.path.join(path, "config.yaml")) as f:
        config = yaml.safe_load(f)
    model = build_model(config["model"])
    weights = load_file(os.path.join(path, "model.safetensors"))
    if model.encoder is not None and not any(k.startswith("encoder.") for k in weights):
        model.encoder = None             # released without the encoder
    model.load_state_dict(weights)
    prior = None
    if "prior" in config:
        prior = LatentPrior(**config["prior"])
        prior.load_state_dict(load_file(os.path.join(path, "prior.safetensors")))
        prior = prior.to(device).eval()
    return model.to(device).eval(), prior, config


def save_prior(path, prior):
    torch.save({"config": prior.config, "state": prior.state_dict()}, path)


def load_prior(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    prior = LatentPrior(**ck["config"])
    prior.load_state_dict(ck["state"])
    return prior.to(device).eval()
