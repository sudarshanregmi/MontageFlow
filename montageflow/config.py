"""YAML configs. A run config is built from a method file and a dataset file:

    cfg = load_config("configs/montageflow.yaml", data="configs/data/tuab.yaml",
                      overrides=["train.steps=1000"])

Later sources override earlier ones: defaults < dataset < method < command line.
"""
from __future__ import annotations

import copy

import yaml

DEFAULTS = {
    "data": {"scale": 0.01},
    "train": {
        "batch_size": 128,
        "micro_batch": 32,
        "lr": 5.0e-5,
        "warmup_epochs": 2.5,
        "weight_decay": 0.0,
        "grad_clip": 0.0,
        "ema": 0.9999,
        "balance_classes": True,
        "seed": 0,
        "save_every": 4000,
        "keep_last": 2,
        "log_every": 50,
    },
    "prior": {"label_conditioned": True, "steps": 12000, "sample_steps": 100, "seed": 0},
    "sampling": {"steps": 50},
}


def merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(item: str) -> dict:
    """"train.steps=1000" -> {"train": {"steps": 1000}}"""
    key, value = item.split("=", 1)
    node = yaml.safe_load(value)
    for part in reversed(key.split(".")):
        node = {part: node}
    return node


def load_config(method: str, data: str = None, overrides=()) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    for path in (data, method):
        if path:
            with open(path) as f:
                cfg = merge(cfg, yaml.safe_load(f))
    for item in overrides:
        cfg = merge(cfg, parse_override(item))
    return cfg
