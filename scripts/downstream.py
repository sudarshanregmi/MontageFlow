"""Downstream utility: train a classifier on generated windows only, then test it on real ones.

The classifier is CBraMod (Wang et al., 2025), fine-tuned with its own recipe. Clone
https://github.com/wjq-learning/CBraMod (pretrained weights in pretrained_weights/) and run

    python scripts/downstream.py release/tuab --data data/tuab --cbramod ~/CBraMod \
        --seed 0 --out results/downstream_montageflow_tuab_s0.json

Pass `real` instead of a model to train on the same number of real training windows. The paper
uses 5,000 windows per class and seeds 0, 1 and 2. It reports balanced accuracy on the real test
split at the epoch with the best validation score (kappa for TUEV, AUROC for TUAB and TUSZ, as in
CBraMod). The data directory needs its val split (prepare_data.py --splits train val test).
"""
import argparse
import copy
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from montageflow.checkpoint import load  # noqa: E402
from montageflow.data import Windows  # noqa: E402
from montageflow.generate import generate  # noqa: E402

SECOND = 200  # CBraMod reads one-second patches at 200 Hz


class Classifier(nn.Module):
    """Pretrained CBraMod backbone with its 'all patch representations' head."""

    def __init__(self, cbramod_dir, n_classes, channels, seconds, dropout=0.1):
        super().__init__()
        sys.path.insert(0, cbramod_dir)
        from models.cbramod import CBraMod
        self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                                seq_len=30, n_layer=12, nhead=8)
        weights = os.path.join(cbramod_dir, "pretrained_weights", "pretrained_weights.pth")
        self.backbone.load_state_dict(torch.load(weights, map_location="cpu"))
        self.backbone.proj_out = nn.Identity()
        self.binary = n_classes == 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * seconds * 200, seconds * 200), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(seconds * 200, 200), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(200, 1 if self.binary else n_classes))

    def forward(self, x):  # x: (B, C, L) in microvolts / 100, which is also CBraMod's scale
        B, C, L = x.shape
        out = self.head(self.backbone(x.reshape(B, C, L // SECOND, SECOND)))
        return out.squeeze(-1) if self.binary else out


def load_split(path):
    """A whole split as float16, in model scale."""
    data = Windows(path)
    x = np.empty((len(data), *data.shape), dtype=np.float16)
    for i in range(0, len(data), 4096):
        idx = np.arange(i, min(i + 4096, len(data)))
        x[idx] = data.get(idx)[0].numpy()
    return x, data.y


def training_windows(args, n_classes, device):
    rs = np.random.RandomState(0)
    if args.model == "real":
        # the same number of real windows; classes with fewer windows are drawn with replacement
        data = Windows(os.path.join(args.data, "train"))
        idx = np.sort(np.concatenate([
            rs.choice(np.flatnonzero(data.y == c), args.per_class, replace=(data.y == c).sum() < args.per_class)
            for c in range(n_classes)]))
        return data.get(idx)[0].numpy().astype(np.float16), data.y[idx]
    if args.cache and os.path.exists(args.cache):
        saved = np.load(args.cache)
        return saved["x"], saved["y"]
    labels = np.repeat(np.arange(n_classes), args.per_class)
    rs.shuffle(labels)
    model, prior, cfg = load(args.model, device=device)
    torch.manual_seed(0)  # the same generated set for every classifier seed
    x = generate(model, labels, prior=prior, steps=cfg["sampling"]["steps"],
                 generator=torch.Generator().manual_seed(10_000)).astype(np.float16)
    if args.cache:
        np.savez(args.cache, x=x, y=labels)
    return x, labels


@torch.no_grad()
def evaluate(model, x, y, device):
    model.eval()
    logits = torch.cat([model(torch.from_numpy(x[i:i + 256]).float().to(device)).float().cpu()
                        for i in range(0, len(x), 256)]).numpy()
    if model.binary:
        score = 1.0 / (1.0 + np.exp(-logits))
        pred = (score > 0.5).astype(np.int64)
        return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)), "auroc": float(roc_auc_score(y, score))}
    pred = logits.argmax(-1)
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)), "kappa": float(cohen_kappa_score(y, pred))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="release dir, Hugging Face path or training checkpoint, or 'real'")
    ap.add_argument("--data", required=True, help="dataset directory with train/, val/ and test/")
    ap.add_argument("--cbramod", required=True, help="clone of github.com/wjq-learning/CBraMod")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cache", default=None, help=".npz to save the generated windows in, and reuse")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = Windows(os.path.join(args.data, "train")).num_classes
    x, y = training_windows(args, n_classes, device)
    val_x, val_y = load_split(os.path.join(args.data, "val"))
    test_x, test_y = load_split(os.path.join(args.data, "test"))
    print(f"training on {len(x)} windows, classes {np.bincount(y).tolist()}", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = Classifier(args.cbramod, n_classes, x.shape[1], x.shape[2] // SECOND).to(device)
    backbone = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": backbone, "lr": 1e-4},
                             {"params": head, "lr": 1e-3 * (args.batch_size / 256) ** 0.5}], weight_decay=5e-2)
    steps = args.epochs * ((len(x) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
    loss_fn = nn.BCEWithLogitsLoss() if model.binary else nn.CrossEntropyLoss(label_smoothing=0.1)
    select = "auroc" if model.binary else "kappa"

    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(x))
        for i in range(0, len(x), args.batch_size):
            idx = np.sort(perm[i:i + args.batch_size])
            xb = torch.from_numpy(x[idx]).float().to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            out = model(xb)
            loss = loss_fn(out, yb.float()) if model.binary else loss_fn(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        val = evaluate(model, val_x, val_y, device)
        print(f"epoch {epoch:3d}  val {select} {val[select]:.4f}  balanced accuracy {val['balanced_accuracy']:.4f}",
              flush=True)
        if best is None or val[select] > best["val"][select]:
            best = {"epoch": epoch, "val": val, "state": copy.deepcopy(model.state_dict())}

    model.load_state_dict(best["state"])
    result = {"model": args.model, "data": args.data, "seed": args.seed, "per_class": args.per_class,
              "best_epoch": best["epoch"], "val": best["val"], "test": evaluate(model, test_x, test_y, device)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
