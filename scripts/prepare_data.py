"""Pack the preprocessed TUH windows into memory-mapped arrays.

scripts/preprocess_{tuab,tuev,tusz}.py write one pickle per window. This script packs each
split into

    <out>/<dataset>/<split>/X.npy   (N, 16, L) float32, microvolts
    <out>/<dataset>/<split>/y.npy   (N,) int64 labels, starting at 0
    <out>/<dataset>/<split>/meta.json

and checks a random subset of rows against the source pickles.

    python scripts/prepare_data.py --src /path/to/windows --out data
"""
import argparse
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np

# the three preprocessing scripts store windows and labels under different keys
KEYS = {
    "tuab": dict(x="X", y="y", offset=0),
    "tuev": dict(x="signal", y="label", offset=1),   # TUEV labels start at 1
    "tusz": dict(x="signal", y="label", offset=0),
}


def read(path, keys):
    with open(path, "rb") as f:
        d = pickle.load(f)
    x = np.asarray(d[keys["x"]], dtype=np.float32)
    y = int(np.asarray(d[keys["y"]]).reshape(-1)[0]) - keys["offset"]
    return x, y


def fill(job):
    x_path, shape, start, files, keys = job
    X = np.lib.format.open_memmap(x_path, mode="r+")
    y = np.empty(len(files), dtype=np.int64)
    for i, f in enumerate(files):
        x, y[i] = read(f, keys)
        X[start + i] = x.reshape(shape[1:])
    X.flush()
    return start, y


def pack(src, out, keys, workers, chunk=512):
    files = sorted(os.path.join(src, f) for f in os.listdir(src) if f.endswith(".pkl"))
    if not files:
        print(f"{src}: no windows, skipped")
        return
    x0, _ = read(files[0], keys)
    shape = (len(files), *x0.reshape(x0.shape[0], -1).shape)
    os.makedirs(out, exist_ok=True)
    x_path = os.path.join(out, "X.npy")
    np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=shape).flush()

    y = np.empty(len(files), dtype=np.int64)
    jobs = [(x_path, shape, s, files[s:s + chunk], keys) for s in range(0, len(files), chunk)]
    with Pool(workers) as pool:
        for start, labels in pool.imap_unordered(fill, jobs):
            y[start:start + len(labels)] = labels
    np.save(os.path.join(out, "y.npy"), y)
    meta = {"N": shape[0], "C": shape[1], "L": shape[2], "units": "microvolts",
            "class_counts": np.bincount(y).tolist()}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{out}: {meta}")

    X = np.load(x_path, mmap_mode="r")
    for i in np.random.RandomState(0).choice(len(files), size=min(200, len(files)), replace=False):
        x, label = read(files[i], keys)
        assert np.array_equal(X[i], x.reshape(shape[1:])) and label == y[i], f"mismatch at row {i}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with <dataset>/<split>/*.pkl from the preprocessing scripts")
    ap.add_argument("--out", default="data")
    ap.add_argument("--datasets", nargs="+", default=list(KEYS))
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    for ds in args.datasets:
        for split in args.splits:
            src = os.path.join(args.src, ds, split)
            if os.path.isdir(src):
                pack(src, os.path.join(args.out, ds, split), KEYS[ds], args.workers)


if __name__ == "__main__":
    main()
