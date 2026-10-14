"""Cut TUEV EDF recordings into 5 s windows around each annotated event (16-channel TCP
bipolar montage, 200 Hz).

Adapted from JET (https://github.com/Y-Research-SBU/JET, MIT License,
Copyright (c) 2026 Y-Research @SBU).
"""
import argparse
import os
import pickle
import shutil
from multiprocessing import Pool
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm


BIPOLAR_PAIRS = [
    ("EEG FP1-REF", "EEG F7-REF"),
    ("EEG F7-REF",  "EEG T3-REF"),
    ("EEG T3-REF",  "EEG T5-REF"),
    ("EEG T5-REF",  "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F8-REF"),
    ("EEG F8-REF",  "EEG T4-REF"),
    ("EEG T4-REF",  "EEG T6-REF"),
    ("EEG T6-REF",  "EEG O2-REF"),
    ("EEG FP1-REF", "EEG F3-REF"),
    ("EEG F3-REF",  "EEG C3-REF"),
    ("EEG C3-REF",  "EEG P3-REF"),
    ("EEG P3-REF",  "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F4-REF"),
    ("EEG F4-REF",  "EEG C4-REF"),
    ("EEG C4-REF",  "EEG P4-REF"),
    ("EEG P4-REF",  "EEG O2-REF"),
]
FS = 200
EVENT_WINDOW_SECONDS = 5


def build_events(signals, times, event_data):
    num_events = event_data.shape[0]
    num_chan = signals.shape[0]
    # float32 halves the disk use; the windows are used in float32 anyway.
    features = np.zeros([num_events, num_chan, FS * EVENT_WINDOW_SECONDS], dtype=np.float32)
    offending_channel = np.zeros([num_events, 1])
    labels = np.zeros([num_events, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(num_events):
        start = np.where(times >= event_data[i, 1])[0][0]
        end = np.where(times >= event_data[i, 2])[0][0]
        features[i, :] = signals[
            :, offset + start - 2 * FS : offset + end + 2 * FS
        ]
        offending_channel[i, :] = int(event_data[i, 0])
        labels[i, :] = int(event_data[i, 3])
    return features, offending_channel, labels


def convert_signals(signals, raw):
    name_to_idx = {ch: i for i, ch in enumerate(raw.info["ch_names"])}
    return np.vstack([
        signals[name_to_idx[a]] - signals[name_to_idx[b]]
        for a, b in BIPOLAR_PAIRS
    ])


def read_edf(file_name):
    raw = mne.io.read_raw_edf(file_name, preload=True)
    raw.resample(FS)
    raw.filter(l_freq=0.3, h_freq=75)
    raw.notch_filter(60)
    _, times = raw[:]
    signals = raw.get_data(units="uV")
    rec_file = file_name[:-3] + "rec"
    event_data = np.genfromtxt(rec_file, delimiter=",")
    raw.close()
    return signals, times, event_data, raw


def _process_one(params):
    """Process a single EDF. Returns (n_events_written, skip_reason_or_None)."""
    dir_name, fname, out_dir = params
    try:
        signals, times, event_data, raw = read_edf(os.path.join(dir_name, fname))
        signals = convert_signals(signals, raw)
    except (ValueError, KeyError) as exc:
        return 0, f"{dir_name}/{fname}: unparseable ({type(exc).__name__})"
    try:
        event_signals, offending_channels, labels = build_events(signals, times, event_data)
    except (ValueError, IndexError) as exc:
        # Skip a file that cannot be processed, and report it, instead of stopping the run.
        return 0, f"{dir_name}/{fname}: bad events ({type(exc).__name__})"
    for idx, (signal, ch, label) in enumerate(zip(event_signals, offending_channels, labels)):
        sample = {"signal": signal, "offending_channel": ch, "label": label}
        with open(os.path.join(out_dir, f"{fname.split('.')[0]}-{idx}.pkl"), "wb") as f:
            pickle.dump(sample, f)
    return len(event_signals), None


def process_split(base_dir, out_dir, num_workers=1):
    """Process every EDF file in base_dir, in parallel when num_workers > 1."""
    tasks = []
    for dir_name, _sub_dirs, file_list in os.walk(base_dir):
        for fname in sorted(file_list):
            if fname.endswith(".edf"):
                tasks.append((dir_name, fname, out_dir))
    print(f"{base_dir}: {len(tasks)} EDF files, {num_workers} workers")

    skipped, total = [], 0
    if num_workers <= 1:
        results = (_process_one(t) for t in tasks)
        for n, skip in tqdm(results, total=len(tasks), ncols=100):
            total += n
            if skip:
                skipped.append(skip)
    else:
        with Pool(processes=num_workers) as pool:
            for n, skip in tqdm(pool.imap_unordered(_process_one, tasks),
                                total=len(tasks), ncols=100):
                total += n
                if skip:
                    skipped.append(skip)

    print(f"{base_dir}: wrote {total} event windows; skipped {len(skipped)} files")
    for s in skipped:
        print(f"  SKIP {s}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Preprocess TUEV EDF + REC data into pickles.")
    parser.add_argument("--input-dir", required=True,
                        help="Root of TUEV v2.0.1 EDF data (contains 'train/' and 'eval/' subdirs).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory under which train/, val/, test/ pickle splits are written.")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of TUEV train subjects held out for validation.")
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    root = args.input_dir
    out = Path(args.output_dir)
    interim = out / "_interim"
    interim_train = interim / "train"
    interim_eval = interim / "eval"
    interim_train.mkdir(parents=True, exist_ok=True)
    interim_eval.mkdir(parents=True, exist_ok=True)

    process_split(os.path.join(root, "train"), str(interim_train), num_workers=args.num_workers)
    process_split(os.path.join(root, "eval"),  str(interim_eval),  num_workers=args.num_workers)

    # Subject-level 80/20 split of the train pool.
    train_files = os.listdir(interim_train)
    all_train_sub = sorted({f.split("_")[0] for f in train_files})
    cut = int(len(all_train_sub) * (1.0 - args.val_ratio))
    train_sub, val_sub = all_train_sub[:cut], all_train_sub[cut:]
    print(f"train subjects: {len(train_sub)}, val subjects: {len(val_sub)}")

    val_files = [f for f in train_files if f.split("_")[0] in set(val_sub)]
    train_files = [f for f in train_files if f.split("_")[0] in set(train_sub)]
    test_files = os.listdir(interim_eval)

    for name in ("train", "val", "test"):
        (out / name).mkdir(parents=True, exist_ok=True)

    # Move instead of copy, so the windows are never stored twice.
    for f in tqdm(train_files, desc="train"):
        shutil.move(str(interim_train / f), str(out / "train" / f))
    for f in tqdm(val_files, desc="val"):
        shutil.move(str(interim_train / f), str(out / "val" / f))
    for f in tqdm(test_files, desc="test"):
        shutil.move(str(interim_eval / f), str(out / "test" / f))

    print("Done!")


if __name__ == "__main__":
    main()
