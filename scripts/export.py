"""Turn a training checkpoint (plus its fitted prior) into a release directory:
config.yaml, model.safetensors (EMA weights, without the encoder) and prior.safetensors.

    python scripts/export.py runs/montageflow_tuab/ckpt_0161999.pt release/tuab
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from montageflow.checkpoint import export, load_prior, model_from_training  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--prior", default=None, help="default: prior.pt next to the checkpoint")
    ap.add_argument("--with-encoder", action="store_true", help="also include the encoder weights")
    args = ap.parse_args()

    model, cfg = model_from_training(args.checkpoint)
    prior = None
    if model.has_latent:
        path = args.prior or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "prior.pt")
        prior = load_prior(path)
    export(args.out, model, cfg, prior, encoder=args.with_encoder)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
