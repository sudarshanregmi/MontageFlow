"""Tiny CPU runs of every model: one training step and a short sample."""
import torch

from montageflow import montage
from montageflow.generate import generate
from montageflow.models import build_model, model_spec
from montageflow.prior import LatentPrior

SMALL = dict(channels=16, length=200, num_classes=2, patch=50, hidden=64, depth=2, heads=4)


def train_step(model):
    x, y = torch.randn(4, 16, 200) * 0.3, torch.tensor([0, 1, 0, 1])
    model.train()
    loss = model(x, y)
    loss.backward()
    assert torch.isfinite(loss)


def test_montageflow_samples_are_valid():
    model = build_model({"type": "flow", **SMALL, "latent_dim": 16, "encoder_width": 32,
                         "encoder_depth": 1, "encoder_heads": 4})
    train_step(model)
    model.eval()
    prior = LatentPrior(16, num_classes=2, width=32, n_blocks=1, sample_steps=4)
    prior.fit(torch.randn(64, 16), torch.randint(0, 2, (64,)), steps=5, log_every=0)
    x = generate(model, [0, 1, 1], prior=prior, steps=4)
    assert x.shape == (3, 16, 200)
    assert montage.montage_index(x) < 1e-5


def test_montagediff_samples_are_valid():
    model = build_model({"type": "diffusion", **SMALL, "timesteps": 50, "projection": True, "latent_dim": 16,
                         "encoder_width": 32, "encoder_depth": 1, "encoder_heads": 4})
    train_step(model)
    model.eval()
    prior = LatentPrior(16, num_classes=2, width=32, n_blocks=1, sample_steps=4)
    prior.fit(torch.randn(64, 16), torch.randint(0, 2, (64,)), steps=5, log_every=0)
    x = generate(model, [0, 1, 1], prior=prior, steps=4)
    assert x.shape == (3, 16, 200)
    assert montage.montage_index(x) < 1e-5


def test_flow_baselines():
    for extra in ({"prediction": "x", "cross_channel": False, "projection": False, "latent_dim": 0,
                   "stat_weight": 1.0, "tv_weight": 0.1, "corr_weight": 0.1},
                  {"projection": False, "latent_dim": 0}):
        model = build_model({"type": "flow", **SMALL, **extra})
        train_step(model)
        assert generate(model.eval(), [0, 1], steps=3).shape == (2, 16, 200)


def test_projection_at_sampling_only():
    model = build_model({"type": "flow", **SMALL, "projection": False, "latent_dim": 0}).eval()
    x = generate(model, [0, 1], steps=3, project=True)
    assert montage.montage_index(x) < 1e-5


def test_diffusion_and_gan():
    for spec in ({"type": "diffusion", **SMALL, "timesteps": 50},
                 {"type": "gan", "channels": 16, "length": 200, "num_classes": 2, "width": 32,
                  "n_blocks": 2, "n_critic": 1}):
        model = build_model(spec)
        train_step(model)
        assert generate(model.eval(), [0, 1], steps=3).shape == (2, 16, 200)


def test_release_without_encoder(tmp_path):
    from montageflow.checkpoint import export, load
    model = build_model({"type": "flow", **SMALL, "projection": True, "latent_dim": 16, "encoder_width": 32,
                         "encoder_depth": 1, "encoder_heads": 4}).eval()
    prior = LatentPrior(16, num_classes=2, width=32, n_blocks=1, sample_steps=4)
    prior.fit(torch.randn(64, 16), torch.randint(0, 2, (64,)), steps=5, log_every=0)
    export(str(tmp_path), model, {"data": {"scale": 0.01}, "sampling": {"steps": 4}}, prior)
    loaded, loaded_prior, _ = load(str(tmp_path))
    assert loaded.encoder is None and loaded.has_latent
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in loaded.state_dict().items())
    x = generate(loaded, [0, 1], prior=loaded_prior, steps=4)
    assert x.shape == (2, 16, 200) and montage.montage_index(x) < 1e-5


def test_spec_round_trip():
    model = build_model({"type": "flow", **SMALL, "latent_dim": 0})
    rebuilt = build_model(model_spec(model))
    assert model_spec(rebuilt) == model_spec(model)
