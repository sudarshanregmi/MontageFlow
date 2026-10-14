from dataclasses import asdict, fields

from .diffusion import DiffusionConfig, DiffusionModel
from .flow import FlowConfig, FlowModel
from .gan import EEGGAN, GANConfig

MODELS = {
    "flow": (FlowConfig, FlowModel),
    "diffusion": (DiffusionConfig, DiffusionModel),
    "gan": (GANConfig, EEGGAN),
}


def build_model(spec: dict):
    """Build a model from a config dict with a `type` key (flow, diffusion or gan)."""
    spec = dict(spec)
    config_cls, model_cls = MODELS[spec.pop("type")]
    known = {f.name for f in fields(config_cls)}
    unknown = set(spec) - known
    if unknown:
        raise ValueError(f"unknown {config_cls.__name__} fields: {sorted(unknown)}")
    return model_cls(config_cls(**spec))


def model_spec(model) -> dict:
    """Inverse of build_model: the config dict that rebuilds `model`."""
    kind = next(k for k, (_, cls) in MODELS.items() if isinstance(model, cls))
    return {"type": kind, **asdict(model.cfg)}


__all__ = ["build_model", "model_spec", "FlowModel", "FlowConfig", "DiffusionModel",
           "DiffusionConfig", "EEGGAN", "GANConfig"]
