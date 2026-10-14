"""MontageFlow: physically valid, high-fidelity EEG generation."""
from .checkpoint import load
from .generate import generate
from .models import build_model

__version__ = "0.1.0"
__all__ = ["load", "generate", "build_model"]
