"""
Model Zoo
=========

Unified access to all model builders.
"""

from __future__ import annotations
from typing import Dict, Type

from .cnn import CNN
from .base import BaseModel

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "cnn": CNN,
    # "resnet": ResNetBuilder,
    # "vit": ViTBuilder,
}

def get_model(name: str, **kwargs):
    """
    Build **and compile** a model by key.

    Parameters
    ----------
    name
        Key in `MODEL_REGISTRY`, e.g. ``"cnn"``.
    **kwargs
        Passed straight to the builder`s constructor, e.g.
        ``dropout_rate=0.2`` or ``num_classes=100``.

    Returns
    -------
    tf.keras.Model
    """
    if name not in MODEL_REGISTRY:  # fail early with a friendly message
        raise KeyError(
            f"{name!r} not found. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs).build()


# what to re-export when someone does “from mpl_tf.models import *”
__all__ = [
    "MODEL_REGISTRY",
    "BaseModel",
    "CNN",
    "get_model",
]
