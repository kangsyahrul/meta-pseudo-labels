from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

import tensorflow as tf


class BaseModel(ABC):
    """Abstract interface every model-builder must follow."""

    def __init__(self, *, compile_kwargs: Dict[str, Any] | None = None) -> None:
        self.compile_kwargs = compile_kwargs or {
            "optimizer": tf.keras.optimizers.Adam(),
            "loss": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "metrics": ["accuracy"],
        }

    @abstractmethod
    def build(self) -> tf.keras.Model:  # noqa: D401
        """Return a **compiled** `tf.keras.Model` instance."""
        raise NotImplementedError
