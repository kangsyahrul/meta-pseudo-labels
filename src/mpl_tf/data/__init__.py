"""
Model Zoo
=========

Unified access to all model builders.
"""

from __future__ import annotations
from typing import Dict, Type

from .base import BaseDataset


# what to re-export when someone does “from mpl_tf.models import *”
__all__ = [
    "BaseDataset",
]
