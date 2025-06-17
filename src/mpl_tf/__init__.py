"""
mpl_tf
~~~~~~
Reference TensorFlow implementation of Meta Pseudo-Labels (MPL).
"""

from __future__ import annotations
from importlib.metadata import version, PackageNotFoundError  # Python ≥3.8

try:
    __version__ = version("mpl_tf")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

# Re-export public sub-packages for convenience
from . import models

__all__ = ["models", "__version__"]
