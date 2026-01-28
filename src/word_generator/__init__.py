"""
French Word Generator

Character-level language models for generating French-like words.
"""

__version__ = "0.1.0"

from .dataset_creation import Dataset
from .layers import (
    Linear,
    BatchNorm1d,
    Embedding,
    Flatten,
    FlattenConsecutive,
    GatedActivation,
    Tanh,
    Sequential,
)
from .generators import (
    RandomGenerator,
    BigramGenerator,
    MLPGenerator,
    WaveNetGenerator,
)

__all__ = [
    "__version__",

    "Dataset",

    "Linear",
    "BatchNorm1d",
    "Embedding",
    "Flatten",
    "FlattenConsecutive",
    "GatedActivation",
    "Tanh",
    "Sequential",

    "RandomGenerator",
    "BigramGenerator",
    "MLPGenerator",
    "WaveNetGenerator",
]
