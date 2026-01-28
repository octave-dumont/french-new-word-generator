"""
Word Generator Models

Available generators:
- RandomGenerator: Uniform random baseline
- BigramGenerator: Character bigram model
- MLPGenerator: Multi-layer perceptron
- WaveNetGenerator: WaveNet-inspired architecture with gated activations
"""

from .random import RandomGenerator
from .bigram import BigramGenerator
from .mlp import MLPGenerator
from .wavenet import WaveNetGenerator

__all__ = [
    "RandomGenerator",
    "BigramGenerator",
    "MLPGenerator",
    "WaveNetGenerator",
]
