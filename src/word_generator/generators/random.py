"""
Random baseline generator for word generation.

Provides a uniform random baseline to compare against learned models.
"""

from __future__ import annotations

import numpy.random as rd
from pydantic import BaseModel, computed_field, model_validator
import torch
import torch.nn.functional as F

from ..dataset_creation import Dataset


class RandomGenerator(BaseModel):
    """
    Uniform random baseline generator.
    
    Generates words by randomly sampling characters with uniform probability
    over the vocabulary. Serves as a theoretical lower bound for model performance;
    any learned model should achieve lower loss than this baseline.
    
    The test loss for uniform random guessing is log(voc_size), which represents
    maximum entropy over the vocabulary.
    
    Word lengths are sampled from a geometric distribution matching the mean
    word length in the training set.
    
    Attributes:
        dataset: Dataset instance with context_len=1.
        mean_word_len: Computed average word length from dataset.wordsset.
    
    Raises:
        ValueError: If dataset.context_len != 1.
    
    Example:
        >>> dataset = Dataset(path=Path("words.txt"), context_len=1).build()
        >>> baseline = RandomGenerator(dataset=dataset)
        >>> print(f"Random baseline loss: {baseline.get_test_loss():.4f}")
        >>> random_words = baseline.generate(num_words=10)
    """
    
    dataset: Dataset

    @computed_field(return_type=float)
    @property
    def mean_word_len(self) -> float:
        """
        Compute the average word length from the dataset.
        
        Returns:
            Mean length of words in dataset.wordsset.
        
        Used to parameterize the geometric distribution for sampling word lengths
        during generation.
        """
        return float(torch.tensor([float(len(w)) for w in self.dataset.wordsset]).mean().item())
    
    @model_validator(mode='after')
    def assert_context_len(self) -> RandomGenerator:
        """
        Validate that dataset has context_len=1.
        
        Returns:
            self: For Pydantic validation chain.
        
        Raises:
            ValueError: If dataset.context_len != 1.
        """
        if self.dataset.context_len != 1:
            raise ValueError(
                f"RandomGenerator requires dataset.context_len=1, "
                f"but got context_len={self.dataset.context_len}"
            )
        return self

    def train(self) -> RandomGenerator:
        """
        No-op training method for API consistency.
        
        Returns:
            self: For method chaining.
        """
        return self
    
    def get_test_loss(self) -> float:
        """
        Compute the theoretical loss for uniform random guessing.
        
        For a uniform distribution over voc_size characters:
            P(any char) = 1 / voc_size
            loss = -log(1 / voc_size) = log(voc_size)
        
        This represents maximum entropy and is the worst possible loss
        for a model that assigns equal probability to all characters.
        
        Returns:
            log(voc_size), the cross-entropy loss for uniform random predictions.
        """
        return float(torch.log(torch.tensor(self.dataset.voc_size)))
    
    def plot_losses(self):
        """
        Print the random baseline loss.
        
        Unlike other generators, RandomGenerator has no training curve,
        so this simply prints the constant theoretical loss value.
        """
        print(f"loss={self.get_test_loss():.5f}")

    def generate(self, num_words: int=10, max_runs: int=1000) -> list[str]:
        """
        Generate random words by uniform character sampling.
        
        For each word:
            1. Sample word length from geometric distribution with mean=mean_word_len
            2. Sample each character uniformly from the vocabulary
            3. Reject words that exist in the training set
        
        Args:
            num_words: Target number of novel words to generate. Default 10.
            max_runs: Maximum generation attempts to prevent infinite loops. Default 1000.
        
        Returns:
            List of generated words. May contain fewer than num_words if max_runs
            is reached or if generated words already exist in the training set.
        
        Note:
            Generated words are typically gibberish since no character patterns
            are learned. This is expected behavior for a random baseline.
        
        Example:
            >>> words = baseline.generate(num_words=5)
            >>> print(words)
            ['xkqmz', 'auwbtelp', 'rnj', 'yofvhcdi', 'zs']
        """
        out = []
        i = 0
        while len(out) < (num_words) and i < max_runs:
            i += 1
            length = rd.geometric(1/self.mean_word_len)
            word = ''.join(rd.choice(list(self.dataset.str_to_idx), length))
            if word not in self.dataset.wordsset:
                out.append(word)
        return out