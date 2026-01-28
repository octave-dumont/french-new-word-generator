"""
Bigram character-level language model for word generation.
"""

from __future__ import annotations
from pydantic import BaseModel, ConfigDict, model_validator
import torch
import matplotlib.pyplot as plt
from typing import Union
from ..dataset_creation import Dataset


class BigramGenerator(BaseModel):
    """
    Character-level bigram language model.
    
    Learns transition probabilities between consecutive characters from the training
    data. Predicts the next character based solely on the current character (context_len=1).
    Uses Laplace smoothing (+1) to handle unseen bigrams.
    
    This is a simple statistical model that serves as a baseline above random guessing.
    It captures basic character patterns (e.g., 'q' is usually followed by 'u') but
    cannot model longer-range dependencies.
    
    Attributes:
        dataset: Dataset instance with context_len=1.
        frequency_mat: Matrix of shape (voc_size, voc_size) counting bigram occurrences.
        probability_mat: Row-normalized frequency matrix giving P(next_char | current_char).
        verbose: Whether to print loss during evaluation. Default True.
        test_loss: Cached test loss after calling get_test_loss().
    
    Raises:
        ValueError: If dataset.context_len != 1.
    
    Example:
        >>> dataset = Dataset(path=Path("words.txt"), context_len=1).build()
        >>> model = BigramGenerator(dataset=dataset)
        >>> model.train()
        >>> print(f"Test loss: {model.get_test_loss():.4f}")
        >>> new_words = model.generate(num_words=10)
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: Dataset
    frequency_mat: torch.Tensor = torch.zeros(0)
    probability_mat: torch.Tensor = torch.zeros(0)
    verbose: bool = True
    test_loss: Union[float, None] = None

    @model_validator(mode='after')
    def assert_context_len(self) -> BigramGenerator:
        """
        Validate that dataset has context_len=1, since bigram models
        only consider the immediately preceding character.
        
        Returns:
            self: For Pydantic validation chain.
        
        Raises:
            ValueError: If dataset.context_len != 1.
        """
        if self.dataset.context_len != 1:
            raise ValueError(
                f"BigramGenerator requires dataset.context_len=1, "
                f"but got context_len={self.dataset.context_len}"
            )
        return self

    def train(self) -> BigramGenerator:
        """
        Build bigram frequency and probability matrices from the dataset.
        
        Iterates through all words in dataset.wordsset, counting transitions
        between consecutive characters (including <BOW> and <EOW> tokens).
        Applies Laplace smoothing by adding 1 to all counts before normalizing.
        
        The resulting probability_mat[i, j] gives P(char_j | char_i).
        
        Returns:
            self: For method chaining.
        
        Example:
            >>> model = BigramGenerator(dataset=dataset).train()
        """
        self.frequency_mat = torch.zeros((self.dataset.voc_size, self.dataset.voc_size), dtype=torch.int32)
        for w in self.dataset.wordsset:
            chs = ["<BOW>"] + list(w) + ["<EOW>"]
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1, ix2 = self.dataset.str_to_idx[ch1], self.dataset.str_to_idx[ch2]
                self.frequency_mat[ix1, ix2] += 1

        self.probability_mat = (self.frequency_mat + 1).float()
        self.probability_mat = self.probability_mat / self.probability_mat.sum(1, keepdim=True)
        return self
    
    def get_test_loss(self) -> float:
        """
        Compute negative log-likelihood loss on the test set.
        
        For each (context, target) pair in the test set, looks up the
        probability assigned by the bigram model and computes the average
        negative log probability.
        
        Returns:
            Average cross-entropy loss on the test set.
        
        Side Effects:
            - Stores result in self.test_loss
            - Prints loss if self.verbose is True
        
        Example:
            >>> loss = model.get_test_loss()
            loss=2.45321
        """
        contexts = self.dataset.Xte[:, -1] 
        targets = self.dataset.Yte       
        
        probs = self.probability_mat[contexts, targets]
        
        self.test_loss = -torch.log(probs).mean().item()
        if self.verbose:
            print(f"loss={self.test_loss:.5f}")
        return float(self.test_loss)
    
    def plot_losses(self) -> None:
        """
        Display a bar chart comparing bigram loss to random baseline.
        
        Random baseline loss is log(voc_size), representing uniform
        probability over all characters.
        
        Side Effects:
            - Calls get_test_loss() to compute bigram loss
            - Displays matplotlib bar chart
        """
        random_loss = float(torch.log(torch.tensor(self.dataset.voc_size)))
        bigram_loss = self.get_test_loss()
        
        models = ['Random', 'Bigram']
        losses = [random_loss, bigram_loss]
        
        plt.bar(models, losses)
        plt.ylabel('Loss')
        plt.show()
    
    
    def generate(self, num_words: int=10, max_runs: int=1000) -> list[str]:
        """
        Generate new words by sampling from the bigram distribution.
        
        Starting from <BOW>, repeatedly samples the next character according
        to the learned probability distribution until <EOW> is sampled.
        Only returns words that are not already in the training set.
        
        Args:
            num_words: Target number of novel words to generate. Default 10.
            max_runs: Maximum generation attempts to avoid infinite loops. Default 1000.
        
        Returns:
            List of generated words (may be fewer than num_words if max_runs is reached
            or if most generated words already exist in the training set).
        
        Example:
            >>> words = model.generate(num_words=5)
            >>> print(words)
        """
        out = []
        i = 0
        while len(out) < (num_words) and i < max_runs:
            i += 1
            ix, word = 0, ""
            while True:
                p = self.probability_mat[ix]
                ix = int(torch.multinomial(p, num_samples=1, replacement=True).item())
                char = self.dataset.idx_to_str[ix]
                if char == "<EOW>":
                    break
                word += char
            if word not in self.dataset.wordsset:
                out.append(word)
        return out