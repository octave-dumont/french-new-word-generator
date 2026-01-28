from __future__ import annotations
from pathlib import Path
from typing import Tuple, Callable, Union
import torch
import random
from pydantic import BaseModel, model_validator, Field, ConfigDict
import regex as re

class Dataset(BaseModel):
    """
    A dataset builder for character-level language modeling.
    
    Loads a word list from a text file, builds character vocabularies, filters rare
    characters, and creates train/dev/test splits formatted for neural network training.
    
    Attributes:
        path: Path to the text file containing one word per line.
        encoding: File encoding. Default "utf-8".
        val_regex: Regex pattern for validating words.
        val_function: Custom validation function. If None, uses val_regex + lowercase check.
        freq_threshold: Minimum character frequency threshold. Characters appearing less
            frequently are filtered out along with words containing them.
        context_len: Number of previous characters used as context for prediction.
        wordsset: List of validated words after filtering.
        train_test_split: Fraction of data used for training. Default 0.8.
        test_dev_split: Fraction of non-training data used for dev. Default 0.9.
        chars: List of all characters including special tokens.
        str_to_idx: Mapping from character string to integer index.
        idx_to_str: Mapping from integer index to character string.
        voc_size: Total vocabulary size including special tokens.
        freq: Dictionary mapping characters to their frequency scores.
        Xtr, Ytr: Training features and labels as tensors.
        Xdev, Ydev: Development features and labels as tensors.
        Xte, Yte: Test features and labels as tensors.
    
    Special Tokens:
        <BOW>: Beginning of word (index 0)
        <EOW>: End of word (index 1)
    
    Example:
        >>> dataset = Dataset(
        ...     path=Path("data/french_words.txt"),
        ...     context_len=8,
        ...     train_test_split=0.8,
        ... ).build()
        >>> print(f"Vocab size: {dataset.voc_size}")
        >>> print(f"Training examples: {dataset.Xtr.shape[0]}")
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    path: Path = Path("data/francais_long.txt")
    encoding: str = "utf-8"
    val_regex: re.Pattern = re.compile(r"^[^\W\d_]+(?:[-''][^\W\d_]+)*$", re.UNICODE)
    val_function: Union[Callable, None] = None
    freq_threshold: float = 5e-3
    context_len: int = 1
    wordsset: list[str] = [""]
    train_test_split: float = Field(gt=0.0, lt=1.0, default=0.8)
    test_dev_split: float = Field(lt=1.0, default=0.9)
    chars: list[str] = []
    str_to_idx: dict[str, int] = {}
    idx_to_str: dict[int, str] = {}
    voc_size: int = 0
    freq: dict[str, float] = {}
    
    Xtr: torch.Tensor = torch.zeros(0)
    Ytr: torch.Tensor = torch.zeros(0)
    Xdev: torch.Tensor = torch.zeros(0)
    Ydev: torch.Tensor = torch.zeros(0)
    Xte: torch.Tensor = torch.zeros(0)
    Yte: torch.Tensor = torch.zeros(0)
    
    @model_validator(mode='after')
    def set_default_validator(self):
        """Sets the default validation function if none provided.
        
        Default accepts lowercase words matching val_regex pattern.
        """
        if self.val_function is None:
            self.val_function = lambda w: bool(self.val_regex.fullmatch(w)) and w.islower()
        return self    
    
    def get_wordsset(self):
        """Load and filter words from the source file.
        
        Reads the file at self.path, splits by newlines, and keeps only
        words that pass the validation function.
        
        Returns:
            self: For method chaining.
        """
        with open(self.path, encoding=self.encoding) as file:
            words = file.read().splitlines()        
        self.wordsset = [w for w in words if self.val_function(w)]
        return self

    def make_dics(self) -> Dataset:
        """Build character-to-index and index-to-character mappings.
        
        Creates vocabulary from all unique characters in wordsset plus
        special tokens <BOW> (index 0) and <EOW> (index 1). Characters
        are sorted alphabetically after special tokens.
        
        Returns:
            self: For method chaining.
        """
        special_tokens: list[str] = ["<BOW>", "<EOW>"]
        self.chars = special_tokens + sorted(list(set(''.join(self.wordsset))))
        self.str_to_idx = {s: i for i, s in enumerate(self.chars)}
        self.idx_to_str = {i: s for i, s in enumerate(self.chars)}
        self.voc_size = len(self.chars)
        return self
    
    def filter_low_freq(self) -> Dataset:
        """Remove words containing rare characters.
        
        Computes frequency score for each character and removes any word
        containing characters below freq_threshold. Rebuilds dictionaries
        after filtering.
        
        Returns:
            self: For method chaining.
        """
        self.freq = {ch: self.voc_size * sum(ch in w for w in self.wordsset) / len(self.wordsset) for ch in self.chars}
        self.freq = dict(sorted(self.freq.items(), key=lambda kv: kv[1]))
        rare_chars = [ch for ch in self.freq if self.freq[ch] < self.freq_threshold]
        self.wordsset = [w for w in self.wordsset if not any(ch in rare_chars for ch in w)]
        self.make_dics()
        return self
    
    def format(self, words: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert words to context-target tensor pairs.
        
        For each word, generates sliding window contexts of length context_len
        and corresponding target characters. Contexts are padded with <BOW>
        tokens at the start.
        
        Args:
            words: List of words to format.
        
        Returns:
            X: Tensor of shape (N, context_len) containing context indices.
            Y: Tensor of shape (N,) containing target character indices.
        
        Example:
            Word "ab" with context_len=3 produces:
                X: [[0,0,0], [0,0,a], [0,a,b]]  (0 = <BOW>)
                Y: [a, b, <EOW>]
        """
        X, Y = [], []
        for w in words:
            context = [self.str_to_idx["<BOW>"]] * self.context_len
            for ch in list(w) + ["<EOW>"]:
                ix = self.str_to_idx[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        return torch.tensor(X), torch.tensor(Y)
    
    def make_split(self) -> Dataset:
        """Split wordsset into train, dev, and test sets.
        
        Shuffles wordsset in place, then splits according to train_test_split
        and test_dev_split ratios. Formats each split into tensors.
        
        Split sizes:
            - Train: 0 to train_test_split (default 80%)
            - Dev: train_test_split to test_dev_split (default 10%)
            - Test: test_dev_split to end (default 10%)
        
        Returns:
            self: For method chaining.
        """
        random.shuffle(self.wordsset)
        length = len(self.wordsset)
        n1, n2 = int(self.train_test_split * length), int(self.test_dev_split * length)
        self.Xtr, self.Ytr = self.format(self.wordsset[:n1])
        self.Xdev, self.Ydev = self.format(self.wordsset[n1:n2])
        self.Xte, self.Yte = self.format(self.wordsset[n2:])
        return self
    
    def build(self) -> Dataset:
        """Execute the full dataset building pipeline.
        
        Returns:
            self: Fully initialized dataset ready for training.
        
        Example:
            >>> dataset = Dataset(path=Path("words.txt"), context_len=8).build()
        """
        self.get_wordsset()
        self.make_dics()
        self.filter_low_freq()
        self.make_split()
        return self