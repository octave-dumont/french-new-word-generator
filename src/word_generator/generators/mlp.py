"""
Multi-layer perceptron (MLP) character-level language model for word generation.
"""

from __future__ import annotations

from typing import Union, Optional, Callable
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from ..dataset_creation import Dataset
from ..layers import Linear, BatchNorm1d, Embedding, Flatten, Sequential, Tanh


class MLPGenerator(BaseModel):
    """
    Multi-layer perceptron for character-level language modeling.
    
    Predicts the next character based on a fixed-size context window of previous
    characters. Uses embeddings, batch normalization, and multiple hidden layers
    with Tanh activations. Trained with SGD and a simple learning rate schedule.
    
    Architecture:
        1. Embedding layer: maps character indices to dense vectors
        2. Flatten: concatenates context embeddings into single vector
        3. Hidden layers: Linear -> BatchNorm -> Tanh (repeated num_mid_layers times)
        4. Output layer: Linear projection to vocabulary size
    
    Attributes:
        dataset: Dataset instance containing training/test data and vocabulary.
        n_embd: Embedding dimension for each character. Default 24.
        n_hidden: Number of units in each hidden layer. Default 125.
        num_mid_layers: Number of hidden layer blocks. Default 2.
        reduce_last_conf: Factor to scale down final layer weights (reduces initial confidence). Default 0.1.
        scaling_factor: Additional weight scaling for Linear layers. Default 1.1.
        model: The Sequential model containing all layers.
        parameters: List of all learnable parameter tensors.
        generator: PyTorch random generator for reproducibility.
        verbose: Whether to print training progress. Default True.
        train_losses: History of training batch losses.
        test_losses: History of test losses (sampled during training).
        ema_train: Exponential moving average of training loss.
    
    Example:
        >>> dataset = Dataset(path=Path("words.txt"), context_len=8).build()
        >>> model = MLPGenerator(dataset=dataset, n_embd=24, n_hidden=128)
        >>> model.train(max_steps=50000)
        >>> words = model.generate(num_words=10)
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: Dataset

    n_embd: int = Field(gt=0, default=24)
    n_hidden: int = Field(gt=0, default=125)
    num_mid_layers: int = Field(gt=0, default=2)
    reduce_last_conf: float = Field(gt=0, default=0.1)
    scaling_factor: float = 1.1 # Based from Torch forum but could use 5/3 or even 1.59253742


    model: Union[Sequential, None] = None
    parameters: Union[list[torch.Tensor], None] = None
    generator: Optional[torch.Generator] = torch.Generator().manual_seed(42)

    verbose: bool = True
    train_losses: list[float] = Field(default_factory=list)
    test_losses: list[float] = Field(default_factory=list)
    ema_train: Union[float, None] = None    


    @computed_field(return_type=list)
    @property
    def first_layer(self) -> list:
        """
        Construct the input layer block.
        
        Returns:
            List of layers: [Embedding, Flatten, Linear, BatchNorm1d, Tanh]
            
        The Linear layer maps from (n_embd * context_len) to n_hidden.
        """
        layer = [Embedding(self.dataset.voc_size, self.n_embd), Flatten(), 
                        Linear(self.n_embd * self.dataset.context_len, self.n_hidden, bias=False), 
                        BatchNorm1d(self.n_hidden),   Tanh()]
        return layer
    
    @computed_field(return_type=list)
    @property
    def mid_layers(self) -> list:
        """
        Construct the hidden layer blocks.
        
        Returns:
            List of layers: [Linear, BatchNorm1d, Tanh] repeated num_mid_layers times.
            
        Each block maintains the hidden dimension (n_hidden -> n_hidden).
        """
        layers = []
        for _ in range(self.num_mid_layers):
            layers += [Linear(self.n_hidden, self.n_hidden),  BatchNorm1d(self.n_hidden),     Tanh()]
        return layers
    
    @computed_field(return_type=list)
    @property
    def last_layer(self) -> list:
        """
        Construct the output layer.
        
        Returns:
            List containing single Linear layer projecting n_hidden to voc_size.
            
        No activation is applied; raw logits are returned for use with cross-entropy loss.
        """
        return [Linear(self.n_hidden, self.dataset.voc_size)]
    
    @model_validator(mode='after')
    def get_model(self) -> MLPGenerator:
        """
        Assemble and initialize the full model after Pydantic validation.
        
        Performs the following initialization steps:
            1. Combines first_layer, mid_layers, and last_layer into Sequential
            2. Scales down final layer weights by reduce_last_conf (reduces initial confidence)
            3. Scales all Linear layer weights to avoid saturation/zeroing-out gradients
            4. Enables gradients for all parameters
            5. Prints parameter count if verbose
        
        Returns:
            self: For Pydantic validation chain.
        """
        self.model = Sequential(self.first_layer + self.mid_layers + self.last_layer)
        with torch.no_grad():
            self.model.layers[-1].weight *= self.reduce_last_conf
            for layer in self.model.layers:
                if isinstance(layer, Linear):
                    layer.weight *= self.scaling_factor

        self.parameters = self.model.parameters()

        for p in self.parameters: 
            p.requires_grad = True

        if self.verbose:
            print(f"Number of parameters: {sum(p.nelement() for p in self.parameters)}")
        return self


    def get_test_loss(self) -> float:
        """
        Compute cross-entropy loss on the full test set.
        
        Sets all layers to evaluation mode (uses running stats for BatchNorm),
        performs a forward pass on the entire test set, and returns the loss.
        
        Returns:
            Cross-entropy loss on the test set.
        
        Note:
            This method sets layer.training = False but does not reset it.
            The train() method sets it back to True at each step.
        """
        for layer in self.model.layers:
            layer.training = False
        with torch.no_grad():
            logits = self.model(self.dataset.Xte)
            loss = F.cross_entropy(logits, self.dataset.Yte)
        return float(loss.item())
    

    def train(self, max_steps: int = 125000, batch_size: int = 64,
              lr_schedule: Callable[[int], float] = lambda i: (0.1 if i < 100000 else 0.01),
              print_every: int = 50, beta: float = 0.98) -> MLPGenerator:
        """
        Train the model using mini-batch SGD.
        
        Args:
            max_steps: Total number of training iterations. Default 125,000.
            batch_size: Number of examples per mini-batch. Default 64.
            lr_schedule: Learning rate as a function of i. Default 0.1 if i < 100000 else 0.01.
            print_every: Number of times to print progress during training. Default 50.
            beta: Decay factor for exponential moving average of training loss. Default 0.98.
        
        Returns:
            self: For method chaining.
        
        Side Effects:
            - Appends to self.train_losses at every step
            - Appends to self.test_losses at print intervals
            - Updates self.ema_train
            - Prints progress if verbose
        
        Example:
            >>> model.train(max_steps=50000, batch_size=128)
            0/50000: loss=3.45123 | ema train=3.45123 | test loss=3.44521
            ...
        """
        
        for i in range(max_steps):
            for layer in self.model.layers:
                layer.training = True
            ix = torch.randint(0, self.dataset.Xtr.shape[0], (batch_size, ), generator=self.generator)
            Xb, Yb = self.dataset.Xtr[ix], self.dataset.Ytr[ix]

            logits = self.model(Xb)
            
            loss = F.cross_entropy(logits, Yb)
            self.train_losses.append(loss.item())
            
            for p in self.parameters:
                p.grad = None
            loss.backward()

            lr = lr_schedule(i)
            with torch.no_grad():
                for p in self.parameters:
                    p.data -= lr * p.grad
                    
            if self.verbose: 
                self.ema_train = loss.item() if self.ema_train is None else beta * self.ema_train + (1 - beta) * loss.item()           

                if i ==0 or i % (max_steps // print_every) == max_steps // print_every - 1:
                    loss_test =  self.get_test_loss()
                    self.test_losses.append(loss_test)
                    print(f"{i}/{max_steps}: loss={loss:.5f} | ema train={self.ema_train:.5f} | test loss={loss_test:.5f}")
        
        return self
    
    def plot_losses(self, target_points: int = 500) -> None:
        """
        Plot training and test loss curves.
        
        Downsamples the training loss history by averaging over blocks to achieve
        approximately target_points data points. Interpolates test losses to align
        with the downsampled training loss x-axis.
        
        Args:
            target_points: Number of points to display. Default 500.
        
        Side Effects:
            Displays a matplotlib figure with train and test loss curves.
        """
        L = len(self.train_losses)
        l = len(self.test_losses)

        view_factor = max(1, (L + target_points - 1) // target_points) 

        K = (L // view_factor) * view_factor

        display_losses_tr = torch.tensor(self.train_losses[:K]).view(-1, view_factor).mean(1)

        block_end_steps = torch.arange(1, K // view_factor + 1) * view_factor - 1  

        display_losses_test = []
        for s in block_end_steps.tolist():
            pos = s * (l - 1) / (K - 1)   
            j = int(pos)
            t = pos - j
            if j >= l - 1:
                display_losses_test.append(self.test_losses[-1])
            else:
                display_losses_test.append((1 - t) * self.test_losses[j] + t * self.test_losses[j + 1])
        
        plt.plot(display_losses_tr, label="Train losses")
        plt.plot(display_losses_test, label="Test losses")
        plt.legend()
        plt.show()

    def generate(self, num_words: int=10, max_runs: int=1000) -> list[str]:
        """
        Generate new words by sampling from the model.
        
        Starting from a context of all <BOW> tokens, repeatedly predicts the next
        character by sampling from the softmax distribution over logits until
        <EOW> is sampled. Only returns words not in the training set.
        
        Args:
            num_words: Target number of novel words to generate. Default 10.
            max_runs: Maximum generation attempts to prevent infinite loops. Default 1000.
        
        Returns:
            List of generated words. May contain fewer than num_words if max_runs
            is reached or if generated words already exist in the training set.
        
        Example:
            >>> words = model.generate(num_words=5)
            >>> print(words)
            ['montier', 'fleurant', 'boiselle', 'chardin', 'preval']
        """
        out = []
        i = 0
        while len(out) < (num_words) and i < max_runs:
            i += 1
            context = [0] * self.dataset.context_len
            word = ""
            while True:
                logits = self.model(torch.tensor([context]))

                probs = F.softmax(logits, dim=1)

                ix = int(torch.multinomial(probs, num_samples=1).item())
                context = context[1:] + [ix]
                char = self.dataset.idx_to_str[ix]
                if char == "<EOW>":
                    break
                word += char
            if word not in self.dataset.wordsset:
                out.append(word)
        return out