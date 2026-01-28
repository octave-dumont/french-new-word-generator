"""
Neural network layer implementations for character-level language models.

Custom implementations of common neural network layers using PyTorch tensors,
designed for educational purposes and compatibility with the word generator models.
"""

from __future__ import annotations

from typing import List, Optional, Iterable
import torch
from dataclasses import dataclass


@dataclass
class Linear:
    """
    Fully connected linear layer.
    
    Performs the transformation: output = input @ weight + bias
    
    Attributes:
        fan_in: Number of input features.
        fan_out: Number of output features.
        bias: Whether to include a bias term. Default True.
        generator: PyTorch random generator for reproducibility.
        gain: Weight initialization strategy. Either "kaiming" or "xavier".
    
    Initialization:
        - kaiming: Scales weights by 1/sqrt(fan_in). Good for ReLU/Tanh networks.
        - xavier: Scales weights by sqrt(2/(fan_in + fan_out)). Good for symmetric activations.
    
    Example:
        >>> layer = Linear(fan_in=64, fan_out=128)
        >>> x = torch.randn(32, 64)
        >>> out = layer(x)  # shape: (32, 128)
    """
    
    fan_in: int
    fan_out: int
    bias: bool = True
    generator: Optional[torch.Generator] = torch.Generator().manual_seed(42)
    gain: Optional[str] = "kaiming"

    def __post_init__(self) -> None:
        """Initialize weights and biases based on the selected gain strategy."""
        if self.gain == "kaiming":
            self.scaling_factor = 1 / self.fan_in**0.5
        elif self.gain == "xavier":
            self.scaling_factor = (2 / (self.fan_in + self.fan_out))**0.5
        else:
            raise ValueError(f"gain {self.gain} is not supported. Use xavier or kaiming.")

        
        self.weight = torch.randn((self.fan_in, self.fan_out), generator=self.generator) * self.scaling_factor
        self.biases = torch.zeros(self.fan_out) if self.bias else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, fan_in).
        
        Returns:
            Output tensor of shape (batch_size, fan_out).
        """
        self.out = x @ self.weight
        if self.bias:
            self.out += self.biases
        return self.out
    
    def parameters(self) -> List[torch.Tensor]:
        """Return list of learnable parameters (weight and optionally bias)."""
        return [self.weight] + ([self.biases] if self.bias else [])
    

@dataclass
class BatchNorm1d:
    """
    1D Batch Normalization layer.
    
    Normalizes activations across the batch dimension, then applies learnable
    scale (gamma) and shift (beta) parameters. Maintains running statistics
    for inference.
    
    Attributes:
        dim: Number of features (channels) to normalize.
        eps: Small constant for numerical stability. Default 1e-5.
        momentum: Momentum for running statistics update. Default 0.001.
        training: Whether in training mode. Set to False for inference.
    
    Learnable Parameters:
        gamma: Scale parameter, initialized to ones.
        beta: Shift parameter, initialized to zeros.
    
    Running Statistics:
        running_mean: Exponential moving average of batch means.
        running_var: Exponential moving average of batch variances.
    
    Example:
        >>> bn = BatchNorm1d(dim=128)
        >>> x = torch.randn(32, 128)
        >>> out = bn(x)  # Normalized, shape: (32, 128)
        >>> bn.training = False  # Switch to eval mode
    """
    
    dim: int
    eps: float = 1e-5
    momentum: float = 0.001
    training: Optional[bool] = True

    def __post_init__(self):
        """Initialize gamma, beta, and running statistics."""
        self.gamma: torch.Tensor = torch.ones(self.dim)
        self.beta: torch.Tensor = torch.zeros(self.dim)

        self.running_mean = torch.zeros(1, self.dim)
        self.running_var  = torch.ones(1, self.dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with batch normalization.
        
        Args:
            x: Input tensor of shape (batch_size, dim) for 2D
               or (batch_size, seq_len, dim) for 3D.
        
        Returns:
            Normalized tensor of same shape as input.
        
        Raises:
            ValueError: If input is not 2D or 3D.
        """

        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            else:
                raise ValueError(f"x has dim={x.ndim}, expected 2 or 3")
            
            xmean, xvar = x.mean(dim, keepdim=True), x.var(dim, keepdim=True)
        else:
            xmean, xvar = self.running_mean, self.running_var

        xnorm = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xnorm + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self) -> List[torch.Tensor]:
        """Return list of learnable parameters (gamma and beta)."""
        return [self.gamma, self.beta]    
    

class Flatten:
    """
    Flattens all dimensions except the batch dimension.
    
    Transforms input of shape (batch_size, d1, d2, ...) to (batch_size, d1*d2*...).
    Commonly used to connect convolutional/embedding layers to fully connected layers.
    
    Example:
        >>> flatten = Flatten()
        >>> x = torch.randn(32, 8, 24)  # (batch, seq_len, embed_dim)
        >>> out = flatten(x)  # shape: (32, 192)
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, ...).
        
        Returns:
            Flattened tensor of shape (batch_size, -1).
        """
        self.out = x.view(x.shape[0], -1)
        return self.out
    
    def parameters(self) -> List[torch.Tensor]:
        """Return empty list (no learnable parameters)."""
        return []
    

@dataclass
class FlattenConsecutive:
    """
    Flattens consecutive elements along the sequence dimension.
    
    Used in WaveNet-style architectures to create hierarchical representations.
    Groups num_concat consecutive time steps and concatenates their features.
    
    Attributes:
        num_concat: Number of consecutive elements to concatenate.
    
    Shape transformation:
        Input: (batch, seq_len, channels)
        Output: (batch, seq_len // num_concat, channels * num_concat)
        
        If seq_len // num_concat == 1, the sequence dimension is squeezed,
        resulting in shape (batch, channels * num_concat).
    
    Example:
        >>> fc = FlattenConsecutive(num_concat=2)
        >>> x = torch.randn(32, 8, 24)  # 8 time steps
        >>> out = fc(x)  # shape: (32, 4, 48) — 4 time steps, doubled features
    """
    
    num_concat: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, channels).
               seq_len must be divisible by num_concat.
        
        Returns:
            Reshaped tensor with consecutive elements concatenated.
        """
        B, T, C = x.shape
        self.out = x.view(B, T // self.num_concat, C * self.num_concat)
        if self.out.shape[1] == 1:
            self.out = self.out.squeeze(1)   
        return self.out
    
    def parameters(self) -> List[torch.Tensor]:
        """Return empty list (no learnable parameters)."""
        return []
    

@dataclass
class Embedding:
    """
    Embedding layer that maps integer indices to dense vectors.
    
    Stores a learnable lookup table of shape (num_embeddings, dim_embeddings).
    Each input index retrieves its corresponding row from the table.
    
    Attributes:
        num_embeddings: Size of the vocabulary (number of unique tokens).
        dim_embeddings: Dimension of each embedding vector.
        generator: PyTorch random generator for reproducibility.
    
    Example:
        >>> emb = Embedding(num_embeddings=100, dim_embeddings=24)
        >>> indices = torch.tensor([[1, 5, 3], [2, 8, 0]])  # (batch=2, seq=3)
        >>> out = emb(indices)  # shape: (2, 3, 24)
    """
    
    num_embeddings: int
    dim_embeddings: int
    generator: Optional[torch.Generator] = torch.Generator().manual_seed(42)

    def __post_init__(self):
        """Initialize embedding weight matrix with random values."""
        self.weight = torch.randn((self.num_embeddings, self.dim_embeddings), generator=self.generator)

    def __call__(self, ix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (lookup).
        
        Args:
            ix: Integer tensor of indices, any shape.
        
        Returns:
            Tensor of shape (*ix.shape, dim_embeddings).
        """
        self.out = self.weight[ix]
        return self.out

    def parameters(self) -> List[torch.Tensor]:
        """Return list containing the embedding weight matrix."""
        return [self.weight]
    


class GatedActivation:
    """
    Gated activation unit from the WaveNet paper.
    
    Splits input channels in half and applies:
        output = tanh(first_half) * sigmoid(second_half)
    
    This gating mechanism allows the network to control information flow,
    with tanh providing the signal and sigmoid providing the gate.
    
    Reference:
        van den Oord et al., "WaveNet: A Generative Model for Raw Audio" (2016)
    
    Shape:
        Input: (..., 2C) where C is the desired output channels
        Output: (..., C)
    
    Example:
        >>> gate = GatedActivation()
        >>> x = torch.randn(32, 256)  # 256 = 2 * 128
        >>> out = gate(x)  # shape: (32, 128)
    """
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated activation.
        
        Args:
            x: Input tensor with even number of channels in last dimension.
        
        Returns:
            Gated output with half the channels.
        """
        C = x.shape[-1] // 2
        filter_part = x[..., :C]
        gate_part = x[..., C:]
        
        self.out = torch.tanh(filter_part) * torch.sigmoid(gate_part)
        return self.out
    
    def parameters(self) -> List[torch.Tensor]:
        """Return empty list (no learnable parameters)."""
        return []
    
    
class Tanh:
    """
    Hyperbolic tangent activation function.
    
    Applies element-wise: output = tanh(input)
    
    Output range: (-1, 1)
    
    Example:
        >>> act = Tanh()
        >>> x = torch.randn(32, 128)
        >>> out = act(x)  # values in (-1, 1)
    """

    def __call__(self, x) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of any shape.
        
        Returns:
            Tensor of same shape with tanh applied element-wise.
        """
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self) -> List[torch.Tensor]:
        """Return empty list (no learnable parameters)."""
        return []
    

@dataclass
class Sequential:
    """
    Container that chains layers sequentially.
    
    Passes input through each layer in order, feeding each layer's output
    as the next layer's input.
    
    Attributes:
        layers: Iterable of layer objects. Each must be callable and have
            a parameters() method.
    
    Example:
        >>> model = Sequential([
        ...     Embedding(100, 24),
        ...     Flatten(),
        ...     Linear(24 * 8, 128),
        ...     BatchNorm1d(128),
        ...     Tanh(),
        ...     Linear(128, 100),
        ... ])
        >>> x = torch.randint(0, 100, (32, 8))
        >>> logits = model(x)  # shape: (32, 100)
        >>> params = model.parameters()  # all learnable params
    """
    
    layers: Iterable

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all layers.
        
        Args:
            x: Input tensor.
        
        Returns:
            Output after passing through all layers sequentially.
        """
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self) -> List[torch.Tensor]:
        """Collect and return all parameters from all layers."""
        return [p for layer in self.layers for p in layer.parameters()]