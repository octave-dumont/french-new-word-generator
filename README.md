# Word Generator

A character-level language modeling library for **generating plausible words in French language**. Implements multiple architectures from simple statistical baselines to neural networks with gated activations.

By **changing the initial wordsset**, you may generate examples similar to any type of word (e.g., medical taxonomy) instead of just french words. This however could imply reconsidering the way the dataset is devised.

Built from scratch using PyTorch tensors with custom layer implementations for educational transparency. Based on Andrej Karpathy's tutorials (makemore series), with additions based on a deeper reading of the original [WaveNet](https://arxiv.org/abs/1609.03499) paper.

## Overview

This project provides four progressively sophisticated word generators:

| Model | Type | Context | Description |
|-------|------|---------|-------------|
| **Random** | Baseline | — | Uniform sampling over vocabulary |
| **Bigram** | Statistical | 1 char | Learns P(next \| current) transition probabilities |
| **MLP** | Neural | N chars | Multi-layer perceptron with embeddings |
| **WaveNet** | Neural | N chars | Hierarchical architecture with gated activations |

## Installation
```bash
git clone https://github.com/yourusername/word_generator.git
cd word_generator
pip install .
```

For development (includes testing and linting tools):
```bash
pip install -e ".[dev]"
```

## Repository Structure

```
word_generator/
├── __init__.py              # Package exports
├── dataset_creation.py      # Dataset loading, filtering, and splitting
├── layers.py                # Custom neural network layer implementations
└── generators/
    ├── __init__.py
    ├── random.py            # Uniform random baseline
    ├── bigram.py            # Character bigram model
    ├── mlp.py               # Multi-layer perceptron
    └── wavenet.py           # WaveNet-inspired architecture

examples/
├── example_random.py        # Random baseline demo
├── example_bigram.py        # Bigram model demo
├── example_mlp.py           # MLP training demo
├── example_wavenet.py       # WaveNet training demo
└── example_compare_all.py   # Side-by-side model comparison

data/
└── francais_long.txt        # Word list (one word per line)
```

## Dataset

### Building a Dataset

```python
from pathlib import Path
from word_generator import Dataset

dataset = Dataset(
    path=Path("data/francais_long.txt"),
    context_len=8,           # Number of previous characters as context
    train_test_split=0.8,    # 80% train, 20% test/dev
    freq_threshold=5e-3,     # Character frequency cutoff
).build()
```

### Special Tokens

The vocabulary includes two special tokens that mark word boundaries:

| Token | Index | Purpose |
|-------|-------|---------|
| `<BOW>` | 0 | **Beginning of Word**: Pads the initial context |
| `<EOW>` | 1 | **End of Word**: Signals generation termination |

During training, each word is transformed into context-target pairs. For example, with `context_len=3`, the word `"chat"` becomes:

```
Context         → Target
[<BOW>, <BOW>, <BOW>] → 'c'
[<BOW>, <BOW>, 'c']   → 'h'
[<BOW>, 'c', 'h']     → 'a'
['c', 'h', 'a']       → 't'
['h', 'a', 't']       → <EOW>
```

### Frequency Threshold Filtering

The `freq_threshold` parameter controls vocabulary pruning. Characters are scored by:

```
score(char) = voc_size × (count of words containing char) / (total words)
```

Characters with `score < freq_threshold` are considered rare. All words containing these rare characters are removed from the dataset, and the vocabulary is rebuilt. This prevents the model from wasting capacity on extremely infrequent characters (accidental typos, unusual diacritics, etc.) while keeping meaningful rare letters.

**Default:** `freq_threshold=5e-3` Removes characters appearing in fewer than ~0.05% of words (scaled by vocabulary size).

## Models

### Random Generator (Baseline)

Samples characters uniformly at random. Provides a theoretical lower bound:

```
loss = log(voc_size) ≈ 3.3 for 27 characters
```

Any learned model should beat this baseline.

### Bigram Generator

A simple statistical model that learns transition probabilities $P(\text{next\ char} | \text{current\ char})$ from character co-occurrence counts. Uses Laplace smoothing (+1) to handle unseen bigrams.

```python
from word_generator import BigramGenerator

model = BigramGenerator(dataset=dataset)
model.train()  # Builds frequency/probability matrices
print(f"Test loss: {model.get_test_loss():.4f}")
```

### MLP Generator

A multi-layer perceptron that predicts the next character from a fixed-size context window.

**Architecture:**
```
Embedding → Flatten → [Linear → BatchNorm → Tanh] × N → Linear → Logits
```

```python
from word_generator import MLPGenerator

model = MLPGenerator(
    dataset=dataset,
    n_embd=24,           # Embedding dimension
    n_hidden=128,        # Hidden layer width
    num_mid_layers=2,    # Number of hidden blocks
)
model.train(max_steps=125000, batch_size=64)
```

### WaveNet Generator

A hierarchical architecture inspired by [WaveNet](https://arxiv.org/abs/1609.03499) that uses **gated activations** for controlled information flow.

**Architecture:**
```
Embedding → [FlattenConsecutive → Linear → BatchNorm → GatedActivation] × N → Linear → Logits
```

**Gated Activation** (from the WaveNet paper):
```
output = tanh(W_f * x) ⊙ sigmoid(W_g * x)
```
The tanh branch provides the signal while the sigmoid branch acts as a gate, allowing the network to learn what information to pass through.

**Hierarchical Structure:** Each `FlattenConsecutive` layer merges `num_concat` consecutive positions, creating an exponentially growing receptive field:
- With `num_concat=2` and 3 layers: receptive field = 2³ = 8 characters

```python
from word_generator import WaveNetGenerator

model = WaveNetGenerator(
    dataset=dataset,
    n_embd=24,
    n_hidden=128,
    num_concat=2,        # Merge 2 consecutive positions per layer
    num_mid_layers=2,
)
model.train(max_steps=130000, batch_size=64)
```

## Training Details

### Simplified Gradient Descent

This implementation uses **manual SGD** rather than PyTorch optimizers, making the training loop fully transparent:

```python
# Forward pass
logits = model(Xb)
loss = F.cross_entropy(logits, Yb)

# Backward pass
for p in parameters:
    p.grad = None
loss.backward()

# Parameter update (vanilla SGD)
with torch.no_grad():
    for p in parameters:
        p.data -= lr * p.grad
```

This explicit implementation is intentional for educational purposes: every step of gradient descent is visible.

### Learning Rate Schedule

Both MLP and WaveNet use a simple step decay schedule:

| Model | Phase 1 | Phase 2 |
|-------|---------|---------|
| MLP | lr=0.1 for steps 0–100k | lr=0.01 for steps 100k+ |
| WaveNet | lr=0.1 for steps 0–90k | lr=0.01 for steps 90k+ |

```python
# Default MLP schedule
lr_schedule = lambda i: 0.1 if i < 100000 else 0.01
```

### Weight Initialization and Scaling Factor

The Linear layers use **Kaiming initialization** by default:

```python
weight = randn(fan_in, fan_out) * (1 / sqrt(fan_in))
```

Additionally, all Linear weights are scaled by a `scaling_factor` (default: **1.1**) after initialization to prevent gradients from vanishing through deep networks with Tanh activations.

**The Scaling Factor Debate:**

There's discussion in the community about the optimal scaling factor for Tanh networks:

| Value | Source | Rationale |
|-------|--------|-----------|
| **1.0** | Standard | No activation |
| **5/3 ≈ 1.67** | Kaiming for Tanh | Accounts for Tanh's bounded output, PyTorch default |
| **1.59253742** | Mathematical derivation | $\sqrt{\frac{1}{V(\tanh(X))}}$ for $X\sim N(0, 1)$ |
| **1.1** | [PyTorch forums](https://discuss.pytorch.org/t/calculate-gain-tanh/20854/3) | Empirically found to work well (including here) |

The "correct" value depends on network depth, batch normalization placement, and other architectural choices. The value 1.1 was chosen through hyperparameter tuning on this specific architecture.

**Final Layer Scaling:** The output layer weights are scaled down by `reduce_last_conf=0.1` to produce lower-confidence predictions, preventing early training instability.

## Quick Start

### Compare All Models

```bash
python examples/example_compare_all.py --quick
```

Output:
```
====================================================================
MODEL COMPARISON: Random vs Bigram vs MLP vs WaveNet
====================================================================

    Model       | Test Loss | Improvement vs Random
    --------------------------------------------------
    Random      | 3.29584   | baseline
    Bigram      | 2.45321   | +25.6%
    MLP         | 2.01234   | +38.9%
    WaveNet     | 1.95678   | +40.6%
```

### Generate Words

```python
from word_generator import Dataset, WaveNetGenerator
from pathlib import Path

# Build dataset
dataset = Dataset(path=Path("data/francais_long.txt"), context_len=8).build()

# Train model
model = WaveNetGenerator(dataset=dataset)
model.train(max_steps=50000)

# Generate new words
words = model.generate(num_words=10)
print(words)
# ['beaumont', 'clavière', 'tournelle', 'grandet', 'valmoir', ...]
```

## Custom Layers

All neural network layers are implemented from scratch in `layers.py`:

| Layer | Description |
|-------|-------------|
| `Linear` | Fully connected layer with Kaiming/Xavier init |
| `BatchNorm1d` | Batch normalization with running statistics |
| `Embedding` | Learnable lookup table |
| `Flatten` | Reshape to (batch, -1) |
| `FlattenConsecutive` | Merge N consecutive time steps |
| `GatedActivation` | tanh(x₁) ⊙ sigmoid(x₂) from WaveNet |
| `Tanh` | Hyperbolic tangent activation |
| `Sequential` | Layer container |

## Acknowledgments

This project is inspired by [Andrej Karpathy's makemore series](https://github.com/karpathy/makemore), which provides an excellent introduction to character-level language modeling. The core concepts follow his pedagogical approach.

**Key modifications from the original:**
- Tuned hyperparameters (embedding dimensions, hidden sizes, learning rate schedules)
- Added **Gated Activations** from the WaveNet paper for the hierarchical model
- Custom `FlattenConsecutive` layer for efficient receptive field expansion
- Pydantic-based configuration for cleaner model initialization

## References

- **WaveNet:** van den Oord, A., et al. (2016). [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499). *arXiv:1609.03499*
- **Makemore:** Karpathy, A. (2022). [makemore](https://github.com/karpathy/makemore). GitHub.
- **Kaiming Initialization:** He, K., et al. (2015). [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852). *ICCV 2015*.

## License

MIT License
