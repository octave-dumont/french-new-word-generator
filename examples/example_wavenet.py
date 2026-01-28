import argparse
from pathlib import Path

from word_generator.dataset_creation import Dataset
from word_generator.generators.wavenet import WaveNetGenerator


def main():
    parser = argparse.ArgumentParser(description="Train WaveNet word generator")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer steps")
    parser.add_argument("--steps", type=int, default=130000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--context-len", type=int, default=8, help="Context window length (must be power of num_concat)")
    parser.add_argument("--n-embd", type=int, default=24, help="Embedding dimension")
    parser.add_argument("--n-hidden", type=int, default=125, help="Hidden layer size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of WaveNet blocks")
    parser.add_argument("--num-concat", type=int, default=2, help="Consecutive elements to flatten per layer")
    args = parser.parse_args()
    
    if args.quick:
        args.steps = 10000
        args.context_len = 8
    
    # Validate context_len is compatible with architecture
    required_context = args.num_concat ** (args.num_layers + 1)
    if args.context_len < required_context:
        print(f"Warning: context_len={args.context_len} may be too small for "
              f"{args.num_layers} layers with num_concat={args.num_concat}. "
              f"Recommended: {required_context}")
        args.context_len = required_context
    
    print("=" * 60)
    print("WAVENET GENERATOR EXAMPLE")
    print("=" * 60)
    
    # Create dataset
    print("\n[1] Building dataset...")
    dataset = Dataset(
        path=Path("data/francais_long.txt"),
        context_len=args.context_len,
        train_test_split=0.8,
    ).build()
    
    print(f"    Vocabulary size: {dataset.voc_size}")
    print(f"    Context length: {dataset.context_len}")
    print(f"    Number of words: {len(dataset.wordsset)}")
    print(f"    Training examples: {dataset.Xtr.shape[0]}")
    print(f"    Test examples: {dataset.Xte.shape[0]}")
    
    # Initialize generator
    print("\n[2] Initializing WaveNetGenerator...")
    print(f"    Embedding dim: {args.n_embd}")
    print(f"    Hidden size: {args.n_hidden}")
    print(f"    WaveNet blocks: {args.num_layers}")
    print(f"    Consecutive concat: {args.num_concat}")
    
    generator = WaveNetGenerator(
        dataset=dataset,
        n_embd=args.n_embd,
        n_hidden=args.n_hidden,
        num_concat=args.num_concat,
        num_mid_layers=args.num_layers,
        verbose=True,
    )
    
    # Print architecture summary
    print("\n    Architecture:")
    for i, layer in enumerate(generator.model.layers):
        print(f"      [{i}] {type(layer).__name__}")
    
    # Train
    print(f"\n[3] Training for {args.steps} steps (batch_size={args.batch_size})...")
    print("-" * 60)
    generator.train(
        max_steps=args.steps,
        batch_size=args.batch_size,
        print_every=50,
    )
    print("-" * 60)
    
    # Final evaluation
    print("\n[4] Final evaluation...")
    final_test_loss = generator.get_test_loss()
    print(f"    Final test loss: {final_test_loss:.5f}")
    
    # Compare to baselines
    import torch
    random_loss = float(torch.log(torch.tensor(dataset.voc_size)))
    print(f"    Random baseline: {random_loss:.5f}")
    print(f"    Improvement: {((random_loss - final_test_loss) / random_loss) * 100:.1f}%")
    
    # Generate words
    print("\n[5] Generating 20 new words...")
    generated_words = generator.generate(num_words=20, max_runs=1000)
    
    print("\n    Generated words:")
    for i, word in enumerate(generated_words, 1):
        print(f"    {i:2d}. {word}")
    
    # Plot losses
    print("\n[6] Plotting training curves...")
    try:
        generator.plot_losses()
    except Exception as e:
        print(f"    Could not display plot: {e}")
        print("    (Run in an environment with display support to see plots)")
    
    print("\n" + "=" * 60)
    print("WaveNet's hierarchical structure efficiently captures")
    print("long-range dependencies with gated activations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
