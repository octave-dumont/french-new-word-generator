import argparse
from pathlib import Path

from word_generator.dataset_creation import Dataset
from word_generator.generators.mlp import MLPGenerator


def main():
    parser = argparse.ArgumentParser(description="Train MLP word generator")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer steps")
    parser.add_argument("--steps", type=int, default=125000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--context-len", type=int, default=8, help="Context window length")
    parser.add_argument("--n-embd", type=int, default=24, help="Embedding dimension")
    parser.add_argument("--n-hidden", type=int, default=125, help="Hidden layer size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of hidden layers")
    args = parser.parse_args()
    
    if args.quick:
        args.steps = 10000
        args.context_len = 4
    
    print("=" * 60)
    print("MLP GENERATOR EXAMPLE")
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
    print("\n[2] Initializing MLPGenerator...")
    print(f"    Embedding dim: {args.n_embd}")
    print(f"    Hidden size: {args.n_hidden}")
    print(f"    Hidden layers: {args.num_layers}")
    
    generator = MLPGenerator(
        dataset=dataset,
        n_embd=args.n_embd,
        n_hidden=args.n_hidden,
        num_mid_layers=args.num_layers,
        verbose=True,
    )
    
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
    print("MLP models can learn multi-character patterns within the")
    print("context window. Increase context_len for longer dependencies.")
    print("=" * 60)


if __name__ == "__main__":
    main()
