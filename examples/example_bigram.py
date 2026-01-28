from pathlib import Path

from word_generator.dataset_creation import Dataset
from word_generator.generators import BigramGenerator


def main():
    # Create dataset with context_len=1 (required for BigramGenerator)
    print("=" * 60)
    print("BIGRAM GENERATOR EXAMPLE")
    print("=" * 60)
    
    print("\n[1] Building dataset...")
    dataset = Dataset(
        path=Path("data/francais_long.txt"),
        context_len=1,
        train_test_split=0.8,
    ).build()
    
    print(f"    Vocabulary size: {dataset.voc_size}")
    print(f"    Number of words: {len(dataset.wordsset)}")
    print(f"    Training examples: {dataset.Xtr.shape[0]}")
    print(f"    Test examples: {dataset.Xte.shape[0]}")
    
    # Initialize generator
    print("\n[2] Initializing BigramGenerator...")
    generator = BigramGenerator(dataset=dataset, verbose=True)
    
    # Train - builds frequency and probability matrices
    print("\n[3] Training (building transition probabilities)...")
    generator.train()
    
    print(f"    Frequency matrix shape: {generator.frequency_mat.shape}")
    print(f"    Probability matrix shape: {generator.probability_mat.shape}")
    
    # Get test loss
    print("\n[4] Evaluating...")
    test_loss = generator.get_test_loss()
    print(f"    Test loss: {test_loss:.5f}")
    
    # Compare to random baseline
    import torch
    random_loss = float(torch.log(torch.tensor(dataset.voc_size)))
    improvement = ((random_loss - test_loss) / random_loss) * 100
    print(f"    Random baseline loss: {random_loss:.5f}")
    print(f"    Improvement over random: {improvement:.1f}%")
    
    # Generate words
    print("\n[5] Generating 20 new words...")
    generated_words = generator.generate(num_words=20, max_runs=1000)
    
    print("\n    Generated words:")
    for i, word in enumerate(generated_words, 1):
        print(f"    {i:2d}. {word}")
    
    # Optional: Plot comparison
    print("\n[6] Plotting loss comparison...")
    try:
        generator.plot_losses()
    except Exception as e:
        print(f"    Could not display plot: {e}")
        print("    (Run in an environment with display support to see plots)")
    
    print("\n" + "=" * 60)
    print("Bigram models capture simple character patterns but lack")
    print("longer-range dependencies. Try MLP or WaveNet for better results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
