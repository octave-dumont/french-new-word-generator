from pathlib import Path

from word_generator.dataset_creation import Dataset
from word_generator.generators.random import RandomGenerator


def main():
    # Create dataset with context_len=1 (required for RandomGenerator)
    print("=" * 60)
    print("RANDOM GENERATOR EXAMPLE")
    print("=" * 60)
    
    print("\n[1] Building dataset...")
    dataset = Dataset(
        path=Path("data/francais_long.txt"),
        context_len=1,
        train_test_split=0.8,
    ).build()
    
    print(f"    Vocabulary size: {dataset.voc_size}")
    print(f"    Number of words: {len(dataset.wordsset)}")
    print(f"    Mean word length: {sum(len(w) for w in dataset.wordsset) / len(dataset.wordsset):.2f}")
    
    # Initialize generator
    print("\n[2] Initializing RandomGenerator...")
    generator = RandomGenerator(dataset=dataset)
    
    # Train (no-op for random generator)
    print("\n[3] Training (no training needed for random baseline)...")
    generator.train()
    
    # Get test loss
    print("\n[4] Evaluating...")
    test_loss = generator.get_test_loss()
    print(f"    Test loss (log(vocab_size)): {test_loss:.5f}")
    
    # Generate words
    print("\n[5] Generating 20 new words...")
    generated_words = generator.generate(num_words=20, max_runs=1000)
    
    print("\n    Generated words:")
    for i, word in enumerate(generated_words, 1):
        print(f"    {i:2d}. {word}")
    
    print("\n" + "=" * 60)
    print("Note: Random generation produces mostly gibberish.")
    print("This serves as a baseline - any real model should beat this loss.")
    print("=" * 60)


if __name__ == "__main__":
    main()
