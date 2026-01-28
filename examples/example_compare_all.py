import argparse
from pathlib import Path
import torch

from word_generator.dataset_creation import Dataset
from word_generator.generators.random import RandomGenerator
from word_generator.generators.bigram import BigramGenerator
from word_generator.generators.mlp import MLPGenerator
from word_generator.generators.wavenet import WaveNetGenerator


def main():
    parser = argparse.ArgumentParser(description="Compare all word generator models")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer steps")
    parser.add_argument("--data-path", type=str, default="data/francais_long.txt", help="Path to word list")
    args = parser.parse_args()
    
    mlp_steps = 5000 if args.quick else 50000
    wavenet_steps = 5000 if args.quick else 50000
    
    print("=" * 70)
    print("MODEL COMPARISON: Random vs Bigram vs MLP vs WaveNet")
    print("=" * 70)
    
    # Create datasets with different context lengths
    print("\n[1] Building datasets...")
    
    dataset_ctx1 = Dataset(
        path=Path(args.data_path),
        context_len=1,
        train_test_split=0.8,
    ).build()
    
    dataset_ctx8 = Dataset(
        path=Path(args.data_path),
        context_len=8,
        train_test_split=0.8,
    ).build()
    
    print(f"    Vocabulary size: {dataset_ctx1.voc_size}")
    print(f"    Number of words: {len(dataset_ctx1.wordsset)}")
    
    # Dictionary to store results
    results = {}
    
    # =========================================================================
    # Random Generator
    # =========================================================================
    print("\n" + "-" * 70)
    print("[2] RANDOM GENERATOR")
    print("-" * 70)
    
    random_gen = RandomGenerator(dataset=dataset_ctx1)
    random_gen.train()
    results['Random'] = {
        'loss': random_gen.get_test_loss(),
        'words': random_gen.generate(num_words=5, max_runs=100)
    }
    print(f"    Test loss: {results['Random']['loss']:.5f}")
    print(f"    Sample words: {', '.join(results['Random']['words'][:5])}")
    
    # =========================================================================
    # Bigram Generator
    # =========================================================================
    print("\n" + "-" * 70)
    print("[3] BIGRAM GENERATOR")
    print("-" * 70)
    
    bigram_gen = BigramGenerator(dataset=dataset_ctx1, verbose=False)
    bigram_gen.train()
    results['Bigram'] = {
        'loss': bigram_gen.get_test_loss(),
        'words': bigram_gen.generate(num_words=5, max_runs=100)
    }
    print(f"    Test loss: {results['Bigram']['loss']:.5f}")
    print(f"    Sample words: {', '.join(results['Bigram']['words'][:5])}")
    
    # =========================================================================
    # MLP Generator
    # =========================================================================
    print("\n" + "-" * 70)
    print(f"[4] MLP GENERATOR (training {mlp_steps} steps)")
    print("-" * 70)
    
    mlp_gen = MLPGenerator(
        dataset=dataset_ctx8,
        n_embd=24,
        n_hidden=128,
        num_mid_layers=2,
        verbose=False,
    )
    print(f"    Parameters: {sum(p.nelement() for p in mlp_gen.parameters)}")
    mlp_gen.train(max_steps=mlp_steps, batch_size=64, print_every=1000)
    results['MLP'] = {
        'loss': mlp_gen.get_test_loss(),
        'words': mlp_gen.generate(num_words=5, max_runs=100)
    }
    print(f"    Test loss: {results['MLP']['loss']:.5f}")
    print(f"    Sample words: {', '.join(results['MLP']['words'][:5])}")
    
    # =========================================================================
    # WaveNet Generator
    # =========================================================================
    print("\n" + "-" * 70)
    print(f"[5] WAVENET GENERATOR (training {wavenet_steps} steps)")
    print("-" * 70)
    
    wavenet_gen = WaveNetGenerator(
        dataset=dataset_ctx8,
        n_embd=24,
        n_hidden=128,
        num_concat=2,
        num_mid_layers=2,
        verbose=False,
    )
    print(f"    Parameters: {sum(p.nelement() for p in wavenet_gen.parameters)}")
    wavenet_gen.train(max_steps=wavenet_steps, batch_size=64, print_every=1000)
    results['WaveNet'] = {
        'loss': wavenet_gen.get_test_loss(),
        'words': wavenet_gen.generate(num_words=5, max_runs=100)
    }
    print(f"    Test loss: {results['WaveNet']['loss']:.5f}")
    print(f"    Sample words: {', '.join(results['WaveNet']['words'][:5])}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n    Model       | Test Loss | Improvement vs Random")
    print("    " + "-" * 50)
    
    random_loss = results['Random']['loss']
    
    for model_name in ['Random', 'Bigram', 'MLP', 'WaveNet']:
        loss = results[model_name]['loss']
        improvement = ((random_loss - loss) / random_loss) * 100
        improvement_str = f"{improvement:+.1f}%" if model_name != 'Random' else "baseline"
        print(f"    {model_name:11} | {loss:.5f}   | {improvement_str}")
    
    print("\n    Generated Words Sample:")
    print("    " + "-" * 50)
    for model_name in ['Random', 'Bigram', 'MLP', 'WaveNet']:
        words = results[model_name]['words'][:5]
        print(f"    {model_name:11} | {', '.join(words)}")
    
    print("\n" + "=" * 70)
    print("Lower loss = better model. Neural models (MLP, WaveNet) should")
    print("significantly outperform statistical models (Random, Bigram).")
    print("=" * 70)
    
    # Optional: Plot comparison
    try:
        import matplotlib.pyplot as plt
        
        models = list(results.keys())
        losses = [results[m]['loss'] for m in models]
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, losses, color=colors, edgecolor='black', linewidth=1.2)
        plt.ylabel('Test Loss (Cross-Entropy)', fontsize=12)
        plt.title('Model Comparison: Test Loss', fontsize=14)
        plt.ylim(0, max(losses) * 1.1)
        
        for bar, loss in zip(bars, losses):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{loss:.3f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"\n    Could not display plot: {e}")


if __name__ == "__main__":
    main()
