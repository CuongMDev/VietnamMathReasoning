"""
Plot the distribution of think lengths in test.json
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_think_lengths(json_path: str = "./data/test.json", output_path: str = "./think_len_distribution.png"):
    # Load data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract think lengths
    think_lens = []
    for item in data:
        think = item.get("think", "") or ""
        think_lens.append(len(think))

    think_lens = np.array(think_lens)

    # Statistics
    print(f"Total samples: {len(think_lens)}")
    print(f"Min think length: {think_lens.min()}")
    print(f"Max think length: {think_lens.max()}")
    print(f"Mean think length: {think_lens.mean():.2f}")
    print(f"Median think length: {np.median(think_lens):.2f}")
    print(f"Std think length: {think_lens.std():.2f}")

    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(think_lens, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Think Length (characters)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Think Lengths")
    axes[0].axvline(think_lens.mean(), color='red', linestyle='--', label=f'Mean: {think_lens.mean():.0f}')
    axes[0].axvline(np.median(think_lens), color='green', linestyle='--', label=f'Median: {np.median(think_lens):.0f}')
    axes[0].legend()

    # Sorted plot (cumulative view)
    sorted_lens = np.sort(think_lens)
    axes[1].plot(range(len(sorted_lens)), sorted_lens, linewidth=1)
    axes[1].set_xlabel("Sample Index (sorted by think length)")
    axes[1].set_ylabel("Think Length (characters)")
    axes[1].set_title("Think Lengths (Sorted)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    plot_think_lengths()
