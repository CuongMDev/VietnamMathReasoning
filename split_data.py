"""
Split data into train/val/test/grpo sets.
- Val/Test: unique questions only (no duplicates)
- Train: remaining samples (can include duplicates)
- GRPO: sampled with log distribution, prioritizing longer think
"""

import json
import os
import numpy as np
from datasets import load_dataset

from config import INSTRUCTION_DATA_PATH, SFT_CFG

VAL_RATIO: float = float(SFT_CFG["dataset"]["val_ratio"])
TEST_RATIO: float = float(SFT_CFG["dataset"]["test_ratio"])
GRPO_RATIO: float = float(SFT_CFG["dataset"]["grpo_ratio"])

np.random.seed(42)


def split_data(
    input_path: str,
    output_dir: str,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    grpo_ratio: float = GRPO_RATIO
):
    """
    Split data into train/val/test/grpo sets.
    - Val/Test: unique questions (filter duplicates), evenly spaced by difficulty
    - Train: remaining samples (can include duplicates)
    - GRPO: sampled with log distribution, prioritizing longer think
    """
    # Load dataset
    dataset = load_dataset("json", data_files=input_path)["train"]
    print(f"📊 Total samples: {len(dataset)}")

    # Add think_len for sorting by difficulty
    def add_think_len(example):
        think = example.get('think', '') or ''
        return {"think_len": len(think)}

    dataset = dataset.map(add_think_len)

    # Sort by difficulty (easy -> hard)
    dataset = dataset.sort("think_len")

    # Step 1: Find unique questions and their first occurrence index
    seen_problems = {}  # problem -> first index
    for i, example in enumerate(dataset):
        problem = example.get('problem', '').strip().lower()
        if problem not in seen_problems:
            seen_problems[problem] = i

    unique_indices = list(seen_problems.values())
    print(f"📊 Unique questions: {len(unique_indices)}")

    # Step 2: Select val/test from unique questions (evenly spaced by difficulty)
    unique_indices_sorted = sorted(unique_indices)  # keep difficulty order

    VAL_TEST_RATIO = val_ratio + test_ratio
    n_unique = len(unique_indices_sorted)
    n_val_test = int(n_unique * VAL_TEST_RATIO)

    if n_val_test > 0:
        step = n_unique / n_val_test
        val_test_unique_positions = [int(i * step) for i in range(n_val_test)]
        val_test_indices = [unique_indices_sorted[pos] for pos in val_test_unique_positions]
    else:
        val_test_indices = []

    # Step 3: Split val_test into val and test
    n_val_test = len(val_test_indices)
    n_val = int(n_val_test * (val_ratio / VAL_TEST_RATIO)) if VAL_TEST_RATIO > 0 else 0

    if n_val > 0:
        val_step = n_val_test / n_val
        val_positions = [int(i * val_step) for i in range(n_val)]
        val_indices = [val_test_indices[pos] for pos in val_positions]
    else:
        val_indices = []

    val_indices_set = set(val_indices)
    test_indices = [idx for idx in val_test_indices if idx not in val_indices_set]

    # Step 4: Get val/test problems for filtering train
    val_problems = set()
    test_problems = set()
    for idx in val_indices:
        val_problems.add(dataset[idx]['problem'].strip().lower())
    for idx in test_indices:
        test_problems.add(dataset[idx]['problem'].strip().lower())

    # Step 5: Get unique train indices (not in val/test)
    train_unique_indices = []
    train_seen = set()
    for i, example in enumerate(dataset):
        problem = example.get('problem', '').strip().lower()
        if problem not in val_problems and problem not in test_problems:
            if problem not in train_seen:
                train_seen.add(problem)
                train_unique_indices.append(i)

    # Step 6: GRPO sampling with log distribution (prioritize longer think)
    n_grpo = int(len(train_unique_indices) * grpo_ratio)
    if n_grpo > 0:
        think_lens = np.array([dataset[i]['think_len'] for i in train_unique_indices], dtype=float)
        log_weights = np.log1p(think_lens)
        probs = log_weights / log_weights.sum()
        grpo_positions = np.random.choice(
            len(train_unique_indices),
            size=min(n_grpo, len(train_unique_indices)),
            replace=False,
            p=probs
        )
        grpo_indices = [train_unique_indices[pos] for pos in grpo_positions]
    else:
        grpo_indices = []

    # Step 7: Get GRPO problems, then Train = all samples NOT in val/test/grpo
    grpo_problems = set()
    for idx in grpo_indices:
        grpo_problems.add(dataset[idx]['problem'].strip().lower())

    train_indices = []
    for i, example in enumerate(dataset):
        problem = example.get('problem', '').strip().lower()
        if problem not in val_problems and problem not in test_problems and problem not in grpo_problems:
            train_indices.append(i)

    # Select datasets
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices) if val_indices else []
    test_dataset = dataset.select(test_indices) if test_indices else []
    grpo_dataset = dataset.select(grpo_indices) if grpo_indices else []

    # Remove helper column
    train_dataset = train_dataset.remove_columns(["think_len"])
    if val_dataset:
        val_dataset = val_dataset.remove_columns(["think_len"])
    if test_dataset:
        test_dataset = test_dataset.remove_columns(["think_len"])
    if grpo_dataset:
        grpo_dataset = grpo_dataset.remove_columns(["think_len"])

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}, GRPO: {len(grpo_dataset)}")

    # Save as JSON
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(list(train_dataset), f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(train_dataset)} train samples to {train_path}")

    if val_dataset:
        val_path = os.path.join(output_dir, "val.json")
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(list(val_dataset), f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(val_dataset)} val samples to {val_path}")

    if test_dataset:
        test_path = os.path.join(output_dir, "test.json")
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(list(test_dataset), f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(test_dataset)} test samples to {test_path}")

    if grpo_dataset:
        grpo_path = os.path.join(output_dir, "grpo.json")
        with open(grpo_path, "w", encoding="utf-8") as f:
            json.dump(list(grpo_dataset), f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(grpo_dataset)} GRPO samples to {grpo_path} (log-weighted, longer think prioritized)")

    return train_dataset, val_dataset, test_dataset, grpo_dataset


if __name__ == "__main__":
    input_file = os.path.join(INSTRUCTION_DATA_PATH, "calculus_data.json")
    split_data(input_file, INSTRUCTION_DATA_PATH)
