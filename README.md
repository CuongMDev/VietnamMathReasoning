### LLM fine-tuning for mathematics with SFT and GRPO training.

## File Description

**Training:**
- `train_sft.py` - Supervised fine-tuning with LoRA
- `train_grpo.py` - GRPO reinforcement learning
- `train_slurm.sh` - SLURM job submission

**Inference & Evaluation:**
- `inference.py` - Run with vLLM (interactive)
- `benchmark.py` - Test on benchmark datasets
- `check_answer.py` - Verify answers

**Data Processing:**
- `data/` - Training datasets (train, val, test, grpo)
- `build_data.py` - Build dataset
- `split_data.py` - Split train/val/test
- `filter.py` - Filter data

**Core Modules:**
- `utils.py` - Utility functions
- `mask.py` - Token masking
- `sft_loss.py` - Custom loss
- `reward.py` - Reward computation
- `Callback.py` - Training callbacks

**Analysis:**
- `plot_think_len.py` - Analyze thinking tokens
- `find_best_val.py` - Find best checkpoint
- `dedup_formula.py` - Deduplicate formulas

**Config & Models:**
- `config.yml`, `grpo_config.yml` - Hyperparameters
- `best_model/` - Fine-tuned LoRA adapter
- `hf_cache/` - Model/dataset cache

## Usage

### 1. SFT Training
```bash
python train_sft.py
```

### 2. GRPO Training
```bash
python train_grpo.py
```

### 3. Inference
```bash
python inference.py
```
Interactive mode - Edit `prompt.txt` → Results in `result.txt`

### 4. Benchmark
```bash
python benchmark.py
```
Results → `benchmark_results.csv`

### 5. Data Processing
```bash
python build_data.py    # Build dataset
python split_data.py    # Split train/val/test
python filter.py        # Filter data
```

## Config

Edit `config.yml` and `grpo_config.yml` before training.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Out of Memory | Reduce batch_size |
| Slow loading | Cache in `hf_cache/` |
| Poor quality | Check config, epochs, dataset |

---

**Updated:** Feb 2026