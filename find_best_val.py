import os
import glob
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from benchmark import evaluate_dataset
from config import GRPO_CFG, MODEL_CACHE_PATH

OUTPUT_DIR = GRPO_CFG["training"]["output_dir"]
BASE_MODEL_NAME = GRPO_CFG["model"]["name"]
VAL_DATA = "data/val.json"

# Find all best_model_step_* directories
pattern = os.path.join(OUTPUT_DIR, "best_model_step_*")
model_dirs = sorted(glob.glob(pattern))

if not model_dirs:
    print(f"No best_model_step_* found in {OUTPUT_DIR}")
    exit(1)

print(f"Found {len(model_dirs)} checkpoints to evaluate:")
for d in model_dirs:
    print(f"  - {d}")

# Load base model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, cache_dir=MODEL_CACHE_PATH)
tokenizer.padding_side = "left"

results = []

for model_dir in model_dirs:
    step_name = os.path.basename(model_dir)
    print(f"\n{'='*60}")
    print(f"Evaluating: {step_name}")
    print(f"{'='*60}")

    # Load base model + LoRA adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=MODEL_CACHE_PATH,
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    acc, avg_tokens = evaluate_dataset(
        VAL_DATA,
        model,
        tokenizer,
        split="train",
        use_local_data=True,
    )

    results.append((step_name, acc, avg_tokens))
    print(f"{step_name}: accuracy={acc:.2f}%, avg_tokens={avg_tokens:.2f}")

    # Free memory
    del model, base_model
    torch.cuda.empty_cache()

# Find best
print(f"\n{'='*60}")
print("Results Summary:")
print(f"{'='*60}")
for name, acc, avg_tok in results:
    print(f"  {name}: {acc:.2f}% (avg tokens: {avg_tok:.2f})")

best_name, best_acc, best_avg_tok = max(results, key=lambda x: x[1])
best_dir = os.path.join(OUTPUT_DIR, best_name)
final_dir = os.path.join(OUTPUT_DIR, "best_model")

print(f"\nBest model: {best_name} with accuracy {best_acc:.2f}%")

# Copy best to best_model
if os.path.exists(final_dir):
    shutil.rmtree(final_dir)
shutil.copytree(best_dir, final_dir)
print(f"Best model copied to: {final_dir}")
