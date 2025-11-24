"""
GRPO Training Script (TRL 0.25.1)
Install:
    pip install trl transformers torch datasets peft accelerate bitsandbytes
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

from config import MODEL_NAME, INSTRUCTION_DATA_PATH, MODEL_CACHE_PATH, PROMPT_TEMPLATE
from reward import (
    compute_reward,
    simple_reward,
    length_and_quality_reward,
    token_efficiency_reward,
    coding_task_reward,
)

# =====================================================
# CONFIG
# =====================================================
OUTPUT_DIR = "./grpo-lora-model"
MAX_LENGTH = 512
MAX_NEW_TOKENS = 256
LR = 5e-6
BATCH_SIZE = 2
EPOCHS = 2
GROUP_SIZE = 4  # số sample mỗi prompt

REWARD_FUNC = compute_reward  # chọn reward ở đây

print("=" * 60)
print("GRPO Training Pipeline - TRL 0.25.1")
print("=" * 60)

# =====================================================
# TOKENIZER
# =====================================================
print("\n[1/6] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer loaded: {MODEL_NAME}")

# =====================================================
# MODEL with 8-bit
# =====================================================
print("\n[2/6] Loading model (8-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    cache_dir=MODEL_CACHE_PATH,
    device_map="auto",
)
print("✓ Model loaded")

# =====================================================
# LoRA
# =====================================================
print("\n[3/6] Applying LoRA...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj",
        "up_proj", "down_proj"
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =====================================================
# DATASET
# =====================================================
print("\n[4/6] Loading dataset...")
dataset = load_dataset("json", data_files=INSTRUCTION_DATA_PATH + "train_grpo.json")["train"]

def convert(example):
    """Bắt buộc phải trả về cột 'prompt' cho TRL 0.25.1"""
    return {
        "prompt": PROMPT_TEMPLATE.format(question=example["problem"])
    }

train_dataset = dataset.map(convert, remove_columns=dataset.column_names)
print("✓ Dataset loaded:", len(train_dataset))

# =====================================================
# GRPO CONFIG (MỚI CHUẨN TRL 0.25)
# =====================================================
print("\n[5/6] Building GRPO Config...")

training_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LR,

    num_generations=GROUP_SIZE,       # số sample mỗi prompt
    max_prompt_length=MAX_LENGTH,
    max_completion_length=MAX_NEW_TOKENS,

    temperature=0.7,
    top_p=0.9,

    beta=0.1,                         # KL penalty

    warmup_ratio=0.03,
    gradient_checkpointing=True,
    bf16=torch.cuda.is_available(),
    logging_steps=10,

    save_steps=50,
    save_total_limit=2,
    logging_dir="./logs",
    #report_to="tensorboard",

    lr_scheduler_type="cosine",
)

print("✓ Config ready")

# =====================================================
# TRAINER
# =====================================================
print("\n[6/6] Initializing GRPOTrainer...")

trainer = GRPOTrainer(
    model=model,
    args=training_config,
    train_dataset=train_dataset,
    reward_funcs=[REWARD_FUNC],  # phải là lis
)

  # phải là list
# =====================================================
# TRAIN
# =====================================================
print("\n🚀 Start training...")
print("-" * 60)

trainer.train()

# =====================================================
# SAVE
# =====================================================
print("\nSaving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("============================================================")
print("🎉 TRAINING DONE — MODEL SAVED!")
print("Path:", OUTPUT_DIR)
print("============================================================")
