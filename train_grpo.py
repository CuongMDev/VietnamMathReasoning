import random
import numpy
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_with_min_lr_schedule_with_warmup
from trl import GRPOTrainer, GRPOConfig
import os

from config import GRPO_CFG, INSTRUCTION_DATA_PATH, MODEL_CACHE_PATH
from utils import make_prompt_template
from reward import *

IGNORE_TOKEN_ID = -100
random.seed(42)
torch.manual_seed(42)

# --- Load tokenizer ---
MODEL_NAME = GRPO_CFG["model"]["name"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)
tokenizer.padding_side = 'left'

# --- Load pretrained model ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map='auto',
    cache_dir=MODEL_CACHE_PATH,
    # attn_implementation="flash_attention_3",
)

# --- LoRA ---
LORA_CONFIG = GRPO_CFG["lora"]
if LORA_CONFIG["using_lora"]:
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# --- Load dataset (pre-split by split_data.py) ---
train_dataset = load_dataset("json", data_files=INSTRUCTION_DATA_PATH + "train.json")["train"]
val_dataset = load_dataset("json", data_files=INSTRUCTION_DATA_PATH + "val.json")["train"]

# --- Combined reward function ---
REWARD_FUNCS_REGISTRY = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "reasoning_steps": reasoning_steps_reward,
    "cosine": get_cosine_scaled_reward(
        min_value_wrong=GRPO_CFG["grpo"]["cosine_min_value_wrong"],
        max_value_wrong=GRPO_CFG["grpo"]["cosine_max_value_wrong"],
        min_value_correct=GRPO_CFG["grpo"]["cosine_min_value_correct"],
        max_value_correct=GRPO_CFG["grpo"]["cosine_max_value_correct"],
        max_len=GRPO_CFG["grpo"]["cosine_max_len"],
    ),
    "repetition_penalty": get_repetition_penalty_reward(
        ngram_size=GRPO_CFG["grpo"]["repetition_n_grams"],
        max_penalty=GRPO_CFG["grpo"]["repetition_max_penalty"],
    ),
    "cosine_word": get_cosine_backtracking_scaled_reward(
        min_value_wrong=GRPO_CFG["grpo"]["cosine_min_value_wrong"],
        max_value_wrong=GRPO_CFG["grpo"]["cosine_max_value_wrong"],
        min_value_correct=GRPO_CFG["grpo"]["cosine_min_value_correct"],
        max_value_correct=GRPO_CFG["grpo"]["cosine_max_value_correct"],
        max_word=GRPO_CFG["grpo"]["cosine_max_word"],
    ),
    "length": len_reward,
}

reward_funcs = [
    REWARD_FUNCS_REGISTRY[name]
    for name in GRPO_CFG["grpo"]["reward_funcs"].keys()
]

reward_weights = list(GRPO_CFG["grpo"]["reward_funcs"].values())

def format_example(example):
    # --- 1️⃣ Tạo messages theo template ---
    messages = make_prompt_template(example['problem'])
    answer = example['answer']

    return {
        "prompt": messages,
        "solution": answer
    }

train_tokenized_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
val_tokenized_dataset = val_dataset.map(format_example, remove_columns=val_dataset.column_names)

# --- GRPO Config ---
OUTPUT_DIR = GRPO_CFG["training"]["output_dir"]
grpo_cfg = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=GRPO_CFG["training"]["epochs"],
    per_device_train_batch_size=GRPO_CFG["training"]["batch_size"],
    per_device_eval_batch_size=GRPO_CFG["training"]["batch_size"],
    gradient_accumulation_steps=GRPO_CFG["training"]["gradient_accumulation_steps"],
    learning_rate=float(GRPO_CFG["training"]["learning_rate"]),
    max_completion_length=GRPO_CFG["grpo"]["max_completion_length"],
    num_generations=GRPO_CFG["grpo"]["num_generations"],  # số generation cho mỗi prompt
    temperature=GRPO_CFG["grpo"]["temperature"],
    top_p=GRPO_CFG["grpo"]["top_p"],
    warmup_ratio=GRPO_CFG["training"]["warmup_ratio"],
    logging_steps=GRPO_CFG["training"]["logging_steps"],
    eval_strategy="steps",
    eval_steps=GRPO_CFG["training"]["eval_steps"],
    save_steps=GRPO_CFG["training"]["save_steps"],
    save_total_limit=GRPO_CFG["training"]["save_total_limit"],
    bf16=True,
    tf32=True,
    load_best_model_at_end=True,
    reward_weights=reward_weights,
    # chat_template_kwargs={"enable_thinking": False}
)

# --- GRPO Trainer ---
trainer = GRPOTrainer(
    model=model,
    args=grpo_cfg,
    train_dataset=train_tokenized_dataset,
    eval_dataset=val_tokenized_dataset,
    reward_funcs=reward_funcs,
    processing_class=tokenizer
)

# --- Scheduler ---
step_in_epoch = trainer.get_train_dataloader().__len__()
num_training_steps = (step_in_epoch + grpo_cfg.gradient_accumulation_steps - 1) // grpo_cfg.gradient_accumulation_steps * grpo_cfg.num_train_epochs
trainer.create_optimizer()
scheduler = get_cosine_with_min_lr_schedule_with_warmup(
    trainer.optimizer,
    num_warmup_steps=grpo_cfg.warmup_steps,
    num_training_steps=num_training_steps,
    min_lr_rate=float(GRPO_CFG["training"]["min_lr_rate"])
)
trainer.lr_scheduler = scheduler

# checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint")]
# checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
# last_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
# print("Resuming from:", last_checkpoint)

# --- Train ---
trainer.train()

# --- Save model ---
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ GRPO training done, model saved at:", OUTPUT_DIR)
