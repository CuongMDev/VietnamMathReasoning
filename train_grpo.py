import random
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_with_min_lr_schedule_with_warmup
from trl import GRPOTrainer, GRPOConfig

from config import GRPO_CFG, MODEL_CACHE_PATH
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
    "sft-cot-model",
    dtype=torch.bfloat16,
    device_map='auto',
    cache_dir=MODEL_CACHE_PATH,
    attn_implementation="flash_attention_2",
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

VAL_RATIO: float = float(GRPO_CFG["dataset"]["val_ratio"])

# --- Load dataset ---
dataset = load_dataset("json", data_files=GRPO_CFG["dataset"]["train_path"])["train"]
# Train/Val split ban đầu
dataset_split = dataset.train_test_split(test_size=VAL_RATIO, seed=42)
train_dataset = dataset_split["train"]
val_dataset = dataset_split["test"]

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
    "length": len_reward,
    "tag_count": tag_count_reward,
}

reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in GRPO_CFG["grpo"]["reward_funcs"]]

def format_example(example):
    # --- 1️⃣ Tạo messages theo template ---
    messages = make_prompt_template(example['problem'])
    answer = example['answer']

    return {
        "prompt": messages,
        "solution": f'${answer}$'
    }

train_tokenized_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
val_tokenized_dataset = val_dataset.map(format_example, remove_columns=dataset.column_names)

# --- GRPO Config ---
OUTPUT_DIR = GRPO_CFG["training"]["output_dir"]
grpo_cfg = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=GRPO_CFG["training"]["epochs"],
    per_device_train_batch_size=GRPO_CFG["training"]["batch_size"],
    gradient_accumulation_steps=GRPO_CFG["training"]["gradient_accumulation_steps"],
    learning_rate=float(GRPO_CFG["training"]["learning_rate"]),
    max_prompt_length=GRPO_CFG["grpo"]["max_prompt_length"],
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
    bf16=torch.cuda.is_available(),
    load_best_model_at_end=True
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

# --- Train ---
trainer.train()

# --- Save model ---
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ GRPO training done, model saved at:", OUTPUT_DIR)
