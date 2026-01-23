import random
import re

import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback, TrainingArguments, Trainer, DataCollatorForSeq2Seq, get_cosine_with_min_lr_schedule_with_warmup, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother

from config import INSTRUCTION_DATA_PATH, SFT_CFG, MODEL_CACHE_PATH
from mask import *
from sft_loss import compute_threshold_loss, plot_token_loss_to_file
from utils import make_prompt_template
from AccuracyEvalCallback import AccuracyEvalCallback

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
random.seed(42)

# --- Cấu hình ---
# Ví dụ: lấy các giá trị
MODEL_NAME: str = str(SFT_CFG["model"]["name"])

OUTPUT_DIR: str = str(SFT_CFG["training"]["output_dir"])
LR: float = float(SFT_CFG["training"]["learning_rate"])
MIN_LR_RATE: float = float(SFT_CFG["training"]["min_lr_rate"])
BATCH_SIZE: int = int(SFT_CFG["training"]["batch_size"])
EPOCHS: int = int(SFT_CFG["training"]["epochs"])

LORA_CONFIG: dict = dict(SFT_CFG["lora"])
TRAIN_PATH: str = str(SFT_CFG["dataset"]["train_path"])
VAL_RATIO: float = float(SFT_CFG["dataset"]["val_ratio"])
TEST_RATIO: float = float(SFT_CFG["dataset"]["test_ratio"])

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)
tokenizer.padding_side = 'left'

# --- Load model ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,  # tiết kiệm VRAM
    device_map='cuda',
    # attn_implementation="flash_attention_2",
    cache_dir=MODEL_CACHE_PATH,
)

# --- Thiết lập LoRA ---
if bool(LORA_CONFIG["using_lora"]):
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

# --- Load dataset ---
dataset = load_dataset("json", data_files=INSTRUCTION_DATA_PATH+TRAIN_PATH)["train"]

# Add think_len for sorting by difficulty
def add_think_len(example):
    think = example.get('think', '') or ''
    return {"think_len": len(think)}

dataset = dataset.map(add_think_len)

# Sort entire dataset by think_len (easy -> hard)
dataset = dataset.sort("think_len")

# Two-stage split: train/val_test, then val_test -> val/test
VAL_TEST_RATIO = VAL_RATIO + TEST_RATIO

n_total = len(dataset)
n_val_test = int(n_total * VAL_TEST_RATIO)
step = n_total / n_val_test

# Stage 1: Select val_test evenly spaced across difficulty
val_test_indices = [int(i * step) for i in range(n_val_test)]
train_indices = [i for i in range(n_total) if i not in set(val_test_indices)]

train_dataset = dataset.select(train_indices)
val_test_dataset = dataset.select(val_test_indices)  # sorted by difficulty

# Stage 2: Split val_test into val and test
# Val: evenly spaced, Test: remaining
n_val_test = len(val_test_dataset)
n_val = int(n_val_test * (VAL_RATIO / VAL_TEST_RATIO))

# Select val evenly spaced across val_test
val_step = n_val_test / n_val
val_indices = [int(i * val_step) for i in range(n_val)]
val_indices_set = set(val_indices)

# Test takes the rest
test_indices = [i for i in range(n_val_test) if i not in val_indices_set]

val_dataset = val_test_dataset.select(val_indices)
test_dataset = val_test_dataset.select(test_indices)

# Remove helper column
train_dataset = train_dataset.remove_columns(["think_len"])
val_dataset = val_dataset.remove_columns(["think_len"])
test_dataset = test_dataset.remove_columns(["think_len"])

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)} (evenly spaced by difficulty)")

# Save as JSON with proper formatting
import json
with open("data/val.json", "w", encoding="utf-8") as f:
    json.dump(list(val_dataset), f, ensure_ascii=False, indent=2)
with open("data/test.json", "w", encoding="utf-8") as f:
    json.dump(list(test_dataset), f, ensure_ascii=False, indent=2)

# split_ratio = 0.7
# # Tách dataset
# train_normal, train_flip = train_dataset.train_test_split(
#     test_size=1 - split_ratio,
#     seed=42
# ).values()
# val_normal, val_flip = val_dataset.train_test_split(
#     test_size=1 - split_ratio,
#     seed=42
# ).values()
# # Flip function (English prompt)
# def flip_example(example):
#     new_problem = f"Please generate a question that would have the following answer:\n\n{example['response']}"
#     new_response = example["problem"]
#     return {
#         "problem": new_problem,
#         "response": new_response,
#         "boxed_force": False
#     }
# # Áp dụng đổi chỗ
# train_flip = train_flip.map(flip_example)
# val_flip = val_flip.map(flip_example)
# # Ghép lại
# train_dataset = concatenate_datasets([train_normal, train_flip])
# val_dataset = concatenate_datasets([val_normal, val_flip])
# Shuffle cuối

# val_no_response = val_dataset.remove_columns(["response"])
# val_no_think = val_dataset.remove_columns(["think"])
# val_dataset = concatenate_datasets([val_no_response, val_no_think])
IGNORE_TOKEN_ID = -100

def format_example(example, mask=True):
    think = example.get('think')
    respond = example.get('response')

    # --- 1️⃣ Tạo messages ---
    messages = make_prompt_template(example['problem'], think=think, respond=respond, boxed_force=example['boxed_force'])

    # --- 2️⃣ Tạo prompt string ---
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
    if respond is None:
        prompt_text = prompt_text[:-len("\n<|im_end|>\n")]

    # --- 3️⃣ Tokenize toàn bộ prompt ---
    tokenized = tokenizer(prompt_text, truncation=False, add_special_tokens=False)
    input_ids = tokenized["input_ids"]

    think_token_id = tokenizer("<think>", add_special_tokens=False)["input_ids"][0]
    think_start_pos = input_ids.index(think_token_id)

    end_think_token_id = tokenizer("</think>", add_special_tokens=False)["input_ids"][0]
    end_think_pos = input_ids.index(end_think_token_id)

    # --- 4️⃣ Tạo labels ---
    end_masking_pos = think_start_pos + 1
    labels = [IGNORE_TOKEN_ID] * end_masking_pos + input_ids[end_masking_pos:]

    # --- 5️⃣ Mask : ---

    if mask:
        labels = mask_step_numbers(tokenizer, prompt_text, input_ids, labels)
        words_to_mask = ["Wait", "Alternatively", "But", "First", "Second", "Third", "first", "second", "third", "Then", "So", "Therefore", "Similarly", "Let", "Now", "Then"]
        labels = mask_words_after_double_newline(tokenizer, input_ids, labels, words_to_mask, mask_ratio=0.30)

    # --- Debug: decode labels, thay -100 bằng <mask> để check ---
    # decoded_labels = []
    # for id in labels:
    #     if id == -100:
    #         decoded_labels.append("<mask>")
    #     else:
    #         decoded_labels.append(tokenizer.decode([id], skip_special_tokens=False))

    # print("=== Labels đã decode (mask hiển thị bằng <mask>) ===\n")
    # print("".join(decoded_labels))

    return {
        "input_ids": input_ids,
        "labels": labels,
    }

train_tokenized_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
val_tokenized_dataset = val_dataset.map(format_example, remove_columns=val_dataset.column_names,
                                        fn_kwargs={
                                            "mask": False,
                                        })

# --- Training arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=float(SFT_CFG["training"]["weight_decay"]),
    bf16=True,
    gradient_accumulation_steps=SFT_CFG["training"]["gradient_accumulation_steps"],
    logging_steps=SFT_CFG["training"]["logging_steps"],
    eval_strategy="no",
    save_steps=SFT_CFG["training"]["save_steps"],
    save_total_limit=SFT_CFG["training"]["save_total_limit"],
    report_to="tensorboard",
    logging_dir="./logs",
    warmup_ratio=SFT_CFG["training"]["warmup_ratio"],
    label_names=["labels"],
)

# Thêm Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=IGNORE_TOKEN_ID,
    padding="longest",
    pad_to_multiple_of=8
)

# --- Callback for accuracy evaluation ---
accuracy_callback = AccuracyEvalCallback(
    val_dataset_raw=val_dataset,
    tokenizer=tokenizer,
    model=model,
    eval_steps=SFT_CFG["training"]["eval_steps"],
    max_new_tokens=3064,
    batch_size=20,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    data_collator=data_collator,
    callbacks=[accuracy_callback],
)

# --- Scheduler ---
step_in_epoch = trainer.get_train_dataloader().__len__()
num_training_steps = (step_in_epoch + training_args.gradient_accumulation_steps - 1) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
trainer.create_optimizer()
scheduler = get_cosine_with_min_lr_schedule_with_warmup(
    trainer.optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=num_training_steps,
    min_lr_rate=MIN_LR_RATE
)
trainer.lr_scheduler = scheduler

# plot_token_loss_to_file(trainer, model, tokenizer, loss_threshold=0.001)

# --- Train ---
trainer.train()

# --- Lưu model LoRA ---
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete, model saved at:", OUTPUT_DIR)
print(f"Best model saved at: {OUTPUT_DIR}/best_model (accuracy: {accuracy_callback.best_accuracy:.4f} at step {accuracy_callback.best_step})")
