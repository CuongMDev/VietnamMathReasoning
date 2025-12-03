import random
import re

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, get_cosine_with_min_lr_schedule_with_warmup, get_scheduler
from transformers.trainer_pt_utils import LabelSmoother

from config import INSTRUCTION_DATA_PATH, SFT_CFG, MODEL_CACHE_PATH
from mask import *
from utils import make_prompt_template

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

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)
tokenizer.padding_side = 'left'

# --- Load model ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,  # tiết kiệm VRAM
    device_map='cuda',
    attn_implementation="flash_attention_2",
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
dataset = load_dataset("json", data_files=INSTRUCTION_DATA_PATH+"train.json")["train"]
# Train/Val split ban đầu
dataset_split = dataset.train_test_split(test_size=VAL_RATIO, seed=42)
train_dataset = dataset_split["train"]
val_dataset = dataset_split["test"]

split_ratio = 0.7
# # Split dataset thành 2 phần
# train_no_response, train_full = train_dataset.train_test_split(test_size=1-split_ratio, seed=42).values()
# # Xoá column tương ứng
# train_no_response = train_no_response.remove_columns(["response"])
# # Ghép lại
# train_dataset = concatenate_datasets([train_no_response, train_full])
# # Shuffle
train_dataset = train_dataset.shuffle(seed=42)

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
        "labels": labels
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
    per_device_eval_batch_size=2*BATCH_SIZE,
    learning_rate=LR,
    weight_decay=float(SFT_CFG["training"]["weight_decay"]),
    bf16=True,
    tf32=True,
    gradient_accumulation_steps=SFT_CFG["training"]["gradient_accumulation_steps"],
    logging_steps=SFT_CFG["training"]["logging_steps"],
    eval_strategy="steps",
    eval_steps=SFT_CFG["training"]["eval_steps"],
    save_steps=SFT_CFG["training"]["save_steps"],
    save_total_limit=SFT_CFG["training"]["save_total_limit"],
    report_to="tensorboard",
    logging_dir="./logs",
    max_grad_norm=0.2,
    warmup_ratio=SFT_CFG["training"]["warmup_ratio"],
    load_best_model_at_end=True,
    label_names=["labels"],
)

# Thêm Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=IGNORE_TOKEN_ID, # Rất quan trọng!
    padding="longest",
    pad_to_multiple_of=8 # Tùy chọn, giúp tối ưu hóa trên GPU tensor core
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=val_tokenized_dataset,   # ✅ thêm tập validation
    data_collator=data_collator,
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

# --- Train ---
trainer.train()
trainer.evaluate()

# --- Lưu model LoRA ---
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Huấn luyện LoRA hoàn tất, model lưu tại:", OUTPUT_DIR)
