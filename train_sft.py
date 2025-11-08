import random

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers.trainer_pt_utils import LabelSmoother

from config import INSTRUCTION_DATA_PATH, CFG, MODEL_CACHE_PATH
from utils import find_sublist_indices, make_prompt_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
random.seed(42)

# --- Cấu hình ---
# Ví dụ: lấy các giá trị
MODEL_NAME: str = str(CFG["model"]["name"])

OUTPUT_DIR: str = str(CFG["training"]["output_dir"])
LR: float = float(CFG["training"]["learning_rate"])
BATCH_SIZE: int = int(CFG["training"]["batch_size"])
EPOCHS: int = int(CFG["training"]["epochs"])

LORA_CONFIG: dict = dict(CFG["lora"])
TRAIN_PATH: str = str(CFG["dataset"]["train_path"])
VAL_RATIO: float = float(CFG["dataset"]["val_ratio"])

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# --- Load model ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # tiết kiệm VRAM
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
dataset_split = dataset.train_test_split(test_size=VAL_RATIO, seed=42)
train_dataset = dataset_split["train"]
val_dataset = dataset_split["test"]

def format_example(example):
    think = example.get('think')
    messages = make_prompt_template(example['instruction'], think=think, respond=example['response'])

    # --- 2️⃣ Tạo prompt string từ messages ---
    # add_generation_prompt=False vì response đã có sẵn, reasoning ẩn nếu enable_thinking=True
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )

    # --- 3️⃣ Tokenize toàn bộ prompt ---
    tokenized = tokenizer(prompt_text, truncation=False, add_special_tokens=False)
    input_ids = tokenized["input_ids"]

    if think is None:
        end_masking_token = "</think>"
    else:
        end_masking_token = "<think>"

    # --- 4️⃣ Tìm vị trí end_masking_token ---
    masking_end_ids = tokenizer(end_masking_token, add_special_tokens=False)["input_ids"]

    end_pos = find_sublist_indices(input_ids, masking_end_ids)
    # mask toàn bộ trước end_masking_token (kể cả prefix)
    labels = [IGNORE_TOKEN_ID] * end_pos + input_ids[end_pos:]

    return {
        "input_ids": input_ids,
        "labels": labels
    }

train_tokenized_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
val_tokenized_dataset = val_dataset.map(format_example, remove_columns=val_dataset.column_names)

# --- Training arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=2*BATCH_SIZE,
    learning_rate=LR,
    weight_decay=CFG["training"]["weight_decay"],
    bf16=True,
    tf32=True,
    gradient_accumulation_steps=CFG["training"]["gradient_accumulation_steps"],
    logging_steps=CFG["training"]["logging_steps"],
    eval_strategy="steps",
    eval_steps=CFG["training"]["eval_steps"],
    save_steps=CFG["training"]["save_steps"],
    save_total_limit=CFG["training"]["save_total_limit"],
    report_to="tensorboard",
    logging_dir="./logs",
    lr_scheduler_type=CFG["training"]["lr_scheduler_type"],
    warmup_ratio=CFG["training"]["warmup_ratio"],
    load_best_model_at_end=True,
    label_names=["labels"]
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

# --- Train ---
trainer.train()
trainer.evaluate()

# --- Lưu model LoRA ---
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Huấn luyện LoRA hoàn tất, model lưu tại:", OUTPUT_DIR)
