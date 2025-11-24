from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, \
    DataCollatorForSeq2Seq
from transformers.trainer_pt_utils import LabelSmoother

from config import MODEL_NAME, INSTRUCTION_DATA_PATH, MODEL_CACHE_PATH, PROMPT_TEMPLATE

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# --- Cấu hình ---
OUTPUT_DIR = "./sft-lora-model"
MAX_LENGTH = 1024
LR = 3e-5
BATCH_SIZE = 2
EPOCHS = 1

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Load model 8-bit ---
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,  # tiết kiệm VRAM
    attn_implementation="flash_attention_2",
    cache_dir=MODEL_CACHE_PATH
)

# --- Thiết lập LoRA ---
lora_config = LoraConfig(
    r=32,                # rank của LoRA
    lora_alpha=32,       # scaling
    target_modules=["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj", "o_proj"],  # tuỳ model
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Load dataset ---
dataset = load_dataset("json", data_files=INSTRUCTION_DATA_PATH+"train.json")["train"]
dataset_split = dataset.train_test_split(test_size=0.01, seed=42)
train_dataset = dataset_split["train"]
val_dataset = dataset_split["test"]

def format_example(example):
    # Tạo text đầy đủ
    instruction = PROMPT_TEMPLATE.format(question=example['instruction'])
    answer = example['response']

    # Tokenize riêng instruction và answer
    tokenized_instruction = tokenizer(instruction, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)
    tokenized_answer = tokenizer(answer, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)

    # Ghép lại
    input_ids = tokenized_instruction["input_ids"] + tokenized_answer["input_ids"]

    labels = [IGNORE_TOKEN_ID] * len(tokenized_instruction["input_ids"]) + tokenized_answer["input_ids"]

    # ✳️ (1) Thêm <eos> cuối nếu model có eos_token
    if tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)

    # Cắt/pad đến MAX_LENGTH
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }

train_tokenized_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
val_tokenized_dataset = val_dataset.map(format_example, remove_columns=val_dataset.column_names)

# --- Training arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    fp16=True,
    gradient_accumulation_steps=8,
    logging_steps=10,
    eval_strategy="steps",   # ✅ thêm dòng này để Trainer chạy eval
    eval_steps=80,             # ✅ mỗi 50 bước đánh giá
    save_steps=80,             # ✅ lưu cùng lúc
    save_total_limit=2,
    report_to="tensorboard",
    logging_dir="./logs",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    adam_beta2=0.999,
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