import os

import random
import json

from transformers import AutoTokenizer

from config import DATA_CACHE_PATH, INSTRUCTION_DATA_PATH, MODEL_CACHE_PATH, CFG
from utils import make_prompt_template

from datasets import load_dataset

MODEL_NAME: str = str(CFG["model"]["name"])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)

def add_dataset(
    all_data,
    name,
    path,
    tokenizer,
    question_key_en,
    answer_key,
    *,
    mask_key=None,
    think_key=None,
    correct_key=None,
    question_key_vi=None,
    subset=None,
    split="train",
    n_samples=0,
    min_length=0,
    max_length=9999999,
    max_think_length=9999999
):
    """
    Tải 1 dataset từ Hugging Face, lọc câu quá dài theo tokenizer,
    và thêm vào list dữ liệu chung.
    """

    print(f"📥 Loading {name}...")
    ds = load_dataset(path, subset, split=split, cache_dir=DATA_CACHE_PATH)
    ds = ds.shuffle(seed=42)
    if n_samples == 0:
        n_samples = len(ds)
    # Lấy nhiều hơn n_samples để lọc bớt những câu dài

    count = 0
    for ex in ds:
        if count >= n_samples:
            break

        if question_key_vi is not None:
            if count % 2 == 0:
                q = ex.get(question_key_en)
            else:
                q = ex.get(question_key_vi)
        else:
            q = ex.get(question_key_en)

        a = ex.get(answer_key)
        if not q or not a:
            continue

        if correct_key is not None:
            c = ex.get(correct_key)
            if c != "Yes":
                continue


        # lines = a.strip().splitlines()
        # last = lines[-1]
        # if last.startswith("The answer is:"):
        #     answer = last.replace("The answer is:", "").strip()
        #     lines[-1] = f"The answer is: \\boxed{{{answer}}}"
        # a = "\n".join(lines)

        if think_key is not None:
            think_tokens = tokenizer(ex.get(think_key))
            if len(think_tokens) > max_think_length:
                continue

        # Tạo text kết hợp instruction + output để check độ dài token
        messages = make_prompt_template(q, a)
        # --- 2️⃣ Tạo prompt string từ messages ---
        # add_generation_prompt=False vì response đã có sẵn, reasoning ẩn nếu enable_thinking=True
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
        # --- 3️⃣ Tokenize toàn bộ prompt ---
        tokens = tokenizer(prompt_text, truncation=False, add_special_tokens=False)
        if len(tokens["input_ids"]) < min_length or len(tokens["input_ids"]) > max_length:
            continue

        data = {
            "instruction": str(q).strip(),
            "response": str(a).strip()
        }
        if mask_key is not None:
            data["mask"] = str(ex.get(mask_key)).strip()
        if think_key is not None:
            data["think"] = str(ex.get(think_key)).strip()

        all_data.append(data)
        count += 1

def build_small_math_reasoning(output_dir=".", test_ratio=0.1):
    """Tạo dataset reasoning toán học và chia train/val/test."""

    data = []

    # Dataset chính
    # add_dataset(
    #     data,
    #     name="mathqa",
    #     tokenizer=tokenizer,
    #     path="nlile/hendrycks-MATH-benchmark",
    #     question_key_en="problem",
    #     answer_key="solution",
    #     n_samples=0,
    #     max_length=768
    # )
    # add_dataset(
    #     data,
    #     name="numinamath",
    #     tokenizer=tokenizer,
    #     path="AI-MO/NuminaMath-CoT",
    #     question_key_en="problem",
    #     answer_key="solution",
    #     n_samples=48000,
    #     max_length=2048
    # )
    # add_dataset(
    #     data,
    #     name="numinamath-tir",
    #     tokenizer=tokenizer,
    #     path="AI-MO/NuminaMath-TIR",
    #     question_key_en="problem",
    #     answer_key="solution",
    #     n_samples=20000,
    #     max_length=1024
    # )
    add_dataset(
        data,
        name="s1K-1.1-germini",
        tokenizer=tokenizer,
        path="simplescaling/s1K-1.1",
        question_key_en="question",
        answer_key="gemini_attempt",
        correct_key="gemini_grade",
        think_key="gemini_thinking_trajectory",
        n_samples=0,
        max_length=3064,
        max_think_length=16000
    )
    add_dataset(
        data,
        name="s1K-1.1-deepseek",
        tokenizer=tokenizer,
        path="simplescaling/s1K-1.1",
        question_key_en="question",
        answer_key="deepseek_attempt",
        correct_key="deepseek_grade",
        think_key="deepseek_thinking_trajectory",
        n_samples=0,
        max_length=3064,
        max_think_length=16000
    )

    random.shuffle(data)
    total = len(data)

    # 🧮 Chia tập (80% train, 10% val, 10% test)
    train_end = int((1-test_ratio) * total)

    splits = {
        "train": data[:train_end],
        "test": data[train_end:]
    }

    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_data in splits.items():
        path = os.path.join(output_dir, f"{split_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(split_data)} samples to {path}")

    print(f"\n🏁 Done! Total: {total} samples")


if __name__ == "__main__":
    build_small_math_reasoning(output_dir=INSTRUCTION_DATA_PATH, test_ratio=0)