import os
from pydoc import text
import random
import json
import re
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

from config import DATA_CACHE_PATH, INSTRUCTION_DATA_PATH, MODEL_CACHE_PATH, CFG
from filter import is_low_quality, has_reasoning_steps, process_mcq
from utils import make_prompt_template

random.seed(42)

MODEL_NAME: str = str(CFG["model"]["name"])
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)

def add_dataset(
    all_data,
    path,
    question_key_en,
    answer_key,
    *,
    mask_key=None,
    think_key=None,
    correct_key=None,
    question_key_vi=None,
    subset=None,
    answer_signal=None,
    data_type="math",
    split="train",
    n_samples=None,
    streaming=False,
    min_length=64,
    max_token=768,
    max_think_token=512,
    max_think_length=None,
    length_sampling=False,
    length_power=0.7,
    max_candidates=None,
    max_data_samples=None,
    keys_get: list = None
):
    print(f"📥 Loading {path}...")
    ds = load_dataset(path, subset, split=split, cache_dir=DATA_CACHE_PATH, streaming=streaming)
    if max_data_samples is not None:
        ds = ds.take(max_data_samples)
    ds = ds.shuffle(seed=42)

    if n_samples is None:
        n_samples = len(ds)

    candidates = []
    for ex in ds:
        if not length_sampling and len(candidates) >= n_samples:
            break
        # --- giới hạn max_candidates ---
        if max_candidates is not None and len(candidates) >= max_candidates:
            break

        if keys_get is not None:
        # keys_get là list of tuples
            if not any(ex.get(k) == v for k, v in keys_get):
                continue   # bỏ record nếu không có cặp nào đúng

        q = ex.get(question_key_en)
        if question_key_vi is not None:
            if len(candidates) % 2 == 1:
                q = ex.get(question_key_vi)

        a = ex.get(answer_key, "")
        if not q:
            continue

        if correct_key is not None and ex.get(correct_key) != "Yes":
            continue

        if answer_signal is not None:
            lines = a.strip().splitlines()
            last = lines[-1]
            if last.startswith(answer_signal):
                answer = last.replace(answer_signal, "").strip()
                lines[-1] = f"The answer is: \\boxed{{{answer}}}"
            a = "\n".join(lines)

        think_text = None
        if think_key is not None:
            think_text = ex.get(think_key, "")

        if answer_key == "</think>":
            think_match = re.search(r'<think>(.*?)</think>', think_text, flags=re.DOTALL)
            if think_match:
                # 2) Lấy phần còn lại *sau thẻ <think>* làm answer
                a = think_text[think_match.end():].strip()

                think_text = think_match.group(1).strip()

                # 3) Check token length
                think_tokens = tokenizer(think_text, truncation=False, add_special_tokens=False)
                if len(think_tokens['input_ids']) > max_think_token:
                    continue
            else:
                # No <think> tags found, skip this record
                continue

            if think_text:
                if max_think_length is not None and len(think_text) > max_think_length:
                    continue
                think_text = re.sub(r'^\s*<[^>]+>\s*', '', think_text, count=1)
                think_text = re.sub(r'\s*</?[^>]+>\s*$', '', think_text, count=1)

            think_tokens = tokenizer(think_text, truncation=False, add_special_tokens=False)
            if len(think_tokens['input_ids']) > max_think_token:
                continue

        messages = make_prompt_template(q, a)
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
        if is_low_quality(prompt_text):
            continue
        if not has_reasoning_steps(think_text, a):
            continue

        tokens = tokenizer(prompt_text, truncation=False, add_special_tokens=False)
        tok_len = len(tokens["input_ids"])
        if tok_len < min_length or tok_len > max_token:
            continue

        # --- xác định độ khó ---
        if tok_len < 256:
            difficulty = "easy"
        elif tok_len < 512:
            difficulty = "medium"
        else:
            difficulty = "hard"

        candidates.append((tok_len, {
            "instruction": q.strip(),
            "response": process_mcq(q.strip(), a.strip()),
            "data_type": data_type,
            "difficulty": difficulty,
            **({ "mask": ex.get(mask_key).strip() } if mask_key else {}),
            **({ "think": think_text.strip() } if think_key else {})
        }))

    # --- Lấy mẫu cuối cùng ---
    if length_sampling and candidates:
        lengths = np.array([c[0] for c in candidates], dtype=float)
        probs = np.power(lengths, length_power)
        probs /= probs.sum()
        selected_indices = np.random.choice(len(candidates), size=min(n_samples, len(candidates)), replace=False, p=probs)
        for i in selected_indices:
            all_data.append(candidates[i][1])
    else:
        for tok_len, data in candidates[:n_samples]:
            all_data.append(data)


def build_small_math_reasoning(output_dir=".", test_ratio=0.1):
    data = []

    # Ví dụ dataset
    add_dataset(
        data,
        path="AI-MO/NuminaMath-CoT",
        question_key_en="problem",
        answer_key="solution",
        n_samples=100,
        max_token=1024
    )
    add_dataset(
        data,
        path="ServiceNow-AI/R1-Distill-SFT",
        question_key_en="problem",
        subset="v0",
        answer_key="solution",
        max_token=768,
        n_samples=100
    )
    add_dataset(
        data,
        path="nvidia/OpenMathReasoning",
        question_key_en="problem",
        split="cot",
        answer_key="</think>",
        think_key="generated_solution",
        max_token=768,
        max_think_token=2304,
        max_think_length=14000,
        max_data_samples=30000,
        n_samples=600,
        streaming=True
    )
    add_dataset(
        data,
        path="gretelai/gretel-math-gsm8k-v0",
        question_key_en="question",
        answer_key="answer",
        answer_signal= "#### ",
        max_token=1024,
    )
    add_dataset(
        data,
        path="simplescaling/s1K-1.1",
        question_key_en="question",
        answer_key="deepseek_attempt",
        think_key="deepseek_thinking_trajectory",
        data_type="other",
        correct_key="deepseek_grade",
        max_token=512,
        max_think_token=2560,
        max_think_length=14000,
        keys_get=[("cot_type", "science")],
    )

    random.shuffle(data)
    total = len(data)

    train_end = int((1 - test_ratio) * total)
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
