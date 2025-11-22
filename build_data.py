from collections import defaultdict
import os
from pydoc import text
import random
import json
import re
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

from config import DATA_CACHE_PATH, INSTRUCTION_DATA_PATH, MODEL_CACHE_PATH, CFG
from filter import clean_reasoning, is_low_quality, is_good_thinking, process_mcq
from utils import extract_boxed, make_prompt_template, remove_all_tags, remove_tags

random.seed(42)
np.random.seed(42)

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
    min_token=None,
    max_token=None,
    length_sampling=False,
    length_power=0.7,
    max_candidates=None,
    max_data_samples=None,
    keys_get: list = None,
    boxed_force=True,
    remove_answer_special_tags_list: list = None,
    max_think_segment_chars=None,
    diversify_key=None, 
    max_per_group=50
):
    print(f"📥 Loading {path}...")
    ds = load_dataset(path, subset, split=split, cache_dir=DATA_CACHE_PATH, streaming=streaming)
    if max_data_samples is not None:
        ds = ds.take(max_data_samples)
    ds = ds.shuffle(seed=42)

    if n_samples is None:
        n_samples = len(ds)

    candidates = []

    seen_groups = defaultdict(int)

    for ex in ds:
        if not length_sampling and len(candidates) >= n_samples:
            break
        # --- giới hạn max_candidates ---
        if max_candidates is not None and len(candidates) >= max_candidates:
            break

        # --- DIVERSITY FILTER ---
        if diversify_key is not None:
            group_val = ex.get(diversify_key)
            # đa dạng: giới hạn số mẫu mỗi nhóm
            if seen_groups[group_val] >= max_per_group:
                continue

        if keys_get is not None:
            # keys_get là list of tuples
            if not any(ex.get(k) == v for k, v in keys_get):
                continue   # bỏ record nếu không có cặp nào đúng

        q = ex.get(question_key_en)
        if question_key_vi is not None:
            if len(candidates) % 2 == 1:
                q = ex.get(question_key_vi)

        a = ex.get(answer_key, "")
        if isinstance(a, list):
            a = a[0]
        
        if not q:
            continue

        if correct_key is not None:
            correct_value = ex.get(correct_key)

            if isinstance(correct_value, list):
                correct_value = correct_value[0]
            
            if correct_value not in ["Yes", True]:
                continue  # chỉ giữ câu trả lời đúng

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
            if isinstance(think_text, list):
                think_text = think_text[0]

            if answer_key == "</think>":
                if "</think>" in think_text:
                    # Vị trí kết thúc của </think>
                    end_idx = think_text.index("</think>")

                    # Answer = phần còn lại sau </think>
                    a = think_text[end_idx + len("</think>"):].strip()

                    # Think = từ đầu đến trước </think>
                    think_text = think_text[:end_idx].strip()
                else:
                    continue  # bỏ nếu không có </think>

            # remove tags <...> first and last
            think_text = remove_tags(think_text)
        if remove_answer_special_tags_list is not None:
            a = remove_all_tags(a, remove_answer_special_tags_list)

        if not is_good_thinking(think_text, a, max_segment_chars=max_think_segment_chars):
            continue

        messages = make_prompt_template(q, think=think_text, respond=a, boxed_force=boxed_force)
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
        if is_low_quality(prompt_text):
            continue

        tokens = tokenizer(prompt_text, truncation=False, add_special_tokens=False)
        tok_len = len(tokens["input_ids"])
        if (min_token is not None and tok_len < min_token) or (max_token is not None and tok_len > max_token):
            continue

        # --- xác định độ khó ---
        if tok_len < 700:
            difficulty = "easy"
        elif tok_len < 1400:
            difficulty = "medium"
        else:
            difficulty = "hard"

        # Tăng bộ đếm
        if diversify_key is not None:
            seen_groups[group_val] += 1

        candidates.append((tok_len, {
            "problem": q.strip(),
            "response": process_mcq(q.strip(), a.strip()),
            "data_type": data_type,
            "difficulty": difficulty,
            "source": path,
            **({ "answer": extract_boxed(a.strip()) } if boxed_force else {}),
            **({ "mask": ex.get(mask_key).strip() } if mask_key else {}),
            **({ "think": think_text.strip() } if think_key else {}),
            "boxed_force": boxed_force
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

    add_dataset(
        data,
        path="nvidia/OpenMathReasoning",
        question_key_en="problem",
        split="cot",
        answer_key="</think>",
        think_key="generated_solution",
        max_token=7000,
        n_samples=90000,
        streaming=True,
        max_data_samples=3000000,
        max_think_segment_chars=390,
        diversify_key="problem_source", 
        max_per_group=30000,
    )

    add_dataset(
        data,
        path="ServiceNow-AI/R1-Distill-SFT",
        question_key_en="problem",
        answer_key="</think>",
        subset='v0',
        think_key="reannotated_assistant_content",
        max_token=7000,
        n_samples=10000,
        max_candidates=20000,
        length_sampling=True,
        max_think_segment_chars=290,
        diversify_key="source", 
        max_per_group=3000,
    )

    add_dataset(
        data,
        path="zwhe99/DeepMath-103K",
        question_key_en="question",
        answer_key="</think>",
        think_key="r1_solution_1",
        max_token=7000,
        n_samples=1000,
        streaming=True,
        max_think_segment_chars=390,
    )

    add_dataset(
        data,
        path="zwhe99/DeepMath-103K",
        question_key_en="question",
        answer_key="</think>",
        think_key="r1_solution_2",
        max_token=7000,
        n_samples=1000,
        streaming=True,
        max_think_segment_chars=390,
    )

    add_dataset(
        data,
        path="zwhe99/DeepMath-103K",
        question_key_en="question",
        answer_key="</think>",
        think_key="r1_solution_3",
        max_token=7000,
        n_samples=1000,
        streaming=True,
        max_think_segment_chars=390,
    )

    add_dataset(
        data,
        path="dvilasuero/natural-science-reasoning",
        question_key_en="question",
        answer_key="</think>",
        think_key="r1-response",
        data_type="science",
        max_token=7000,
        boxed_force=False,
        remove_answer_special_tags_list=["think", "answer"]
    )

    random.shuffle(data)
    total = len(data)

    train_end = int((1 - test_ratio) * total)
    splits = {
        "train": data[:train_end],
        **({ "test": data[train_end:]} if train_end < total else {})
    }

    os.makedirs(output_dir, exist_ok=True)
    for split_name, split_data in splits.items():
        path = os.path.join(output_dir, f"{split_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(split_data)} samples to {path}")

    print(f"\n🏁 Done! Total: {total} samples")

def build_test_math(output_dir="."):
    data = []
    add_dataset(
        data,
        path="brucewlee1/mmlu-high-school-mathematics",
        question_key_en="centerpiece",
        split="test",
        answer_signal="",
        answer_key="correct_options_literal",
        boxed_force=True,
    )

    random.shuffle(data)
    total = len(data)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "test.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(data)} samples to {path}")


if __name__ == "__main__":
    build_small_math_reasoning(output_dir=INSTRUCTION_DATA_PATH, test_ratio=0)
    # build_test_math(output_dir=INSTRUCTION_DATA_PATH)
