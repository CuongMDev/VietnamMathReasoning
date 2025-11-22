import csv
import os
import re
from datetime import datetime

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from build_data import add_dataset
from generate_answers import generate_answers, generate_answers_budget_forcing

from config import RESULT_FILE, DEVICE, MODEL_CACHE_PATH, DATA_CACHE_PATH, CFG
from utils import extract_boxed, is_answer_equal, make_prompt_template

MODEL_NAME = CFG["model"]["name"]

# 🧠 Tải model và tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)
tokenizer.padding_side = "left"
# --- Load model ---
model = AutoModelForCausalLM.from_pretrained(
    "sft-cot-model",
    torch_dtype=torch.float16,  # tiết kiệm VRAM
    device_map='cuda',
    attn_implementation="flash_attention_2",
    cache_dir=MODEL_CACHE_PATH
)
# model = PeftModel.from_pretrained(model, "sft-cot-model")
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.eval()

BATCH_SIZE = 8

# 📊 Đánh giá accuracy và lưu chi tiết (kèm output đầy đủ)
def evaluate_dataset(dataset_name, config_name=None, use_local_data=False, eval_size=0, problem="problem", answer="answer", split="test"):
    if use_local_data:
        dataset = load_dataset("json", data_files=dataset_name)[split]
    else:
        dataset = load_dataset(dataset_name, config_name, split=split, cache_dir=DATA_CACHE_PATH)

    if eval_size > 0:
        dataset = dataset.select(range(min(eval_size, len(dataset))))

    correct = 0
    total_tokens_generated = 0

    total = len(dataset)

    # 🗂️ File lưu chi tiết
    detail_file = RESULT_FILE.replace(".csv", f"_{os.path.basename(dataset_name).replace('.json','')}_details.csv")
    with open(detail_file, "w", newline="", encoding="utf-8") as f_detail:
        writer = csv.writer(f_detail)
        # ➕ Thêm cột raw_output + token_generated
        writer.writerow(["question", "ground_truth", "prediction", "raw_output", "token_generated", "is_correct"])

        progress_bar = tqdm(range(0, total, BATCH_SIZE), desc=f"Evaluating {dataset_name}")
        for i in progress_bar:
            batch_indices = list(range(i, min(i + BATCH_SIZE, total)))
            batch_samples = dataset.select(batch_indices)
            batch_questions = [make_prompt_template(s[problem]) for s in batch_samples]

            # Batch inference → trả về (outputs, token_counts)
            batch_outputs, batch_token_counts = generate_answers_budget_forcing(
                model,
                tokenizer,
                batch_questions,
                max_new_tokens=1024, 
                max_tokens_thinking_tmp=4000,
            )

            for sample, output, tok_count in zip(batch_samples, batch_outputs, batch_token_counts):
                question = sample[problem]
                gt = str(sample[answer]).strip()
                pred = extract_boxed(output)

                # Lấy dòng cuối ground truth
                lines = gt.strip().splitlines()
                last = lines[-1]
                gt = last.replace("#### ", "").strip()

                raw_output = output.strip()  # ✨ Lưu đầy đủ output

                # So sánh kết quả
                is_correct = is_answer_equal(pred, gt)

                if is_correct:
                    correct += 1
                total_tokens_generated += tok_count

                # Ghi vào CSV: thêm cột token_generated
                writer.writerow([question, gt, pred, raw_output, tok_count, int(is_correct)])

            current_acc = correct / (i + len(batch_samples)) * 100
            progress_bar.set_postfix(correct=correct, acc=f"{current_acc:.2f}%")

    acc = correct / total * 100
    avg_tokens = total_tokens_generated / total

    print(f"{dataset_name}: {correct}/{total} = {acc:.2f}%")
    print(f"📄 Chi tiết đã lưu vào: {detail_file}")
    print(f"🧮 Số token sinh trung bình: {avg_tokens:.2f}")

    return acc, avg_tokens

# 💾 Lưu kết quả tổng
def save_result(model_name, dataset_name, accuracy, avg_tokens_generated):
    header = ["timestamp", "model", "dataset", "accuracy (%)", "avg_tokens_generated"]
    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_name, dataset_name, f"{accuracy:.2f}", f"{avg_tokens_generated:.2f}"]

    try:
        with open(RESULT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as e:
        print(f"⚠️ Không thể lưu kết quả: {e}")


if __name__ == "__main__":
    # hendrycks = evaluate_dataset("nlile/hendrycks-MATH-benchmark", split="test", eval_size=30)
    # save_result(MODEL_NAME, "hendrycks_math", hendrycks)
    # aime_acc = evaluate_dataset("HuggingFaceH4/aime_2024", split="train")
    # save_result(MODEL_NAME, "AIME24-1", *aime_acc)
    # aime_acc = evaluate_dataset("data/aime_2024-vie.json", split="train", use_local_data=True)
    # save_result(MODEL_NAME, "AIME24-1-vie", aime_acc)
    #
    # math_acc = evaluate_dataset("HuggingFaceH4/MATH-500", eval_size=30)
    # save_result(MODEL_NAME, "MATH-500", math_acc)

    # gsm8k_acc = evaluate_dataset("openai/gsm8k", config_name="main", problem="question", answer="answer", eval_size=1000)
    # save_result(MODEL_NAME, "GSM8K", *gsm8k_acc)

    gretelai_gsm8k_acc = evaluate_dataset("data/test.json", split="train", use_local_data=True, eval_size=30)
    save_result(MODEL_NAME, "gretelai-GSM8K", *gretelai_gsm8k_acc)

    print("\n📊 Benchmark Summary")
    # print(f"AIME24-1 Accuracy: {aime_acc:.2f}%")
    # print(f"MATH-500 Accuracy: {math_acc:.2f}%")
    print(f"\n✅ Kết quả đã được lưu vào file: {RESULT_FILE}")
