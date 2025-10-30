import csv
import re
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm
from datetime import datetime
import os

from config import MODEL_NAME, RESULT_FILE, DEVICE, MODEL_CACHE_PATH, DATA_CACHE_PATH, INSTRUCTION_DATA_PATH, \
    PROMPT_TEMPLATE

# 🧠 Tải model và tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)
tokenizer.padding_side = "left"
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_CACHE_PATH,
    quantization_config=bnb_config,
)
model = PeftModel.from_pretrained(model, "sft-lora-model")
model.generation_config.pad_token_id = tokenizer.eos_token_id
model.eval()

BATCH_SIZE = 8

# 🔢 Hàm trích xuất kết quả trong \boxed{...}
def extract_boxed(s: str):
    start = s.find(r"\boxed{")
    if start != -1:
        start += len(r"\boxed{")
        depth = 1
        i = start
        while i < len(s) and depth > 0:
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
            i += 1
        content = s[start:i-1].strip()
        return content

    eq_match = re.search(r"=\s*([^\n]*)$", s)
    if eq_match:
        return eq_match.group(1).strip()

    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return "?"


# ⚙️ Hàm sinh câu trả lời
def generate_answers(prompts, max_new_tokens=512):
    # prompts: list of str
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return texts

# 📊 Đánh giá accuracy và lưu chi tiết
def evaluate_dataset(dataset_name, use_local_data=False, eval_size=0, problem="problem", answer="answer", split="test"):
    if use_local_data:
        dataset = load_dataset("json", data_files=dataset_name)[split]
    else:
        dataset = load_dataset(dataset_name, split=split, cache_dir=DATA_CACHE_PATH)

    if eval_size > 0:
        dataset = dataset.select(range(min(eval_size, len(dataset))))

    correct = 0
    total = len(dataset)

    # 🗂️ File lưu chi tiết
    detail_file = RESULT_FILE.replace(".csv", f"_{os.path.basename(dataset_name).replace('.json','')}_details.csv")
    with open(detail_file, "w", newline="", encoding="utf-8") as f_detail:
        writer = csv.writer(f_detail)
        writer.writerow(["question", "ground_truth", "prediction", "is_correct"])

        progress_bar = tqdm(range(0, total, BATCH_SIZE), desc=f"Evaluating {dataset_name}")
        for i in progress_bar:
            batch_indices = list(range(i, min(i + BATCH_SIZE, total)))
            batch_samples = dataset.select(batch_indices)
            batch_questions = [PROMPT_TEMPLATE.format(question=s[problem]) for s in batch_samples]

            # Batch inference
            batch_outputs = generate_answers(batch_questions, max_new_tokens=3584)

            for sample, output in zip(batch_samples, batch_outputs):
                question = sample[problem]
                gt = str(sample[answer]).strip()
                pred = extract_boxed(output)

                # So sánh kết quả
                is_correct = False
                try:
                    if float(pred) == float(gt):
                        is_correct = True
                except:
                    if pred == gt:
                        is_correct = True

                if is_correct:
                    correct += 1

                # Ghi từng dòng vào file chi tiết
                writer.writerow([question, gt, pred, int(is_correct)])

            current_acc = correct / (i + len(batch_samples)) * 100
            progress_bar.set_postfix(correct=correct, acc=f"{current_acc:.2f}%")

    acc = correct / total * 100
    print(f"{dataset_name}: {correct}/{total} = {acc:.2f}%")
    print(f"📄 Chi tiết đã lưu vào: {detail_file}")
    return acc

# 💾 Lưu kết quả tổng
def save_result(model_name, dataset_name, accuracy):
    header = ["timestamp", "model", "dataset", "accuracy (%)"]
    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_name, dataset_name, f"{accuracy:.2f}"]

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
    aime_acc = evaluate_dataset("HuggingFaceH4/aime_2024", split="train")
    save_result(MODEL_NAME, "AIME24-1", aime_acc)
    #
    # math_acc = evaluate_dataset("HuggingFaceH4/MATH-500", eval_size=30)
    # save_result(MODEL_NAME, "MATH-500", math_acc)

    print("\n📊 Benchmark Summary")
    # print(f"AIME24-1 Accuracy: {aime_acc:.2f}%")
    # print(f"MATH-500 Accuracy: {math_acc:.2f}%")
    print(f"\n✅ Kết quả đã được lưu vào file: {RESULT_FILE}")
