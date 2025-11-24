import csv
import re
import tempfile
import subprocess
import os
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm
from datetime import datetime
import warnings

# Tắt warnings
warnings.filterwarnings('ignore')

# Tăng timeout cho HuggingFace
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 phút
os.environ['CURL_CA_BUNDLE'] = ''  # Fix SSL issues nếu có

from config import (
    MODEL_NAME, RESULT_FILE, DEVICE, MODEL_CACHE_PATH, DATA_CACHE_PATH,
    INSTRUCTION_DATA_PATH, PROMPT_TEMPLATE, LORA_MODEL_PATH, USE_LORA
)
from calculation_detector import CalculationDetector

# 🧠 Tải model và tokenizer
print(f"📥 Loading tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_CACHE_PATH,
    local_files_only=True  # Chỉ dùng file local
)
tokenizer.padding_side = "left"

# Fix pad_token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"📥 Loading base model from {MODEL_NAME}...")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_CACHE_PATH,
    quantization_config=bnb_config,
    local_files_only=True  # Chỉ dùng file local
)

# Load LoRA adapter nếu USE_LORA = True
if USE_LORA:
    print(f"📥 Loading LoRA adapter from {LORA_MODEL_PATH}...")
    model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
    print(f"✅ Loaded model: {MODEL_NAME} + {LORA_MODEL_PATH}")
else:
    print(f"✅ Loaded base model: {MODEL_NAME}")

# Set pad_token_id for model
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

model.eval()

BATCH_SIZE = 1


# 🔧 Hàm chạy code trong sandbox
def run_in_sandbox(code: str, timeout: float = 5.0) -> str:
    """Chạy code Python trong môi trường cô lập, lấy stdout hoặc stderr"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout.strip() or result.stderr.strip()
        os.unlink(tmp_path)
    except subprocess.TimeoutExpired:
        output = "⚠️ Timeout: code chạy quá lâu"
        try:
            os.unlink(tmp_path)
        except:
            pass
    except Exception as e:
        output = f"⚠️ Lỗi sandbox: {e}"
        try:
            os.unlink(tmp_path)
        except:
            pass

    return output


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
        content = s[start:i - 1].strip()
        return content

    eq_match = re.search(r"=\s*([^\n]*)$", s)
    if eq_match:
        return eq_match.group(1).strip()

    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return "?"


# ⚙️ Hàm sinh câu trả lời với sandbox execution (SILENT MODE)
def generate_answer_with_execution(
        prompt,
        max_new_tokens=512,
        chunk_size=50,
        max_calculations=3
):
    """
    Sinh token từng chunk, detect calculation, execute code, continue
    SILENT MODE - không in ra console
    """
    detector = CalculationDetector()

    # Tokenize prompt ban đầu với attention_mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    current_input_ids = inputs.input_ids
    current_attention_mask = inputs.attention_mask

    full_text = prompt
    calculation_count = 0
    tokens_generated = 0

    while tokens_generated < max_new_tokens:
        # Sinh một chunk tokens với attention_mask
        with torch.no_grad():
            outputs = model.generate(
                current_input_ids,
                attention_mask=current_attention_mask,
                max_new_tokens=min(chunk_size, max_new_tokens - tokens_generated),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode text mới
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_text = generated_text[len(full_text):] if len(generated_text) > len(full_text) else ""

        # Cập nhật full text
        full_text = generated_text
        tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])

        # Kiểm tra EOS
        if outputs[0][-1] == tokenizer.eos_token_id:
            break

        # Kiểm tra xem có cần calculation không
        if calculation_count < max_calculations:
            needs_calc = detector.detect_needs_calculation(full_text)

            if needs_calc:
                # Extract context và generate code
                context = detector.extract_calculation_context(full_text)
                code = detector.generate_calculation_code(context, prompt)

                if code:
                    # Chạy code trong sandbox
                    execution_output = run_in_sandbox(code)

                    # Append output vào full_text
                    calculation_result = f"\n\n[Calculation: {execution_output}]\n\n"
                    full_text += calculation_result

                    # Re-tokenize với context mới và attention_mask
                    new_inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                    current_input_ids = new_inputs.input_ids
                    current_attention_mask = new_inputs.attention_mask

                    calculation_count += 1

                    # Continue generation với context mới
                    continue

        # Cập nhật input_ids và attention_mask cho lần generate tiếp theo
        current_input_ids = outputs
        # Tạo attention_mask mới cho output
        current_attention_mask = torch.ones_like(outputs).to(DEVICE)

        # Kiểm tra xem có nên tiếp tục không
        if not detector.should_continue_generation(full_text):
            break

        # Nếu không có text mới được sinh ra, dừng
        if not new_text or len(new_text.strip()) == 0:
            break

    # Trả về phần text sau prompt
    result = full_text[len(prompt):] if len(full_text) > len(prompt) else full_text
    return result.strip()

# 📊 Đánh giá accuracy và lưu chi tiết (SILENT MODE)
def evaluate_dataset(
        dataset_name,
        use_local_data=False,
        eval_size=0,
        problem="problem",
        answer="answer",
        split="test",
        use_execution=True
):
    if use_local_data:
        dataset = load_dataset("json", data_files=dataset_name)[split]
    else:
        dataset = load_dataset(dataset_name, split=split, cache_dir=DATA_CACHE_PATH)

    if eval_size > 0:
        dataset = dataset.select(range(min(eval_size, len(dataset))))

    correct = 0
    total = len(dataset)

    # 🗂️ File lưu chi tiết
    model_suffix = LORA_MODEL_PATH.replace('/', '_') if USE_LORA else "base"
    suffix = f"_{model_suffix}_with_execution" if use_execution else f"_{model_suffix}_baseline"
    detail_file = RESULT_FILE.replace(".csv",
                                      f"_{os.path.basename(dataset_name).replace('.json', '')}{suffix}_details.csv")

    with open(detail_file, "w", newline="", encoding="utf-8") as f_detail:
        writer = csv.writer(f_detail)
        writer.writerow(["question", "ground_truth", "prediction", "raw_output", "is_correct"])

        # 🟩 Thêm thanh tiến độ tqdm
        print(f"🔍 Evaluating {total} samples from {dataset_name}...\n")
        for idx, sample in enumerate(tqdm(dataset, desc="Progress", ncols=80, colour="green")):
            question = sample[problem]
            gt = str(sample[answer]).strip()

            # Format prompt
            prompt_text = PROMPT_TEMPLATE.format(question=question)

            # Generate với hoặc không có execution
            try:
                if use_execution:
                    output = generate_answer_with_execution(
                        prompt_text,
                        max_new_tokens=2048,
                        chunk_size=50,
                        max_calculations=3
                    )
                else:
                    # Generate bình thường không có execution
                    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    output = output[len(prompt_text):].strip()
            except Exception as e:
                output = ""

            raw_output = output.strip()
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
            writer.writerow([question, gt, pred, raw_output, int(is_correct)])

    acc = correct / total * 100
    print(f"\n✅ {dataset_name}: {correct}/{total} = {acc:.2f}%")
    print(f"📄 Saved to: {detail_file}")
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
        print(f"⚠️ Cannot save results: {e}")


if __name__ == "__main__":
    model_display_name = f"{MODEL_NAME}+{LORA_MODEL_PATH}" if USE_LORA else MODEL_NAME

    print("\n" + "=" * 80)
    print(f"EVALUATING: {model_display_name}")
    print("=" * 80 + "\n")

    # Evaluate
    aime_acc = evaluate_dataset(
        "HuggingFaceH4/aime_2024",
        split="train",
        eval_size=5,
        use_execution=True
    )
    save_result(model_display_name + "_with_execution", "AIME24-1", aime_acc)

    print(f"\n✅ Results saved to: {RESULT_FILE}\n")