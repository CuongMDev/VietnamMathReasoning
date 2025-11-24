# """
# Quick test script để test model sft-lora-model-tir với một vài câu hỏi đơn giản
# """
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel
# import torch
# from config import MODEL_NAME, MODEL_CACHE_PATH, PROMPT_TEMPLATE, LORA_MODEL_PATH
# from calculation_detector import CalculationDetector
# import tempfile
# import subprocess
# import os
# import random
#
#
#
# def run_in_sandbox(code: str, timeout: float = 5.0) -> str:
#     """Chạy code trong sandbox"""
#     with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
#         f.write(code)
#         tmp_path = f.name
#
#     try:
#         result = subprocess.run(
#             ["python", tmp_path],
#             capture_output=True,
#             text=True,
#             timeout=timeout
#         )
#         output = result.stdout.strip() or result.stderr.strip()
#         os.unlink(tmp_path)
#     except subprocess.TimeoutExpired:
#         output = "⚠️ Timeout"
#         try:
#             os.unlink(tmp_path)
#         except:
#             pass
#     except Exception as e:
#         output = f"⚠️ Error: {e}"
#         try:
#             os.unlink(tmp_path)
#         except:
#             pass
#     return output
#
#
# def test_single_question(model, tokenizer, question, use_execution=True):
#     """Test một câu hỏi"""
#
#     print("\n" + "=" * 80)
#     print(f"QUESTION: {question}")
#     print("=" * 80)
#
#     prompt = PROMPT_TEMPLATE.format(question=question)
#
#     if use_execution:
#         print("\n🔧 Mode: WITH EXECUTION\n")
#
#         detector = CalculationDetector()
#         full_text = prompt
#
#         # Sinh text ban đầu
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#         with torch.no_grad():
#             outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
#
#         generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         full_text = generated
#
#         print("📝 Initial generation:")
#         print(generated[len(prompt):])
#
#         # Check calculation
#         if detector.detect_needs_calculation(full_text):
#             print("\n🔍 Calculation detected!")
#
#             context = detector.extract_calculation_context(full_text)
#             code = detector.generate_calculation_code(context, question)
#
#             if code:
#                 print("\n🔧 Generated code:")
#                 print(code)
#
#                 result = run_in_sandbox(code)
#                 print(f"\n📤 Execution result: {result}")
#
#                 # Continue generation với result
#                 full_text += f"\n\n[Calculation: {result}]\n\n"
#
#                 inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
#                 with torch.no_grad():
#                     outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
#
#                 final = tokenizer.decode(outputs[0], skip_special_tokens=True)
#                 print("\n📝 Final generation:")
#                 print(final[len(full_text):])
#         else:
#             print("\n⚠️ No calculation detected")
#
#     else:
#         print("\n📝 Mode: BASELINE (no execution)\n")
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#         with torch.no_grad():
#             outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
#
#         generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print(generated[len(prompt):])
#
#     print("\n" + "=" * 80)
#
#
# def main():
#     """Main test function"""
#
#     # Load model
#     print("📥 Loading model...")
#     print(f"   Base: {MODEL_NAME}")
#     print(f"   LoRA: {LORA_MODEL_PATH}")
#
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)
#
#     bnb_config = BitsAndBytesConfig(load_in_8bit=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         cache_dir=MODEL_CACHE_PATH,
#         quantization_config=bnb_config,
#     )
#
#     model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
#     model.eval()
#
#     print("✅ Model loaded!\n")
#
#     # Test cases
#     test_cases = [
#         "Calculate the sum of all even numbers from 1 to 100.",
#         "What is 15 factorial?",
#         "Compute 2 to the power of 10.",
#         "Find the square root of 144.",
#         "What is 123 multiplied by 456?",
#     ]
#
#     # Test với execution
#     print("\n" + "#" * 80)
#     print("# TESTING WITH EXECUTION")
#     print("#" * 80)
#
#     random_questions = random.sample(test_cases, 2)
#     for question in random_questions:
#         test_single_question(model, tokenizer, question, use_execution=True)
#         input("\nPress Enter to continue...")
#
#     # Test without execution
#     print("\n" + "#" * 80)
#     print("# TESTING WITHOUT EXECUTION (Baseline)")
#     print("#" * 80)
#
#     random_questions = random.sample(test_cases, 2)
#     for question in random_questions:
#         test_single_question(model, tokenizer, question, use_execution=False)
#         input("\nPress Enter to continue...")
#
#
# if __name__ == "__main__":
#     main()

"""
Quick evaluation script for sft-lora-model-tir
"""
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from config import MODEL_NAME, MODEL_CACHE_PATH, LORA_MODEL_PATH, PROMPT_TEMPLATE, DEVICE, USE_LORA, DATA_CACHE_PATH


def load_test_data(path, limit=None):
    """Load test samples từ file JSONL hoặc tạo sẵn nếu không có file."""
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}. Using built-in test samples instead.")
        samples = [
            {"question": "What is 12 + 8?", "answer": "20"},
            {"question": "If you multiply 7 by 9, what do you get?", "answer": "63"},
            {"question": "What is 100 divided by 4?", "answer": "25"},
            {"question": "What is the square of 15?", "answer": "225"},
            {"question": "A car travels 60 km in 2 hours. What is its speed in km/h?", "answer": "30"}
        ]
        return samples[:limit] if limit else samples

    # Nếu file tồn tại → đọc như bình thường
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data[:limit] if limit else data


def load_model():
    """Load model and LoRA (if enabled)"""
    print("📥 Loading model...")
    print(f"   Base: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_PATH,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if USE_LORA:
        print(f"   LoRA: {LORA_MODEL_PATH}")
        model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
    else:
        print("   ⚙️ Running base model only")

    model.eval()
    print("✅ Model loaded!\n")
    return model, tokenizer


def evaluate(model, tokenizer, samples):
    """Run evaluation on test samples"""
    print("\n🧩 Running quick test...\n")
    results = []

    for i, sample in enumerate(samples, 1):
        question = sample["question"]
        expected = str(sample["answer"])

        print("=" * 100)
        print(f"🧠 Question {i}: {question}")
        print("=" * 100)

        prompt = PROMPT_TEMPLATE.format(question=question)

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract phần model trả lời
        answer_text = generated[len(prompt):].strip()

        print("📝 Model output:")
        print(answer_text)
        print(f"🎯 Expected: {expected}")

        correct = expected in answer_text
        results.append((question, expected, answer_text, "✅" if correct else "❌"))
        print(f"✅ Correct? {correct}\n")

    return results


def print_summary(results):
    """In bảng tổng hợp kết quả"""
    print("\n" + "#" * 100)
    print("# SUMMARY RESULTS")
    print("#" * 100)

    for q, exp, pred, status in results:
        print(f"Q: {q}")
        print(f"Expected: {exp}")
        print(f"Predicted: {pred}")
        print(f"Result: {status}")
        print("-" * 100)

    correct_count = sum(1 for r in results if r[-1] == "✅")
    print(f"\n🎯 Accuracy: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")


def main():
    test_file = os.path.join(DATA_CACHE_PATH, "quick_test.jsonl")
    samples = load_test_data(test_file, limit=5)
    model, tokenizer = load_model()
    results = evaluate(model, tokenizer, samples)
    print_summary(results)


if __name__ == "__main__":
    main()
