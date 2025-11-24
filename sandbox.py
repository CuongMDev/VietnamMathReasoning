import torch
import tempfile
import subprocess
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


# ======================================================
# 1️⃣ Load model
# ======================================================
def load_model(model_name: str, device="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    return model, tokenizer


# ======================================================
# 2️⃣ Sinh code Python từ prompt (hỗ trợ batch + streamer)
# ======================================================
def generate_code_batch(model, tokenizer, prompts, max_new_tokens=256):
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    full_text = ""
    for chunk in streamer:
        print(chunk, end="", flush=True)
        full_text += chunk

    thread.join()
    return full_text


# ======================================================
# 3️⃣ Hàm thực thi code trong sandbox
# ======================================================
def run_in_sandbox(code: str, timeout: float = 3.0) -> str:
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
    except subprocess.TimeoutExpired:
        output = "⚠️ Timeout: code chạy quá lâu"
    except Exception as e:
        output = f"⚠️ Lỗi sandbox: {e}"

    return output


# ======================================================
# 4️⃣ Pipeline hoàn chỉnh: từ prompt → code → kết quả thực thi
# ======================================================
def run_pipeline(model, tokenizer, prompts):
    codes = []
    outputs = []
    print("\n=== 🔹 Sinh code từ mô hình ===")
    generated = generate_code_batch(model, tokenizer, prompts)

    # Nếu model sinh nhiều code, bạn có thể cắt bằng <|end|> hoặc dấu ```output
    codes = [generated.strip()]

    print("\n\n=== 🔹 Thực thi sandbox ===")
    for i, code in enumerate(codes):
        print(f"\n🧮 Batch {i}:")
        result = run_in_sandbox(code)
        print(result)
        outputs.append(result)

    return outputs


# ======================================================
# 5️⃣ Ví dụ sử dụng
# ======================================================
if __name__ == "__main__":
    model_name = "Qwen/Qwen2-1.5B-Instruct"
    model, tokenizer = load_model(model_name)

    prompts = [
        """Viết code Python sau:
Tính thể tích tứ diện có ma trận Cayley-Menger như sau:
C = np.array([
    [0, 1, 1, 1, 1],
    [1, 0, 41, 80, 89],
    [1, 41, 0, 80, 89],
    [1, 80, 80, 0, 89],
    [1, 89, 89, 89, 0]
])
Sau đó in ra thể tích V.
""",
    ]

    run_pipeline(model, tokenizer, prompts)
