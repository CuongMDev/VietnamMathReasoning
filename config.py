# ⚙️ Cấu hình
import torch

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULT_FILE = "benchmark_results.csv"
MODEL_CACHE_PATH = "./hf_cache/model"
DATA_CACHE_PATH = "./hf_cache/data"
INSTRUCTION_DATA_PATH = "data/"       # file jsonl như mô tả ở trên

# ⭐ Thêm cấu hình cho model test
LORA_MODEL_PATH = "sft-lora-model-tir"  # Model bạn muốn test
USE_LORA = True  # Set False nếu muốn test base model

PROMPT_TEMPLATE = """### Instruction:
Solve the problem step-by-step and reply in English only.
Put final answer in \\boxed

### Question:
{question}

### Answer:
"""