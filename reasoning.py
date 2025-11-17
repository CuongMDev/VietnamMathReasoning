# 1. Cài đặt + import
#pip install -q --upgrade transformers accelerate bitsandbytes sentencepiece
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import re

# 2. Load model
model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ),
    trust_remote_code=True
)

print("Model đã sẵn sàng")

# 3. Hàm giải toán

def solve_math(question):
    prompt = f"""You are a math expert. Solve the following problem step by step.
Put your final answer inside \boxed{{}}.

Question: {question}"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.0,
        do_sample=False,
        repetition_penalty=1.1
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Lấy toàn bộ phần suy luận (từ sau "Question:" trở đi cho đẹp)
    think = text.split("Question:")[1] if "Question:" in text else text
    think = think.strip()
    
    # === CÁCH LẤY RESPONSE CHẮC CHẮN NHẤT ===
    boxed_match = re.search(r'\\boxed\{(.*?)\}', text)
    if boxed_match:
        response = boxed_match.group(1).strip()
    else:
        # Nếu model không dùng \boxed (rất hiếm), lấy số cuối cùng
        numbers = re.findall(r'\d+', text)
        response = numbers[-1] if numbers else "Không tìm thấy"
    
    return {
        "instruction": question.strip(),
        "think": think,
        "response": response
    }

# 4. Test
question = "Kirsty collects small models of animals. The last time she counted she had 240 models in total. She had 3 times as many dogs as cats, and twice as many cats as birds. How many dog models does Kirsty have?"

result = solve_math(question)
print(json.dumps(result, indent=2, ensure_ascii=False))