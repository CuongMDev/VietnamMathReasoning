import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from config import MODEL_CACHE_PATH

device = "cuda:0"
model_name = "lmsys/vicuna-7b-v1.3"



# ==== LOAD MODEL ====
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=MODEL_CACHE_PATH
).half().eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)


# ==== ĐỌC PROMPT TEMPLATE TỪ FILE ====
with open("equivalence_prompt.txt", "r", encoding="utf-8") as f:
    raw_prompt = f.read()


def fill_prompt(instruction, ans1, ans2):
    prompt = raw_prompt
    prompt = prompt.replace('""{instruction}""', instruction)
    prompt = prompt.replace('""{ans1}""', ans1)
    prompt = prompt.replace('""{ans2}""', ans2)
    return prompt


def run_equivalence_model(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded

def extract_equivalence(text):
    m = re.search(r'boxed\s*\{\s*(TRUE|FALSE)\s*\}', text, re.I)
    if not m:
        return None
    val = m.group(1).upper()
    return True if val == "TRUE" else False

rows = []
with open("answers_to_compare.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

output_file = "equivalence_results.jsonl"
total_true = 0
total_false = 0

with open(output_file, "w", encoding="utf-8") as f_out:
    for item in tqdm(rows, desc="Checking equivalence"):

        prompt = fill_prompt(item["instruction"], item["ans1"], item["ans2"])
        result_text = run_equivalence_model(prompt)
        eq = extract_equivalence(result_text)

        item["equivalent"] = eq
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

        if eq is True:
            total_true += 1
        elif eq is False:
            total_false += 1
        else:
            print("NO BOXED FOUND:", result_text)

    summary = {"true": total_true, "false": total_false}
    f_out.write(json.dumps(summary, ensure_ascii=False) + "\n")

print("DONE! Saved:", output_file)