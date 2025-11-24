import json
import re
import os
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Disable HF warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load model
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Pattern để giữ nguyên số + công thức + ký tự đặc biệt
PATTERN = r"(\d+|<<.*?>>|[\(\)\[\]\{\}%]|[\S]*[<>/*+=-^][\S]*|\\\S)"

def translate_fragment(text, src, tgt):
    # Tách theo newline trước
    lines = text.split('\n')
    translated_lines = []

    for line in lines:
        parts = re.split(PATTERN, line)
        result = []

        for p in parts:
            if re.fullmatch(PATTERN, p):
                # Thêm dấu cách trước và sau phần khớp PATTERN
                result.append(f"{p} ")
                continue

            if p.strip() == "":
                result.append(p)
                continue

            tokenizer.src_lang = src
            encoded = tokenizer(p, return_tensors="pt", truncation=True)
            bos = tokenizer.get_lang_id(tgt)

            with torch.no_grad():
                out = model.generate(**encoded, forced_bos_token_id=bos, max_length=512)

            translated = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            result.append(translated)

        translated_lines.append("".join(result))

    # Ghép lại các dòng với \n
    return "\n".join(translated_lines)


def translate_item(obj, src, tgt):
    if not isinstance(obj, dict):
        return obj

    new_obj = obj.copy()

    if "instruction" in new_obj:
        new_obj["instruction"] = translate_fragment(new_obj["instruction"], src, tgt)

    if "response" in new_obj:
        new_obj["response"] = translate_fragment(new_obj["response"], src, tgt)

    return new_obj


def main():
    input_file = "train.json"
    output_file = "result.json"

    src, tgt = "en", "vi"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Nếu file là list → dịch từng phần tử
    if isinstance(data, list):
        translated = [translate_item(x, src, tgt) for x in data]
    else:
        translated = translate_item(data, src, tgt)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    print("\nHoàn tất! JSON chỉ dịch instruction + response.")
    print("Lưu tại:", output_file)

def output():
    src = "en"
    tgt = "vi"

    # Nhập instruction
    print("\nNhập instruction:")
    instruction = input("> ")

    # Nhập response
    print("\nNhập response:")
    response = input("> ")

    # Đóng gói vào object như JSON
    obj = {
        "instruction": instruction,
        "response": response
    }

    # Dịch bằng hàm đã có
    translated = translate_item(obj, src, tgt)

    # In kết quả
    print("\n----- KẾT QUẢ -----")
    print("Instruction đã dịch:")
    print(translated["instruction"])

    print("\nResponse đã dịch:")
    print(translated["response"])

if __name__ == "__main__":
    main()
    #output()
