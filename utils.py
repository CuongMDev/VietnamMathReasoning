import re


def find_sublist_indices(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i + len(sub)
    return -1


def make_prompt_template(user_prompt: str, think=None, respond=None):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful and harmless assistant. " 
                       "You are Qwen developed by Alibaba. "
                       "You may reason internally but output only the final answer in LaTeX \\boxed."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    assistant_content = ""

    # Nếu có think (suy luận nội bộ)
    if think is not None:
        assistant_content += f"<think>\n{think}\n</think>\n"

    # Nếu có respond, thêm message của assistant
    if respond is not None:
        assistant_content += respond

    if assistant_content:
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })

    return messages

# 🔢 Hàm trích xuất kết quả trong \boxed{...}
def extract_boxed(s: str):
    start = s.rfind(r"\boxed{")
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

def normalize_answer(ans: str) -> str:
    """
    Chuẩn hóa answer:
    - Bỏ LaTeX \text{} và \boxed{}
    - Bỏ { }, $, ,
    - Trim whitespace
    """
    if not ans:
        return ""
    # Bỏ \text{} và \boxed{}
    ans = re.sub(r'\\text\{[^}]*\}', '', ans)
    ans = re.sub(r'\\boxed\{([^}]*)\}', r'\1', ans)
    # Bỏ ký tự { } $ ,
    ans = re.sub(r'[${},]', '', ans)
    # Trim
    return ans.strip()


def is_answer_equal(pred: str, gt: str) -> bool:
    """
    So sánh 2 answer:
    1. Chuẩn hóa
    2. Nếu convert được sang float thì so sánh float
    3. Nếu không thì so sánh string
    """
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    
    is_correct = False
    try:
        if float(pred_norm) == float(gt_norm):
            is_correct = True
    except:
        if pred_norm.lower() == gt_norm.lower():  # ignore case
            is_correct = True
    
    return is_correct