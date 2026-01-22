import re
import sympy as sp

def find_sublist_indices(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i + len(sub)
    return -1


def make_prompt_template(user_prompt: str, think=None, respond=None, boxed_force=True, add_system=True):
    messages = []
    if add_system:
        messages.append({
            "role": "system",
            "content": "You are a helpful and harmless assistant. " 
                       "You are Qwen developed by Alibaba. "
        })
        if boxed_force:
            messages[-1]["content"] += "Please reason step by step, and put your final answer within \\boxed{}."
    messages.append({
        "role": "user",
        "content": user_prompt
    })

    assistant_content = ""

    # Nếu có think (suy luận nội bộ)
    if think is not None:
        assistant_content += f"<think>\n{think}\n</think>\n"

    # Nếu có respond, thêm message của assistant
    if respond is not None:
        if think is None:
            assistant_content += f"<think>\n\n</think>\n"
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

def remove_leading_zeros(s: str) -> str:
    # Nếu s là số nguyên hoặc phân số đơn giản
    try:
        return str(int(s))  # '025' -> '25'
    except:
        return s

def normalize_expression(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace("\\dfrac", "\\frac")

    # 1) Xử lý phần trăm
    expr = re.sub(r'(\d+)\s*%', r'(\1/100)', expr)
    expr = expr.replace("\\%", "/100")

    # 2) Xử lý dấu phẩy hàng nghìn: 3,003 → 3003
    #    Giữ được cấu trúc số dạng 123,456.78 → 123456.78
    expr = re.sub(r'(?<=\d),(?=\d{3}(\D|$))', '', expr)

    expr = expr.replace(" ", "")
    return expr


def latex_to_sympy(expr: str):
    expr = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', expr)
    expr = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', expr)
    return expr


def is_answer_equal(pred: str, gt: str) -> bool:
    pred_norm = remove_leading_zeros(latex_to_sympy(normalize_expression(pred)))
    gt_norm   = remove_leading_zeros(latex_to_sympy(normalize_expression(gt)))

    try:
        return sp.simplify(f"({pred_norm}) - ({gt_norm})") == 0
    except Exception:
        return pred_norm == gt_norm
    
# remove first and last tags <...>
def remove_tags(text: str) -> str:
    text = re.sub(r'^\s*<[^>]+>\s*', '', text, count=1)
    text = re.sub(r'\s*</?[^>]+>\s*$', '', text, count=1)
    return text

# just remove all tag in tags list, not remove content inside
def remove_all_tags(text: str, tags: list) -> str:
    for tag in tags:
        text = re.sub(fr'</?{tag}\b[^>]*>', '', text, flags=re.IGNORECASE)
    return text
