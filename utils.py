import re
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, verify, parse

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
                       "You are an expert in calculus (limits, derivatives, integrals, differential equations). "
        })
        if boxed_force:
            messages[-1]["content"] += "Please reason step by step, and put your final answer within \\boxed{}. The \\boxed{} should contain ONLY the final answer (number, expression, or value) without any explanation or units unless the problem specifically asks for it."
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
    if start == -1:
        return ""
    
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

def extract_after_equals(s: str) -> str:
    """Extract the part after '=' if present, otherwise return original string."""
    if '=' in s:
        return s.split('=')[-1].strip()
    return s.strip()

def __is_answer_equal(pred: str, gt: str, ignore_not_parseable=False) -> bool:
    pred = f"${pred}$"
    gt = f"${gt}$"
    gold_parsed = parse(gt, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
    if len(gold_parsed) == 0 and ignore_not_parseable:
        return True
        
    answer_parsed = parse(pred, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])

    is_correct = verify(answer_parsed, gold_parsed)
    return is_correct

def _normalize_choice_letter(s: str) -> str:
    """Extract choice letter from plain 'A' or LaTeX like \\text{A}."""
    s = s.strip()
    m = re.match(r'\\text\{([A-Da-d])\}$', s)
    if m:
        return m.group(1).upper()
    if s.upper() in ("A", "B", "C", "D"):
        return s.upper()
    return ""

def _normalize_text_cmd(s: str) -> str:
    r"""Normalize \text{...} -> lowercase content. e.g. \text{DNE} -> dne"""
    return re.sub(r'\\text\{([^}]*)\}', lambda m: m.group(1).lower(), s)

def _normalize_dne(s: str) -> str:
    """Normalize 'does not exist' variants to 'dne'."""
    if re.match(r'does\s+not\s+exist$', s.strip(), re.IGNORECASE):
        return "dne"
    return s

def _extract_choice_value(letter: str, question: str) -> str:
    """If letter is A/B/C/D, extract the corresponding value from the question's choices."""
    letter = letter.strip().upper()
    if letter not in ("A", "B", "C", "D"):
        return ""
    # Match patterns like: A. 5, A) 5, A: 5, **A.** 5, $A.$ 5
    pattern = rf'(?:\*\*|\$)?{letter}[\.\)\:](?:\*\*|\$)?\s*(.+?)(?:\s*(?:(?:\*\*|\$)?[B-D][\.\)\:])|$)'
    match = re.search(pattern, question, re.DOTALL)
    if match:
        return match.group(1).strip().rstrip('.').strip()
    return ""

def is_answer_equal(question: str, pred: str, gt: str, ignore_not_parseable=False) -> bool:
    # Normalize \text{...} -> lowercase, then DNE
    pred = _normalize_text_cmd(pred)
    gt = _normalize_text_cmd(gt)
    pred = _normalize_dne(pred)
    gt = _normalize_dne(gt)

    # Normalize choice letters (A, \text{A})
    pred_letter = _normalize_choice_letter(pred)
    if pred_letter:
        resolved = _extract_choice_value(pred_letter, question)
        if resolved:
            pred = resolved
    gt_letter = _normalize_choice_letter(gt)
    if gt_letter:
        resolved = _extract_choice_value(gt_letter, question)
        if resolved:
            gt = resolved

    # Extract part after '=' if present
    pred_after_equals = extract_after_equals(pred)
    gt_after_equals = extract_after_equals(gt)

    return __is_answer_equal(pred_after_equals, gt_after_equals, ignore_not_parseable) \
        or __is_answer_equal(pred, gt, ignore_not_parseable)


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
