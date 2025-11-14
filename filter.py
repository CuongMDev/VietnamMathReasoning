import re

def is_low_quality(text: str) -> bool:
    """Kiểm tra xem câu hỏi có lỗi định dạng (low quality) không."""
    patterns = [
        r"```",               # chứa code block chưa đóng
        r"===+|---+",         # ASCII art kiểu gạch dài
        r"\[image.*\]",       # tham chiếu hình ảnh không tồn tại
        r"Figure\s*\d+",      # tham chiếu đến hình bị lỗi
        r"Question\s*\d+\s*:",# lỗi đánh số câu hỏi không đồng nhất
        r"\.\s*\.\s*\.",      # chuỗi dấu chấm lặp kiểu ASCII art
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)

def has_reasoning_steps(think, solution: str) -> bool:
    """Kiểm tra xem có dòng giải tuần tự không."""
    # 1️⃣ Check từ khóa trong think
    keywords = ["step", "→", "=>", "first", "therefore", "so", "then", "hence"]
    if think is not None and not any(k in think.lower() for k in keywords):
        return False
    
    # 1️⃣b Check bắt buộc phải có "wait" trong think
    if think is not None and "wait" not in think.lower():
        return False

    # 2️⃣ Check solution có dạng numbered steps: 1., 2., 3. …
    numbered_steps = re.findall(r'^\s*\d+\.', solution, flags=re.MULTILINE)
    if len(numbered_steps) < 2:
        return False
    return True

def process_mcq(instruction: str, solution: str) -> str:
    """
    Xử lý MCQ:
    - Phát hiện nhiều dạng nhãn A), A., (A), Answer: A, ...
    - Nếu có \boxed{...} thì chỉ giữ letter A-D trong \boxed{} (nếu tìm được).
    - Nếu không tìm letter, giữ nguyên \boxed{...}.
    """
    has_mcq_instr = bool(re.search(r'(?i)(?:\([A-D]\)|[A-D][\)\.]|Answer\s*[:]\s*[A-D])', instruction or ""))
    
    if has_mcq_instr:
        def replace_boxed(match):
            content = match.group(1)
            # Bắt chữ cái A-D (case-insensitive)
            letter_match = re.search(r'(?i)([A-D])(?=[\s\.\)\(,:;]|$)', content)
            if letter_match:
                return f"\\boxed{{{letter_match.group(1).upper()}}}"
            return match.group(0)

        # Match \boxed{...} với capture group cho nội dung bên trong
        pattern = r'\\boxed\{((?:[^{}]|(?:\{[^}]*\}))*)\}'
        solution_clean = re.sub(pattern, replace_boxed, solution or "")
        return solution_clean.strip()

    return (solution or "").strip()