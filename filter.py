import random
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
        r"\\frac\s*\d"        # \frac ngay sau đó là số mà không có { }
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)

def is_step_by_step(solution: str) -> bool:
    # 2️⃣ Check solution có numbered steps dạng \n1., \n2., ...
    numbered_steps = re.findall(r'(?<=\n)\s*(\d+)\.', solution)
    if len(numbered_steps) < 2:
        return False
    
    return True
def is_good_thinking(think, solution: str, max_segment_chars=None) -> bool:
    """Kiểm tra think và numbered steps kiểu newline + số + dấu chấm."""
    
    # 1️⃣ Kiểm tra think
    if think is not None:
        segments = think.split("\n\n")
        for seg in segments:
            if max_segment_chars is not None and len(seg.strip()) > max_segment_chars:
                return False
        keywords = ["step", "→", "=>", "first", "therefore", "so", "then", "hence"]
        if not any(k in think.lower() for k in keywords):
            return False

    if "\\begin{" in solution:
        return False

    return True

def fix_multiple_choice_answer(solution):
    def replace_boxed(match):
        content = match.group(1)
        # Bắt chữ cái A-D (case-insensitive), có thể có dấu . ) : , ; hoặc space theo sau
        letter_match = re.search(r'(?i)([A-D])(?=[\s\.\)\,:;]|$)', content)
        if letter_match:
            return f"\\boxed{{{letter_match.group(1).upper()}}}"
        return match.group(0)

    # Match \boxed{...} với capture group cho nội dung bên trong (hỗ trợ nested {})
    pattern = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
    solution = re.sub(pattern, replace_boxed, solution or "")

    return solution

def process_mcq(instruction: str, solution: str) -> str:
    """
    Xử lý MCQ:
    - Phát hiện nhiều dạng nhãn A), A., (A), Answer: A, ...
    - Nếu có \boxed{...} thì chỉ giữ letter A-D trong \boxed{} (nếu tìm được).
    - Nếu không tìm letter, giữ nguyên \boxed{...}.
    """
    # has_mcq_instr = bool(re.search(r'(?i)(?:\([A-D]\)|[A-D][\)\.]|Answer\s*[:]\s*[A-D])', instruction or ""))
    
    # if has_mcq_instr:
    #     solution = fix_multiple_choice_answer(solution)
    
    # 3️⃣ Chuyển \n1., \n2., ... thành ### Step 1, ### Step 2
    """
    Chuyển các dòng kiểu \n1, \n2, ... thành \n### Step 1:, \n### Step 2:, ...
    """
    def repl(match):
        num = match.group(1)
        return f"\n\n---\n\n### Step {num}: "
    
    # Regex: tìm \n theo sau là số và dấu chấm
    solution = re.sub(r'\n\s*(\d+)\.', repl, solution)
    
    return solution.strip()

def clean_reasoning(text: str) -> str:
    # 1) Remove filler words (case-insensitive) together with following commas/spaces
    fillers = [
        "hmm", "okay", "let me think", "i should probably", "i remember",
        "wait", "hold on", "check", "seems", "perhaps", "i will",
        "first", "alright", "so", "right"
    ]
    # build a single regex for fillers (word-boundary), case-insensitive
    filler_pattern = r'\b(?:' + '|'.join(re.escape(f) for f in fillers) + r')\b[ ,]*'
    filler_re = re.compile(filler_pattern, re.IGNORECASE)
    text = filler_re.sub('', text)

    # 2) Normalize some whitespace but do NOT touch operator spacing in math expressions
    text = re.sub(r'\u00A0', ' ', text)      # NBSP -> space
    text = re.sub(r'[ \t\f\v]+', ' ', text)  # collapse weird whitespace
    text = text.strip()

    # 3) Sentence split (prefer splitting on .!? followed by whitespace)
    if re.search(r'[.!?]', text):
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    else:
        raw_sentences = text.splitlines()

    # 4) Compile patterns used in loop (performance + correctness)
    keep_re = re.compile(
        r'[0-9]|=|\+|-|\*|/|\^|\(|\)|sqrt|sin|cos|tan|π|pi|≈|≤|≥',
        re.IGNORECASE
    )
    # connectors at sentence start to remove (use IGNORECASE)
    connector_re = re.compile(r'^(?:so|but|and|then|thus|therefore)\b[ ,]*', re.IGNORECASE)
    # leading punctuation to strip
    leading_punct_re = re.compile(r'^[,;:\-\s"\'\(\)]+')

    out_sentences = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue

        # remove leading connectors like "so", "but", etc.
        s = connector_re.sub('', s)
        # remove leading punctuation/spaces (if any)
        s = leading_punct_re.sub('', s)

        if not s:
            continue

        # keep only math-relevant sentences
        if keep_re.search(s):
            # collapse repeated spaces (but DO NOT add spaces around operators)
            s = re.sub(r' {2,}', ' ', s)
            out_sentences.append(s)

    # join with one sentence per line, no extra blank lines or trailing spaces
    clean_text = "\n".join(out_sentences).strip()
    return clean_text