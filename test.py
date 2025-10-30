import re


def extract_boxed(s: str):
    """
    Lấy toàn bộ nội dung trong \\boxed{...}, kể cả khi có ngoặc lồng nhau.
    Nếu không có boxed, lấy phần sau dấu '=' hoặc dòng cuối cùng.
    """
    start = s.find(r"\boxed{")
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

    # Nếu không có boxed, thử lấy phần sau dấu '='
    eq_match = re.search(r"=\s*([^\n]*)$", s)
    if eq_match:
        return eq_match.group(1).strip()

    # Nếu vẫn không có, lấy dòng cuối cùng không rỗng
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if lines:
        return lines[-1]

    return "?"

print(extract_boxed("= 1}^\infty \frac{1}{n^3} \\&= \\boxed{p - q}.\end{align*}"))