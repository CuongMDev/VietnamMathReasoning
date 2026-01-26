import re
import sys
from collections import Counter

def count_thinking_words(text):
    """Đếm số lần xuất hiện của các từ thinking/reasoning trong string."""

    thinking_words = [
        "Wait", "Alternatively", "But"
    ]

    # Đếm từng từ (chỉ đếm khi là từ độc lập, không phải một phần của từ khác)
    word_counts = Counter()
    for word in thinking_words:
        # Sử dụng regex để match từ độc lập (word boundary)
        pattern = r'\b' + re.escape(word) + r'\b'
        count = len(re.findall(pattern, text))
        word_counts[word] = count

    total = sum(word_counts.values())
    return word_counts, total, thinking_words


if __name__ == "__main__":
    file_path = "test.txt"
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    word_counts, total, thinking_words = count_thinking_words(text)

    # In kết quả
    print(f"File: {file_path}")
    print("-" * 40)

    for word in thinking_words:
        count = word_counts[word]
        if count > 0:
            print(f"  {word}: {count}")

    print("-" * 40)
    print(f"Tổng cộng: {total}")
