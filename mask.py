import random
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def mask_step_numbers(tokenizer, prompt_text, input_ids, labels, ignore_id=IGNORE_TOKEN_ID):
    step_token_ids = tokenizer("### Step ", add_special_tokens=False)["input_ids"]
    positions = []  # lưu danh sách (start_pos, length)

    search_text_pos = 0
    search_input_pos = 0

    while True:
        idx = prompt_text.find("### Step ", search_text_pos)
        if idx == -1:
            break

        step_pos = None
        for i in range(search_input_pos, len(input_ids) - len(step_token_ids) + 1):
            if input_ids[i:i+len(step_token_ids)] == step_token_ids:
                step_pos = i
                break

        if step_pos is None:
            search_text_pos = idx + len("### Step ")
            continue

        colon_idx = prompt_text.find(":", idx)
        if colon_idx == -1:
            break

        num_text = prompt_text[idx + len("### Step "): colon_idx]
        num_token_ids = tokenizer(num_text, add_special_tokens=False)["input_ids"]

        # Lưu vị trí bước này
        positions.append((step_pos + len(step_token_ids), len(num_token_ids)))

        search_text_pos = colon_idx + 1
        search_input_pos = step_pos + len(step_token_ids) + len(num_token_ids)

    # ❗ Nếu không tìm được step nào thì return luôn
    if not positions:
        return labels

    # --- Chọn ngẫu nhiên 1 step để giữ nguyên ---
    keep_idx = random.randrange(len(positions))

    # --- Mask tất cả các step còn lại ---
    for i, (start, L) in enumerate(positions):
        if i == keep_idx:
            continue  # skip, không mask
        for j in range(start, start + L):
            labels[j] = ignore_id

    return labels

def mask_words_after_double_newline(tokenizer, input_ids, labels, words_to_mask, mask_ratio, ignore_id=IGNORE_TOKEN_ID):
    # 1. Tạo token sequence của word
    sequence_token_ids = {
        word: tokenizer(word, add_special_tokens=False)["input_ids"]
        for word in words_to_mask
    }

    input_len = len(input_ids)

    # 2. Tìm tất cả các đoạn có thể mask
    candidate_spans = []   # list[(start, end)]

    search_input_pos = 0
    while search_input_pos < input_len:
        matched = False
        for word, seq_ids in sequence_token_ids.items():
            seq_len = len(seq_ids)

            if (search_input_pos + seq_len <= input_len 
                and input_ids[search_input_pos:search_input_pos+seq_len] == seq_ids):

                # Đây là một span có thể mask
                candidate_spans.append((search_input_pos, search_input_pos + seq_len))

                search_input_pos += seq_len
                matched = True
                break

        if not matched:
            search_input_pos += 1

    # 3. Tính số span sẽ mask theo mask_ratio
    total = len(candidate_spans)
    num_mask = int(total * mask_ratio)

    # num_mask spans sẽ được chọn random
    spans_to_mask = set(random.sample(range(total), num_mask))

    # 4. Mask thật sự
    for i, (start, end) in enumerate(candidate_spans):
        if i in spans_to_mask:
            for j in range(start, end):
                labels[j] = ignore_id

    return labels