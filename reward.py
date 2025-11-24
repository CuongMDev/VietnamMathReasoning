"""
Reward Functions cho GRPO Training
File này chứa các hàm tính reward khác nhau
"""


def compute_reward(samples, prompts, outputs, tokenizer, device):
    """
    Reward function phức tạp với nhiều tiêu chí

    Args:
        samples: Generated samples từ model
        prompts: List of original prompts
        outputs: List of generated responses
        tokenizer: Tokenizer object
        device: torch device

    Returns:
        List of reward scores (float)
    """
    rewards = []

    for prompt, output in zip(prompts, outputs):
        score = 0.0
        response = output.strip()

        # 1. Length reward (khuyến khích responses đủ dài)
        word_count = len(response.split())
        if word_count < 10:
            length_score = -0.5  # Penalty cho quá ngắn
        elif word_count > 200:
            length_score = 0.3  # Tốt nhưng ko quá dài
        else:
            length_score = min(word_count / 100, 0.5)

        # 2. Quality keywords (từ khóa tích cực)
        quality_keywords = {
            'helpful': 0.2,
            'useful': 0.2,
            'important': 0.15,
            'effective': 0.15,
            'benefit': 0.1,
            'advantage': 0.1,
        }
        keyword_score = sum(
            bonus for word, bonus in quality_keywords.items()
            if word in response.lower()
        )

        # 3. Structure reward (có cấu trúc tốt)
        has_structure = any([
            '\n' in response,  # Có xuống dòng
            response.count('.') > 2,  # Nhiều câu
            ':' in response,  # Có liệt kê
        ])
        structure_score = 0.2 if has_structure else 0

        # 4. Completeness (kết thúc đúng cách)
        ends_properly = response.endswith(('.', '!', '?', ':', '\n'))
        completion_score = 0.1 if ends_properly else -0.1

        # 5. Relevance (check từ khóa của prompt có trong response)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words) / max(len(prompt_words), 1)
        relevance_score = min(overlap * 0.3, 0.3)

        # Tổng hợp reward
        score = (
                length_score +
                keyword_score +
                structure_score +
                completion_score +
                relevance_score
        )

        # Clamp vào range [-1, 2]
        score = max(-1.0, min(2.0, score))
        rewards.append(score)

    return rewards


def simple_reward(samples, prompts, outputs, tokenizer, device):
    """
    Reward function đơn giản chỉ dựa trên length

    Args:
        samples: Generated samples từ model
        prompts: List of original prompts
        outputs: List of generated responses
        tokenizer: Tokenizer object
        device: torch device

    Returns:
        List of reward scores (float)
    """
    rewards = []
    for output in outputs:
        # Score dựa trên độ dài (10-100 từ là tốt)
        length = len(output.split())
        if length < 10:
            score = -0.5
        elif length > 150:
            score = 0.3
        else:
            score = min(length / 100, 1.0)
        rewards.append(score)
    return rewards


def length_and_quality_reward(samples, prompts, outputs, tokenizer, device):
    """
    Reward dựa trên độ dài và một số quality checks cơ bản

    Args:
        samples: Generated samples từ model
        prompts: List of original prompts
        outputs: List of generated responses
        tokenizer: Tokenizer object
        device: torch device

    Returns:
        List of reward scores (float)
    """
    rewards = []

    for prompt, output in zip(prompts, outputs):
        score = 0.0
        response = output.strip()
        length = len(response.split())

        # Length score
        if 20 <= length <= 100:
            length_score = 0.5
        elif 10 <= length < 20 or 100 < length <= 150:
            length_score = 0.2
        else:
            length_score = -0.3

        # Quality checks
        has_multiple_sentences = response.count('.') >= 2
        has_proper_ending = response.endswith(('.', '!', '?'))
        has_structure = '\n' in response or ':' in response

        quality_score = 0.1 * sum([
            has_multiple_sentences,
            has_proper_ending,
            has_structure
        ])

        score = length_score + quality_score
        score = max(-1.0, min(1.5, score))
        rewards.append(score)

    return rewards


def token_efficiency_reward(samples, prompts, outputs, tokenizer, device):
    """
    Reward ưu tiên responses hiệu quả (không dài dòng)

    Args:
        samples: Generated samples từ model
        prompts: List of original prompts
        outputs: List of generated responses
        tokenizer: Tokenizer object
        device: torch device

    Returns:
        List of reward scores (float)
    """
    rewards = []

    for prompt, output in zip(prompts, outputs):
        response = output.strip()

        # Token count
        tokens = tokenizer.encode(response)
        token_count = len(tokens)

        # Reward ngắn gọn nhưng đầy đủ (50-150 tokens)
        if 50 <= token_count <= 150:
            score = 0.8
        elif 30 <= token_count < 50 or 150 < token_count <= 200:
            score = 0.4
        elif token_count < 30:
            score = -0.5  # Quá ngắn
        else:
            score = -0.3  # Quá dài

        # Bonus cho completeness
        if response.endswith(('.', '!', '?')):
            score += 0.2

        rewards.append(max(-1.0, min(1.0, score)))

    return rewards


def coding_task_reward(samples, prompts, outputs, tokenizer, device):
    """
    Reward function chuyên cho coding tasks

    Args:
        samples: Generated samples từ model
        prompts: List of original prompts
        outputs: List of generated responses
        tokenizer: Tokenizer object
        device: torch device

    Returns:
        List of reward scores (float)
    """
    rewards = []

    for prompt, output in zip(prompts, outputs):
        score = 0.0
        response = output.strip()

        # Check for code indicators
        has_code_block = '```' in response
        has_function = 'def ' in response or 'function' in response
        has_comments = '#' in response or '//' in response
        has_imports = 'import ' in response or 'from ' in response

        # Scoring
        if has_code_block:
            score += 0.5
        if has_function:
            score += 0.3
        if has_comments:
            score += 0.2
        if has_imports:
            score += 0.1

        # Length check (code nên đủ dài)
        word_count = len(response.split())
        if word_count < 20:
            score -= 0.4
        elif word_count > 50:
            score += 0.2

        # Check for common bad patterns
        if response.count('...') > 2:  # Too many placeholders
            score -= 0.3

        rewards.append(max(-1.0, min(2.0, score)))

    return rewards


def custom_reward_template(samples, prompts, outputs, tokenizer, device):
    """
    Template để tạo reward function riêng

    Args:
        samples: Generated samples từ model
        prompts: List of original prompts
        outputs: List of generated responses
        tokenizer: Tokenizer object
        device: torch device

    Returns:
        List of reward scores (float)
    """
    rewards = []

    for prompt, output in zip(prompts, outputs):
        score = 0.0
        response = output.strip()

        # TODO: Thêm logic tính reward của bạn ở đây
        # Ví dụ:
        # - Check keywords
        # - Check format
        # - Check length
        # - Check quality

        # Placeholder logic
        score = len(response.split()) / 100

        # Clamp score
        score = max(-1.0, min(2.0, score))
        rewards.append(score)

    return rewards