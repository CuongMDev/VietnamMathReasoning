"""Reward functions for GRPO training."""

import torch
import math
import re
from typing import Dict

from count_thinking_words import count_thinking_words
from utils import extract_boxed, is_answer_equal, is_answer_equal

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        pred = extract_boxed(content)
        reward = float(is_answer_equal(pred, sol, ignore_not_parseable=True))
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    
    def count_tags(text: str) -> float:
        count = 0.0
        # We only count </think> tag, because <think> tag is available in system prompt
        if text.count("\n</think>\n") == 1:
            count += 1.0
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


# def english_reward(completions, **kwargs):
#     """Reward function that checks if the reasoning process is in English."""
#     import unicodedata
#     from langdetect import detect, LangDetectException

#     def is_non_english(text):
#         """
#         Checks if the given text contains languages other than English.
#         Ignores LaTeX notation.
        
#         Args:
#             text (str): The text to analyze
            
#         Returns:
#             bool: False if the text is in English (with LaTeX allowed),
#                 True if it contains non-English languages
#         """
#         # Skip if empty
#         if not text or text.strip() == "":
#             return False
        
#         # First, remove LaTeX notation to avoid false positives
#         # This pattern matches typical LaTeX structures like $...$ or \begin{...}...\end{...}
#         latex_pattern = r'\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]|\\begin\{.*?\}.*?\\end\{.*?\}'
#         text_without_latex = re.sub(latex_pattern, '', text, flags=re.DOTALL)
        
#         # Also remove common LaTeX commands
#         latex_commands = r'\\[a-zA-Z]+((\{[^{}]*\})?|(\[[^\[\]]*\])?)+'
#         text_without_latex = re.sub(latex_commands, '', text_without_latex)
        
#         # Check if we have non-ASCII characters that are not typical in English text
#         # First, normalize unicode characters
#         normalized_text = unicodedata.normalize('NFKD', text_without_latex)
        
#         # Common non-English character sets (excluding common punctuation and symbols)
#         non_english_patterns = [
#             # Cyrillic characters
#             r'[\u0400-\u04FF]',
#             # Chinese/Japanese/Korean characters
#             r'[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]',
#             # Arabic characters
#             r'[\u0600-\u06FF]',
#             # Hebrew characters
#             r'[\u0590-\u05FF]',
#             # Thai characters
#             r'[\u0E00-\u0E7F]',
#             # Greek characters
#             r'[\u0370-\u03FF]',
#         ]
        
#         for pattern in non_english_patterns:
#             if re.search(pattern, normalized_text):
#                 return True
        
#         # If no obvious non-English characters found, try language detection
#         # Clean text further - remove URLs, numbers, punctuation
#         cleaned_text = re.sub(r'http\S+|www\S+|\d+|[^\w\s]', ' ', text_without_latex)
#         cleaned_text = ' '.join(cleaned_text.split())
        
#         # Only perform language detection if we have enough text
#         if len(cleaned_text.split()) >= 5:
#             try:
#                 detected_lang = detect(cleaned_text)
#                 return detected_lang != 'en'
#             except LangDetectException:
#                 # If detection fails, rely on character-based detection above
#                 pass
        
#         # Default to assuming it's English
#         return False

#     contents = [completion[0]["content"] for completion in completions]
#     return [0 if has_non_english(content) else 1 for content in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        pred = extract_boxed(content)
        reward = float(is_answer_equal(pred, sol, ignore_not_parseable=True))
        correctness.append(reward)

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            pred = extract_boxed(content)
            is_correct = is_answer_equal(pred, sol, ignore_not_parseable=True)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = min(gen_len / max_len, 1.0)
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward

def get_cosine_backtracking_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_word: int = 20,
):
    
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_word: Maximum word count for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            pred = extract_boxed(content)
            is_correct = is_answer_equal(pred, sol, ignore_not_parseable=True)
            gen_word = count_thinking_words(content)[1]

            # Apply cosine scaling based on length
            progress = min(gen_word / max_word, 1.0)
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward

def logprob_confidence_reward(completions, **kwargs):
    """
    Reward based on average token log-probability (model confidence).

    Maps mean logprob from [-5, 0] -> [0, 1].

    Higher confidence => higher reward.
    """

    rewards = []

    for comp in completions:

        # Handle TRL formats: [ {..} ] or {..}
        if isinstance(comp, list):
            c = comp[0]
        elif isinstance(comp, dict):
            c = comp
        else:
            rewards.append(0.0)
            continue

        logps = c.get("logprobs", None)

        if logps is None or len(logps) == 0:
            rewards.append(0.0)
            continue

        if not isinstance(logps, torch.Tensor):
            logps = torch.tensor(logps)

        # Important: detach from graph
        logps = logps.detach()

        # Mean token log-prob
        avg_logp = logps.mean()

        # Clamp for numerical stability
        avg_logp = torch.clamp(avg_logp, -5.0, 0.0)

        # Normalize [-5, 0] -> [0, 1]
        reward = (avg_logp + 5.0) / 5.0

        rewards.append(float(reward))

    return rewards
