"""
Filter calculus data using local AI model.
Calculus topics:
- Limits
- Derivatives
- Integrals
- Differential equations
- Continuity
- Function analysis
"""

import json
import os
import re
import torch
from typing import Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_CACHE_PATH


# ==================== CALCULUS KEYWORDS ====================
CALCULUS_KEYWORDS_EN = [
    # Limits
    r"\blimit\b", r"\blim\b", r"\\lim", r"approaches", r"tends to",
    # Derivatives
    r"\bderivative\b", r"\bdifferentiate\b", r"\bdifferentiation\b",
    r"f'\(", r"f''", r"dy/dx", r"d/dx", r"\\frac\{d", r"\\partial",
    # Integrals (exclude "integral number/part" = whole number)
    r"\bintegral\b(?!\s+(?:number|part|value))", r"\bintegrate\b", r"\bintegration\b",
    r"\\int", r"antiderivative", r"primitive",
    # Series
    r"convergent", r"divergent",
    r"taylor", r"maclaurin", r"\\sum.*n=",
    # Differential equations
    r"differential equation", r"\bODE\b", r"\bPDE\b",
    # Continuity (require math context)
    r"\bcontinuous\b(?=\s*(?:\(|function|at|on|over|from|if|and\s+differentiable))",
    r"\bcontinuity\b", r"discontinuous",
    # Function analysis
    r"critical point", r"inflection", r"concav", r"extrema",
    r"maximum.*function", r"minimum.*function", r"monoton",
    r"increasing.*function", r"decreasing.*function",
    # Other calculus terms
    r"rate of change", r"instantaneous", r"tangent line",
    r"area under", r"volume of revolution", r"l'h[oô]pital",
]

CALCULUS_PATTERNS = CALCULUS_KEYWORDS_EN


def has_calculus_keywords(text: str) -> bool:
    """Quick check using keyword matching."""
    text_lower = text.lower()
    for pattern in CALCULUS_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def is_duplicate(text: str, seen: set) -> bool:
    """Check if text is duplicate (and add to seen set if not)."""
    normalized = text.strip().lower()
    if normalized in seen:
        return True
    seen.add(normalized)
    return False


def has_boxed_answer(response: str) -> bool:
    """Check if response contains \\boxed{} answer."""
    return '\\boxed' in (response or '')


# ==================== AI CLASSIFICATION ====================

SYSTEM_PROMPT = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You are a math topic classifier. Your task is to determine if a math problem belongs to CALCULUS.

Calculus topics include:
- Limits
- Derivatives and differentiation
- Integrals and integration
- Differential equations
- Continuity
- Function analysis: extrema, monotonicity, concavity, inflection points

NOT calculus:
- Basic algebra, equations, inequalities
- Geometry (without calculus methods)
- Combinatorics, probability (without limits/integrals)
- Number theory
- Linear algebra (matrices, vectors without calculus)

Answer with ONLY one word: "CALCULUS" or "OTHER"
"""

USER_PROMPT = """Classify this problem:

{problem}"""


class AIClassifier:
    """Base class for AI-based calculus classification."""

    def classify(self, problem: str) -> bool:
        """Return True if problem is calculus-related."""
        raise NotImplementedError

    def classify_batch(self, problems: list[str]) -> list[bool]:
        """Classify multiple problems."""
        return [self.classify(p) for p in problems]


class LocalModelClassifier(AIClassifier):
    """Classify using a local LLM (transformers)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 10,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Loading model {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=MODEL_CACHE_PATH,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=MODEL_CACHE_PATH,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        print(f"✅ Model loaded successfully!")

    def _prepare_messages(self, problem: str) -> list[dict]:
        """Prepare system and user messages for classification."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(problem=problem[:1500])}
        ]

    def classify(self, problem: str) -> bool:
        """Classify a single problem."""
        messages = self._prepare_messages(problem)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip().upper()

        return "CALCULUS" in response

    def classify_batch(self, problems: list[str], batch_size: int = 8) -> list[bool]:
        """Classify multiple problems with batching for efficiency."""
        results = []

        for i in tqdm(range(0, len(problems), batch_size), desc="Batch classifying", disable=True):
            batch = problems[i:i + batch_size]
            batch_results = [self.classify(p) for p in batch]
            results.extend(batch_results)

        return results


class KeywordClassifier(AIClassifier):
    """Simple keyword-based classifier (no model needed, very fast)."""

    def classify(self, problem: str) -> bool:
        return has_calculus_keywords(problem)


class HybridClassifier(AIClassifier):
    """
    Hybrid approach:
    1. First check with keywords (fast)
    2. If no keyword found, use AI for deeper analysis
    """

    def __init__(self, ai_classifier: AIClassifier):
        self.ai_classifier = ai_classifier

    def classify(self, problem: str) -> bool:
        # Fast keyword check first
        if has_calculus_keywords(problem):
            return True

        # Use AI for uncertain cases
        return self.ai_classifier.classify(problem)


# ==================== DATA FILTERING ====================

def filter_calculus_data(
    input_path: str,
    output_path: str,
    classifier: AIClassifier,
    problem_key: str = "problem",
    response_key: str = "response",
    batch_size: int = 100,
    save_rejected: bool = False,
    filter_duplicates: bool = True,
    filter_boxed: bool = True
):
    """
    Filter calculus problems from a JSON dataset.

    Args:
        input_path: Path to input JSON file (list of dicts)
        output_path: Path to save filtered data
        classifier: AIClassifier instance
        problem_key: Key containing the problem text
        response_key: Key containing the response text
        batch_size: Number of problems to classify in each batch
        save_rejected: Whether to save rejected problems to a separate file
        filter_boxed: Whether to filter out samples without \\boxed in response
    """
    print(f"📂 Loading data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📊 Total samples: {len(data)}")

    calculus_data = []
    other_data = []
    seen = set()
    duplicate_count = 0
    no_boxed_count = 0

    # Process in batches
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch = data[i:i + batch_size]
        problems = [item.get(problem_key, "") for item in batch]

        # Classify batch
        if isinstance(classifier, LocalModelClassifier):
            results = classifier.classify_batch(problems, batch_size=8)
        else:
            results = classifier.classify_batch(problems)

        for item, is_calc in zip(batch, results):
            problem = item.get(problem_key, "")
            response = item.get(response_key, "")
            # Filter duplicates
            if filter_duplicates and is_duplicate(problem, seen):
                duplicate_count += 1
                continue
            # Filter boxed
            if filter_boxed and not has_boxed_answer(response):
                no_boxed_count += 1
                continue
            if is_calc:
                calculus_data.append(item)
            else:
                other_data.append(item)

    # Save filtered data
    print(f"\n✅ Calculus samples: {len(calculus_data)}")
    print(f"❌ Other samples: {len(other_data)}")
    if filter_duplicates:
        print(f"🔄 Duplicates removed: {duplicate_count}")
    if filter_boxed:
        print(f"📦 No boxed removed: {no_boxed_count}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calculus_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved calculus data to {output_path}")

    if save_rejected:
        rejected_path = output_path.replace(".json", "_rejected.json")
        with open(rejected_path, "w", encoding="utf-8") as f:
            json.dump(other_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved rejected data to {rejected_path}")

    return calculus_data, other_data


def filter_calculus_from_hf_dataset(
    dataset_path: str,
    output_path: str,
    classifier: AIClassifier,
    problem_key: str = "problem",
    response_key: str = "response",
    subset: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = True,
    filter_duplicates: bool = True,
    filter_boxed: bool = True
):
    """
    Filter calculus problems directly from a HuggingFace dataset.
    """
    from datasets import load_dataset
    from config import DATA_CACHE_PATH

    print(f"📥 Loading {dataset_path}...")
    ds = load_dataset(dataset_path, subset, split=split, cache_dir=DATA_CACHE_PATH, streaming=streaming)

    if max_samples:
        ds = ds.take(max_samples)

    calculus_data = []
    seen = set()
    duplicate_count = 0
    no_boxed_count = 0

    for ex in tqdm(ds, desc="Filtering"):
        problem = ex.get(problem_key, "")
        response = ex.get(response_key, "")
        # Filter duplicates
        if filter_duplicates and is_duplicate(problem, seen):
            duplicate_count += 1
            continue
        # Filter boxed
        if filter_boxed and not has_boxed_answer(response):
            no_boxed_count += 1
            continue
        if classifier.classify(problem):
            calculus_data.append(dict(ex))

    print(f"\n✅ Found {len(calculus_data)} calculus samples")
    if filter_duplicates:
        print(f"🔄 Duplicates removed: {duplicate_count}")
    if filter_boxed:
        print(f"📦 No boxed removed: {no_boxed_count}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calculus_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved to {output_path}")

    return calculus_data


# ==================== MAIN ====================

if __name__ == "__main__":
    # Example usage

    # 1. Keyword-based classifier (no model needed, very fast but less accurate)
    # classifier = KeywordClassifier()

    # 2. Local model classifier (accurate, runs on GPU/CPU)
    # classifier = LocalModelClassifier(model_name="Qwen/Qwen2.5-1.5B-Instruct")

    # 3. Hybrid: keywords first, then local model for uncertain cases
    # classifier = HybridClassifier(LocalModelClassifier(model_name="Qwen/Qwen2.5-1.5B-Instruct"))

    # Use keyword classifier by default (fast, no GPU needed)
    classifier = KeywordClassifier()

    # Uncomment to use local model:
    # classifier = LocalModelClassifier(
    #     model_name="Qwen/Qwen3-1.7B",
    #     torch_dtype=torch.bfloat16,  # or torch.float16 for older GPUs
    # )

    # Filter from local JSON file
    input_file = "./data/calculus_data_raw.json"
    output_file = "./data/calculus_data.json"

    if os.path.exists(input_file):
        filter_calculus_data(
            input_path=input_file,
            output_path=output_file,
            classifier=classifier,
            problem_key="problem",
            save_rejected=True,
            filter_duplicates=False
        )
    else:
        print(f"⚠️ Input file not found: {input_file}")
        print("Run build_data.py first or specify a different input path.")

    # Alternative: Filter directly from HuggingFace dataset
    # filter_calculus_from_hf_dataset(
    #     dataset_path="nvidia/OpenMathReasoning",
    #     output_path="./data/calculus_from_nvidia.json",
    #     classifier=classifier,
    #     problem_key="problem",
    #     split="cot",
    #     max_samples=10000
    # )
