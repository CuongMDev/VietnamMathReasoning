import re
from typing import Optional, Tuple, Dict


class CalculationDetector:
    """
    Phát hiện khi nào model cần thực hiện calculation và sinh code tương ứng
    """

    # Patterns để phát hiện cần tính toán
    CALCULATION_PATTERNS = [
        r'calculate\s+(?:the\s+)?(.+)',
        r'compute\s+(?:the\s+)?(.+)',
        r'find\s+(?:the\s+)?(?:value\s+of\s+)?(.+)',
        r'what\s+is\s+(.+\s*[\+\-\*/\^]\s*.+)',
        r'evaluate\s+(.+)',
        r'solve\s+for\s+(.+)',
        r'determine\s+(?:the\s+)?(.+)',
    ]

    # Patterns cho các phép toán cụ thể (FIXED)
    MATH_OPERATIONS = {
        'factorial': r'(\d+)\s*!',
        'power': r'(\d+(?:\.\d+)?)\s*\^\s*(\d+(?:\.\d+)?)',
        'sqrt': r'(?:sqrt|√)\s*\(?(\d+(?:\.\d+)?)\)?',
        'basic_ops': r'(\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(\d+(?:\.\d+)?)',
        # FIXED: Pattern linh hoạt hơn cho sum/product range
        'sum_range': r'sum\s+(?:of\s+)?(?:all\s+)?(?:numbers?\s+)?(?:from\s+)?(\d+)\s+to\s+(\d+)',
        'product_range': r'product\s+(?:of\s+)?(?:all\s+)?(?:numbers?\s+)?(?:from\s+)?(\d+)\s+to\s+(\d+)',
    }

    def __init__(self):
        self.last_detected_position = -1

    def detect_needs_calculation(self, text: str, min_gap: int = 100) -> bool:
        """
        Phát hiện xem có cần tính toán không
        min_gap: khoảng cách tối thiểu (chars) từ lần detect trước để tránh spam
        """
        # Nếu quá gần lần detect trước, skip
        if len(text) - self.last_detected_position < min_gap:
            return False

        text_lower = text.lower()

        # ⭐ PRIORITY: Check các phép toán cụ thể TRƯỚC (detect sớm hơn)

        # 1. Factorial - detect ngay khi thấy keyword VÀ có số
        if 'factorial' in text_lower:
            # FIXED: Kiểm tra có số không
            if re.search(r'\d+', text):
                self.last_detected_position = len(text)
                return True

        # 2. Sum/Product of even/odd numbers
        if ('even' in text_lower or 'odd' in text_lower) and 'sum' in text_lower:
            if 'from' in text_lower and 'to' in text_lower:
                self.last_detected_position = len(text)
                return True

        # 3. Power/exponent
        if '^' in text or 'power' in text_lower or '**' in text:
            self.last_detected_position = len(text)
            return True

        # 4. Square root
        if 'sqrt' in text_lower or 'square root' in text_lower:
            self.last_detected_position = len(text)
            return True

        # 5. Check calculation keywords
        for pattern in self.CALCULATION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                self.last_detected_position = len(text)
                return True

        # 6. Check có phép toán phức tạp không (nhiều chữ số, nhiều phép toán)
        complex_calc = re.search(r'\d{3,}\s*[\+\-\*/\^]\s*\d{3,}', text)
        if complex_calc:
            self.last_detected_position = len(text)
            return True

        # 7. Check có nhiều phép toán liên tiếp không
        multi_ops = re.findall(r'\d+(?:\.\d+)?\s*[\+\-\*/\^]\s*\d+(?:\.\d+)?', text)
        if len(multi_ops) >= 2:
            self.last_detected_position = len(text)
            return True

        return False

    def extract_calculation_context(self, text: str, window: int = 200) -> str:
        """
        Trích xuất context xung quanh vị trí cần tính toán
        """
        # Lấy 200 ký tự cuối (hoặc toàn bộ nếu ngắn hơn)
        return text[-window:] if len(text) > window else text

    def generate_calculation_code(self, context: str, question: str = "") -> Optional[str]:
        """
        Sinh Python code để thực hiện calculation dựa trên context
        FIXED: Better error handling và pattern matching
        """
        code_lines = ["import math", "import numpy as np", ""]

        context_lower = context.lower()

        # 1. Phát hiện sum range: "sum from 1 to 100"
        sum_match = re.search(self.MATH_OPERATIONS['sum_range'], context_lower)
        if sum_match:
            start, end = sum_match.groups()
            code_lines.append(f"# Sum from {start} to {end}")
            code_lines.append(f"result = sum(range({start}, {end}+1))")
            code_lines.append("print(result)")
            return "\n".join(code_lines)

        # 2. Phát hiện sum of even/odd numbers
        if 'even' in context_lower and 'sum' in context_lower:
            match = re.search(r'from\s+(\d+)\s+to\s+(\d+)', context_lower)
            if match:
                start, end = match.groups()
                code_lines.append(f"# Sum of even numbers from {start} to {end}")
                code_lines.append(f"result = sum(i for i in range({start}, {end}+1) if i % 2 == 0)")
                code_lines.append("print(result)")
                return "\n".join(code_lines)

        if 'odd' in context_lower and 'sum' in context_lower:
            match = re.search(r'from\s+(\d+)\s+to\s+(\d+)', context_lower)
            if match:
                start, end = match.groups()
                code_lines.append(f"# Sum of odd numbers from {start} to {end}")
                code_lines.append(f"result = sum(i for i in range({start}, {end}+1) if i % 2 == 1)")
                code_lines.append("print(result)")
                return "\n".join(code_lines)

        # 3. Phát hiện factorial - FIXED: Better number extraction
        fact_match = re.search(self.MATH_OPERATIONS['factorial'], context)
        if fact_match or 'factorial' in context_lower:
            # Try multiple patterns to find the number
            num = None
            num_match = re.search(r'factorial\s+of\s+(\d+)', context_lower)
            if num_match:
                num = num_match.group(1)
            elif fact_match:
                num = fact_match.group(1)
            else:
                # Look for any number near "factorial"
                words = context_lower.split()
                for i, word in enumerate(words):
                    if 'factorial' in word:
                        # Check surrounding words for numbers
                        for offset in [-2, -1, 1, 2]:
                            idx = i + offset
                            if 0 <= idx < len(words):
                                found = re.search(r'(\d+)', words[idx])
                                if found:
                                    num = found.group(1)
                                    break
                        break

            # FIXED: Only generate code if we found a number
            if num:
                code_lines.append(f"# Factorial of {num}")
                code_lines.append(f"result = math.factorial({num})")
                code_lines.append("print(result)")
                return "\n".join(code_lines)

        # 4. Phát hiện power/exponent - FIXED: Better error handling
        power_match = re.search(self.MATH_OPERATIONS['power'], context)
        if power_match:
            base, exp = power_match.groups()
        elif '^' in context or '**' in context:
            # FIXED: Try alternative patterns
            alt_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:\*\*|\^)\s*(\d+(?:\.\d+)?)', context)
            if alt_match:
                base, exp = alt_match.groups()
            else:
                base, exp = None, None
        else:
            base, exp = None, None

        # FIXED: Only generate if we have both base and exp
        if base and exp:
            code_lines.append(f"# Calculate {base}^{exp}")
            code_lines.append(f"result = {base} ** {exp}")
            code_lines.append("print(result)")
            return "\n".join(code_lines)

        # 5. Phát hiện sqrt
        sqrt_match = re.search(self.MATH_OPERATIONS['sqrt'], context_lower)
        if sqrt_match or 'sqrt' in context_lower or 'square root' in context_lower:
            if sqrt_match:
                num = sqrt_match.group(1)
            else:
                num_search = re.search(r'(?:of|root\s+of)\s+(\d+)', context_lower)
                if not num_search:
                    num_search = re.search(r'sqrt\s*\(?\s*(\d+)', context_lower)
                num = num_search.group(1) if num_search else None

            # FIXED: Only generate if number found
            if num:
                code_lines.append(f"# Square root of {num}")
                code_lines.append(f"result = math.sqrt({num})")
                code_lines.append("print(result)")
                return "\n".join(code_lines)

        # 6. Phát hiện basic operations phức tạp
        # Tìm tất cả expressions dạng "123 * 456" hoặc "123 + 456 - 789"
        expressions = re.findall(r'\d+(?:\.\d+)?(?:\s*[\+\-\*/]\s*\d+(?:\.\d+)?)+', context)
        if expressions:
            # Lấy expression dài nhất/phức tạp nhất
            expr = max(expressions, key=len)
            # Clean up expression
            expr = expr.strip()
            code_lines.append(f"# Calculate: {expr}")
            code_lines.append(f"result = {expr}")
            code_lines.append("print(result)")
            return "\n".join(code_lines)

        # 7. Fallback: tìm bất kỳ expression nào có phép toán
        simple_calc = re.search(r'(\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(\d+(?:\.\d+)?)', context)
        if simple_calc:
            full_expr = simple_calc.group(0)
            code_lines.append(f"# Calculate: {full_expr}")
            code_lines.append(f"result = {full_expr}")
            code_lines.append("print(result)")
            return "\n".join(code_lines)

        # Không tìm thấy pattern nào
        return None

    def should_continue_generation(self, text: str) -> bool:
        """
        Kiểm tra xem có nên tiếp tục generation không
        """
        # Dừng nếu gặp các markers kết thúc
        end_markers = [
            r'\\boxed\{',
            'the answer is',
            'therefore,',
            'in conclusion',
        ]

        text_lower = text.lower()
        for marker in end_markers:
            if re.search(marker, text_lower):
                return False

        return True


def test_detector():
    """Test function"""
    detector = CalculationDetector()

    test_cases = [
        "Calculate the sum of all even numbers from 1 to 100",
        "What is 123 * 456?",
        "Find the factorial of 10",
        "Compute 5^10",
        "The square root of 144 is",
        "Let's evaluate 789 + 456 - 123",
        "First, we need to calculate 999 * 888",
        "Sum from 1 to 100",  # ADDED: Test simplified sum
    ]

    print("Testing Calculation Detector:\n")
    for i, case in enumerate(test_cases, 1):
        print(f"Test {i}: {case}")
        needs_calc = detector.detect_needs_calculation(case)
        print(f"  Needs calculation: {needs_calc}")

        if needs_calc:
            code = detector.generate_calculation_code(case)
            if code:
                print(f"  Generated code:")
                for line in code.split('\n'):
                    print(f"    {line}")
            else:
                print(f"  ⚠️  Detection passed but code generation failed!")
        print()


if __name__ == "__main__":
    test_detector()