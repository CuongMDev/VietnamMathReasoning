"""
Quick test script cho CalculationDetector:
- Kiểm tra generate_calculation_code() có sinh code không
- Kiểm tra code chạy ra kết quả đúng không
"""

from calculation_detector import CalculationDetector

def run_generated_code(code: str):
    """Chạy code sinh ra và trả về kết quả"""
    local_vars = {}
    try:
        exec(code, {"__builtins__": __builtins__}, local_vars)
        return local_vars.get("result", None)
    except Exception as e:
        print(f"⚠️ Lỗi khi chạy code: {e}")
        print("Code:\n", code)
        return None


def main():
    detector = CalculationDetector()

    # Các test cơ bản
    test_cases = [
        ("Sum from 1 to 10", 55),
        ("Factorial of 5", 120),
        ("What is 10 * 5", 50),
        ("Square root of 16", 4.0),
        ("Sum of even numbers from 1 to 10", 30),  # 2+4+6+8+10
        ("Calculate 100 - 25", 75),
        ("What is 7 + 8", 15),
    ]

    print("🧪 Testing CalculationDetector...\n")

    passed = 0
    failed = 0

    for text, expected in test_cases:
        print("=" * 70)
        print(f"🧩 Input: {text}")

        code = detector.generate_calculation_code(text, text)
        if not code:
            print("❌ Không sinh ra được code!")
            failed += 1
            continue

        print("\n📜 Generated code:\n", code)

        result = run_generated_code(code)

        if result is None:
            print("❌ Không có biến `result` sau khi chạy.")
            failed += 1
        elif abs(result - expected) < 1e-6:
            print(f"✅ Kết quả đúng: {result}")
            passed += 1
        else:
            print(f"❌ Sai kết quả! Expected {expected}, got {result}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"🎯 Tổng kết: {passed} passed, {failed} failed")
    print("=" * 70)


if __name__ == "__main__":
    main()
