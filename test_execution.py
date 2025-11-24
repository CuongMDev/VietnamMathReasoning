"""
Script test đơn giản để verify sandbox execution hoạt động
"""
from calculation_detector import CalculationDetector
import tempfile
import subprocess
import os


def run_in_sandbox(code: str, timeout: float = 5.0) -> str:
    """Chạy code Python trong môi trường cô lập"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout.strip() or result.stderr.strip()
        os.unlink(tmp_path)
    except subprocess.TimeoutExpired:
        output = "⚠️ Timeout"
        try:
            os.unlink(tmp_path)
        except:
            pass
    except Exception as e:
        output = f"⚠️ Error: {e}"
        try:
            os.unlink(tmp_path)
        except:
            pass

    return output


def test_full_pipeline():
    """Test toàn bộ pipeline: detect -> generate code -> execute"""

    detector = CalculationDetector()

    test_cases = [
        {
            "name": "Sum of even numbers",
            "text": "Let me calculate the sum of all even numbers from 1 to 100.",
            "expected_result": "2550"
        },
        {
            "name": "Factorial",
            "text": "Now I need to find the factorial of 10.",
            "expected_result": "3628800"
        },
        {
            "name": "Power calculation",
            "text": "Calculate 2^10 to get the result.",
            "expected_result": "1024"
        },
        {
            "name": "Square root",
            "text": "The square root of 144 is needed here.",
            "expected_result": "12.0"
        },
        {
            "name": "Basic arithmetic",
            "text": "We need to compute 123 * 456 for this problem.",
            "expected_result": "56088"
        },
    ]

    print("=" * 80)
    print("TESTING FULL PIPELINE: Detection -> Code Generation -> Execution")
    print("=" * 80 + "\n")

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"Test {i}: {test['name']}")
        print(f"{'─' * 80}")
        print(f"Input text: {test['text']}")

        # Step 1: Detection
        needs_calc = detector.detect_needs_calculation(test['text'])
        print(f"\n✓ Detection: {'YES' if needs_calc else 'NO'}")

        if not needs_calc:
            print("❌ FAILED: Should detect calculation")
            failed += 1
            continue

        # Step 2: Code generation
        code = detector.generate_calculation_code(test['text'])

        if not code:
            print("❌ FAILED: Could not generate code")
            failed += 1
            continue

        print(f"\n✓ Generated code:")
        print("```python")
        for line in code.split('\n'):
            print(f"  {line}")
        print("```")

        # Step 3: Execution
        output = run_in_sandbox(code)
        print(f"\n✓ Execution output: {output}")
        print(f"  Expected: {test['expected_result']}")

        # Verify result
        if output.strip() == test['expected_result']:
            print(f"\n✅ PASSED")
            passed += 1
        else:
            print(f"\n⚠️ PARTIAL: Output differs but might be equivalent")
            # Check if numerically equivalent
            try:
                if float(output.strip()) == float(test['expected_result']):
                    print(f"✅ PASSED (numerically equivalent)")
                    passed += 1
                else:
                    print(f"❌ FAILED")
                    failed += 1
            except:
                failed += 1

    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed}/{len(test_cases)} tests passed")
    print("=" * 80 + "\n")

    return passed, failed


def test_detector_only():
    """Test chỉ detection logic"""

    detector = CalculationDetector()

    print("=" * 80)
    print("TESTING DETECTION ONLY")
    print("=" * 80 + "\n")

    positive_cases = [
        "Calculate the sum of numbers from 1 to 100",
        "What is 123 * 456?",
        "Find the factorial of 10",
        "Compute 2^15",
        "We need to evaluate 999 + 888 - 777",
        "The square root of 256 is",
    ]

    negative_cases = [
        "This is a simple sentence",
        "Let x = 5",  # Simple assignment
        "The answer is 42",
        "We can see that",
        "Therefore, the result",
    ]

    print("POSITIVE CASES (should detect):")
    for text in positive_cases:
        result = detector.detect_needs_calculation(text)
        status = "✅" if result else "❌"
        print(f"  {status} {text}")
        # Reset detector state
        detector.last_detected_position = -1

    print("\nNEGATIVE CASES (should NOT detect):")
    for text in negative_cases:
        result = detector.detect_needs_calculation(text)
        status = "✅" if not result else "❌"
        print(f"  {status} {text}")
        detector.last_detected_position = -1


if __name__ == "__main__":
    # Test detection only
    test_detector_only()

    print("\n\n")

    # Test full pipeline
    test_full_pipeline()