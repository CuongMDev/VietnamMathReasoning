import tempfile
import subprocess
import os
from calculation_detector import CalculationDetector


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
        return output
    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp_path)
        except:
            pass
        return "⚠️ Timeout"
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except:
            pass
        return f"⚠️ Error: {e}"


detector = CalculationDetector()

test_cases = [
    ("Calculate 15 + 27", "42"),
    ("What is 144 * 12", "1728"),
    ("Find the factorial of 5", "120"),
    ("Compute 2^10", "1024"),
    ("Sum from 1 to 10", "55"),
    ("Calculate 100 - 37", "63"),
    ("What is 50 / 2", "25"),
]

print("🧪 DEBUG: Testing CalculationDetector\n")
print("=" * 80)

for i, (context, expected) in enumerate(test_cases, 1):
    # Reset detector mỗi test
    detector = CalculationDetector()

    print(f"\n📝 Test {i}: {context}")
    print(f"   Expected: {expected}")

    # Check detection
    needs = detector.detect_needs_calculation(context)
    print(f"   Detection: {'✅ YES' if needs else '❌ NO'}")

    if not needs:
        print(f"   ⚠️ PROBLEM: Should detect but didn't!")
        continue

    # Generate code
    code = detector.generate_calculation_code(context, "")

    if not code:
        print(f"   ❌ No code generated!")
        continue

    print(f"\n   Generated Code:")
    for line in code.split('\n'):
        print(f"      {line}")

    # Execute code
    output = run_in_sandbox(code)
    print(f"\n   Execution Output: {output}")

    # Check correctness
    if output == expected or output == str(float(expected)):
        print(f"   ✅ CORRECT")
    else:
        print(f"   ❌ WRONG (expected {expected}, got {output})")

    print("-" * 80)

print("\n✅ Debug completed")