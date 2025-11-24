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


if __name__ == "__main__":
    detector = CalculationDetector()

    print("🧪 Testing Code Generation from CalculationDetector\n")
    print("=" * 80)

    # Test case 1: Simple arithmetic
    print("\n📝 Test 1: Simple Arithmetic")
    prompt1 = "Question: Calculate 15 + 27"
    context1 = "We need to calculate 15 + 27"

    print(f"Prompt: {prompt1}")
    print(f"Context: {context1}")

    code1 = detector.generate_calculation_code(context1, prompt1)
    print(f"\n🔧 Generated Code:")
    print("```python")
    print(code1 if code1 else "❌ No code generated")
    print("```")

    if code1:
        output1 = run_in_sandbox(code1)
        print(f"\n📤 Execution Output: {output1}")
        print(f"✅ PASS" if output1 == "42" else f"⚠️ Got: {output1}")

    # Test case 2: Multiplication and division
    print("\n" + "=" * 80)
    print("\n📝 Test 2: Multiplication and Division")
    prompt2 = "Question: What is 144 divided by 12?"
    context2 = "First, I need to calculate 144 / 12"

    print(f"Prompt: {prompt2}")
    print(f"Context: {context2}")

    code2 = detector.generate_calculation_code(context2, prompt2)
    print(f"\n🔧 Generated Code:")
    print("```python")
    print(code2 if code2 else "❌ No code generated")
    print("```")

    if code2:
        output2 = run_in_sandbox(code2)
        print(f"\n📤 Execution Output: {output2}")
        print(f"✅ PASS" if output2 == "12" or output2 == "12.0" else f"⚠️ Got: {output2}")

    # Test case 3: Complex calculation
    print("\n" + "=" * 80)
    print("\n📝 Test 3: Complex Calculation")
    prompt3 = "Question: Find the sum of squares of 3, 4, and 5"
    context3 = "We need to calculate 3^2 + 4^2 + 5^2"

    print(f"Prompt: {prompt3}")
    print(f"Context: {context3}")

    code3 = detector.generate_calculation_code(context3, prompt3)
    print(f"\n🔧 Generated Code:")
    print("```python")
    print(code3 if code3 else "❌ No code generated")
    print("```")

    if code3:
        output3 = run_in_sandbox(code3)
        print(f"\n📤 Execution Output: {output3}")
        print(f"✅ PASS" if output3 == "50" else f"⚠️ Got: {output3}")

    # Test case 4: Percentage calculation
    print("\n" + "=" * 80)
    print("\n📝 Test 4: Percentage")
    prompt4 = "Question: What is 15% of 200?"
    context4 = "Let me calculate 15% of 200, which is 0.15 * 200"

    print(f"Prompt: {prompt4}")
    print(f"Context: {context4}")

    code4 = detector.generate_calculation_code(context4, prompt4)
    print(f"\n🔧 Generated Code:")
    print("```python")
    print(code4 if code4 else "❌ No code generated")
    print("```")

    if code4:
        output4 = run_in_sandbox(code4)
        print(f"\n📤 Execution Output: {output4}")
        print(f"✅ PASS" if output4 == "30" or output4 == "30.0" else f"⚠️ Got: {output4}")

    # Test case 5: Detection test
    print("\n" + "=" * 80)
    print("\n📝 Test 5: Detection Test")

    test_texts = [
        "Now I need to calculate the sum",
        "Let me compute this value",
        "To find the answer, I should calculate",
        "This is just explaining the concept",
    ]

    for i, text in enumerate(test_texts, 1):
        needs_calc = detector.detect_needs_calculation(text)
        print(f"\nText {i}: '{text}'")
        print(f"  Needs calculation: {'✅ Yes' if needs_calc else '❌ No'}")

    print("\n" + "=" * 80)
    print("\n✅ Code generation testing completed!")