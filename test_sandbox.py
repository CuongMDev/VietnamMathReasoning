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
        return output
    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp_path)
        except:
            pass
        return "⚠️ Timeout: code chạy quá lâu"
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except:
            pass
        return f"⚠️ Lỗi sandbox: {e}"


if __name__ == "__main__":
    print("🧪 Testing Sandbox Execution\n")
    print("=" * 60)

    # Test 1: Simple calculation
    print("\n📝 Test 1: Simple Math")
    code1 = """
result = 2 + 3 * 4
print(result)
"""
    output1 = run_in_sandbox(code1)
    print(f"Code: {code1.strip()}")
    print(f"Output: {output1}")
    print(f"✅ PASS" if output1 == "14" else f"❌ FAIL (expected 14, got {output1})")

    # Test 2: Loop calculation
    print("\n" + "=" * 60)
    print("\n📝 Test 2: Loop Sum")
    code2 = """
total = sum(range(1, 11))
print(total)
"""
    output2 = run_in_sandbox(code2)
    print(f"Code: {code2.strip()}")
    print(f"Output: {output2}")
    print(f"✅ PASS" if output2 == "55" else f"❌ FAIL (expected 55, got {output2})")

    # Test 3: Float calculation
    print("\n" + "=" * 60)
    print("\n📝 Test 3: Float Division")
    code3 = """
result = 22 / 7
print(round(result, 2))
"""
    output3 = run_in_sandbox(code3)
    print(f"Code: {code3.strip()}")
    print(f"Output: {output3}")
    print(f"✅ PASS" if output3 == "3.14" else f"❌ FAIL (expected 3.14, got {output3})")

    # Test 4: Multiple operations
    print("\n" + "=" * 60)
    print("\n📝 Test 4: Complex Calculation")
    code4 = """
x = 10
y = 20
z = (x ** 2 + y ** 2) ** 0.5
print(int(z))
"""
    output4 = run_in_sandbox(code4)
    print(f"Code: {code4.strip()}")
    print(f"Output: {output4}")
    print(f"✅ PASS" if output4 == "22" else f"❌ FAIL (expected 22, got {output4})")

    # Test 5: Error handling
    print("\n" + "=" * 60)
    print("\n📝 Test 5: Error Handling")
    code5 = """
x = 1 / 0
"""
    output5 = run_in_sandbox(code5)
    print(f"Code: {code5.strip()}")
    print(f"Output: {output5}")
    print(f"✅ PASS (Error caught)" if "ZeroDivisionError" in output5 or "division by zero" in output5 else f"❌ FAIL")

    # Test 6: Timeout test (optional, commented out by default)
    # print("\n" + "="*60)
    # print("\n📝 Test 6: Timeout Test")
    # code6 = """
    # import time
    # time.sleep(10)
    # print("Done")
    # """
    # output6 = run_in_sandbox(code6, timeout=2.0)
    # print(f"Output: {output6}")
    # print(f"✅ PASS (Timeout caught)" if "Timeout" in output6 else f"❌ FAIL")

    print("\n" + "=" * 60)
    print("\n✅ Sandbox testing completed!")