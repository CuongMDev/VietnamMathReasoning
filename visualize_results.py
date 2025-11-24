"""
Visualization script để so sánh kết quả với và không có execution
"""
import csv
import matplotlib.pyplot as plt
import os
from collections import defaultdict


def load_details_csv(filepath):
    """Load chi tiết từ CSV file"""
    if not os.path.exists(filepath):
        print(f"⚠️ File not found: {filepath}")
        return None

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    return data


def analyze_results(data):
    """Phân tích kết quả"""
    if not data:
        return None

    total = len(data)
    correct = sum(1 for row in data if row['is_correct'] == '1')

    # Phân tích độ dài output
    output_lengths = [len(row['raw_output']) for row in data]

    # Phân tích prediction patterns
    predictions = [row['prediction'] for row in data]

    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total * 100 if total > 0 else 0,
        'avg_output_length': sum(output_lengths) / len(output_lengths) if output_lengths else 0,
        'predictions': predictions,
    }


def compare_results(baseline_file, execution_file):
    """So sánh 2 kết quả"""

    print("=" * 80)
    print("COMPARING RESULTS: Baseline vs With Execution")
    print("=" * 80 + "\n")

    baseline_data = load_details_csv(baseline_file)
    execution_data = load_details_csv(execution_file)

    if not baseline_data or not execution_data:
        print("❌ Cannot load data files")
        return

    baseline_stats = analyze_results(baseline_data)
    execution_stats = analyze_results(execution_data)

    print("📊 BASELINE (No Execution):")
    print(f"   Total: {baseline_stats['total']}")
    print(f"   Correct: {baseline_stats['correct']}")
    print(f"   Accuracy: {baseline_stats['accuracy']:.2f}%")
    print(f"   Avg output length: {baseline_stats['avg_output_length']:.0f} chars")

    print("\n📊 WITH EXECUTION:")
    print(f"   Total: {execution_stats['total']}")
    print(f"   Correct: {execution_stats['correct']}")
    print(f"   Accuracy: {execution_stats['accuracy']:.2f}%")
    print(f"   Avg output length: {execution_stats['avg_output_length']:.0f} chars")

    print("\n🔍 IMPROVEMENT:")
    acc_diff = execution_stats['accuracy'] - baseline_stats['accuracy']
    print(f"   Accuracy: {acc_diff:+.2f}%")

    correct_diff = execution_stats['correct'] - baseline_stats['correct']
    print(f"   Correct answers: {correct_diff:+d}")

    # Chi tiết về những câu khác biệt
    print("\n🔬 DETAILED COMPARISON:")
    differences = []

    for i, (b_row, e_row) in enumerate(zip(baseline_data, execution_data)):
        b_correct = b_row['is_correct'] == '1'
        e_correct = e_row['is_correct'] == '1'

        if b_correct != e_correct:
            differences.append({
                'index': i,
                'question': b_row['question'][:80] + "..." if len(b_row['question']) > 80 else b_row['question'],
                'ground_truth': b_row['ground_truth'],
                'baseline_pred': b_row['prediction'],
                'execution_pred': e_row['prediction'],
                'baseline_correct': b_correct,
                'execution_correct': e_correct,
            })

    if differences:
        print(f"\n   Found {len(differences)} differences:")
        for diff in differences[:5]:  # Show first 5
            print(f"\n   Question {diff['index'] + 1}: {diff['question']}")
            print(f"      Ground truth: {diff['ground_truth']}")
            print(f"      Baseline pred: {diff['baseline_pred']} ({'✅' if diff['baseline_correct'] else '❌'})")
            print(f"      Execution pred: {diff['execution_pred']} ({'✅' if diff['execution_correct'] else '❌'})")

        if len(differences) > 5:
            print(f"\n   ... and {len(differences) - 5} more differences")
    else:
        print("   No differences in correctness found")

    # Visualize
    visualize_comparison(baseline_stats, execution_stats)


def visualize_comparison(baseline_stats, execution_stats):
    """Tạo biểu đồ so sánh"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Accuracy comparison
    ax1 = axes[0]
    methods = ['Baseline', 'With Execution']
    accuracies = [baseline_stats['accuracy'], execution_stats['accuracy']]
    colors = ['#3498db', '#2ecc71']

    bars = ax1.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Subplot 2: Correct answers comparison
    ax2 = axes[1]
    correct_counts = [baseline_stats['correct'], execution_stats['correct']]
    total = baseline_stats['total']

    bars = ax2.bar(methods, correct_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Correct Answers', fontsize=12)
    ax2.set_title(f'Correct Answers (out of {total})', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, total])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = 'comparison_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")

    # Try to show
    try:
        plt.show()
    except:
        print("   (Cannot display plot in this environment)")


def analyze_execution_patterns(execution_file):
    """Phân tích patterns trong outputs có execution"""

    data = load_details_csv(execution_file)
    if not data:
        return

    print("\n" + "=" * 80)
    print("ANALYZING EXECUTION PATTERNS")
    print("=" * 80 + "\n")

    # Count outputs có chứa [Calculation: ...]
    has_calculation = 0
    calculation_marker = "[Calculation:"

    for row in data:
        if calculation_marker in row['raw_output']:
            has_calculation += 1

    print(
        f"📊 Outputs containing calculations: {has_calculation}/{len(data)} ({has_calculation / len(data) * 100:.1f}%)")

    # Show some examples
    print("\n📝 Example outputs with calculations:\n")

    count = 0
    for i, row in enumerate(data):
        if calculation_marker in row['raw_output'] and count < 3:
            print(f"Example {count + 1}:")
            print(f"  Question: {row['question'][:60]}...")
            print(f"  Correct: {'Yes' if row['is_correct'] == '1' else 'No'}")

            # Extract calculation part
            output = row['raw_output']
            if calculation_marker in output:
                calc_start = output.index(calculation_marker)
                calc_end = output.index(']', calc_start) + 1
                calc_text = output[calc_start:calc_end]
                print(f"  {calc_text}")

            print()
            count += 1


if __name__ == "__main__":
    # Paths to CSV files (adjust based on your actual files)
    baseline_file = "benchmark_results_aime_2024_details.csv"
    execution_file = "benchmark_results_aime_2024_with_execution_details.csv"

    # Check if files exist
    if os.path.exists(baseline_file) and os.path.exists(execution_file):
        compare_results(baseline_file, execution_file)
        analyze_execution_patterns(execution_file)
    else:
        print("⚠️ CSV files not found. Please run benchmark.py first to generate results.")
        print(f"\nLooking for:")
        print(f"  - {baseline_file}")
        print(f"  - {execution_file}")
        print("\nYou can also modify the file paths in this script if your files have different names.")