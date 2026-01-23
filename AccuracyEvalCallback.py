# --- TrainerCallback for accuracy evaluation ---

from tqdm import tqdm
from transformers import TrainerCallback
from utils import extract_boxed, is_answer_equal, make_prompt_template
from generate_answers import generate_answers


class AccuracyEvalCallback(TrainerCallback):
    """Callback to compute accuracy by generating answers during evaluation."""

    def __init__(self, val_dataset_raw, tokenizer, model, eval_steps, max_new_tokens=3064, batch_size=4):
        self.val_dataset_raw = val_dataset_raw  # Raw dataset with problem, answer fields
        self.tokenizer = tokenizer
        self.model = model
        self.eval_steps = eval_steps
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.best_accuracy = 0.0
        self.best_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Generate answers and compute accuracy at eval_steps intervals."""
        # Only evaluate at eval_steps intervals
        if state.global_step % self.eval_steps != 0 or state.global_step == 0:
            return

        print(f"\n[AccuracyEvalCallback] Step {state.global_step}: Generating answers for validation...")

        self.model.eval()
        correct = 0
        total = len(self.val_dataset_raw)

        # Prepare all prompts
        all_prompts = [make_prompt_template(ex["problem"]) for ex in self.val_dataset_raw]
        all_answers = [ex.get("answer", "") for ex in self.val_dataset_raw]

        # Process in batches
        all_pred_answers = []
        all_token_counts = []
        num_batches = (total + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total)

            batch_prompts = all_prompts[start_idx:end_idx]

            # Use generate_answers for batch inference
            batch_outputs, batch_token_counts = generate_answers(
                self.model,
                self.tokenizer,
                batch_prompts,
                max_new_tokens=self.max_new_tokens,
            )

            for output, tok_count in zip(batch_outputs, batch_token_counts):
                pred_answer = extract_boxed(output)
                all_pred_answers.append(pred_answer)
                all_token_counts.append(tok_count)

        # Calculate accuracy
        for i, (pred_answer, gt_answer) in enumerate(zip(all_pred_answers, all_answers)):
            if is_answer_equal(pred_answer, gt_answer):
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        avg_token_count = sum(all_token_counts) / total if total > 0 else 0.0

        print(f"\n[Accuracy] {correct}/{total} = {accuracy:.4f}")
        print(f"[Avg Token Count] {avg_token_count:.2f}")

        # Log to state metrics
        state.log_history.append({
            "step": state.global_step,
            "eval_accuracy": accuracy,
            "eval_avg_token_count": avg_token_count,
        })

        # Track best model
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_step = state.global_step
            print(f"[New Best] Accuracy: {accuracy:.4f} at step {state.global_step}")
            # Save best model
            self.model.save_pretrained(f"{args.output_dir}/best_model")
            self.tokenizer.save_pretrained(f"{args.output_dir}/best_model")
