import math
from matplotlib import pyplot as plt
import numpy as np
from transformers import Trainer
import torch
import torch.nn as nn

loss_fct = nn.CrossEntropyLoss(reduction='none')

_tokenizer = None

def set_tokenizer(tokenizer):
    global _tokenizer
    _tokenizer = tokenizer

def _build_weights(labels, response_weight=0.3, ignore_index=-100):
    """Tạo weight tensor cho batch dựa vào labels.
    Think: cosine scale 0→1 (tăng chậm đầu, nhanh cuối), response: response_weight.
    Think start = phần tử đầu tiên không bị mask (-100).
    Think end = vị trí </think> token trong labels.
    """
    end_think_token_id = _tokenizer("</think>", add_special_tokens=False)["input_ids"][0]

    batch_size, seq_len = labels.shape
    weights = torch.zeros(batch_size, seq_len, device=labels.device)

    for b in range(batch_size):
        lbl = labels[b].tolist()

        # Tìm think_start: phần tử đầu tiên không bị mask
        think_start = None
        for i, l in enumerate(lbl):
            if l != ignore_index:
                think_start = i
                break

        # Tìm end_think: vị trí </think> trong labels
        end_think = None
        for i, l in enumerate(lbl):
            if l == end_think_token_id:
                end_think = i
                break

        if think_start is None or end_think is None:
            weights[b, :] = response_weight
            continue

        # Think: cosine scale từ 0 đến 1 (tăng chậm đầu, nhanh cuối)
        think_len = end_think - think_start
        if think_len > 1:
            for i in range(think_len):
                weights[b, think_start + i] = 1 - math.cos(math.pi / 2 * i / (think_len - 1))
        elif think_len == 1:
            weights[b, think_start] = 1.0

        # Response: sau </think> đến hết
        weights[b, end_think:] = response_weight

    return weights


def compute_threshold_loss(
    outputs,
    labels,
    num_items_in_batch=None,
    epsilon=0.0,
    ignore_index=-100,
    loss_threshold=0.0,
    shift_labels=True,
    gradient_accumulation_steps=1,
    response_weight=0.3,
):
    """
    Hàm tính Loss có áp dụng Label Smoothing + threshold + token weights.
    Token weights: think phần log scale 0→1, response phần = response_weight.
    Gọi set_tokenizer() trước khi dùng để set tokenizer.
    """
    # 1. Lấy logits từ outputs
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

    # Build weights từ labels trước khi shift
    token_weights = None
    if _tokenizer is not None:
        token_weights = _build_weights(labels, response_weight, ignore_index)
        if shift_labels:
            token_weights = token_weights[..., 1:].contiguous()

    # 2. Xử lý shift_labels
    if shift_labels:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    # 3. Tính Log Softmax
    log_probs = -nn.functional.log_softmax(logits, dim=-1)

    # 4. Điều chỉnh chiều của labels nếu cần
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    # 5. Tạo mask cho các vị trí cần ignore
    padding_mask = labels.eq(ignore_index)

    # 6. Clamp labels để tránh lỗi gather
    labels = torch.clamp(labels, min=0)

    # 7. Tính NLL Loss
    nll_loss = log_probs.gather(dim=-1, index=labels)

    # 8. Tính Smoothed Loss
    smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

    # 9. Áp dụng mask padding
    nll_loss.masked_fill_(padding_mask, 0.0)
    smoothed_loss.masked_fill_(padding_mask, 0.0)

    # 10. Chỉ giữ token có loss >= threshold
    threshold_mask = nll_loss >= loss_threshold
    combined_mask = threshold_mask & (~padding_mask)
    nll_loss = nll_loss * combined_mask
    smoothed_loss = smoothed_loss * combined_mask

    # 10.5. Áp dụng token weights
    if token_weights is not None:
        token_weights = token_weights.unsqueeze(-1)  # [B, L] -> [B, L, 1]
        nll_loss = nll_loss * token_weights
        smoothed_loss = smoothed_loss * token_weights

    # 11. Tính tổng số phần tử active
    num_active_elements = combined_mask.sum()
    if num_active_elements == 0:
        return torch.tensor(0.0).to(logits.device)

    # 12. Chuẩn hóa loss
    nll_loss = nll_loss.sum() / num_active_elements
    smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])

    # 13. Kết hợp loss
    loss = (1 - epsilon) * nll_loss + epsilon * smoothed_loss
    
    return loss / gradient_accumulation_steps

def plot_token_loss_to_file(trainer, model, tokenizer, filename="token_loss_hist.png", loss_threshold=0.15, shift_labels=True):
    """
    Lấy 1 batch từ dataloader, tính loss per token, in thông tin debug và plot histogram.
    Đồng thời in số lượng token < và >= loss_threshold.
    """
    # Lấy 1 batch từ dataloader
    batch = next(iter(trainer.get_train_dataloader()))
    input_ids = batch["input_ids"].to(model.device)
    labels = batch["labels"].to(model.device)

    # Forward pass (không tính gradient)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    # --- Shift labels cho causal LM nếu cần ---
    if shift_labels:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    # Tính loss token-level (CrossEntropy không reduce, ignore_index)
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    loss_per_token = loss_fct(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    ).view(labels.size())  # [B, L]

    # Flatten và bỏ ignore_index
    loss_values = loss_per_token[labels != -100].float().cpu().numpy()

    # Tính số lượng token dưới và >= threshold
    num_below_threshold = np.sum(loss_values < loss_threshold)
    num_above_equal_threshold = np.sum(loss_values >= loss_threshold)

    print(f"✅ Token loss statistics:")
    print(f"   Total tokens (valid): {len(loss_values)}")
    print(f"   Tokens < threshold {loss_threshold}: {num_below_threshold}")
    print(f"   Tokens >= threshold {loss_threshold}: {num_above_equal_threshold}")
    print(f"   Min: {loss_values.min():.6f}, Max: {loss_values.max():.6f}, Mean: {loss_values.mean():.6f}")

    # Vẽ histogram
    plt.figure(figsize=(10,5))
    plt.hist(loss_values, bins=1000, color='skyblue', alpha=0.7)
    plt.axvline(loss_threshold, color='red', linestyle='--', label=f"Threshold={loss_threshold}")
    plt.xlabel("Loss per token")
    plt.ylabel("Number of tokens")
    plt.title("Distribution of token-level loss (first batch)")
    plt.legend()
    plt.tight_layout()

    # Lưu ra file
    plt.savefig(filename)
    plt.close()
    print(f"✅ Histogram token loss đã lưu tại: {filename}")