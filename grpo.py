from typing import List

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- CONFIG ---
MODEL_NAME = "gpt2"      # thay bằng model bạn muốn
BATCH_SIZE = 4           # số prompt mỗi batch
G = 4                    # số sample per prompt (group size)
MAX_NEW_TOKENS = 64
KL_COEF = 0.01
CLIP_EPS = 0.2
LR = 1e-5
EPS = 1e-8

# --- load model & tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
# giữ một bản "old" policy bằng clone (đơn giản)
old_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
old_model.eval()
for p in old_model.parameters():
    p.requires_grad = False

optimizer = AdamW(model.parameters(), lr=LR)

# --- user-defined reward function ---
# reward_fn(prompts: List[str], gens: List[str]) -> List[float]
# Ví dụ đơn giản: reward = độ dài + một số heuristics; trong thực tế dùng reward model.
def reward_fn(prompts: List[str], gens: List[str]) -> List[float]:
    # placeholder heuristic: khuyến khích vừa đủ độ dài và presence of digit (ví dụ)
    rewards = []
    for gen in gens:
        r = len(gen.split())
        r += 5.0 * any(ch.isdigit() for ch in gen)
        rewards.append(float(r))
    return rewards

# --- helpers ---
def generate_group(prompts: List[str], group_size: int) -> List[List[str]]:
    """
    Với mỗi prompt trả về list của group_size generated texts.
    Trả về danh sách chiều batch x group_size
    """
    all_groups = []
    model.eval()
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        gens = model.generate(
            **inputs,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.eos_token_id
        )
        texts = [tokenizer.decode(g[len(inputs["input_ids"][0]):], skip_special_tokens=True) for g in gens]
        all_groups.append(texts)
    return all_groups  # List[List[str]]

def compute_logprobs_for_sequences(model_ref, prompts: List[str], gens_flat: List[str]):
    """
    Tính log-prob của mỗi generated text dưới model_ref.
    Trả về tensor shape (N,)
    Cách đơn giản: compute negative cross-entropy (labels=shifted ids) per example.
    """
    logps = []
    model_ref.eval()
    with torch.no_grad():
        for prompt, gen in zip(prompts, gens_flat):
            full = prompt + gen
            toks = tokenizer(full, return_tensors="pt").to(device)
            input_ids = toks["input_ids"]
            # tính loss trên phần generated tokens: label phần generated, mask prompt part
            prompt_len = len(tokenizer(prompt)["input_ids"])
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100  # không tính loss cho prompt
            outputs = model_ref(input_ids, labels=labels)
            # outputs.loss là mean token loss -> chuyển sang log-prob tổng token
            loss = outputs.loss.item()
            seq_len = (labels != -100).sum().item()
            # approx total logprob = - loss * seq_len
            total_logprob = -loss * max(seq_len, 1)
            logps.append(total_logprob)
    return torch.tensor(logps, dtype=torch.float32, device=device)  # (N,)

# --- main training step (1 batch) ---
def grpo_update(prompts: List[str]):
    # 1) generate G completions per prompt
    groups = generate_group(prompts, G)  # list len=B, each is list len=G

    # flatten gens to compute rewards and logprobs easily
    gens_flat = []
    prompts_flat_for_gen = []
    for p, gl in zip(prompts, groups):
        for g in gl:
            gens_flat.append(g)
            prompts_flat_for_gen.append(p)

    # 2) compute scalar rewards for each generated sample
    rewards_flat = reward_fn(prompts_flat_for_gen, gens_flat)
    rewards = torch.tensor(rewards_flat, dtype=torch.float32, device=device)  # (B*G,)

    # 3) group-wise normalization -> compute advantage A_i = (r_i - mean_group) / (std_group + eps)
    B = len(prompts)
    rewards_grouped = rewards.view(B, G)  # (B, G)
    mean_grp = rewards_grouped.mean(dim=1, keepdim=True)  # (B,1)
    std_grp = rewards_grouped.std(dim=1, unbiased=False, keepdim=True)  # (B,1)
    advantages = (rewards_grouped - mean_grp) / (std_grp + EPS)  # (B,G)
    advantages = advantages.view(-1)  # (B*G,)

    # 4) compute old logprobs and new logprobs
    old_logps = compute_logprobs_for_sequences(old_model, prompts_flat_for_gen, gens_flat)  # (N,)
    new_logps = compute_logprobs_for_sequences(model, prompts_flat_for_gen, gens_flat)      # (N,)

    # 5) ratio and surrogate loss (clipped)
    ratios = torch.exp(new_logps - old_logps)  # (N,)
    # GRPO uses advantage normalized in group; keep sign
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
    policy_loss = -torch.mean(torch.min(surr1, surr2))

    # 6) optional KL penalty to keep close to old policy (approx via token-level KL)
    # here approximate KL by difference in per-seq avg logp (simple)
    approx_kl = torch.mean(old_logps - new_logps)  # E[log pi_old - log pi_new]
    kl_loss = KL_COEF * approx_kl

    loss = policy_loss + kl_loss

    # 7) backward and step
    model.train()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # 8) update old_model parameters (simple polyak: copy or soft update)
    # Here: hard copy for simplicity
    old_model.load_state_dict({k: v.clone().detach() for k, v in model.state_dict().items()})

    return {
        "policy_loss": policy_loss.item(),
        "kl": approx_kl.item(),
        "loss": loss.item(),
        "reward_mean": rewards.mean().item()
    }

# --- Example usage ---
if __name__ == "__main__":
    prompts = [
        "Solve: If 2x+3=7 then x =",
        "Explain why the sky is blue in simple terms.",
        "Write a short poem about autumn.",
        "What is the capital of France?"
    ]  # length should be BATCH_SIZE or less
    stats = grpo_update(prompts[:BATCH_SIZE])
    print("GRPO step stats:", stats)
