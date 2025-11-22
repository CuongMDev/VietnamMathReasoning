# ⚙️ Hàm sinh câu trả lời
import torch

def get_generated_tokens_count(outputs, input_length, tokenizer):
    # Extract generated tokens count (remove input tokens and padding)
    pad_token_id = tokenizer.pad_token_id
    generated_tokens_count = []
    for output in outputs.sequences:
        # Find the last non-pad token
        non_pad_indices = (output != pad_token_id).nonzero(as_tuple=True)[0]
        if len(non_pad_indices) > 0:
            last_non_pad = non_pad_indices[-1].item()
            count = max(0, last_non_pad - input_length + 1)
        else:
            count = 0
        generated_tokens_count.append(count)

    return generated_tokens_count

def generate_answers(model, tokenizer, messages, max_new_tokens=512, enable_thinking=True):
    texts = []
    for message in messages:
        text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if enable_thinking and not text.endswith("<think>\n"):
            text += "<think>\n"
        texts.append(text)

    # messages: list of message
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            return_dict_in_generate=True,
            do_sample=False,
        )
    
    # Decode texts
    texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs.sequences]
    
    return texts, get_generated_tokens_count(outputs, input_length, tokenizer)


def generate_answers_budget_forcing(
    model, tokenizer, messages,
    max_new_tokens=512,
    max_tokens_thinking_tmp=512,
    num_ignore=1
):
    """
    Budget forcing:
    - Dừng ngay khi gặp </think>
    - Nếu ignore_count < num_ignore → thêm "Wait\n" và infer tiếp
    - Khi think xong hoặc hết ignore → infer answer
    - Trả về (output_text, num_generated_tokens)
    """

    results = []
    token_counts = []

    # Token id cho stop-on-think
    stop_id = tokenizer.convert_tokens_to_ids("</think>")

    for message in messages:
        current_max_tokens_thinking_tmp = max_tokens_thinking_tmp
        # --- Build initial prompt ---
        text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        if not text.endswith("<think>\n"):
            text += "<think>\n"

        ignore_count = 0
        current_text = text
        input_length = None

        # --- THINKING PHASE ---
        while True:
            inputs = tokenizer([current_text], return_tensors="pt", padding=True).to(model.device)
            if input_length is None:
                input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=current_max_tokens_thinking_tmp,
                    eos_token_id=stop_id,
                    return_dict_in_generate=True,
                    do_sample=False,
                )

            sequence = outputs.sequences[0]
            total_tokens = sequence.shape[0]
            
            # Số token sinh ra trong lần generate này
            new_tokens = total_tokens - inputs["input_ids"].shape[1]

            current_text = tokenizer.decode(sequence, skip_special_tokens=False)

            # Nếu gặp </think>
            if "</think>" in current_text:
                ignore_count += 1
                if ignore_count <= num_ignore:
                    # Trừ số token sinh ra (trừ 1 cho </think>)
                    tokens_to_subtract = new_tokens - 1
                    current_max_tokens_thinking_tmp = max(0, current_max_tokens_thinking_tmp - tokens_to_subtract)

                    # Bỏ </think> trước khi append Wait
                    current_text = current_text[:-len("</think>\n")] + "Wait, "
                    continue
                else:
                    break

            # Safety: nếu chưa gặp </think> nhưng quá nhiều token
            current_text += "\n</think>\n"
            break

        
        # --- ANSWERING PHASE ---
        inputs = tokenizer([current_text], return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                do_sample=False,
            )

        final_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        total_tokens = outputs.sequences[0].shape[0]
        generated_tokens = max(0, total_tokens - input_length)

        results.append(final_output)
        token_counts.append(generated_tokens)

    return results, token_counts