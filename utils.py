def find_sublist_indices(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i + len(sub)
    return -1


def make_prompt_template(user_prompt: str, think=None, respond=None):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful and harmless assistant. "
                       "You are Qwen developed by Alibaba. "
                       "You may reason internally but output only the final answer in LaTeX \\boxed."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    assistant_content = ""

    # Nếu có think (suy luận nội bộ)
    if think is not None:
        assistant_content += f"<think>\n{think}\n</think>\n"

    # Nếu có respond, thêm message của assistant
    if respond is not None:
        assistant_content += respond

    if assistant_content:
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })

    return messages
