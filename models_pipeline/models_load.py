import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

MODELS = [
    'Qwen/Qwen3-8B',
    'SeaLLMs/SeaLLMs-v3-7B-Chat',
    'google/gemma-4-E4B-it',
]
BATCH_SIZE = 16


def load_model(model_id: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map='cuda',
    )
    return model, tokenizer


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    # qwen3 would otherwise write a reasoning trace before answering, the other templates ignore it
    chats: list[str] = [
        tokenizer.apply_chat_template(  # pyright: ignore[reportAssignmentType]
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    responses = []
    for start in range(0, len(chats), BATCH_SIZE):
        inputs = tokenizer(
            chats[start : start + BATCH_SIZE],
            return_tensors='pt',
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(  # pyright: ignore[reportCallIssue]
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        new = outputs[:, inputs['input_ids'].shape[1] :]
        responses += [
            r.strip() for r in tokenizer.batch_decode(new, skip_special_tokens=True)
        ]
    return responses
