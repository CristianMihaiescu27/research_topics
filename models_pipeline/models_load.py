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
# qwen3's generation_config holds its thinking-mode settings, these are the ones its model card
# gives for enable_thinking=False. the other two models sample with their own generation_config
SAMPLING = {
    'Qwen/Qwen3-8B': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20},
}
GREEDY = {'do_sample': False, 'temperature': None, 'top_p': None, 'top_k': None}


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
    sample: bool = False,
) -> list[str]:
    decoding = (
        {'do_sample': True, **SAMPLING.get(model.name_or_path or '', {})}
        if sample
        else GREEDY
    )
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
                **decoding,
            )
        new = outputs[:, inputs['input_ids'].shape[1] :]
        responses += [
            r.strip() for r in tokenizer.batch_decode(new, skip_special_tokens=True)
        ]
    return responses
