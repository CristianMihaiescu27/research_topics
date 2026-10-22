import sys
from pathlib import Path

import pandas as pd
from transformers import set_seed

from models_pipeline.models_load import generate_response, load_model

LANGUAGES = [
    'English',
    'Chinese',
    'Vietnamese',
    'Thai',
    'Indonesian',
    'Malay',
    'Khmer',
    'Lao',
    'Dutch',
    'Romanian',
]
# khmer, thai and lao take several times more tokens per word than english
MAX_NEW_TOKENS = 512


def name(model_id: str, seed: int) -> str:
    return f'{model_id.split("/")[1]}_seed{seed}'


if __name__ == '__main__':
    model_id, seed = sys.argv[1], int(sys.argv[2])
    set_seed(seed)
    rows = pd.read_csv('data/questions.csv').melt(
        id_vars='id',
        value_vars=LANGUAGES,
        var_name='language',
        value_name='question',
    )
    model, tokenizer = load_model(model_id)
    rows['response'] = generate_response(
        model,
        tokenizer,
        rows.question.tolist(),
        MAX_NEW_TOKENS,
        sample=True,
    )
    Path('results').mkdir(exist_ok=True)
    rows.to_json(
        f'results/{name(model_id, seed)}.jsonl',
        orient='records',
        lines=True,
        force_ascii=False,
    )
