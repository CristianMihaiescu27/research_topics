import sys
from pathlib import Path

import pandas as pd

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

if __name__ == '__main__':
    model_id = sys.argv[1]
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
    )
    Path('results').mkdir(exist_ok=True)
    rows.to_json(
        f'results/{model_id.split("/")[1]}.jsonl',
        orient='records',
        lines=True,
        force_ascii=False,
    )
