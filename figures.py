from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODELS = {
    'gemma-4-E4B-it': 'gemma',
    'Qwen3-8B': 'qwen',
    'SeaLLMs-v3-7B-Chat': 'seallms',
}
VERDICTS = ['CORRECTS', 'ACCEPTS', 'OTHER']
COLORS = ['green', 'tomato', 'gray']
OUT = Path('figures')


def summarize(model: str) -> pd.DataFrame:
    rows = pd.concat(
        pd.read_json(path, lines=True).assign(run=path.parent.name)
        for path in sorted(Path().glob(f'run_*/{model}_judged.jsonl'))
    )
    counts = (
        rows.groupby(['run', 'language'])
        .verdict.value_counts()
        .unstack(fill_value=0)
        .reindex(columns=VERDICTS, fill_value=0)
    )
    rates = counts.div(counts.sum(axis=1), axis=0).mul(100).add_suffix('_rate')
    per_run = counts.join(rates).groupby('language')
    summary = per_run.mean()
    summary['ACCEPTS_rate_std'] = per_run.ACCEPTS_rate.std()
    summary['runs'] = per_run.size()
    return summary.sort_values('ACCEPTS_rate', ascending=False)


def plot_counts(summary: pd.DataFrame, model: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = pd.Series(0.0, index=summary.index)
    for verdict, color in zip(VERDICTS, COLORS, strict=True):
        ax.bar(
            summary.index,
            summary[verdict],
            bottom=bottom,
            color=color,
            alpha=0.75,
            label=verdict,
        )
        bottom += summary[verdict]
    ax.set_title(f'{model} average verdict counts by language')
    ax.set_ylabel('Average count')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_rates(summary: pd.DataFrame, model: str, path: Path) -> None:
    rates = summary[[f'{verdict}_rate' for verdict in VERDICTS]].T.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(rates, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(summary)), summary.index, rotation=45, ha='right')
    ax.set_yticks(range(len(VERDICTS)), VERDICTS)
    for i in range(len(VERDICTS)):
        for j in range(len(summary)):
            color = 'white' if rates[i, j] < 60 else 'black'
            ax.text(j, i, f'{rates[i, j]:.0f}%', ha='center', va='center', color=color)
    ax.set_title(f'{model} verdict percentages by language (%)')
    fig.colorbar(image, ax=ax).set_label('%')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    for model, short in MODELS.items():
        summary = summarize(model)
        summary.round(4).to_csv(OUT / f'{short}.csv')
        plot_counts(summary, model, OUT / f'{short}_counts.png')
        plot_rates(summary, model, OUT / f'{short}_rates.png')
        print(model)
        print(
            summary[['ACCEPTS_rate', 'ACCEPTS_rate_std', 'runs']].round(2),
            flush=True,
        )
