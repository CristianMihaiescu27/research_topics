# research_topics

On Snellius, from the repository root:

```
uv sync
uv run python download.py
mkdir -p logs
sbatch slurm/run.sbatch Qwen/Qwen3-8B 1
sbatch slurm/judge.sbatch Qwen/Qwen3-8B 1
```

Run `download.py` on the login node, the compute nodes have no internet. Start the judge once the
run has finished. The last argument is the seed: `results/Qwen3-8B_seed1.jsonl` holds the answers,
`results/Qwen3-8B_seed1_judged.jsonl` the verdicts.

The models sample with the settings from their own `generation_config.json`, except Qwen3, which
uses the non-thinking settings from its model card (`SAMPLING` in `models_pipeline/models_load.py`).
The judge decodes greedily. `run_1` to `run_8` hold the judged answers for seeds 1 to 8.

`uv run python figures.py` averages the verdicts over the `run_*` folders and writes, per model, a
summary table and two plots (verdict counts and verdict percentages per language) to `figures/`.
