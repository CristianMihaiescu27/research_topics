# research_topics

On Snellius, from the repository root:

```
uv sync
uv run python download.py
mkdir -p logs
sbatch slurm/run.sbatch Qwen/Qwen3-8B
sbatch slurm/judge.sbatch Qwen/Qwen3-8B
```

Run `download.py` on the login node, the compute nodes have no internet. Start the judge once the
run has finished. Results go to `results/`.
