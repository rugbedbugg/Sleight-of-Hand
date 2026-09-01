# Contributing to Sleight of Hand

Sleight of Hand is a heads-up Leduc hold'em agent built from a game engine, Bayesian opponent model, expectiminimax search, and genetic optimization.

## Development setup

Use the uv-managed Python selected by `.python-version`:

```sh
uv venv
uv pip install -r requirements.txt
uv run pytest tests/ -q
```

Use `uv run python demo.py --hands 5 --seed 1` for a short interactive smoke test. Full experiment scripts can take several minutes and write into `results/`.

## Change guidelines

- Keep engine, agents, belief, search, and genetic code in their existing packages.
- Pass explicit seeds through stochastic code and make evaluation comparisons reproducible.
- Add unit coverage for game invariants, probability normalization, search decisions, and genome operations.
- Do not commit regenerated plots or result tables unless the pull request intentionally updates a documented benchmark.

## Pull requests

Include the pytest result, the seed and hand count for performance claims, and a brief explanation of any policy, reward, or opponent-model change.
