# Contributing to Sleight of Hand

Sleight of Hand is a heads-up Leduc hold'em agent built from a game engine, Bayesian opponent model, expectiminimax search, and genetic optimization.

## Development setup

Use the uv-managed Python selected by `.python-version`. Dependencies come
from `pyproject.toml` and the committed `uv.lock`:

```sh
uv sync --locked
uv run pytest tests/ -q
```

Change dependencies with `uv add` / `uv remove` (or edit `pyproject.toml`
and run `uv lock`), and commit `uv.lock` in the same change. A ChipZen SDK
or WebSocket pin change must also update `bots/chipzen/requirements.txt`,
the uploaded runtime's install list; `tests/test_dependencies.py` fails if
the two differ. CI runs `uv sync --locked` and never updates the lock.

Use `uv run python demo.py --hands 5 --seed 1` for a short interactive smoke test. Full experiment scripts can take several minutes and write into `results/`.

## Change guidelines

- Keep engine, agents, belief, search, and genetic code in their existing packages.
- Pass explicit seeds through stochastic code and make evaluation comparisons reproducible.
- Add unit coverage for game invariants, probability normalization, search decisions, and genome operations.
- Do not commit regenerated plots or result tables unless the pull request intentionally updates a documented benchmark.

## Pull requests

Include the pytest result, the seed and hand count for performance claims, and a brief explanation of any policy, reward, or opponent-model change.
