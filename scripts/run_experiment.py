"""Run every unfinished shard of one or more experiment specs.

Example:
    python scripts/run_experiment.py --spec experiments/specs/E0001-canonical.json \\
        --spec experiments/specs/E0001-memory-off.json --workers 2

Completed shards are skipped, so re-running resumes. Results land under
``--root`` (default ``runs/``, git-ignored) and are research evidence only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleight_of_hand.experiments import scheduler
from sleight_of_hand.experiments.spec import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, action="append", required=True)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-free-mib", type=float, default=1024)
    args = parser.parse_args()
    specs = [load(path) for path in args.spec]
    outcome = scheduler.run(specs, args.root, args.workers, args.min_free_mib)
    print(json.dumps({k: v for k, v in outcome.items() if k != "results"}, indent=1))
    raise SystemExit(1 if outcome["failed"] else 0)


if __name__ == "__main__":
    main()
