"""Analyze one experiment's completed runs (research evidence only).

Example:
    python scripts/analyze_experiment.py --experiment E0001

Verifies every run's raw-journal checksum, recomputes metrics per arm from
the stored evidence, compares comparable arms within one platform and
provenance stratum, and writes ``<root>/analysis/<experiment>.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleight_of_hand.experiments import analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    report = analysis.analyze(args.root, args.experiment, not args.no_verify)
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.experiment}.json").write_text(
        json.dumps(report, indent=1, default=str) + "\n"
    )
    text = analysis.markdown(report)
    (out / f"{args.experiment}.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
