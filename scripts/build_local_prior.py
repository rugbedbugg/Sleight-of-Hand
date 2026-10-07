"""Compile a local historical-prior bundle from completed local runs.

Example:
    python scripts/build_local_prior.py --experiment E0002 --arm canonical \\
        --opponent script.tag_simple --data-version e0002-tag-v1 \\
        --output runs/priors/e0002-tag.json

Uses the runtime's own observation parser and the existing deterministic
prior compiler. The bundle feeds only the policy-inert OpponentMemory in
local validation experiments; it is never staged for Chipzen.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sleight_of_hand.experiments.model import sha256
from sleight_of_hand.experiments.platforms.priors import (
    observation_dataset,
)
from sleight_of_hand.experiments.storage import Index
from sleight_of_hand.holdem.profiles import encode


def compiler():
    spec = importlib.util.spec_from_file_location(
        "build_opponent_priors", ROOT / "scripts" / "build_opponent_priors.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compile_priors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = Index(args.root / "index.sqlite")
    (spec,) = [s for s in index.specs(args.experiment) if s.arm == args.arm]
    run_dirs = [
        r["run_dir"] for r in index.runs(spec.spec_hash) if r["status"] == "COMPLETED"
    ]
    dataset = observation_dataset(run_dirs, args.opponent, args.data_version)
    bundle = compiler()(encode(dataset))
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bundle)
    print(
        json.dumps(
            {
                "bundle": str(args.output),
                "bundle_sha256": "sha256:" + sha256(bundle),
                "observations": len(dataset["observations"]),
                "source_runs": len(run_dirs),
                "source_spec": spec.spec_hash,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
