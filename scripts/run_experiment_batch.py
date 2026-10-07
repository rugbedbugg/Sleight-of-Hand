"""Run a resumable batch of specs with host resource sampling.

Example:
    python scripts/run_experiment_batch.py --specs-dir experiments/specs \\
        --match 'E0001-*.json' --workers 2

Samples available memory, swap and load while the batch runs and writes a
batch report to ``<root>/batches/``. Specs whose platform is not available
(for example missing online credentials) are reported and skipped, never
forced.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleight_of_hand.experiments import scheduler
from sleight_of_hand.experiments.journal import utc_now
from sleight_of_hand.experiments.platforms import get_platform
from sleight_of_hand.experiments.spec import load


def meminfo() -> dict:
    values = {}
    with open("/proc/meminfo", encoding="ascii") as stream:
        for line in stream:
            key, value = line.split(":", 1)
            values[key] = int(value.split()[0]) / 1024
    return values


class Sampler(threading.Thread):
    def __init__(self, every: float):
        super().__init__(daemon=True)
        self.every, self.samples, self.stop = every, [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            info = meminfo()
            self.samples.append(
                {
                    "t": round(time.time(), 1),
                    "available_mib": round(info["MemAvailable"]),
                    "swap_used_mib": round(info["SwapTotal"] - info["SwapFree"]),
                    "load1": os.getloadavg()[0],
                }
            )
            self.stop.wait(self.every)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--specs-dir", type=Path, default=Path("experiments/specs"))
    parser.add_argument("--match", default="*.json")
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-free-mib", type=float, default=1024)
    parser.add_argument("--sample-seconds", type=float, default=5)
    args = parser.parse_args()
    runnable, skipped = [], []
    for path in sorted(args.specs_dir.glob(args.match)):
        spec = load(path)
        state = get_platform(spec.platform).status(spec)
        if state.availability.value == "AVAILABLE":
            runnable.append(spec)
        else:
            skipped.append(
                {
                    "spec": path.name,
                    "status": state.availability.value,
                    "reason": state.reason,
                }
            )
    sampler = Sampler(args.sample_seconds)
    started = time.time()
    sampler.start()
    try:
        outcome = scheduler.run(runnable, args.root, args.workers, args.min_free_mib)
    finally:
        sampler.stop.set()
        sampler.join()
    samples = sampler.samples
    report = {
        "finished_at": utc_now(),
        "wall_seconds": round(time.time() - started, 1),
        "workers": args.workers,
        "skipped": skipped,
        "outcome": outcome,
        "host": {
            "cpus": os.cpu_count(),
            "min_available_mib": min(
                (s["available_mib"] for s in samples), default=None
            ),
            "max_swap_used_mib": max(
                (s["swap_used_mib"] for s in samples), default=None
            ),
            "max_load1": max((s["load1"] for s in samples), default=None),
            "samples": len(samples),
        },
    }
    out = args.root / "batches"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"batch-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(report, indent=1) + "\n")
    summary = {k: v for k, v in report.items() if k != "outcome"}
    summary["completed"] = outcome["completed"]
    summary["failed"] = outcome["failed"]
    print(json.dumps(summary, indent=1))
    print(f"batch report: {path}")
    raise SystemExit(1 if outcome["failed"] else 0)


if __name__ == "__main__":
    main()
