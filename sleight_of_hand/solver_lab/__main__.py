"""Bounded offline solve/validate commands. No network or production entry point."""

import argparse
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .contract import digest, generator_hashes, read_json, validate_manifest, write_json
from .dataset import TeacherPolicy, validate_dataset


def run(source, manifest_path, output):
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    if output.exists():
        raise ValueError(
            "output directory already exists; use a fresh evidence directory"
        )
    results = []
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    with tempfile.TemporaryDirectory(prefix="soh-m1-") as temporary:
        for i in range(len(manifest["games"])):
            path = Path(temporary) / f"{i}.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sleight_of_hand.solver_lab.worker",
                    "--source",
                    str(source.resolve()),
                    "--manifest",
                    str(manifest_path.resolve()),
                    "--index",
                    str(i),
                    "--output",
                    str(path),
                ],
                env=env,
                check=True,
                timeout=manifest["limits"]["wall_seconds"],
            )
            results.append(read_json(path))
    payload = {
        "manifest": manifest,
        "input_sha256": digest(manifest),
        "source_files": results[0]["source_files"],
        "generator_files": generator_hashes(),
        "python": platform.python_version(),
        "games": [r["solved"] for r in results],
    }
    artifact = {"payload": payload, "sha256": digest(payload)}
    validate_dataset(artifact)
    load_started = time.perf_counter()
    teacher = TeacherPolicy(artifact)
    load_seconds = time.perf_counter() - load_started
    latency_us = []
    count = 0
    for game, solved in zip(manifest["games"], payload["games"]):
        for sample in solved["samples"]:
            for _ in range(20):
                lookup_started = time.perf_counter_ns()
                actual = teacher.lookup(game, sample["state"])
                latency_us.append((time.perf_counter_ns() - lookup_started) / 1000)
            if (
                actual["status"] != "supported"
                or actual["probabilities"] != sample["probabilities"]
            ):
                raise ValueError("teacher-policy disagreement")
            count += 1
    output.mkdir(parents=True)
    write_json(output / "dataset.json", artifact)
    write_json(
        output / "execution.json",
        {
            "dataset_sha256": artifact["sha256"],
            "benchmarks": [r["benchmark"] for r in results],
            "teacher_agreement": {
                "samples": count,
                "exact_matches": count,
                "max_absolute_error": 0.0,
            },
            "student_trained": False,
            "adapter": {
                "load_seconds": load_seconds,
                "queries": len(latency_us),
                "median_us": statistics.median(latency_us),
                "p99_us": sorted(latency_us)[int(len(latency_us) * 0.99)],
                "max_us": max(latency_us),
            },
        },
    )
    print(f"Generated {count} teacher infosets: {artifact['sha256']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    solve_parser = commands.add_parser("solve")
    solve_parser.add_argument("--source", type=Path, required=True)
    solve_parser.add_argument("--manifest", type=Path, required=True)
    solve_parser.add_argument("--output", type=Path, required=True)
    validator = commands.add_parser("validate")
    validator.add_argument("dataset", type=Path)
    args = parser.parse_args()
    if args.command == "solve":
        run(args.source, args.manifest, args.output)
    else:
        validate_dataset(read_json(args.dataset))
        print("Dataset validated")


if __name__ == "__main__":
    main()
