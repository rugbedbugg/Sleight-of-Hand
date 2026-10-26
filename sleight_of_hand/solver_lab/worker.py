"""One resource-limited invocation of the pinned, unmodified teacher."""

from __future__ import annotations

import argparse
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

from .contract import (
    REVISION,
    ROOT,
    digest,
    file_digest,
    read_json,
    validate_manifest,
    write_json,
)
from .dataset import expected_samples, validate_profile
from .oracle import evaluate
from .tree import actions, nodes


def impose_limits(limits):
    """Linux hard limits are set before importing/constructing the external solver."""
    memory = limits["address_space_mb"] * 1024**2
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    cpu = limits["cpu_seconds"]
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2_000_000, 2_000_000))
    allowed = sorted(os.sched_getaffinity(0))[: limits["cpus"]]
    os.sched_setaffinity(0, allowed)


def verify_source(source):
    source = Path(source).resolve()
    lock = read_json(ROOT / "source.lock.json")
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.stdout.strip() != REVISION:
        raise ValueError("solver revision mismatch")
    # Reject additional Python modules as well as modified tracked modules.
    found = {
        p.relative_to(source).as_posix(): file_digest(p)
        for p in (source / "python/src").rglob("*.py")
    }
    found["LICENSE"] = file_digest(source / "LICENSE")
    if found != lock["files"]:
        raise ValueError("solver checkout differs from pinned source")
    return source, found


def solve(source, manifest, index):
    validate_manifest(manifest)
    source, hashes = verify_source(source)
    sys.path.insert(0, str(source / "python/src"))
    from algorithms.naive_eval import exploitability_naive
    from algorithms.vector_cfr import VectorCFRConfig, VectorCFRTrainer
    from games.river_holdem import RiverHoldemConfig, RiverHoldemGame, card_str

    spec = manifest["games"][index]
    game = RiverHoldemGame(
        RiverHoldemConfig(
            board=spec["board"],
            pot=spec["pot"],
            stacks=tuple(spec["stacks"]),
            bet_sizes=(0.5, 1.0),
            include_all_in=True,
            max_raises=2,
            ranges=tuple(["".join(h["cards"]) for h in r] for r in spec["ranges"]),
            range_weights=tuple([h["weight"] for h in r] for r in spec["ranges"]),
        )
    )
    # Confirm card translation and range order; upstream must not silently drop cards.
    for p in (0, 1):
        actual = [{card_str(c) for c in h.cards} for h in game.hands[p]]
        if actual != [set(h["cards"]) for h in spec["ranges"][p]]:
            raise ValueError("solver private-card translation mismatch")
    # Validate the complete external tree against our independently implemented accounting.
    for node in nodes(spec):
        state = game.initial_state()
        for token in node.history:
            options = {
                a.label if a.label in ("c", "f") else f"{a.label}{a.amount}": a
                for a in game.legal_actions(state)
            }
            state = game.next_state(state, options[token])
        if (
            state.contrib != node.contributions
            or game.is_terminal(state) != node.terminal
            or state.terminal_winner != node.winner
        ):
            raise ValueError("solver accounting mismatch")
        if not node.terminal and state.player != node.player:
            raise ValueError("solver position mismatch")
        options = game.legal_actions(state)
        tokens = [
            a.label if a.label in ("c", "f") else f"{a.label}{a.amount}"
            for a in options
        ]
        if tokens != [a["id"] for a in actions(spec, node)]:
            raise ValueError("solver action abstraction mismatch")

    trainer = VectorCFRTrainer(
        game,
        VectorCFRConfig(
            use_plus=True, linear_weighting=True, alternating=True, use_naive_eval=True
        ),
    )
    diagnostics = []
    completed = 0
    for target in manifest["checkpoints"]:
        trainer.run(target - completed)
        completed = target
        raw = trainer.average_strategy_profile()
        profile = {
            str(p): {
                key: {"actions": tokens, "strategy": matrix}
                for key, (tokens, matrix) in raw[p].items()
            }
            for p in (0, 1)
        }
        validate_profile(spec, profile)
        diagnostics.append(
            {
                "iterations": target,
                "solver_exploitability_chips": exploitability_naive(game, raw),
                "oracle": evaluate(spec, profile),
            }
        )
    return {
        "game_sha256": digest(spec),
        "teacher_profile": profile,
        "diagnostics": diagnostics,
        "samples": expected_samples(spec, profile),
    }, hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    validate_manifest(manifest)
    impose_limits(manifest["limits"])
    started = time.monotonic()
    solved, hashes = solve(args.source, manifest, args.index)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write_json(
        args.output,
        {
            "solved": solved,
            "source_files": hashes,
            "benchmark": {
                "wall_seconds": time.monotonic() - started,
                "cpu_seconds": usage.ru_utime + usage.ru_stime,
                "peak_rss_kib": usage.ru_maxrss,
                "affinity_cpus": len(os.sched_getaffinity(0)),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "limits": manifest["limits"],
                "threads": 1,
            },
        },
    )


if __name__ == "__main__":
    main()
