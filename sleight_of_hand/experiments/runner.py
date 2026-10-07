"""Execute one shard of one spec: journal, normalize, measure, index.

Runs inside a worker process. Any exception fails only this run: its raw
evidence is kept read-only, its manifest says FAILED, and other workers are
unaffected. The policy under test is fingerprinted and must equal the
spec's ``policy_revision``; experiments never run against altered policy.
"""

from __future__ import annotations

import os
import platform as host_platform
import resource
import subprocess
import sys
import time
from pathlib import Path

from . import metrics, normalization
from .journal import RunJournal, read_raw, utc_now
from .model import RunStatus, sha256
from .platforms import get_platform
from .platforms.base import MatchContext
from .spec import ExperimentSpec
from .storage import Index

ROOT = Path(__file__).resolve().parents[2]
#: Every file whose behavior the experiments evaluate (the runtime policy).
POLICY_FILES = (
    "bots/chipzen/bot.py",
    "bots/chipzen/accounting_observer.py",
    "sleight_of_hand/__init__.py",
    "sleight_of_hand/engine",
    "sleight_of_hand/policy",
    "sleight_of_hand/holdem",
)


def policy_revision(root: Path = ROOT) -> str:
    """Line-ending-insensitive digest of the runtime policy sources."""
    paths = []
    for entry in POLICY_FILES:
        path = root / entry
        paths += sorted(path.glob("*.py")) if path.is_dir() else [path]
    material = b"".join(
        p.relative_to(root).as_posix().encode()
        + b"\0"
        + p.read_bytes().replace(b"\r\n", b"\n")
        + b"\0"
        for p in paths
    )
    return "sha256:" + sha256(material)


def _git(*args) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_state() -> dict:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_head": _git("rev-parse", "HEAD"),
        "tracked_changes": None if status is None else bool(status),
        "policy_revision": policy_revision(),
    }


def runtime_versions() -> dict:
    import chipzen

    return {
        "python": sys.version.split()[0],
        "implementation": sys.implementation.name,
        "os": host_platform.platform(),
        "chipzen_sdk": chipzen.__version__,
    }


def match_seeds(spec: ExperimentSpec, match_index: int) -> dict:
    policy = spec.seed_policy
    # Without common random numbers each arm gets its own streams.
    salt = () if policy.common_random_numbers else (spec.spec_hash,)
    return {
        role: policy.derive(*salt, role, match_index)
        for role in ("soh", "opponent", "deck")
    }


def run_shard(spec_dict: dict, shard: tuple[int, int], worker_id: str, root: str):
    """Worker entry point (top-level so it can be pickled)."""
    spec = ExperimentSpec.from_dict(spec_dict)
    root = Path(root)
    index = Index(root / "index.sqlite")
    started = time.time()
    cpu_start = time.process_time()
    source = source_state()
    if source["policy_revision"] != spec.policy_revision:
        raise RuntimeError(
            "policy revision differs from the spec; refusing to run "
            f"({source['policy_revision']} != {spec.policy_revision})"
        )
    platform = get_platform(spec.platform)
    platform.require_available(spec)
    platform.prepare(spec)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = (
        f"{spec.experiment_id}.{spec.arm}.{spec.short_hash}."
        f"s{shard[0]:06d}-{shard[1]:06d}.{stamp}.{os.getpid()}"
    )
    run_dir = root / "runs" / run_id
    index.register_spec(spec)
    manifest = {
        "run_id": run_id,
        "spec_hash": spec.spec_hash,
        "spec": spec.to_dict(),
        "provenance": spec.provenance.value,
        "shard": list(shard),
        "worker_id": worker_id,
        "started_at": utc_now(),
        "source": source,
        "runtime": runtime_versions(),
        "platform": {
            "name": platform.name,
            "adapter_version": platform.adapter_version,
            "metadata": platform.metadata(),
        },
        "normalizer": normalization.VERSION,
    }
    journal = RunJournal.create(run_dir, manifest)
    index.start_run(
        run_id, spec.spec_hash, shard, worker_id, manifest["started_at"], run_dir
    )
    hands = matches = 0
    try:
        for match_index in range(*shard):
            seeds = match_seeds(spec, match_index)
            context = MatchContext(
                spec, match_index, spec.opponent_for(match_index), seeds
            )

            def emit(event, _m=match_index):
                journal.append({"match": _m, **event})

            summary = platform.play_match(context, emit)
            journal.append(
                {
                    "match": match_index,
                    "kind": "match_summary",
                    "hands": summary.hands,
                    "ended": summary.ended,
                    "platform_ids": summary.platform_ids,
                    "uncontrolled": list(summary.uncontrolled),
                }
            )
            matches += 1
            hands += summary.hands
            journal.checkpoint(matches, hands)
            if hands > spec.stopping.max_hands:
                raise RuntimeError("hand ceiling exceeded")
        checksums = journal.seal_raw()
        raw = read_raw(run_dir)  # derive only from the sealed evidence
        normalized = normalization.normalize(raw)
        derived = {
            f"normalized/{name}.jsonl": journal.write_derived(
                f"normalized/{name}.jsonl", rows
            )
            for name, rows in normalized.items()
        }
        measured = metrics.compute(spec.metrics, normalized, raw, spec.policy_config)
        derived["metrics.json"] = journal.write_json("metrics.json", measured)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        summary = {
            "matches": matches,
            "hands": hands,
            "wall_seconds": round(time.time() - started, 3),
            "cpu_seconds": round(time.process_time() - cpu_start, 3),
            "max_rss_mib": round(usage.ru_maxrss / 1024, 1),
            "hands_per_second": round(hands / max(1e-9, time.time() - started), 2),
        }
        derived["result.json"] = journal.write_json(
            "result.json",
            {
                "run_id": run_id,
                "status": RunStatus.COMPLETED.value,
                "spec_hash": spec.spec_hash,
                "summary": summary,
                "raw_sha256": checksums["raw_sha256"],
            },
        )
        journal.complete(checksums, derived, summary)
        index.finish_run(
            run_id,
            RunStatus.COMPLETED,
            utc_now(),
            matches=matches,
            hands=hands,
            raw_sha256=checksums["raw_sha256"],
        )
        return {"run_id": run_id, "status": "COMPLETED", **summary}
    except BaseException as exc:
        journal.fail(exc)
        index.finish_run(
            run_id,
            RunStatus.FAILED,
            utc_now(),
            matches=matches,
            hands=hands,
            error=f"{type(exc).__name__}: {str(exc)[:300]}",
        )
        raise
