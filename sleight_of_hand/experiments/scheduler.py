"""Bounded concurrent execution of spec shards.

Workers are separate processes (``spawn``), so a crash or exception in one
cannot corrupt another's state; each shard writes only its own run
directory. At most ``max_workers`` shards are in flight, and a new shard is
admitted only while the host reports at least ``min_free_mib`` of available
memory. Completed shards are skipped on the next invocation, so batches are
resumable. Runs left RUNNING by a dead worker are marked INCOMPLETE.
"""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

from . import journal
from .model import RunStatus
from .runner import run_shard
from .spec import ExperimentSpec
from .storage import Index, transaction


@dataclass(frozen=True)
class Task:
    spec: ExperimentSpec
    shard: tuple[int, int]


def available_mib() -> float | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


def plan(specs: list[ExperimentSpec], index: Index) -> list[Task]:
    """Every shard without a COMPLETED run, interleaved across specs."""
    queues = []
    for spec in specs:
        done = {
            (r["shard_start"], r["shard_end"])
            for r in index.runs(spec.spec_hash)
            if r["status"] == RunStatus.COMPLETED.value
        }
        queues.append([Task(spec, s) for s in spec.stopping.shards() if s not in done])
    tasks = []
    while any(queues):  # round-robin so paired arms progress together
        for queue in queues:
            if queue:
                tasks.append(queue.pop(0))
    return tasks


def mark_incomplete(index: Index, runs_root: Path) -> list[str]:
    found = journal.recover(runs_root)
    if found:
        with index.connect() as db, transaction(db):
            for run_id in found:
                db.execute(
                    "UPDATE runs SET status = ?, finished_at = ? "
                    "WHERE run_id = ? AND status = ?",
                    (
                        RunStatus.INCOMPLETE.value,
                        journal.utc_now(),
                        run_id,
                        RunStatus.RUNNING.value,
                    ),
                )
    return found


def _interim(index: Index, spec: ExperimentSpec) -> None:
    """Record non-binding looks at each evaluation interval boundary."""
    runs = [
        r
        for r in index.runs(spec.spec_hash)
        if r["status"] == RunStatus.COMPLETED.value
    ]
    completed = sum(r["matches"] or 0 for r in runs)
    step = spec.stopping.evaluation_interval_matches
    boundary = completed - completed % step
    if boundary:
        index.record_evaluation(
            spec.spec_hash,
            boundary,
            journal.utc_now(),
            {
                "completed_matches": completed,
                "completed_hands": sum(r["hands"] or 0 for r in runs),
                "binding": False,
                "rule": "fixed horizon; interim looks cannot stop the run",
            },
        )


def run(
    specs: list[ExperimentSpec],
    root: Path,
    max_workers: int = 2,
    min_free_mib: float = 1024,
    worker_prefix: str = "local-worker",
    log=print,
    *,
    stop_on_failure: bool = False,
    require_resources: bool = False,
) -> dict:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    root = Path(root)
    index = Index(root / "index.sqlite")
    for spec in specs:
        index.register_spec(spec)
    recovered = mark_incomplete(index, root / "runs")
    tasks = plan(specs, index)
    log(f"planned {len(tasks)} shard(s); recovered {len(recovered)} incomplete run(s)")
    results, failures = [], []
    free_slots = [f"{worker_prefix}-{i:02d}" for i in range(1, max_workers + 1)]
    low_memory_waits = 0
    resource_blocked = None
    lowest_free = available_mib()
    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as pool:
            running = {}
            pending = list(tasks)
            while pending or running:
                while pending and free_slots:
                    if stop_on_failure and failures:
                        break
                    free = available_mib()
                    if free is not None:
                        lowest_free = min(lowest_free or free, free)
                    if require_resources and (free is None or free < min_free_mib):
                        low_memory_waits += 1
                        if not running:
                            resource_blocked = "insufficient or unknown free memory"
                        break
                    if free is not None and free < min_free_mib and running:
                        low_memory_waits += 1
                        break  # wait for a running shard to finish first
                    task = pending.pop(0)
                    slot = free_slots.pop(0)
                    future = pool.submit(
                        run_shard, task.spec.to_dict(), task.shard, slot, str(root)
                    )
                    running[future] = (task, slot)
                if not running:
                    if resource_blocked or (stop_on_failure and failures):
                        break
                    time.sleep(1)
                    continue
                done, _ = wait(running, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    task, slot = running.pop(future)
                    free_slots.append(slot)
                    label = f"{task.spec.experiment_id}/{task.spec.arm} {task.shard}"
                    try:
                        result = future.result()
                        results.append(result)
                        log(
                            f"{slot} COMPLETED {label}: {result['hands']} hands, "
                            f"{result['hands_per_second']} hands/s, "
                            f"rss {result['max_rss_mib']} MiB"
                        )
                        _interim(index, task.spec)
                    except BrokenProcessPool:
                        raise
                    except Exception as exc:  # noqa: BLE001 - isolate one shard
                        failures.append(
                            {"task": label, "error": f"{type(exc).__name__}: {exc}"}
                        )
                        log(f"{slot} FAILED {label}: {type(exc).__name__}: {exc}")
    finally:
        # A crashed worker leaves its run RUNNING; never report it as done.
        mark_incomplete(index, root / "runs")
    return {
        "planned": len(tasks),
        "completed": len(results),
        "failed": failures,
        "results": results,
        "recovered_incomplete": recovered,
        "low_memory_waits": low_memory_waits,
        "resource_blocked": resource_blocked,
        "lowest_available_mib": None if lowest_free is None else round(lowest_free),
    }
