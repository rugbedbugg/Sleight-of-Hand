"""Finite orchestration of approved research, with no policy authority."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

from . import analysis, journal, scheduler
from .model import Availability, Redactor, canonical, sha256
from .platforms import get_platform
from .platforms.priors import memory_for
from .programme import Item, Programme
from .runner import source_state
from .spec import ExperimentSpec
from .storage import MIGRATIONS, schema_version

VERSION = 1


class Snapshot:
    """Read-only metadata view implementing the scheduler's planning interface.

    A missing index is an empty programme, not a reason to create a database.
    Never migrate or recover journals while planning.
    """

    def __init__(self, root: Path):
        self.records, self.registered = [], []
        path = root / "index.sqlite"
        if path.exists():
            with closing(
                sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            ) as db:
                if schema_version(db) != len(MIGRATIONS):
                    raise RuntimeError("index schema requires review")
                db.row_factory = sqlite3.Row
                self.records = [
                    dict(row)
                    for row in db.execute("SELECT * FROM runs ORDER BY shard_start")
                ]
                self.registered = [
                    ExperimentSpec.from_dict(json.loads(row[0]))
                    for row in db.execute("SELECT spec_json FROM specs")
                ]

    def runs(self, spec_hash=None):
        return [
            r for r in self.records if spec_hash is None or r["spec_hash"] == spec_hash
        ]


def _running(run: dict) -> bool:
    manifest = json.loads((Path(run["run_dir"]) / "manifest.json").read_text())
    return (
        journal._owner_alive(manifest.get("owner", {}))
        and time.time() - float(manifest.get("heartbeat", 0)) <= journal.STALE_SECONDS
    )


def _evidence_key(item: Item, runs: list[dict]) -> str:
    return sha256(
        canonical(
            {
                "specs": [s.spec_hash for s in item.specs],
                "runs": sorted(
                    (r["run_id"], r["raw_sha256"])
                    for r in runs
                    if r["status"] == "COMPLETED"
                ),
            }
        )
    )


def _analysis_path(root: Path, item: Item) -> Path:
    return root / "analysis" / f"{item.experiment_id}.supervisor.json"


def _analysis_current(root: Path, item: Item, key: str) -> bool:
    path = _analysis_path(root, item)
    if not path.exists():
        return False
    try:
        receipt = json.loads(path.read_text())
        report = root / "analysis" / f"{item.experiment_id}.json"
        return (
            receipt["schema_version"] == VERSION
            and receipt["evidence_key"] == key
            and receipt["report_sha256"] == sha256(report.read_bytes())
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _past_failures(root: Path, digest: str) -> set[str]:
    failed = set()
    for path in sorted((root / "supervisor").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue  # abrupt termination can leave a torn final record
            if event.get("kind") == "end" and event.get("programme_hash") == digest:
                failed.update(
                    f.get("experiment", f.get("campaign"))
                    for f in event.get("failures", [])
                )
    return failed


def plan(programme: Programme, root: Path) -> dict:
    """Inspect authority, availability and shards without starting any work."""
    root = Path(root)
    source = source_state()
    snapshot = Snapshot(root)
    rows, preceding = [], {}
    free = scheduler.available_mib()
    past_failures = _past_failures(root, programme.digest)
    for item in programme.items:
        specs = item.specs
        hashes = {s.spec_hash for s in specs}
        runs = [r for r in snapshot.records if r["spec_hash"] in hashes]
        pending = scheduler.plan(list(specs), snapshot)
        total = sum(len(s.stopping.shards()) for s in specs)
        platforms = []
        status, reason, action = "PENDING", "approved unfinished work", "RUN"
        key = _evidence_key(item, runs)
        current = _analysis_current(root, item, key)
        row = {
            "experiment": item.experiment_id,
            "arms": [
                {"arm": s.arm, "spec_hash": s.spec_hash, "source_sha": s.source_sha}
                for s in specs
            ],
            "platform": specs[0].platform,
            "provenance": specs[0].provenance.value,
            "governance": item.governance,
            "governance_reason": item.reason,
            "pending_shards": len(pending),
            "completed_shards": total - len(pending),
            "incomplete_runs": sum(r["status"] == "INCOMPLETE" for r in runs),
            "running_runs": sum(r["status"] == "RUNNING" for r in runs),
            "analysis_status": "CURRENT" if current else "PENDING",
            "platform_statuses": platforms,
        }
        try:
            for spec in specs:
                platform = get_platform(spec.platform)
                state = platform.status(spec)
                platforms.append(
                    {
                        "arm": spec.arm,
                        "status": state.availability.value,
                        "reason": state.reason,
                    }
                )
                platform.prepare(spec)  # configuration validation only; no matches
            unapproved = [
                s
                for s in snapshot.registered
                if s.experiment_id == item.experiment_id and s.spec_hash not in hashes
            ]
            live = any(_running(r) for r in runs if r["status"] == "RUNNING")
            row["incomplete_runs"] += sum(
                not _running(r) for r in runs if r["status"] == "RUNNING"
            )
            completed = {
                (r["spec_hash"], r["shard_start"], r["shard_end"])
                for r in runs
                if r["status"] == "COMPLETED"
            }
            failed = any(
                r["status"] == "FAILED"
                and (r["spec_hash"], r["shard_start"], r["shard_end"]) not in completed
                for r in runs
            )
            allowed_shards = {
                (s.spec_hash, start, end)
                for s in specs
                for start, end in s.stopping.shards()
            }
            outside_horizon = any(
                (r["spec_hash"], r["shard_start"], r["shard_end"]) not in allowed_shards
                for r in runs
            )
            missing_prior = False
            for spec in specs:
                prior = spec.policy_config.get("historical_prior")
                if prior:
                    if not Path(prior["bundle_path"]).is_file():
                        missing_prior = True
                    else:
                        memory_for(spec.policy_config)
            if item.governance == "REQUIRES_REVIEW":
                status, reason, action = "REQUIRES_REVIEW", item.reason, "REVIEW"
            elif any(s.policy_revision != source["policy_revision"] for s in specs):
                status, reason, action = (
                    "REQUIRES_REVIEW",
                    "stale policy fingerprint; immutable specs were not changed",
                    "REVIEW",
                )
            elif unapproved:
                status, reason, action = (
                    "REQUIRES_REVIEW",
                    "index contains specs outside this programme's approved arm set",
                    "REVIEW",
                )
            elif outside_horizon:
                status, reason, action = (
                    "REQUIRES_REVIEW",
                    "indexed shard is outside the immutable fixed horizon",
                    "REVIEW",
                )
            elif failed or (pending and item.experiment_id in past_failures):
                status, reason, action = (
                    "FAILED",
                    "failed shard requires investigation; evidence retained",
                    "INVESTIGATE",
                )
            elif live:
                status, reason, action = (
                    "BLOCKED",
                    "another worker owns unfinished work",
                    "WAIT_FOR_USER",
                )
            elif any(preceding[d]["status"] != "COMPLETE" for d in item.depends_on):
                status, reason, action = (
                    "BLOCKED",
                    "dependencies incomplete: " + ", ".join(item.depends_on),
                    "WAIT_FOR_DEPENDENCY",
                )
            elif missing_prior:
                status, reason, action = (
                    "BLOCKED",
                    "pinned prior artifact missing; rebuild with the existing local-prior compiler and review its hash",
                    "WAIT_FOR_ARTIFACT",
                )
            elif not pending:
                status, reason, action = (
                    ("COMPLETE", "fixed work and analysis complete", "NONE")
                    if current
                    else (
                        "PENDING",
                        "fixed work complete; verified analysis required",
                        "ANALYZE",
                    )
                )
            elif any(s["status"] != Availability.AVAILABLE.value for s in platforms):
                status, reason, action = (
                    "BLOCKED",
                    "; ".join(
                        s["reason"] for s in platforms if s["status"] != "AVAILABLE"
                    ),
                    "WAIT_FOR_USER",
                )
            elif free is None or free < programme.min_free_mib:
                status, reason, action = (
                    "BLOCKED",
                    "insufficient or unknown free memory; no safe admission",
                    "WAIT_FOR_RESOURCES",
                )
        except Exception as exc:  # noqa: BLE001 - one bad item must be inspectable
            status, reason, action = (
                "FAILED",
                f"{type(exc).__name__}: {exc}",
                "INVESTIGATE",
            )
        row.update(status=status, reason=reason, intended_action=action)
        rows.append(row)
        preceding[item.experiment_id] = row
    from . import optimizer

    campaigns = [
        optimizer.inspect(c, programme, root, source, preceding)
        for c in programme.campaigns
    ]
    return Redactor()(
        {
            "schema_version": VERSION,
            "source": source,
            "programme_hash": programme.digest,
            "workers": programme.workers,
            "min_free_mib": programme.min_free_mib,
            "available_mib": free,
            "production": "LOCKED / untouched",
            "items": rows,
            "campaigns": campaigns,
        }
    )


def _analyze(root: Path, item: Item, *, familywise: bool = False) -> dict:
    _verify_evidence(root, item)
    # Existing analysis checks raw checksums and recomputes all statistics.
    report = analysis.analyze(
        root, item.experiment_id, check_raw=True, familywise=familywise
    )
    for stratum in report["strata"]:
        for arm in stratum["arms"].values():
            if arm["problems"]:
                raise RuntimeError("raw evidence verification failed")
    data = canonical(report) + b"\n"
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    journal.atomic_write(out / f"{item.experiment_id}.json", data)
    journal.atomic_write(
        out / f"{item.experiment_id}.md", (analysis.markdown(report) + "\n").encode()
    )
    conclusions = []
    for stratum in report["strata"]:
        for comparison in stratum["comparisons"]:
            verdict = comparison["recommendation"]["verdict"]
            conclusions.append(
                {
                    "status": "REQUIRES_REVIEW"
                    if verdict == "DIFFERENCE_DETECTED"
                    else "INCONCLUSIVE",
                    "comparison": comparison["arms"],
                    "evidence": comparison["recommendation"],
                }
            )
        for arm, result in stratum["arms"].items():
            if "accounting" in result["metrics"]:
                conclusions.append(
                    {
                        "status": "REQUIRES_REVIEW",
                        "arm": arm,
                        "accounting": result["metrics"]["accounting"],
                    }
                )
    if not conclusions:
        conclusions.append(
            {
                "status": "INCONCLUSIVE",
                "reason": "descriptive evidence; no comparative conclusion",
            }
        )
    hashes = {s.spec_hash for s in item.specs}
    runs = [r for r in Snapshot(root).records if r["spec_hash"] in hashes]
    receipt = {
        "schema_version": VERSION,
        "evidence_key": _evidence_key(item, runs),
        "report_sha256": sha256(data),
        "conclusions": conclusions,
        "binding": False,
        "note": "Additional samples or policy changes require review and a new approved spec.",
    }
    journal.atomic_write(_analysis_path(root, item), canonical(receipt) + b"\n")
    return receipt


def _verify_evidence(root: Path, item: Item) -> None:
    hashes = {s.spec_hash for s in item.specs}
    for run in Snapshot(root).records:
        if run["spec_hash"] not in hashes or run["status"] != "COMPLETED":
            continue
        directory = Path(run["run_dir"])
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest["spec_hash"] != run["spec_hash"]
            or manifest["shard"] != [run["shard_start"], run["shard_end"]]
            or manifest["checksums"]["raw_sha256"] != run["raw_sha256"]
            or not journal.verify(directory)["ok"]
        ):
            raise RuntimeError(
                "completed evidence does not match its indexed spec/shard"
            )
        for relative, expected in manifest["checksums"]["derived"].items():
            if sha256((directory / relative).read_bytes()) != expected:
                raise RuntimeError(
                    "derived evidence checksum failed; rebuild requires review"
                )


@contextmanager
def _cycle(root: Path, initial: dict, arguments: list[str]):
    directory = root / "supervisor"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another supervisor cycle is active") from exc
        path = directory / f"{uuid.uuid4().hex}.jsonl"
        with path.open("xb") as stream:
            sequence = 0
            redact = Redactor()

            def record(kind, **values):
                nonlocal sequence
                sequence += 1
                stream.write(
                    canonical(
                        redact(
                            {
                                "seq": sequence,
                                "kind": kind,
                                "at": journal.utc_now(),
                                **values,
                            }
                        )
                    )
                    + b"\n"
                )
                stream.flush()
                os.fsync(stream.fileno())

            record("start", schema_version=VERSION, arguments=arguments, plan=initial)
            try:
                yield record, path
            except BaseException as exc:
                record(
                    "end",
                    disposition="FAILED",
                    failure={"type": type(exc).__name__, "message": str(exc)},
                )
                raise
            finally:
                path.chmod(0o400)


def execute(
    programme: Programme, root: Path, command="auto", experiment=None, arguments=()
) -> dict:
    """One bounded pass; each approved campaign advances at most one batch."""
    from . import optimizer

    if command == "optimize":
        if experiment not in {c.identifier for c in programme.campaigns}:
            raise ValueError("campaign is not in the approved programme")
    elif experiment is not None:
        programme.item(experiment)  # refuse unapproved IDs before creating files
    root = Path(root).resolve()
    initial = plan(programme, root)
    actions, failures, stops = [], [], []
    with _cycle(root, initial, list(arguments)) as (record, decision_path):
        # Re-plan under the lock, and after each item so dependencies can advance.
        for item in programme.items:
            if command == "optimize":
                continue
            if experiment is not None and item.experiment_id != experiment:
                continue
            row = next(
                r
                for r in plan(programme, root)["items"]
                if r["experiment"] == item.experiment_id
            )
            record("inspect", item=row)
            if row["status"] in {"FAILED", "REQUIRES_REVIEW"}:
                (failures if row["status"] == "FAILED" else stops).append(row)
                break
            if row["status"] == "BLOCKED":
                if row["intended_action"] == "WAIT_FOR_RESOURCES":
                    stops.append(row)
                    break
                continue
            if command == "analyze" and row["pending_shards"]:
                stops.append(
                    {
                        "experiment": item.experiment_id,
                        "status": "REQUIRES_REVIEW",
                        "reason": "complete the immutable fixed horizon before analysis",
                    }
                )
                break
            try:
                _verify_evidence(root, item)
                if row["status"] == "COMPLETE" and command != "analyze":
                    record("verified", experiment=item.experiment_id)
                    continue
                if row["pending_shards"]:
                    record(
                        "action_started", experiment=item.experiment_id, action="RUN"
                    )
                    outcome = scheduler.run(
                        list(item.specs),
                        root,
                        programme.workers,
                        programme.min_free_mib,
                        log=lambda _: None,
                        stop_on_failure=True,
                        require_resources=True,
                    )
                    actions.append(
                        {
                            "experiment": item.experiment_id,
                            "action": "RUN",
                            "outcome": outcome,
                        }
                    )
                    record("action_finished", **actions[-1])
                    if outcome["failed"]:
                        failures.append(
                            {
                                "experiment": item.experiment_id,
                                "status": "FAILED",
                                "reason": "experiment failed; investigation required",
                            }
                        )
                        break
                    if outcome["resource_blocked"]:
                        stops.append(
                            {
                                "experiment": item.experiment_id,
                                "status": "BLOCKED",
                                "reason": outcome["resource_blocked"],
                            }
                        )
                        break
                # Recheck exact horizon and authority before accepting analysis.
                row = next(
                    r
                    for r in plan(programme, root)["items"]
                    if r["experiment"] == item.experiment_id
                )
                if row["pending_shards"] or row["status"] not in {
                    "PENDING",
                    "COMPLETE",
                }:
                    raise RuntimeError(
                        "experiment did not reach a complete analyzable horizon"
                    )
                record(
                    "action_started", experiment=item.experiment_id, action="ANALYZE"
                )
                receipt = _analyze(root, item)
                actions.append(
                    {
                        "experiment": item.experiment_id,
                        "action": "ANALYZE",
                        "analysis": receipt,
                    }
                )
                record("action_finished", **actions[-1])
            except Exception as exc:  # noqa: BLE001 - durable failed cycle, no retries
                failures.append(
                    {
                        "experiment": item.experiment_id,
                        "status": "FAILED",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                record("failure", **failures[-1])
                break
        if command in {"auto", "optimize"} and not failures and not stops:
            for campaign in programme.campaigns:
                if experiment is not None and campaign.identifier != experiment:
                    continue
                current = plan(programme, root)
                row = next(
                    r
                    for r in current["campaigns"]
                    if r["campaign"] == campaign.identifier
                )
                record("inspect_campaign", item=row)
                if row["status"] in {"FAILED", "REQUIRES_REVIEW"}:
                    (failures if row["status"] == "FAILED" else stops).append(row)
                    break
                if row["status"] == "BLOCKED":
                    continue
                try:
                    progress = optimizer.advance(
                        campaign, programme, root, current["source"]
                    )
                    actions.extend(progress)
                    for action in progress:
                        record("action_finished", **action)
                    after = next(
                        r
                        for r in plan(programme, root)["campaigns"]
                        if r["campaign"] == campaign.identifier
                    )
                    if after["status"] in {"FAILED", "REQUIRES_REVIEW", "BLOCKED"}:
                        break
                except Exception as exc:  # noqa: BLE001 - durable failed cycle
                    failure = {
                        "campaign": campaign.identifier,
                        "status": "REQUIRES_REVIEW"
                        if isinstance(exc, optimizer.ReviewRequired)
                        else "FAILED",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                    (
                        stops if failure["status"] == "REQUIRES_REVIEW" else failures
                    ).append(failure)
                    record("failure", **failure)
                    break
        final = plan(programme, root)
        for stop in failures + stops:
            key = "campaign" if "campaign" in stop else "experiment"
            row = next(
                r
                for r in final["items"] + final["campaigns"]
                if r.get(key) == stop[key]
            )
            row.update(
                status=stop["status"],
                reason=stop["reason"],
                intended_action={
                    "FAILED": "INVESTIGATE",
                    "REQUIRES_REVIEW": "REVIEW",
                    "BLOCKED": "WAIT_FOR_RESOURCES",
                }[stop["status"]],
            )
            if key == "campaign":
                row["classification"] = stop["status"]
        selected = [
            r
            for r in final["items"] + final["campaigns"]
            if experiment is None
            or r.get("experiment", r.get("campaign")) == experiment
        ]
        disposition = (
            "FAILED"
            if failures or any(r["status"] == "FAILED" for r in selected)
            else "REQUIRES_REVIEW"
            if any(s["status"] == "REQUIRES_REVIEW" for s in stops + selected)
            else "BLOCKED"
            if stops or any(r["status"] == "BLOCKED" for r in selected)
            else "COMPLETE"
            if all(r["status"] == "COMPLETE" for r in selected)
            else "PENDING"
        )
        final.update(
            disposition=disposition,
            actions=actions,
            failures=failures,
            governance_stops=stops,
            decision_journal=str(decision_path),
            nothing_to_do=not actions,
        )
        record("end", **final)
    return Redactor()(final)
