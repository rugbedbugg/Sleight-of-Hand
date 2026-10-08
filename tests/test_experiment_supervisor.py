"""Supervisor authority, offline orchestration and the installed public CLI."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from sleight_of_hand.experiments import (
    cli,
    journal,
    programme,
    runner,
    scheduler,
    supervisor,
)
from sleight_of_hand.experiments.model import Availability, Provenance
from sleight_of_hand.experiments.platforms.base import PlatformStatus
from sleight_of_hand.experiments.platforms.chipzen import ChipzenExternalPlatform
from sleight_of_hand.experiments.spec import StoppingRule, dump
from sleight_of_hand.experiments.storage import Index
from tests.test_experiment_chipzen import ENV, chipzen_spec
from tests.test_experiment_scheduler import tiny


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key in ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(scheduler, "available_mib", lambda: 4096)
    monkeypatch.setattr(
        ChipzenExternalPlatform,
        "play_match",
        Mock(side_effect=AssertionError("no live matches in supervisor tests")),
    )


def small(identifier="E9999", arm="canonical"):
    return tiny(
        experiment_id=identifier,
        arm=arm,
        policy_config={"samples": 2},
        metrics=("outcome",),
        stopping=StoppingRule(2, 2, 4, 4, 2, 1, 1),
    )


def write_programme(directory, groups=None):
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for specs in groups or [[small()]]:
        refs = []
        for spec in specs:
            path = directory / f"{spec.experiment_id}-{spec.arm}.json"
            path.write_bytes(dump(spec))
            refs.append({"path": path.name, "spec_hash": spec.spec_hash})
        entries.append(
            {
                "id": specs[0].experiment_id,
                "governance": "LOCAL_APPROVED"
                if specs[0].platform == "local"
                else "UNRATED_APPROVED",
                "reason": "approved fixed test programme",
                "specs": refs,
                "depends_on": [],
            }
        )
    value = {
        "schema_version": 1,
        "resources": {"workers": 2, "min_free_mib": 0},
        "experiments": entries,
    }
    path = directory / "programme.json"
    path.write_text(json.dumps(value))
    return path


def amend(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_programme_roundtrip_pins_and_historical_fingerprint(tmp_path):
    path = write_programme(tmp_path)
    approved = programme.load(path)
    assert approved.workers == 2
    assert approved.item("E9999").specs == (small(),)
    assert programme.load(path).digest == approved.digest
    historical = programme.load(runner.ROOT / "experiments/programme.json")
    assert {s.policy_revision for i in historical.items for s in i.specs} == {
        runner.policy_revision()
    }
    assert all(
        s.source_sha != runner.source_state()["git_head"]
        for i in historical.items
        for s in i.specs
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(schema_version=2),
        lambda v: v.update(unknown=True),
        lambda v: v["resources"].update(unknown=True),
        lambda v: v["resources"].update(workers=True),
        lambda v: v["resources"].update(min_free_mib=-1),
        lambda v: v["experiments"].append(copy.deepcopy(v["experiments"][0])),
        lambda v: v["experiments"][0].update(unknown=True),
        lambda v: v["experiments"][0].update(governance="UNRATED_APPROVED"),
        lambda v: v["experiments"][0].update(depends_on=["E9999"]),
        lambda v: v["experiments"][0]["specs"].append(
            copy.deepcopy(v["experiments"][0]["specs"][0])
        ),
        lambda v: v["experiments"][0]["specs"][0].update(spec_hash="0" * 64),
        lambda v: v["experiments"][0]["specs"][0].update(path="../outside.json"),
        lambda v: v["experiments"][0].update(id="different"),
    ],
)
def test_programme_rejects_ambiguous_or_unknown_authority(tmp_path, change):
    path = write_programme(tmp_path)
    amend(path, change)
    with pytest.raises(ValueError):
        programme.load(path)


@pytest.mark.parametrize(
    "platform, provenance",
    [
        ("local", Provenance.LIVE_RATED),
        ("chipzen", Provenance.LIVE_RATED),
        ("chipzen", Provenance.SYNTHETIC),
    ],
)
def test_unsupported_provenance_never_eligible(tmp_path, platform, provenance):
    spec = small() if platform == "local" else chipzen_spec()
    path = write_programme(tmp_path, [[replace(spec, provenance=provenance)]])
    with pytest.raises(ValueError, match="provenance"):
        programme.load(path)


def test_planning_has_no_execution_or_filesystem_side_effects(tmp_path, monkeypatch):
    path = write_programme(tmp_path / "specs")
    approved = programme.load(path)
    root = tmp_path / "runs"
    execute = Mock(side_effect=AssertionError("planning must not execute"))
    monkeypatch.setattr(scheduler, "run", execute)
    monkeypatch.setattr(supervisor.analysis, "analyze", execute)
    result = supervisor.plan(approved, root)
    assert not root.exists()
    assert result["items"][0]["intended_action"] == "RUN"
    assert result["items"][0]["pending_shards"] == 2
    assert result["source"]["policy_revision"] == runner.policy_revision()
    assert result["source"]["git_head"] == runner.source_state()["git_head"]
    assert result["production"] == "LOCKED / untouched"
    execute.assert_not_called()


def test_unapproved_experiment_cannot_run(tmp_path):
    approved = programme.load(write_programme(tmp_path / "specs"))
    with pytest.raises(ValueError, match="not in the approved"):
        supervisor.execute(approved, tmp_path / "runs", "run", "E1234")
    assert not (tmp_path / "runs").exists()


def test_real_auto_seals_analyzes_and_second_cycle_has_no_duplicate_work(
    tmp_path, monkeypatch
):
    a = small()
    b = a.derive_arm(
        "memory-off",
        a.created_at,
        policy_config={"samples": 2, "opponent_memory": False},
    )
    path = write_programme(tmp_path / "specs", [[a, b]])
    before = {p.name: p.read_bytes() for p in path.parent.glob("*.json")}
    approved = programme.load(path)
    root = tmp_path / "runs"
    first = supervisor.execute(approved, root, arguments=["auto"])
    assert first["disposition"] == "COMPLETE"
    assert [a["action"] for a in first["actions"]] == ["RUN", "ANALYZE"]
    records = Index(root / "index.sqlite").runs()
    assert len(records) == 4
    assert all(journal.verify(r["run_dir"])["ok"] for r in records)
    report = json.loads((root / "analysis/E9999.json").read_text())
    assert report["strata"][0]["comparisons"]
    assert all(a["matches"] == 2 for a in report["strata"][0]["arms"].values())
    monkeypatch.setattr(
        scheduler, "run", Mock(side_effect=AssertionError("completed work reran"))
    )
    second = supervisor.execute(approved, root, arguments=["auto"])
    assert second["disposition"] == "COMPLETE" and second["nothing_to_do"]
    assert second["actions"] == []
    assert Index(root / "index.sqlite").runs() == records
    assert {p.name: p.read_bytes() for p in path.parent.glob("*.json")} == before
    for result in (first, second):
        events = [
            json.loads(line)
            for line in Path(result["decision_journal"]).read_text().splitlines()
        ]
        assert events[0]["kind"] == "start" and events[-1]["kind"] == "end"
        assert events[-1]["disposition"] == "COMPLETE"
        assert events[0]["plan"]["programme_hash"] == approved.digest
        assert not os.access(result["decision_journal"], os.W_OK)
    # Completion is not permission to trust altered evidence on future cycles.
    derived = Path(records[0]["run_dir"]) / "normalized/hands.jsonl"
    derived.write_text("[]\n")
    assert supervisor.execute(approved, root)["disposition"] == "FAILED"


def test_incomplete_shard_resumes_through_existing_scheduler(tmp_path):
    spec = small()
    path = write_programme(tmp_path / "specs", [[spec]])
    root = tmp_path / "runs"
    index = Index(root / "index.sqlite")
    index.register_spec(spec)
    directory = root / "runs/crashed"
    raw = journal.RunJournal.create(directory, {"run_id": "crashed"})
    raw.append({"kind": "partial"})
    raw._close_raw()
    value = raw.manifest()
    value["owner"]["pid"] = 2**22 + 77
    (directory / "manifest.json").write_text(json.dumps(value))
    index.start_run(
        "crashed", spec.spec_hash, (0, 1), "w", journal.utc_now(), directory
    )
    approved = programme.load(path)
    assert supervisor.plan(approved, root)["items"][0]["incomplete_runs"] == 1
    result = supervisor.execute(approved, root)
    assert result["disposition"] == "COMPLETE"
    assert result["actions"][0]["outcome"]["recovered_incomplete"] == ["crashed"]
    assert sum(r["status"] == "COMPLETED" for r in index.runs()) == 2
    assert (
        next(r for r in index.runs() if r["run_id"] == "crashed")["status"]
        == "INCOMPLETE"
    )


def test_chipzen_blocked_does_not_stop_local_work(tmp_path):
    online = replace(
        chipzen_spec(), experiment_id="E0001", policy_revision=runner.policy_revision()
    )
    approved = programme.load(
        write_programme(tmp_path / "specs", [[online], [small()]])
    )
    result = supervisor.execute(approved, tmp_path / "runs")
    assert [(r["experiment"], r["status"]) for r in result["items"]] == [
        ("E0001", "BLOCKED"),
        ("E9999", "COMPLETE"),
    ]
    assert result["disposition"] == "BLOCKED"
    assert all(a["experiment"] == "E9999" for a in result["actions"])


def test_available_mock_chipzen_is_eligible_without_playing(tmp_path, monkeypatch):
    spec = replace(chipzen_spec(), policy_revision=runner.policy_revision())
    approved = programme.load(write_programme(tmp_path / "specs", [[spec]]))
    monkeypatch.setattr(
        ChipzenExternalPlatform,
        "status",
        lambda *a: PlatformStatus(Availability.AVAILABLE, "offline fixture"),
    )
    row = supervisor.plan(approved, tmp_path / "runs")["items"][0]
    assert row["status"] == "PENDING" and row["intended_action"] == "RUN"


def test_failure_stops_cycle_and_is_not_retried_next_time(tmp_path, monkeypatch):
    approved = programme.load(
        write_programme(tmp_path / "specs", [[small("E9998")], [small()]])
    )
    run = Mock(
        return_value={
            "failed": [{"error": "offline failure"}],
            "resource_blocked": None,
        }
    )
    monkeypatch.setattr(scheduler, "run", run)
    root = tmp_path / "runs"
    result = supervisor.execute(approved, root)
    assert result["disposition"] == "FAILED" and len(result["failures"]) == 1
    assert result["items"][0]["status"] == "FAILED"
    assert run.call_count == 1
    again = supervisor.execute(approved, root)
    assert again["disposition"] == "FAILED" and run.call_count == 1


@pytest.mark.parametrize("stale", [False, True])
def test_review_boundary_never_executes(tmp_path, monkeypatch, stale):
    spec = replace(small(), policy_revision="sha256:" + "0" * 64) if stale else small()
    path = write_programme(tmp_path / "specs", [[spec]])
    if not stale:
        amend(
            path,
            lambda v: v["experiments"][0].update(
                governance="REQUIRES_REVIEW", reason="human approval needed"
            ),
        )
    run = Mock(side_effect=AssertionError("governance bypass"))
    monkeypatch.setattr(scheduler, "run", run)
    result = supervisor.execute(programme.load(path), tmp_path / "runs")
    assert result["disposition"] == "REQUIRES_REVIEW"
    assert result["governance_stops"] and not result["actions"]


def test_journal_and_output_never_echo_secrets(tmp_path, monkeypatch):
    path = write_programme(tmp_path / "specs")
    monkeypatch.setenv("CHIPZEN_RESEARCH_TOKEN", "private-test-credential")
    monkeypatch.setattr(
        scheduler, "run", Mock(side_effect=RuntimeError("echo private-test-credential"))
    )
    result = supervisor.execute(
        programme.load(path),
        tmp_path / "runs",
        arguments=["auto", "private-test-credential"],
    )
    assert result["disposition"] == "FAILED"
    assert "private-test-credential" not in json.dumps(result)
    assert "private-test-credential" not in Path(result["decision_journal"]).read_text()


def test_resources_stop_without_waiting_or_starting_a_worker(tmp_path, monkeypatch):
    path = write_programme(tmp_path / "specs")
    amend(path, lambda v: v["resources"].update(min_free_mib=1024))
    monkeypatch.setattr(scheduler, "available_mib", lambda: 0)
    result = supervisor.execute(programme.load(path), tmp_path / "runs")
    assert result["disposition"] == "BLOCKED" and not result["actions"]
    # Also exercise the scheduler admission guard, including its first worker.
    outcome = scheduler.run(
        [small()], tmp_path / "strict", min_free_mib=1024, require_resources=True
    )
    assert outcome["completed"] == 0 and outcome["resource_blocked"]


def test_scheduler_stop_on_failure_does_not_admit_more_shards(tmp_path):
    stale = replace(small(), policy_revision="sha256:" + "0" * 64)
    outcome = scheduler.run(
        [stale], tmp_path, max_workers=1, min_free_mib=0, stop_on_failure=True
    )
    assert len(outcome["failed"]) == 1 and outcome["completed"] == 0


def test_dependencies_advance_in_the_same_cycle(tmp_path):
    path = write_programme(tmp_path / "specs", [[small("E9998")], [small()]])
    amend(path, lambda v: v["experiments"][1].update(depends_on=["E9998"]))
    approved = programme.load(path)
    assert (
        supervisor.plan(approved, tmp_path / "runs")["items"][1]["status"] == "BLOCKED"
    )
    result = supervisor.execute(approved, tmp_path / "runs")
    assert result["disposition"] == "COMPLETE"
    assert [a["experiment"] for a in result["actions"] if a["action"] == "RUN"] == [
        "E9998",
        "E9999",
    ]


def test_no_eligible_work_exits_cleanly(tmp_path):
    path = write_programme(tmp_path / "specs")
    amend(path, lambda v: v.update(experiments=[]))
    result = supervisor.execute(programme.load(path), tmp_path / "runs")
    assert result["disposition"] == "COMPLETE" and result["nothing_to_do"]


@pytest.mark.parametrize("command", ["status", "plan", "run", "analyze", "auto"])
def test_cli_parsing(command):
    args = [command] + (["E9999"] if command in {"run", "analyze"} else [])
    parsed = cli.parser().parse_args(args + ["--json"])
    assert parsed.command == command and parsed.json


def test_installed_uv_research_cli_works_without_pythonpath(tmp_path):
    path = write_programme(tmp_path / "specs")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for command in ("status", "plan", "auto", "auto", "analyze", "run"):
        args = [
            "uv",
            "run",
            "--locked",
            "--python",
            sys.executable,
            "research",
            command,
        ]
        if command in {"analyze", "run"}:
            args += ["E9999"]
        args += ["--programme", str(path), "--root", str(tmp_path / "runs"), "--json"]
        process = subprocess.run(
            args,
            cwd=runner.ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        assert process.returncode == 0, process.stderr
        result = json.loads(process.stdout)
        assert result["production"] == "LOCKED / untouched"
    assert len(Index(tmp_path / "runs/index.sqlite").runs()) == 2


def test_live_worker_blocks_duplicate_execution(tmp_path):
    path = write_programme(tmp_path / "specs")
    spec = small()
    index = Index(tmp_path / "runs/index.sqlite")
    index.register_spec(spec)
    directory = tmp_path / "runs/runs/live"
    raw = journal.RunJournal.create(directory, {"run_id": "live"})
    raw._close_raw()
    index.start_run("live", spec.spec_hash, (0, 1), "w", journal.utc_now(), directory)
    row = supervisor.plan(programme.load(path), tmp_path / "runs")["items"][0]
    assert row["status"] == "BLOCKED" and "another worker" in row["reason"]


def test_cycle_lock_prevents_two_supervisors(tmp_path):
    approved = programme.load(write_programme(tmp_path / "specs"))
    root = tmp_path / "runs"
    with (
        supervisor._cycle(root, supervisor.plan(approved, root), ["auto"]),
        pytest.raises(RuntimeError, match="another supervisor"),
    ):
        supervisor.execute(approved, root)


def test_duplicate_json_fields_are_rejected(tmp_path):
    path = write_programme(tmp_path)
    path.write_text(
        path.read_text().replace(
            '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
        )
    )
    with pytest.raises(ValueError, match="duplicate programme field"):
        programme.load(path)


def test_missing_pinned_prior_is_blocked_without_spec_rewrite(tmp_path):
    spec = replace(
        small(),
        policy_config_json=json.dumps(
            {
                "historical_prior": {
                    "bundle_path": str(tmp_path / "missing"),
                    "bundle_sha256": "sha256:" + "0" * 64,
                    "opponent_key": "script.tag_simple",
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    path = write_programme(tmp_path / "specs", [[spec]])
    row = supervisor.plan(programme.load(path), tmp_path / "runs")["items"][0]
    assert row["status"] == "BLOCKED" and row["intended_action"] == "WAIT_FOR_ARTIFACT"


def test_analyze_incomplete_does_not_run_or_claim_completion(tmp_path, monkeypatch):
    approved = programme.load(write_programme(tmp_path / "specs"))
    monkeypatch.setattr(
        scheduler,
        "run",
        Mock(side_effect=AssertionError("analysis must not execute matches")),
    )
    result = supervisor.execute(approved, tmp_path / "runs", "analyze", "E9999")
    assert result["disposition"] == "REQUIRES_REVIEW" and not result["actions"]


def test_indexed_unapproved_arm_requires_review(tmp_path):
    approved = programme.load(write_programme(tmp_path / "specs"))
    Index(tmp_path / "runs/index.sqlite").register_spec(small(arm="unapproved"))
    row = supervisor.plan(approved, tmp_path / "runs")["items"][0]
    assert row["status"] == "REQUIRES_REVIEW"


def test_analysis_evidence_is_bound_to_fixed_shard_horizon(tmp_path):
    spec = small()
    approved = programme.load(write_programme(tmp_path / "specs"))
    index = Index(tmp_path / "runs/index.sqlite")
    index.register_spec(spec)
    with index.connect() as db:
        db.execute(
            "INSERT INTO runs (run_id,spec_hash,shard_start,shard_end,worker_id,status,started_at,run_dir) VALUES (?,?,0,99,'w','COMPLETED','','unused')",
            ("bad-shard", spec.spec_hash),
        )
    row = supervisor.plan(approved, tmp_path / "runs")["items"][0]
    assert row["status"] == "REQUIRES_REVIEW" and "fixed horizon" in row["reason"]
