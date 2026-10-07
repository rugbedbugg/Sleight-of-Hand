"""Bounded concurrent workers, resumption, isolation and crash recovery."""

import json

import pytest

from sleight_of_hand.experiments import journal, runner, scheduler
from sleight_of_hand.experiments.model import OpponentRef, RunStatus
from sleight_of_hand.experiments.platforms.local import LocalPlatform
from sleight_of_hand.experiments.spec import StoppingRule
from sleight_of_hand.experiments.storage import Index
from tests.test_experiment_spec import make


def tiny(arm="canonical", **changes):
    values = {
        "arm": arm,
        "policy_revision": runner.policy_revision(),
        "opponent_cohort": (
            OpponentRef("local", "script", "calling_station"),
            OpponentRef("local", "script", "random_legal"),
        ),
        "stopping": StoppingRule(
            min_matches=4,
            max_matches=4,
            min_hands=20,
            max_hands=20,
            hands_per_match=5,
            matches_per_shard=1,
            evaluation_interval_matches=2,
        ),
    }
    values.update(changes)
    return make(**values)


def test_policy_revision_is_line_ending_insensitive(tmp_path):
    for name in ("engine", "policy", "holdem"):
        (tmp_path / "sleight_of_hand" / name).mkdir(parents=True)
    (tmp_path / "bots/chipzen").mkdir(parents=True)
    (tmp_path / "sleight_of_hand/__init__.py").write_bytes(b"")
    for name in ("bot.py", "accounting_observer.py"):
        (tmp_path / "bots/chipzen" / name).write_bytes(b"x = 1\n")
    (tmp_path / "sleight_of_hand/holdem/a.py").write_bytes(b"a = 1\n")
    unix = runner.policy_revision(tmp_path)
    (tmp_path / "sleight_of_hand/holdem/a.py").write_bytes(b"a = 1\r\n")
    assert runner.policy_revision(tmp_path) == unix
    (tmp_path / "sleight_of_hand/holdem/a.py").write_bytes(b"a = 2\n")
    assert runner.policy_revision(tmp_path) != unix


def test_concurrent_workers_complete_resume_and_record_looks(tmp_path):
    specs = [tiny(), tiny("memory-off", policy_config={"opponent_memory": False})]
    outcome = scheduler.run(
        specs, tmp_path, max_workers=2, min_free_mib=0, log=lambda _: None
    )
    assert (
        outcome["planned"] == 8 and outcome["completed"] == 8 and not outcome["failed"]
    )
    index = Index(tmp_path / "index.sqlite")
    runs = index.runs()
    assert {r["worker_id"] for r in runs} <= {"local-worker-01", "local-worker-02"}
    assert all(r["status"] == "COMPLETED" for r in runs)
    for run in runs:
        assert journal.verify(run["run_dir"])["ok"]
    looks = index.evaluations(specs[0].spec_hash)
    assert looks and all(look["binding"] == 0 for look in looks)
    again = scheduler.run(
        specs, tmp_path, max_workers=2, min_free_mib=0, log=lambda _: None
    )
    assert again["planned"] == 0  # resumable: nothing left to do


def test_one_failing_spec_does_not_affect_another(tmp_path):
    good = tiny()
    stale = tiny("stale", policy_revision="sha256:" + "0" * 64)
    outcome = scheduler.run([stale, good], tmp_path, 2, 0, log=lambda _: None)
    assert outcome["completed"] == 4
    assert len(outcome["failed"]) == 4
    assert all("policy revision" in f["error"] for f in outcome["failed"])


def test_failure_mid_run_keeps_evidence_and_is_indexed(tmp_path, monkeypatch):
    spec = tiny()
    original = LocalPlatform.play_match

    def explode(self, context, emit):
        original(self, context, emit)
        raise RuntimeError("platform dropped")

    monkeypatch.setattr(LocalPlatform, "play_match", explode)
    with pytest.raises(RuntimeError, match="platform dropped"):
        runner.run_shard(spec.to_dict(), (0, 1), "w", str(tmp_path))
    (run,) = Index(tmp_path / "index.sqlite").runs()
    assert run["status"] == "FAILED" and "platform dropped" in run["error"]
    manifest = json.loads(
        (tmp_path / "runs" / run["run_id"] / "manifest.json").read_text()
    )
    assert manifest["status"] == "FAILED"
    assert journal.read_raw(tmp_path / "runs" / run["run_id"])  # raw retained


def test_crashed_run_is_reported_incomplete_and_its_shard_rerun(tmp_path):
    spec = tiny(stopping=StoppingRule(1, 1, 5, 5, 5, 1, 1))
    index = Index(tmp_path / "index.sqlite")
    index.register_spec(spec)
    run_dir = tmp_path / "runs" / "crashed"
    j = journal.RunJournal.create(run_dir, {"run_id": "crashed"})
    index.start_run("crashed", spec.spec_hash, (0, 1), "w", journal.utc_now(), run_dir)
    j.append({"match": 0, "kind": "deal"})
    value = j.manifest()
    value["owner"]["pid"] = 2**22 + 4321
    (run_dir / "manifest.json").write_text(json.dumps(value))
    outcome = scheduler.run([spec], tmp_path, 1, 0, log=lambda _: None)
    assert outcome["recovered_incomplete"] == ["crashed"]
    assert outcome["completed"] == 1
    statuses = {r["run_id"]: r["status"] for r in index.runs(spec.spec_hash)}
    assert statuses.pop("crashed") == RunStatus.INCOMPLETE.value
    assert list(statuses.values()) == ["COMPLETED"]


def test_memory_guard_waits_instead_of_overcommitting(tmp_path):
    outcome = scheduler.run(
        [tiny()], tmp_path, 2, min_free_mib=10**9, log=lambda _: None
    )
    assert outcome["completed"] == 4  # still progresses one shard at a time
    assert outcome["low_memory_waits"] > 0


def test_worker_count_is_bounded():
    with pytest.raises(ValueError):
        scheduler.run([], "unused", max_workers=0)
