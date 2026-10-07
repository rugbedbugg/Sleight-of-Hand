"""Journal immutability, crash handling, and the SQLite index migrations."""

import gzip
import json
import os
import sqlite3
import threading

import pytest

from sleight_of_hand.experiments import journal
from sleight_of_hand.experiments.model import RunStatus, sha256
from sleight_of_hand.experiments.storage import MIGRATIONS, Index, migrate
from tests.test_experiment_spec import make


def open_journal(tmp_path, name="run-1"):
    return journal.RunJournal.create(tmp_path / name, {"run_id": name})


def test_append_checkpoint_seal_is_deterministic_and_read_only(tmp_path):
    first = open_journal(tmp_path, "a")
    second = open_journal(tmp_path, "b")
    for j in (first, second):
        j.append({"match": 0, "kind": "message", "message": {"type": "x"}})
        j.checkpoint(1, 3)
        assert j.manifest()["progress"] == {"matches": 1, "hands": 3}
        j.append({"match": 1, "kind": "action", "seat": 0})
    sealed = [j.seal_raw() for j in (first, second)]
    assert sealed[0] == sealed[1]  # identical bytes, identical gzip
    raw_gz = tmp_path / "a" / journal.RAW_GZ
    assert not os.access(raw_gz, os.W_OK)
    assert not (tmp_path / "a" / journal.RAW).exists()
    events = journal.read_raw(tmp_path / "a")
    assert [e["seq"] for e in events] == [1, 2]
    assert sha256(gzip.decompress(raw_gz.read_bytes())) == sealed[0]["raw_sha256"]
    first.complete(sealed[0], {}, {"hands": 3})
    assert journal.verify(tmp_path / "a")["ok"]


def test_tampering_is_detected(tmp_path):
    j = open_journal(tmp_path)
    j.append({"match": 0, "kind": "deal"})
    checksums = j.seal_raw()
    j.complete(checksums, {}, {})
    path = tmp_path / "run-1" / journal.RAW_GZ
    path.chmod(0o600)
    path.write_bytes(journal.deterministic_gzip(b'{"forged":1}\n'))
    path.chmod(0o400)
    assert not journal.verify(tmp_path / "run-1")["ok"]


def test_a_run_directory_is_never_reused(tmp_path):
    open_journal(tmp_path)
    with pytest.raises(FileExistsError):
        open_journal(tmp_path)


def test_failure_keeps_raw_evidence_read_only(tmp_path):
    j = open_journal(tmp_path)
    j.append({"match": 0, "kind": "deal"})
    j.fail(RuntimeError("boom"))
    value = j.manifest()
    assert value["status"] == RunStatus.FAILED.value
    assert value["error"] == {"type": "RuntimeError", "message": "boom"}
    raw = tmp_path / "run-1" / journal.RAW
    assert raw.exists() and not os.access(raw, os.W_OK)
    assert journal.read_raw(tmp_path / "run-1")[0]["kind"] == "deal"


def test_dead_owner_is_marked_incomplete_never_completed(tmp_path):
    j = open_journal(tmp_path)
    j.append({"match": 0, "kind": "deal"})
    j.checkpoint(1, 1)
    j._raw.write(b'{"torn":')  # a crash mid-line
    j._raw.flush()
    value = j.manifest()
    value["owner"]["pid"] = 2**22 + 12345  # no such process
    (tmp_path / "run-1" / "manifest.json").write_text(json.dumps(value))
    assert journal.recover(tmp_path) == ["run-1"]
    recovered = j.manifest()
    assert recovered["status"] == RunStatus.INCOMPLETE.value
    assert recovered["progress"] == {"matches": 1, "hands": 1}
    assert [e["kind"] for e in journal.read_raw(tmp_path / "run-1")] == ["deal"]
    assert journal.recover(tmp_path) == []  # idempotent
    assert not journal.verify(tmp_path / "run-1")["ok"]


def test_live_owner_is_left_alone(tmp_path):
    open_journal(tmp_path)  # owned by this live process, fresh heartbeat
    assert journal.recover(tmp_path) == []


def test_journal_redacts_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("CHIPZEN_RESEARCH_TOKEN", "cz_extbot_secretvalue1")
    j = journal.RunJournal.create(tmp_path / "r", {"run_id": "r"}, journal.Redactor())
    j.append(
        {
            "match": 0,
            "kind": "message",
            "message": {"ticket": "abc", "echo": "cz_extbot_secretvalue1"},
        }
    )
    j.seal_raw()
    text = gzip.decompress((tmp_path / "r" / journal.RAW_GZ).read_bytes()).decode()
    assert "secretvalue1" not in text and '"abc"' not in text


# --- index ---------------------------------------------------------------------


def schema(path):
    with sqlite3.connect(path) as db:
        return sorted(
            row[0]
            for row in db.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL")
        )


def test_migrations_are_deterministic_and_idempotent(tmp_path):
    for name in ("a.sqlite", "b.sqlite"):
        with sqlite3.connect(tmp_path / name, isolation_level=None) as db:
            assert migrate(db) == len(MIGRATIONS)
            assert migrate(db) == len(MIGRATIONS)
            assert db.execute(
                "SELECT version, name FROM schema_migrations"
            ).fetchall() == [(1, "initial")]
    assert schema(tmp_path / "a.sqlite") == schema(tmp_path / "b.sqlite")


def test_concurrent_first_connections_migrate_once(tmp_path):
    index = Index(tmp_path / "index.sqlite")
    errors = []

    def touch():
        try:
            with index.connect():
                pass
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=touch) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    with sqlite3.connect(tmp_path / "index.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM schema_migrations").fetchone() == (1,)


def test_newer_database_is_refused(tmp_path):
    with sqlite3.connect(tmp_path / "x.sqlite", isolation_level=None) as db:
        migrate(db)
        db.execute("INSERT INTO schema_migrations VALUES (99, 'future')")
        with pytest.raises(RuntimeError, match="newer"):
            migrate(db)


def test_index_run_lifecycle_and_spec_integrity(tmp_path):
    index = Index(tmp_path / "index.sqlite")
    spec = make()
    index.register_spec(spec)
    index.register_spec(spec)  # idempotent
    assert index.specs("E9999") == [spec]
    index.start_run(
        "r1", spec.spec_hash, (0, 3), "w1", "2026-10-08T00:00:00Z", tmp_path
    )
    index.finish_run("r1", RunStatus.COMPLETED, "2026-10-08T00:01:00Z", hands=30)
    with pytest.raises(RuntimeError, match="only a running run"):
        index.finish_run("r1", RunStatus.FAILED, "2026-10-08T00:02:00Z")
    assert index.runs(spec.spec_hash)[0]["status"] == "COMPLETED"
    index.record_evaluation(spec.spec_hash, 2, "2026-10-08T00:03:00Z", {"n": 2})
    assert index.evaluations(spec.spec_hash)[0]["binding"] == 0
    with sqlite3.connect(tmp_path / "index.sqlite") as db:
        stored = db.execute("SELECT spec_json FROM specs").fetchone()[0]
    assert "token" not in stored.lower()
