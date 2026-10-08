"""Journal immutability, crash handling, and the SQLite index migrations."""

import gzip
import json
import multiprocessing
import os
import queue
import sqlite3
import threading
import time
from contextlib import closing
from unittest.mock import Mock

import pytest

from sleight_of_hand.experiments import journal, storage
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


def assert_index_ready(db):
    assert db.isolation_level is None and not db.in_transaction
    assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    assert db.execute("PRAGMA foreign_keys").fetchone() == (1,)
    assert db.execute("PRAGMA busy_timeout").fetchone() == (30000,)
    assert storage.schema_version(db) == len(MIGRATIONS)
    assert db.execute(
        "SELECT version, name FROM schema_migrations ORDER BY version"
    ).fetchall() == [(v, name) for v, name, _ in MIGRATIONS]
    for table in ("specs", "runs", "evaluations"):
        assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)
    assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        db.execute("INSERT INTO evaluations VALUES ('missing', 1, '', 0, '{}')")


def touch_index(path, barrier, results):
    # Top-level for the scheduler's actual spawn process boundary.
    try:
        barrier.wait(timeout=15)
        with Index(path).connect() as db:
            assert_index_ready(db)
        results.put((os.getpid(), None))
    except Exception as exc:  # noqa: BLE001 - report child failures to the parent
        results.put((os.getpid(), f"{type(exc).__name__}: {exc}"))


def prepare_index(path, initial):
    if initial == "empty":
        path.touch()
    elif initial == "wal":
        with Index(path).connect() as db:
            assert_index_ready(db)


@pytest.mark.parametrize("initial", ["missing", "empty", "wal"])
def test_concurrent_first_connections_migrate_once(tmp_path, initial):
    path = tmp_path / "index.sqlite"
    prepare_index(path, initial)
    barrier, results = threading.Barrier(6), queue.Queue()
    threads = [
        threading.Thread(target=touch_index, args=(path, barrier, results), daemon=True)
        for _ in range(6)
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 45
    for thread in threads:
        thread.join(max(0, deadline - time.monotonic()))
    assert not any(thread.is_alive() for thread in threads)
    reports = [results.get(timeout=1) for _ in threads]
    assert [error for _, error in reports] == [None] * 6
    with Index(path).connect() as db:
        assert_index_ready(db)


@pytest.mark.parametrize("initial", ["missing", "empty", "wal"])
def test_concurrent_spawned_process_connections_migrate_once(tmp_path, initial):
    path = tmp_path / "index.sqlite"
    prepare_index(path, initial)
    context = multiprocessing.get_context("spawn")
    barrier, results = context.Barrier(6), context.Queue()
    processes = [
        context.Process(target=touch_index, args=(path, barrier, results))
        for _ in range(6)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 45
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [p.exitcode for p in processes] == [0] * 6
        reports = [results.get(timeout=1) for _ in processes]
        assert len({pid for pid, _ in reports}) == 6
        assert [error for _, error in reports] == [None] * 6
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
        results.join_thread()
    with Index(path).connect() as db:
        assert_index_ready(db)


def test_wal_bootstrap_rechecks_after_real_lock_contention(tmp_path, monkeypatch):
    path = tmp_path / "index.sqlite"
    with (
        closing(sqlite3.connect(path, isolation_level=None)) as blocker,
        closing(sqlite3.connect(path, isolation_level=None)) as candidate,
    ):
        blocker.execute("BEGIN IMMEDIATE")
        statements = []
        candidate.set_trace_callback(statements.append)

        def finish_competing_bootstrap(delay):
            blocker.execute("ROLLBACK")
            with Index(path).connect() as db:
                assert_index_ready(db)

        pause = Mock(side_effect=finish_competing_bootstrap)
        monkeypatch.setattr(storage.time, "sleep", pause)
        storage._enable_wal(candidate)
        pause.assert_called_once()
        # Re-read rather than assuming the other opener succeeded. SQLite can
        # still report this connection's cached DELETE mode; the second mode
        # assignment refreshes it and must itself return WAL.
        assert 1 <= statements.count("PRAGMA journal_mode = WAL") <= 2
        assert statements.count("PRAGMA journal_mode") == 2
        assert not candidate.in_transaction
        candidate.execute("PRAGMA foreign_keys = ON")
        migrate(candidate)
        assert_index_ready(candidate)


def test_existing_wal_does_not_repeat_transition(tmp_path, monkeypatch):
    path = tmp_path / "index.sqlite"
    with Index(path).connect():
        pass
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        statements = []
        db.set_trace_callback(statements.append)
        pause = Mock(side_effect=AssertionError("steady-state open must not sleep"))
        monkeypatch.setattr(storage.time, "sleep", pause)
        storage._enable_wal(db)
        assert "PRAGMA journal_mode = WAL" not in statements
        assert statements.count("PRAGMA journal_mode") == 1


@pytest.mark.parametrize("lock", ["BEGIN", "BEGIN IMMEDIATE", "BEGIN EXCLUSIVE"])
def test_wal_bootstrap_contention_has_one_bounded_budget(tmp_path, lock):
    path = tmp_path / "index.sqlite"
    with (
        closing(sqlite3.connect(path, isolation_level=None)) as blocker,
        closing(sqlite3.connect(path, isolation_level=None)) as candidate,
    ):
        blocker.execute("CREATE TABLE held (x)")
        blocker.execute(lock)
        blocker.execute("SELECT * FROM held").fetchall()
        start = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="^database is locked$"):
            storage._enable_wal(candidate, timeout=0.04)
        assert 0.03 <= time.monotonic() - start < 2
        assert not candidate.in_transaction
        assert candidate.execute("PRAGMA busy_timeout").fetchone() == (40,)


@pytest.mark.parametrize(
    "operation", ["PRAGMA journal_mode", "PRAGMA journal_mode = WAL"]
)
@pytest.mark.parametrize(
    "error, code",
    [
        (sqlite3.OperationalError("disk I/O error"), 10),
        (sqlite3.OperationalError("database table is locked"), 6),
        (sqlite3.OperationalError("database is locked"), 10),
        (sqlite3.OperationalError("database is locked unexpectedly"), None),
        (sqlite3.DatabaseError("database disk image is malformed"), 11),
    ],
)
def test_wal_bootstrap_propagates_non_busy_errors(monkeypatch, operation, error, code):
    if code is not None:
        error.sqlite_errorcode = code
    with closing(sqlite3.connect(":memory:", isolation_level=None)) as real:

        def execute(sql):
            if sql == operation:
                raise error
            return real.execute(sql)

        db = Mock(execute=Mock(side_effect=execute))
        pause = Mock(side_effect=AssertionError("non-busy errors must not retry"))
        monkeypatch.setattr(storage.time, "sleep", pause)
        with pytest.raises(type(error)) as caught:
            storage._enable_wal(db)
        assert caught.value is error
        pause.assert_not_called()


@pytest.mark.parametrize("code", [None, 5, 261])
def test_wal_bootstrap_busy_classification_and_deadline(monkeypatch, code):
    # Attribute-less errors exercise the Python 3.10 compatibility path;
    # 261 is SQLITE_BUSY_RECOVERY, an extended SQLITE_BUSY result.
    error = sqlite3.OperationalError("database is locked")
    if code is not None:
        error.sqlite_errorcode = code
    clock = [0.0]
    waits = []

    def sleep(delay):
        assert 0 < delay <= 0.05
        waits.append(delay)
        clock[0] += delay

    def execute(sql):
        if sql == "PRAGMA journal_mode":
            raise error

    monkeypatch.setattr(storage.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(storage.time, "sleep", sleep)
    with pytest.raises(sqlite3.OperationalError) as caught:
        storage._enable_wal(Mock(execute=execute), timeout=0.2)
    assert caught.value is error
    assert clock[0] == 0.2 and len(waits) < 20


def test_wal_bootstrap_rejects_unsupported_mode():
    with (
        closing(sqlite3.connect(":memory:", isolation_level=None)) as db,
        pytest.raises(RuntimeError, match="requires WAL.*memory"),
    ):
        storage._enable_wal(db)


def test_index_closes_connection_when_bootstrap_fails(tmp_path, monkeypatch):
    db = sqlite3.connect(":memory:", isolation_level=None)
    monkeypatch.setattr(storage.sqlite3, "connect", lambda *a, **kw: db)
    with (
        pytest.raises(RuntimeError, match="requires WAL"),
        Index(tmp_path / "index.sqlite").connect(),
    ):
        pytest.fail("must not yield without WAL")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        db.execute("SELECT 1")


def test_failed_migration_rolls_back_schema_and_version(tmp_path, monkeypatch):
    with closing(
        sqlite3.connect(tmp_path / "index.sqlite", isolation_level=None)
    ) as db:
        with monkeypatch.context() as patch:
            patch.setattr(
                storage,
                "MIGRATIONS",
                ((1, "broken", "CREATE TABLE partial (x); INVALID SQL"),),
            )
            with pytest.raises(sqlite3.OperationalError):
                migrate(db)
        assert storage.schema_version(db) == 0
        assert (
            db.execute("SELECT name FROM sqlite_master WHERE name='partial'").fetchall()
            == []
        )
        assert migrate(db) == len(MIGRATIONS)


def test_connect_waits_for_complete_migration(tmp_path, monkeypatch):
    path = tmp_path / "index.sqlite"
    real_connect = sqlite3.connect
    waiting = threading.Event()
    results = queue.Queue()

    def connect(*args, **kwargs):
        db = real_connect(*args, **kwargs)
        db.set_trace_callback(
            lambda sql: waiting.set() if sql == "BEGIN IMMEDIATE" else None
        )
        return db

    with closing(real_connect(path, isolation_level=None)) as migrating:
        migrating.execute("PRAGMA journal_mode = WAL")
        migrating.execute("BEGIN IMMEDIATE")
        migrating.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL)"
        )
        statements = [s.strip() for s in MIGRATIONS[0][2].split(";") if s.strip()]
        migrating.execute(statements[0])
        monkeypatch.setattr(storage.sqlite3, "connect", connect)
        thread = threading.Thread(
            target=touch_index,
            args=(path, threading.Barrier(1), results),
            daemon=True,
        )
        thread.start()
        try:
            assert waiting.wait(timeout=5)
            assert results.empty()  # connect cannot yield a partial schema
            with closing(real_connect(path)) as reader:
                assert storage.schema_version(reader) == 0
                assert (
                    reader.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                    == []
                )
            for statement in statements[1:]:
                migrating.execute(statement)
            migrating.execute("INSERT INTO schema_migrations VALUES (1, 'initial')")
            migrating.execute("COMMIT")
        finally:
            if migrating.in_transaction:
                migrating.execute("ROLLBACK")
            thread.join(timeout=35)
        assert not thread.is_alive()
        assert results.get(timeout=1)[1] is None


def test_newer_database_is_refused(tmp_path):
    with sqlite3.connect(tmp_path / "x.sqlite", isolation_level=None) as db:
        migrate(db)
        db.execute("INSERT INTO schema_migrations VALUES (99, 'future')")
        with pytest.raises(RuntimeError, match="newer"):
            migrate(db)
    with (
        pytest.raises(RuntimeError, match="newer"),
        Index(tmp_path / "x.sqlite").connect(),
    ):
        pytest.fail("must not yield a newer schema")


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
