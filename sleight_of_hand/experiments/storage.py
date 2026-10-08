"""SQLite index of specs, runs and interim evaluations.

The index is metadata only; raw evidence lives in each run's journal. The
schema is built by an ordered, append-only list of migrations, each applied
once inside a transaction. Never edit a released migration: add a new one.
Credentials are never written here (specs reject them on construction).
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .model import RunStatus, canonical
from .spec import ExperimentSpec

MIGRATIONS: tuple[tuple[int, str, str], ...] = (
    (
        1,
        "initial",
        """
        CREATE TABLE specs (
            spec_hash TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            arm TEXT NOT NULL,
            platform TEXT NOT NULL,
            provenance TEXT NOT NULL,
            policy_config_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            spec_json TEXT NOT NULL
        );
        CREATE INDEX specs_experiment ON specs (experiment_id, arm);
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            spec_hash TEXT NOT NULL REFERENCES specs (spec_hash),
            shard_start INTEGER NOT NULL,
            shard_end INTEGER NOT NULL,
            worker_id TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            matches INTEGER,
            hands INTEGER,
            raw_sha256 TEXT,
            run_dir TEXT NOT NULL,
            error TEXT
        );
        CREATE INDEX runs_spec ON runs (spec_hash, shard_start, status);
        CREATE TABLE evaluations (
            spec_hash TEXT NOT NULL REFERENCES specs (spec_hash),
            completed_matches INTEGER NOT NULL,
            recorded_at TEXT NOT NULL,
            binding INTEGER NOT NULL CHECK (binding IN (0, 1)),
            summary_json TEXT NOT NULL,
            PRIMARY KEY (spec_hash, completed_matches)
        );
        """,
    ),
)


def schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone()
    if row is None:
        return 0
    return connection.execute(
        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
    ).fetchone()[0]


@contextmanager
def transaction(connection: sqlite3.Connection):
    """Explicit write transaction (the connection is in autocommit mode)."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


def migrate(connection: sqlite3.Connection) -> int:
    """Apply pending migrations in order, each atomically; idempotent.

    The version is re-read inside each write transaction, so concurrent
    processes opening a fresh database cannot apply a migration twice.
    """
    known = [m[0] for m in MIGRATIONS]
    if known != list(range(1, len(MIGRATIONS) + 1)):
        raise RuntimeError("migrations must be numbered 1..n")
    with transaction(connection):
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, name TEXT NOT NULL)"
        )
    for version, label, sql in MIGRATIONS:
        with transaction(connection):
            current = schema_version(connection)
            if current > len(MIGRATIONS):
                raise RuntimeError("database is newer than this code")
            if current >= version:
                continue
            for statement in filter(None, (s.strip() for s in sql.split(";"))):
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                (version, label),
            )
    return schema_version(connection)


def _enable_wal(connection: sqlite3.Connection, *, timeout: float = 30) -> None:
    """Enter WAL in autocommit mode, within one contention timeout budget.

    The journal-mode lock upgrade can return SQLITE_BUSY without invoking
    SQLite's busy handler. Even reading the mode can contend with another
    opener. Retry only these operations, without retaining a transaction or
    cursor across attempts, and re-read the mode after every busy result.
    """
    deadline = time.monotonic() + timeout
    delay = 0.001
    # Own the wait budget here: SQLite must not spend another full timeout
    # inside each attempt. Ordinary transactions keep their native timeout.
    connection.execute("PRAGMA busy_timeout = 0")
    try:
        while True:
            try:
                mode = connection.execute("PRAGMA journal_mode").fetchone()
                if mode != ("wal",):
                    mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            except sqlite3.OperationalError as exc:
                code = getattr(exc, "sqlite_errorcode", None)
                # Error codes/constants were added in Python 3.11. On 3.10
                # accept only SQLite's exact SQLITE_BUSY message. SQLITE_LOCKED
                # (same-connection/shared-cache misuse) is not retried.
                busy = (
                    code & 0xFF == 5
                    if code is not None
                    else exc.args == ("database is locked",)
                )
                remaining = deadline - time.monotonic()
                if not busy or remaining <= 0:
                    raise
                time.sleep(min(delay, remaining))
                if time.monotonic() >= deadline:
                    raise
                delay = min(delay * 2, 0.05)
            else:
                if mode != ("wal",):
                    raise RuntimeError(f"index requires WAL journal mode, got {mode!r}")
                return
    finally:
        connection.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")


class Index:
    """Short-lived connections so concurrent worker processes can share it."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            _enable_wal(connection)
            connection.execute("PRAGMA foreign_keys = ON")
            migrate(connection)
            yield connection
        finally:
            connection.close()

    def register_spec(self, spec: ExperimentSpec) -> None:
        with self.connect() as db, transaction(db):
            row = db.execute(
                "SELECT spec_json FROM specs WHERE spec_hash = ?", (spec.spec_hash,)
            ).fetchone()
            text = canonical(spec.to_dict()).decode("ascii")
            if row is not None:
                if row[0] != text:
                    raise RuntimeError("stored spec differs for the same hash")
                return
            db.execute(
                "INSERT INTO specs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    spec.spec_hash,
                    spec.experiment_id,
                    spec.arm,
                    spec.platform,
                    spec.provenance.value,
                    spec.policy_config_hash,
                    spec.created_at,
                    text,
                ),
            )

    def start_run(self, run_id, spec_hash, shard, worker_id, started_at, run_dir):
        with self.connect() as db, transaction(db):
            db.execute(
                "INSERT INTO runs (run_id, spec_hash, shard_start, shard_end, "
                "worker_id, status, started_at, run_dir) VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    spec_hash,
                    shard[0],
                    shard[1],
                    worker_id,
                    RunStatus.RUNNING.value,
                    started_at,
                    str(run_dir),
                ),
            )

    def finish_run(self, run_id, status: RunStatus, finished_at, **values):
        allowed = {"matches", "hands", "raw_sha256", "error"}
        if not set(values) <= allowed:
            raise ValueError("unknown run fields")
        with self.connect() as db, transaction(db):
            current = db.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if current is None or current[0] != RunStatus.RUNNING.value:
                raise RuntimeError("only a running run can be finished")
            columns = ["status = ?", "finished_at = ?"] + [f"{k} = ?" for k in values]
            db.execute(
                f"UPDATE runs SET {', '.join(columns)} WHERE run_id = ?",
                (status.value, finished_at, *values.values(), run_id),
            )

    def runs(self, spec_hash: str | None = None) -> list[dict]:
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            query = "SELECT * FROM runs"
            args: tuple = ()
            if spec_hash is not None:
                query += " WHERE spec_hash = ?"
                args = (spec_hash,)
            return [dict(r) for r in db.execute(query + " ORDER BY shard_start", args)]

    def specs(self, experiment_id: str | None = None) -> list[ExperimentSpec]:
        with self.connect() as db:
            query, args = "SELECT spec_json FROM specs", ()
            if experiment_id is not None:
                query, args = query + " WHERE experiment_id = ?", (experiment_id,)
            return [
                ExperimentSpec.from_dict(json.loads(r[0]))
                for r in db.execute(query + " ORDER BY arm", args)
            ]

    def record_evaluation(self, spec_hash, completed_matches, recorded_at, summary):
        # Fixed-horizon specs: every interim look is recorded and non-binding.
        with self.connect() as db, transaction(db):
            db.execute(
                "INSERT OR IGNORE INTO evaluations VALUES (?, ?, ?, 0, ?)",
                (
                    spec_hash,
                    completed_matches,
                    recorded_at,
                    canonical(summary).decode("ascii"),
                ),
            )

    def evaluations(self, spec_hash: str) -> list[dict]:
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM evaluations WHERE spec_hash = ? "
                    "ORDER BY completed_matches",
                    (spec_hash,),
                )
            ]
