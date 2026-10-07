"""Per-run immutable evidence journal.

Layout::

    runs/<run-id>/
        manifest.json                 status, provenance, versions, checksums
        raw/events.jsonl[.gz]         platform events exactly as delivered
        normalized/{hands,decisions,observations}.jsonl
        metrics.json
        result.json

Raw events are appended during the run and checkpointed (fsync) after each
match. Finalizing compresses them deterministically, verifies the round
trip, records the SHA-256 of the uncompressed stream and makes the raw files
read-only. Normalized data is derived and can always be regenerated from
raw. A run left RUNNING without a live owner is reported INCOMPLETE, never
completed.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import socket
import stat
import time
from datetime import datetime, timezone
from pathlib import Path

from sleight_of_hand.holdem.profile_store import atomic_write

from .model import Redactor, RunStatus, canonical, sha256

RAW = Path("raw") / "events.jsonl"
RAW_GZ = Path("raw") / "events.jsonl.gz"
#: A run whose owner has not checkpointed for this long is presumed dead.
STALE_SECONDS = 600


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_line(value) -> bytes:
    return canonical(value) + b"\n"


def _read_only(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def deterministic_gzip(data: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as stream:
        stream.write(data)
    return buffer.getvalue()


class RunJournal:
    def __init__(self, run_dir: Path, redactor: Redactor | None = None):
        self.dir = Path(run_dir)
        self.redact = redactor or Redactor()
        self._raw = None
        self._seq = 0

    # --- lifecycle -----------------------------------------------------------

    @classmethod
    def create(cls, run_dir: Path, manifest: dict, redactor=None) -> RunJournal:
        journal = cls(run_dir, redactor)
        journal.dir.mkdir(parents=True, exist_ok=False)  # never reuse a run
        (journal.dir / "raw").mkdir()
        (journal.dir / "normalized").mkdir()
        value = dict(manifest)
        value.update(
            status=RunStatus.RUNNING.value,
            owner={"host": socket.gethostname(), "pid": os.getpid()},
            heartbeat=time.time(),
            progress={"matches": 0, "hands": 0},
        )
        journal._write_manifest(value)
        journal._raw = open(journal.dir / RAW, "ab")  # noqa: SIM115 - long-lived
        return journal

    def manifest(self) -> dict:
        return json.loads((self.dir / "manifest.json").read_text(encoding="ascii"))

    def _write_manifest(self, value: dict) -> None:
        atomic_write(
            self.dir / "manifest.json",
            json.dumps(self.redact(value), indent=1, sort_keys=True).encode("ascii")
            + b"\n",
        )

    def append(self, event: dict) -> None:
        """Append one redacted raw event with a run-local sequence number."""
        if self._raw is None:
            raise RuntimeError("journal is not open for appends")
        self._seq += 1
        self._raw.write(_json_line({"seq": self._seq, **self.redact(event)}))

    def checkpoint(self, matches: int, hands: int) -> None:
        """Make every event so far durable and record progress."""
        self._raw.flush()
        os.fsync(self._raw.fileno())
        value = self.manifest()
        value.update(
            heartbeat=time.time(), progress={"matches": matches, "hands": hands}
        )
        self._write_manifest(value)

    def _close_raw(self) -> bytes:
        if self._raw is not None:
            self._raw.flush()
            os.fsync(self._raw.fileno())
            self._raw.close()
            self._raw = None
        return (self.dir / RAW).read_bytes()

    def write_derived(self, relative: str, lines: list[dict]) -> str:
        data = b"".join(_json_line(line) for line in lines)
        atomic_write(self.dir / relative, data)
        return sha256(data)

    def write_json(self, relative: str, value: dict) -> str:
        data = json.dumps(value, indent=1, sort_keys=True).encode("ascii") + b"\n"
        atomic_write(self.dir / relative, data)
        return sha256(data)

    def seal_raw(self) -> dict:
        """Compress, verify and freeze the raw stream; return its checksums."""
        raw = self._close_raw()
        packed = deterministic_gzip(raw)
        if gzip.decompress(packed) != raw:
            raise RuntimeError("raw journal compression round trip failed")
        atomic_write(self.dir / RAW_GZ, packed)
        (self.dir / RAW).unlink()
        _read_only(self.dir / RAW_GZ)
        return {
            "raw_sha256": sha256(raw),
            "raw_gzip_sha256": sha256(packed),
            "raw_events": raw.count(b"\n"),
            "raw_bytes": len(raw),
        }

    def complete(self, checksums: dict, derived: dict, summary: dict) -> None:
        value = self.manifest()
        value.update(
            status=RunStatus.COMPLETED.value,
            finished_at=utc_now(),
            heartbeat=time.time(),
            checksums={**checksums, "derived": derived},
            summary=summary,
        )
        self._write_manifest(value)

    def fail(self, error: BaseException) -> None:
        # Keep whatever raw evidence exists, durable and read-only.
        try:
            self._close_raw()
            if (self.dir / RAW).exists():
                _read_only(self.dir / RAW)
        except OSError:
            pass
        value = self.manifest()
        value.update(
            status=RunStatus.FAILED.value,
            finished_at=utc_now(),
            error={"type": type(error).__name__, "message": str(error)[:500]},
        )
        self._write_manifest(value)


def read_raw(run_dir: Path) -> list[dict]:
    """Raw events of a sealed (or failed/incomplete, unsealed) run."""
    run_dir = Path(run_dir)
    if (run_dir / RAW_GZ).exists():
        data = gzip.decompress((run_dir / RAW_GZ).read_bytes())
    else:
        data = (run_dir / RAW).read_bytes()
        # A crash can leave a torn final line; it is not evidence.
        if data and not data.endswith(b"\n"):
            data = data[: data.rfind(b"\n") + 1]
    return [json.loads(line) for line in data.splitlines()]


def verify(run_dir: Path) -> dict:
    """Recompute the raw checksum of a completed run."""
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    if manifest["status"] != RunStatus.COMPLETED.value:
        return {"ok": False, "reason": f"status {manifest['status']}"}
    packed = (Path(run_dir) / RAW_GZ).read_bytes()
    raw = gzip.decompress(packed)
    expected = manifest["checksums"]
    ok = (
        sha256(raw) == expected["raw_sha256"]
        and sha256(packed) == expected["raw_gzip_sha256"]
        and not os.access(Path(run_dir) / RAW_GZ, os.W_OK)
    )
    return {"ok": ok, "raw_sha256": sha256(raw)}


def _owner_alive(owner: dict) -> bool:
    if owner.get("host") != socket.gethostname():
        return True  # cannot judge another host; leave it alone
    try:
        os.kill(int(owner["pid"]), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, KeyError, TypeError, ValueError):
        return True
    return True


def recover(runs_root: Path, now: float | None = None) -> list[str]:
    """Mark RUNNING runs with a dead or silent owner INCOMPLETE; return them."""
    now = time.time() if now is None else now
    found = []
    for manifest_path in sorted(Path(runs_root).glob("*/manifest.json")):
        value = json.loads(manifest_path.read_text())
        if value.get("status") != RunStatus.RUNNING.value:
            continue
        stale = now - float(value.get("heartbeat", 0)) > STALE_SECONDS
        if _owner_alive(value.get("owner", {})) and not stale:
            continue
        value.update(
            status=RunStatus.INCOMPLETE.value,
            recovered_at=utc_now(),
            recovery_reason="owner exited" if not stale else "heartbeat stale",
        )
        atomic_write(
            manifest_path,
            json.dumps(value, indent=1, sort_keys=True).encode("ascii") + b"\n",
        )
        raw = manifest_path.parent / RAW
        if raw.exists():
            _read_only(raw)
        found.append(manifest_path.parent.name)
    return found
