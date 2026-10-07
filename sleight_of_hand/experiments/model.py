"""Shared vocabulary: provenance, statuses, identities, hashing and secrets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from enum import Enum


class Provenance(str, Enum):
    """Where evidence came from. Analysis never pools different classes."""

    LOCAL_SELFPLAY = "LOCAL_SELFPLAY"
    SCRIPTED_PROBE = "SCRIPTED_PROBE"
    LIVE_UNRATED = "LIVE_UNRATED"
    LIVE_RATED = "LIVE_RATED"
    BENCHMARK = "BENCHMARK"
    HISTORICAL_PUBLIC = "HISTORICAL_PUBLIC"
    SYNTHETIC = "SYNTHETIC"


class RunStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    # Found RUNNING with no live owner: never promoted to COMPLETED.
    INCOMPLETE = "INCOMPLETE"


class Availability(str, Enum):
    AVAILABLE = "AVAILABLE"
    READY_FOR_CREDENTIALS = "READY_FOR_CREDENTIALS"
    BLOCKED_PENDING_USER_ACTION = "BLOCKED_PENDING_USER_ACTION"
    BLOCKED_PENDING_RULE_CONFIRMATION = "BLOCKED_PENDING_RULE_CONFIRMATION"


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def name(value: str, what: str = "identifier") -> str:
    if type(value) is not str or not _NAME.fullmatch(value):
        raise ValueError(f"invalid {what}")
    return value


@dataclass(frozen=True)
class OpponentRef:
    """A namespaced opponent. Identities never match across platforms.

    Equality and the qualified key include the platform and key kind, so
    ``chipzen/bot_uuid/X`` and ``other/player_id/X`` are unrelated. The
    display name is metadata and never part of identity.
    """

    platform: str
    kind: str
    key: str
    display_name: str | None = None

    def __post_init__(self):
        name(self.platform, "platform")
        name(self.kind, "identity kind")
        name(self.key, "identity key")
        if self.display_name is not None and (
            type(self.display_name) is not str or len(self.display_name) > 128
        ):
            raise ValueError("invalid display name")

    @property
    def qualified(self) -> str:
        return f"{self.platform}/{self.kind}/{self.key}"

    def __eq__(self, other):
        if not isinstance(other, OpponentRef):
            return NotImplemented
        return self.qualified == other.qualified

    def __hash__(self):
        return hash(self.qualified)

    def to_dict(self) -> dict:
        value = {"platform": self.platform, "kind": self.kind, "key": self.key}
        if self.display_name is not None:
            value["display_name"] = self.display_name
        return value

    @classmethod
    def from_dict(cls, value: dict) -> OpponentRef:
        exact(value, {"platform", "kind", "key"}, {"display_name"})
        return cls(**value)


def exact(value, required: set, optional: set = frozenset()) -> dict:
    if type(value) is not dict:
        raise ValueError("expected an object")
    keys = set(value)
    if not required <= keys or not keys <= required | set(optional):
        raise ValueError(
            f"unexpected fields: missing {sorted(required - keys)}, "
            f"unknown {sorted(keys - required - set(optional))}"
        )
    return value


def _check_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("non-string key")
            _check_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_json(item)


def canonical(value) -> bytes:
    """Deterministic JSON bytes: sorted keys, compact, finite numbers only."""
    _check_json(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- secrets -----------------------------------------------------------------

#: Environment variables whose values are credentials. Values are never read
#: into specs or journals; they are only used to scrub accidental echoes.
SECRET_ENV = (
    "CHIPZEN_TOKEN",
    "CHIPZEN_TICKET",
    "CHIPZEN_RESEARCH_TOKEN",
    "CHIPZEN_PROBE_TOKEN",
)
SECRET_KEY = re.compile(
    r"(token|ticket|secret|password|passwd|api[_-]?key|authorization|cookie|"
    r"session)",
    re.IGNORECASE,
)
SECRET_VALUE = re.compile(
    r"(cz_extbot_[A-Za-z0-9_\-]+|Bearer\s+[A-Za-z0-9._\-]+|"
    r"([?&](token|ticket)=)[^&\s\"']+)"
)
REDACTED = "[REDACTED]"


class Redactor:
    """Scrubs credential-like keys and values from anything journaled."""

    def __init__(self, extra: tuple[str, ...] = ()):
        values = [os.environ.get(k) for k in SECRET_ENV] + list(extra)
        self._values = sorted({v for v in values if v and len(v) >= 6}, key=len)[::-1]

    def text(self, value: str) -> str:
        for secret in self._values:
            value = value.replace(secret, REDACTED)
        return SECRET_VALUE.sub(REDACTED, value)

    def __call__(self, value):
        if isinstance(value, dict):
            return {
                k: REDACTED if SECRET_KEY.search(str(k)) else self(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self(v) for v in value]
        if isinstance(value, str):
            return self.text(value)
        return value


def assert_no_secrets(value, where: str = "value") -> None:
    """Reject (rather than silently scrub) secrets in durable configuration."""
    if isinstance(value, dict):
        for key, item in value.items():
            if SECRET_KEY.search(str(key)):
                raise ValueError(f"credential-like key in {where}: {key!r}")
            assert_no_secrets(item, where)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_no_secrets(item, where)
    elif isinstance(value, str) and (
        SECRET_VALUE.search(value) or Redactor().text(value) != value
    ):
        raise ValueError(f"credential-like value in {where}")
