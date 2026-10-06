"""Versioned evidence models. Identity locates evidence, never policy actions."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from uuid import UUID

SCHEMA_VERSION = 1
TENDENCIES = frozenset(
    (
        "btn_open",
        "btn_limp",
        "btn_fold",
        "bb_iso",
        "bb_check_limp",
        "bb_3bet",
        "bb_call_open",
        "bb_fold_open",
    )
    + tuple(
        f"{kind}/{bucket}" for kind in ("open_shove", "reshove") for bucket in range(6)
    )
)


def identifier(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value):
        raise ValueError("invalid identifier")
    return value


def natural(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**53:
        raise ValueError("invalid count")
    return value


def fields(value: object, expected: set[str]) -> dict:
    if type(value) is not dict or set(value) != expected:
        raise ValueError("invalid schema fields")
    return value


def digest(value: object) -> str:
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid digest")
    return value


def encode(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("ascii")


def decode(data: bytes) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=pairs)


@dataclass(frozen=True)
class OpponentIdentity:
    provider: str
    key: str | None = None
    key_kind: str = "unknown"
    scope: str = "unknown"
    persistence_evidence: str | None = None
    display_name: str | None = None  # Never serialized or used for lookup.

    def __post_init__(self):
        identifier(self.provider)
        if self.key is not None:
            identifier(self.key)
        if self.key_kind not in {
            "bot_uuid",
            "external_bot_id",
            "participant_id",
            "unknown",
        }:
            raise ValueError("invalid identity kind")
        if self.scope not in {"persistent", "match", "unknown"}:
            raise ValueError("invalid identity scope")
        if (
            self.key_kind == "bot_uuid"
            and self.key is not None
            and str(UUID(self.key)) != self.key
        ):
            raise ValueError("bot UUID must be canonical")
        if self.persistence_evidence is not None:
            identifier(self.persistence_evidence)
        if self.scope == "persistent" and (
            self.key is None
            or self.key_kind not in {"bot_uuid", "external_bot_id"}
            or not self.persistence_evidence
        ):
            raise ValueError("persistent identity requires a proven bot key")
        if self.display_name is not None and (
            type(self.display_name) is not str or len(self.display_name) > 128
        ):
            raise ValueError("invalid display metadata")

    @property
    def persistent_key(self) -> str | None:
        if self.scope != "persistent":
            return None
        # Tuple encoding is collision-free even when opaque keys contain ':'.
        return json.dumps(
            [self.provider, self.key_kind, self.key], separators=(",", ":")
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "display_name"}

    @classmethod
    def from_dict(cls, value):
        fields(value, {"provider", "key", "key_kind", "scope", "persistence_evidence"})
        return cls(**value)


@dataclass(frozen=True)
class IdentityResolution:
    identity: OpponentIdentity
    reason: str
    opponent_seat: int | None = None


def resolve_match_identity(
    provider: str, match_key: str, seats: object
) -> IdentityResolution:
    """Transport participant IDs are ONLY match-scoped; ignore any bot_id claim.

    A caller with independently verified stable bot mapping may inject a
    different resolver. Display names alone never establish a lookup key.
    """
    anonymous = IdentityResolution(OpponentIdentity(provider), "identity_unavailable")
    if not isinstance(seats, list) or len(seats) != 2:
        return anonymous
    if any(type(s) is not dict or type(s.get("seat")) is not int for s in seats):
        return anonymous
    if {s["seat"] for s in seats} != {0, 1}:
        return anonymous
    if sum(s.get("is_self") is True for s in seats) != 1:
        return anonymous
    other = next(s for s in seats if s.get("is_self") is not True)
    name = other.get("display_name")
    if type(name) is not str or len(name) > 128:
        name = None
    try:
        key = identifier(other.get("participant_id"))
    except ValueError:
        return IdentityResolution(
            OpponentIdentity(provider, display_name=name), "no_usable_id", other["seat"]
        )
    return IdentityResolution(
        OpponentIdentity(provider, key, "participant_id", "match", display_name=name),
        "participant_scope_unproven",
        other["seat"],
    )


@dataclass(frozen=True)
class Counts:
    successes: int = 0
    opportunities: int = 0

    def __post_init__(self):
        natural(self.successes)
        natural(self.opportunities)
        if self.successes > self.opportunities:
            raise ValueError("successes exceed opportunities")

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            self.successes + other.successes, self.opportunities + other.opportunities
        )


@dataclass(frozen=True)
class BeliefConfig:
    alpha_base: float = 1.0
    beta_base: float = 1.0
    historical_ess_cap: float = 10.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("nonfinite prior configuration")
            if value < 0 or (name != "historical_ess_cap" and value == 0):
                raise ValueError("invalid prior configuration")
            if value > 1e9:
                raise ValueError("prior configuration too large")


def validate_stats(stats: tuple[tuple[str, Counts], ...]) -> None:
    if type(stats) is not tuple or len(stats) > len(TENDENCIES):
        raise ValueError("invalid statistics")
    names = []
    for name, count in stats:
        if name not in TENDENCIES or not isinstance(count, Counts):
            raise ValueError("unknown tendency")
        names.append(name)
    if names != sorted(set(names)):
        raise ValueError("statistics must be unique and sorted")


@dataclass(frozen=True)
class Provenance:
    data_version: str
    input_hashes: tuple[str, ...]
    source_ids: tuple[str, ...]
    observation_count: int
    input_schema_version: int = 1

    def __post_init__(self):
        identifier(self.data_version)
        natural(self.observation_count)
        if type(self.input_schema_version) is not int or self.input_schema_version != 1:
            raise ValueError("unsupported input schema")
        for values, validate in (
            (self.input_hashes, digest),
            (self.source_ids, identifier),
        ):
            if (
                type(values) is not tuple
                or not values
                or tuple(sorted(set(values))) != values
            ):
                raise ValueError("invalid provenance")
            for v in values:
                validate(v)


@dataclass(frozen=True)
class HistoricalOpponentProfile:
    identity: OpponentIdentity
    stats: tuple[tuple[str, Counts], ...]
    total_hands: int
    total_matches: int
    provenance: Provenance
    config: BeliefConfig = BeliefConfig()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError("unsupported profile schema")
        if self.identity.persistent_key is None:
            raise ValueError("historical profiles require persistent identity")
        validate_stats(self.stats)
        natural(self.total_hands)
        natural(self.total_matches)
        if (
            not 0
            < self.total_matches
            <= self.total_hands
            == self.provenance.observation_count
        ):
            raise ValueError("inconsistent profile coverage")
        if any(c.opportunities > self.total_hands for _, c in self.stats):
            raise ValueError("opportunities exceed hands")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "stats": {k: asdict(v) for k, v in self.stats},
            "total_hands": self.total_hands,
            "total_matches": self.total_matches,
            "provenance": {
                **asdict(self.provenance),
                "input_hashes": list(self.provenance.input_hashes),
                "source_ids": list(self.provenance.source_ids),
            },
            "config": asdict(self.config),
            "historical_ess": {
                k: min(c.opportunities, self.config.historical_ess_cap)
                for k, c in self.stats
            },
        }

    @classmethod
    def from_dict(cls, data):
        fields(
            data,
            {
                "schema_version",
                "identity",
                "stats",
                "total_hands",
                "total_matches",
                "provenance",
                "config",
                "historical_ess",
            },
        )
        fields(data["config"], {"alpha_base", "beta_base", "historical_ess_cap"})
        p = fields(
            data["provenance"],
            {
                "data_version",
                "input_hashes",
                "source_ids",
                "observation_count",
                "input_schema_version",
            },
        )
        if type(p["input_hashes"]) is not list or type(p["source_ids"]) is not list:
            raise ValueError("invalid provenance lists")
        if type(data["historical_ess"]) is not dict or any(
            type(x) not in (int, float) or not math.isfinite(x)
            for x in data["historical_ess"].values()
        ):
            raise ValueError("invalid historical ESS")
        if type(data["stats"]) is not dict:
            raise ValueError("invalid statistics")
        stats = tuple(
            (k, Counts(**fields(v, {"successes", "opportunities"})))
            for k, v in sorted(data["stats"].items())
        )
        result = cls(
            OpponentIdentity.from_dict(data["identity"]),
            stats,
            data["total_hands"],
            data["total_matches"],
            Provenance(
                p["data_version"],
                tuple(p["input_hashes"]),
                tuple(p["source_ids"]),
                p["observation_count"],
                p["input_schema_version"],
            ),
            BeliefConfig(**data["config"]),
            data["schema_version"],
        )
        if data["historical_ess"] != result.to_dict()["historical_ess"]:
            raise ValueError("inconsistent historical ESS")
        return result
