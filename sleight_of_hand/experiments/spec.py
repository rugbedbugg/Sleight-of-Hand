"""Immutable experiment specifications.

A spec is frozen, validated on construction and identified by the SHA-256 of
its canonical JSON. Changing any configuration means deriving a new arm (a
new spec with a new hash); a spec is never edited in place. Stopping rules
are fixed before the first match and recorded in the spec itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from .model import (
    OpponentRef,
    Provenance,
    assert_no_secrets,
    canonical,
    exact,
    name,
    sha256,
)

SCHEMA_VERSION = 1
GAME_VARIANTS = frozenset({"nlhe_hu"})
METRIC_GROUPS = frozenset({"outcome", "behavior", "model", "accounting"})
POLICY_KEYS = frozenset({"opponent_memory", "params", "samples", "historical_prior"})
PARAM_NAMES = frozenset(
    {"value_bet_threshold", "call_threshold", "bluff_freq", "aggression", "steepness"}
)
_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _int(value, low, high, what):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid {what}")
    return value


@dataclass(frozen=True)
class SeedPolicy:
    """Every random stream derives from ``base_seed`` and the match index.

    Arms sharing a base seed and ``common_random_numbers`` see identical
    decks and bot seeds per match index, so their results pair exactly.
    """

    base_seed: int
    common_random_numbers: bool = True

    def __post_init__(self):
        _int(self.base_seed, 0, 2**63 - 1, "base seed")
        if type(self.common_random_numbers) is not bool:
            raise ValueError("invalid common_random_numbers")

    def derive(self, *parts) -> int:
        material = ":".join(str(p) for p in (self.base_seed, *parts))
        return int(sha256(material.encode("ascii"))[:16], 16)


@dataclass(frozen=True)
class StoppingRule:
    """A fixed-horizon rule declared before the experiment starts.

    ``max_matches`` and ``hands_per_match`` fix the plan; ``max_hands`` is a
    hard ceiling the plan may not exceed. ``min_*`` are the sample sizes an
    analysis needs before it reports a comparison. Interim evaluations every
    ``evaluation_interval_matches`` are recorded as non-binding looks:
    ``early_stopping`` is ``"none"``, so no result can stop a run early.
    """

    min_matches: int
    max_matches: int
    min_hands: int
    max_hands: int
    hands_per_match: int
    matches_per_shard: int
    evaluation_interval_matches: int
    early_stopping: str = "none"

    def __post_init__(self):
        _int(self.min_matches, 1, 10**7, "min_matches")
        _int(self.max_matches, self.min_matches, 10**7, "max_matches")
        _int(self.hands_per_match, 1, 10**5, "hands_per_match")
        _int(self.min_hands, 1, 10**9, "min_hands")
        _int(self.max_hands, self.min_hands, 10**9, "max_hands")
        _int(self.matches_per_shard, 1, self.max_matches, "matches_per_shard")
        _int(self.evaluation_interval_matches, 1, self.max_matches, "interval")
        if self.max_matches * self.hands_per_match > self.max_hands:
            raise ValueError("planned hands exceed max_hands")
        if self.early_stopping != "none":
            raise ValueError("only fixed-horizon stopping is implemented")

    def shards(self) -> list[tuple[int, int]]:
        size = self.matches_per_shard
        return [
            (start, min(start + size, self.max_matches))
            for start in range(0, self.max_matches, size)
        ]


def _policy(config: dict) -> dict:
    if type(config) is not dict or not set(config) <= POLICY_KEYS:
        raise ValueError("unknown policy configuration keys")
    if type(config.get("opponent_memory", True)) is not bool:
        raise ValueError("invalid opponent_memory")
    params = config.get("params", {})
    if type(params) is not dict or not set(params) <= PARAM_NAMES:
        raise ValueError("unknown policy parameters")
    for value in params.values():
        if type(value) not in (int, float):
            raise ValueError("invalid policy parameter")
    _int(config.get("samples", 128), 1, 512, "samples")
    prior = config.get("historical_prior")
    if prior is not None and type(prior) is not dict:
        raise ValueError("invalid historical_prior")
    return config


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    arm: str
    description: str
    source_sha: str
    policy_revision: str
    policy_config_json: str
    platform: str
    platform_config_json: str
    game_variant: str
    opponent_cohort: tuple[OpponentRef, ...]
    seed_policy: SeedPolicy
    stopping: StoppingRule
    metrics: tuple[str, ...]
    provenance: Provenance
    created_at: str
    schema_version: int = SCHEMA_VERSION
    _hash: str = field(default="", repr=False, compare=False)

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported spec schema")
        name(self.experiment_id, "experiment_id")
        name(self.arm, "arm")
        if type(self.description) is not str or len(self.description) > 2000:
            raise ValueError("invalid description")
        if type(self.source_sha) is not str or not _SHA.fullmatch(self.source_sha):
            raise ValueError("source_sha must be a full commit SHA")
        if type(self.policy_revision) is not str or not _DIGEST.fullmatch(
            self.policy_revision
        ):
            raise ValueError("policy_revision must be sha256:<hex>")
        name(self.platform, "platform")
        if self.game_variant not in GAME_VARIANTS:
            raise ValueError("unsupported game variant")
        if not self.opponent_cohort or not all(
            isinstance(o, OpponentRef) and o.platform == self.platform
            for o in self.opponent_cohort
        ):
            # Opponents are identities on the spec's own platform only.
            raise ValueError("cohort must be non-empty and on the spec platform")
        if len(set(self.opponent_cohort)) != len(self.opponent_cohort):
            raise ValueError("duplicate cohort member")
        if not isinstance(self.seed_policy, SeedPolicy) or not isinstance(
            self.stopping, StoppingRule
        ):
            raise TypeError("invalid seed policy or stopping rule")
        if not self.metrics or not set(self.metrics) <= METRIC_GROUPS:
            raise ValueError("unknown metric group")
        if list(self.metrics) != sorted(set(self.metrics)):
            raise ValueError("metrics must be sorted and unique")
        if not isinstance(self.provenance, Provenance):
            raise TypeError("invalid provenance")
        if type(self.created_at) is not str or not _STAMP.fullmatch(self.created_at):
            raise ValueError("created_at must be YYYY-MM-DDTHH:MM:SSZ")
        for text in (self.policy_config_json, self.platform_config_json):
            if json.dumps(json.loads(text), sort_keys=True, separators=(",", ":")) != (
                text
            ):
                raise ValueError("configuration must be canonical JSON")
        _policy(self.policy_config)
        assert_no_secrets(self.to_dict(), "experiment spec")
        object.__setattr__(self, "_hash", sha256(canonical(self.to_dict())))

    @classmethod
    def create(cls, *, policy_config: dict, platform_config: dict, **values):
        return cls(
            policy_config_json=canonical(policy_config).decode("ascii"),
            platform_config_json=canonical(platform_config).decode("ascii"),
            **values,
        )

    @property
    def policy_config(self) -> dict:
        return json.loads(self.policy_config_json)  # a fresh copy each time

    @property
    def platform_config(self) -> dict:
        return json.loads(self.platform_config_json)

    @property
    def policy_config_hash(self) -> str:
        return "sha256:" + sha256(self.policy_config_json.encode("ascii"))

    @property
    def spec_hash(self) -> str:
        return self._hash

    @property
    def short_hash(self) -> str:
        return self._hash[:12]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "arm": self.arm,
            "description": self.description,
            "source_sha": self.source_sha,
            "policy_revision": self.policy_revision,
            "policy_config": json.loads(self.policy_config_json),
            "platform": self.platform,
            "platform_config": json.loads(self.platform_config_json),
            "game_variant": self.game_variant,
            "opponent_cohort": [o.to_dict() for o in self.opponent_cohort],
            "seed_policy": {
                "base_seed": self.seed_policy.base_seed,
                "common_random_numbers": self.seed_policy.common_random_numbers,
            },
            "stopping": {
                "min_matches": self.stopping.min_matches,
                "max_matches": self.stopping.max_matches,
                "min_hands": self.stopping.min_hands,
                "max_hands": self.stopping.max_hands,
                "hands_per_match": self.stopping.hands_per_match,
                "matches_per_shard": self.stopping.matches_per_shard,
                "evaluation_interval_matches": (
                    self.stopping.evaluation_interval_matches
                ),
                "early_stopping": self.stopping.early_stopping,
            },
            "metrics": list(self.metrics),
            "provenance": self.provenance.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: dict) -> ExperimentSpec:
        exact(
            value,
            {
                "schema_version",
                "experiment_id",
                "arm",
                "description",
                "source_sha",
                "policy_revision",
                "policy_config",
                "platform",
                "platform_config",
                "game_variant",
                "opponent_cohort",
                "seed_policy",
                "stopping",
                "metrics",
                "provenance",
                "created_at",
            },
        )
        seeds = exact(value["seed_policy"], {"base_seed", "common_random_numbers"})
        stopping = exact(
            value["stopping"],
            {
                "min_matches",
                "max_matches",
                "min_hands",
                "max_hands",
                "hands_per_match",
                "matches_per_shard",
                "evaluation_interval_matches",
                "early_stopping",
            },
        )
        if type(value["opponent_cohort"]) is not list:
            raise ValueError("invalid cohort")
        if type(value["metrics"]) is not list:
            raise ValueError("invalid metrics")
        return cls.create(
            schema_version=value["schema_version"],
            experiment_id=value["experiment_id"],
            arm=value["arm"],
            description=value["description"],
            source_sha=value["source_sha"],
            policy_revision=value["policy_revision"],
            policy_config=value["policy_config"],
            platform=value["platform"],
            platform_config=value["platform_config"],
            game_variant=value["game_variant"],
            opponent_cohort=tuple(
                OpponentRef.from_dict(o) for o in value["opponent_cohort"]
            ),
            seed_policy=SeedPolicy(**seeds),
            stopping=StoppingRule(**stopping),
            metrics=tuple(value["metrics"]),
            provenance=Provenance(value["provenance"]),
            created_at=value["created_at"],
        )

    def derive_arm(self, arm: str, created_at: str, **changes) -> ExperimentSpec:
        """A new arm with changed configuration; the original is untouched."""
        if arm == self.arm:
            raise ValueError("a configuration change needs a new arm name")
        if "policy_config" in changes:
            changes["policy_config_json"] = canonical(
                changes.pop("policy_config")
            ).decode("ascii")
        if "platform_config" in changes:
            changes["platform_config_json"] = canonical(
                changes.pop("platform_config")
            ).decode("ascii")
        if "experiment_id" in changes:
            raise ValueError("arms belong to one experiment")
        return replace(self, arm=arm, created_at=created_at, **changes)

    def opponent_for(self, match_index: int) -> OpponentRef:
        return self.opponent_cohort[match_index % len(self.opponent_cohort)]


def dump(spec: ExperimentSpec) -> bytes:
    """Spec file bytes: the spec plus its hash, verified again on load."""
    return (
        json.dumps(
            {"spec": spec.to_dict(), "spec_hash": spec.spec_hash},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def load(path: Path) -> ExperimentSpec:
    value = json.loads(Path(path).read_text(encoding="ascii"))
    exact(value, {"spec", "spec_hash"})
    spec = ExperimentSpec.from_dict(value["spec"])
    if spec.spec_hash != value["spec_hash"]:
        raise ValueError(f"spec hash mismatch in {path}")
    return spec
