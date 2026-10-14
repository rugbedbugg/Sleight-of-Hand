"""Reviewed, local-only authority for bounded parameter proposals."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .model import Provenance, assert_no_secrets, canonical, exact, name, sha256
from .spec import (
    PARAM_NAMES,
    ExperimentSpec,
    SeedPolicy,
    StoppingRule,
)
from .spec import (
    load as load_spec,
)

VERSION = 1
ALGORITHM = {"kind": "bounded-pattern-search", "version": VERSION}
OBJECTIVE = "paired-bb-per-100-bonferroni-all-pairs"
SCREENING = "resolution-v-screening"
#: Non-binding classifications of a campaign that reached COMPLETE.
COMPLETIONS = frozenset(
    {"NO_IMPROVEMENT_DETECTED", "PROMISING", "INCONCLUSIVE", "REJECTED"}
)


class ReviewRequired(ValueError):
    """Reviewed authority no longer matches its declared identity."""


def unique_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate campaign field or parameter")
        value[key] = item
    return value


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("campaign numbers must be finite")
    return Decimal(str(value))


@dataclass(frozen=True)
class Campaign:
    config_json: str
    template: ExperimentSpec

    @property
    def config(self):
        return json.loads(self.config_json)

    @property
    def digest(self):
        return "sha256:" + sha256(self.config_json.encode("ascii"))

    @property
    def identifier(self):
        return self.config["id"]

    @property
    def kind(self):
        return self.config["algorithm"]["kind"]

    @property
    def initial(self):
        """The canonical starting vector every batch is compared against."""
        c = self.config
        if self.kind == SCREENING:
            return {f["parameter"]: f["center"] for f in c["factors"]}
        return c["initial"]

    @property
    def dependencies(self):
        """Normalized ``{id, status, classification}`` dependency conditions."""
        deps = self.config["depends_on"]
        if self.kind == SCREENING:
            return tuple(deps)
        return tuple(
            {"id": d, "status": "COMPLETE", "classification": None} for d in deps
        )


def in_domain(key, value) -> bool:
    """The existing approved PolicyParams domain."""
    if key == "steepness":
        return Decimal("0.5") <= value <= 30
    return 0 <= value <= 1


def validate_vector(campaign, vector):
    if campaign.kind == SCREENING:
        from . import screening

        return screening.validate_vector(campaign, vector)
    exact(vector, set(PARAM_NAMES))
    config = campaign.config
    bounds = {p["parameter"]: p for p in config["parameters"]}
    for key, value in vector.items():
        v = number(value)
        if key in bounds:
            p = bounds[key]
            if not number(p["min"]) <= v <= number(p["max"]):
                raise ValueError("candidate outside approved bounds")
        elif v != number(config["initial"][key]):
            raise ValueError("candidate changed an unauthorized parameter")


def load(path: Path) -> Campaign:
    """Dispatch on the reviewed algorithm; each kind has a strict schema."""
    path = Path(path).resolve()
    value = json.loads(path.read_text(), object_pairs_hook=unique_fields)
    algorithm = value.get("algorithm") if type(value) is dict else None
    if type(algorithm) is dict and algorithm.get("kind") == SCREENING:
        from . import screening

        return screening.load(path, value)
    return _load_pattern(path, value)


def common(path, value, schema_version):
    """Checks shared by every campaign kind; returns the pinned template."""
    assert_no_secrets(value, "campaign")
    if type(value["schema_version"]) is not int or value["schema_version"] != (
        schema_version
    ):
        raise ValueError("unsupported campaign schema")
    name(value["id"], "campaign ID")
    if len(value["id"]) > 64:
        raise ValueError("campaign ID too long")
    if value["governance"] not in {"LOCAL_APPROVED", "REQUIRES_REVIEW"}:
        raise ValueError("optimizer is local only")
    if not isinstance(value["purpose"], str) or not value["purpose"].strip():
        raise ValueError("campaign purpose required")
    SeedPolicy(value["campaign_seed"])
    resources = exact(value["resources"], {"workers", "min_free_mib"})
    if type(resources["workers"]) is not int or not 1 <= resources["workers"] <= 2:
        raise ValueError("optimizer workers must be 1..2")
    if not 0 <= number(resources["min_free_mib"]) <= 10**9:
        raise ValueError("invalid campaign memory floor")
    fixed = exact(
        value["fixed_policy"], {"samples", "opponent_memory", "historical_prior"}
    )
    if (
        type(fixed["samples"]) is not int
        or fixed["samples"] != 128
        or fixed["opponent_memory"] is not True
        or fixed["historical_prior"] is not None
    ):
        raise ValueError("optimizer fixed policy must remain canonical")
    reference = exact(value["template"], {"path", "spec_hash"})
    relative = Path(reference["path"])
    # Campaigns and their templates share the reviewed experiments directory.
    parent = path.parent
    source = (parent / relative).resolve()
    if relative.is_absolute() or not source.is_relative_to(parent):
        raise ValueError("campaign template must stay inside its directory")
    template = load_spec(source)
    if template.spec_hash != reference["spec_hash"]:
        raise ReviewRequired("campaign template hash mismatch")
    if template.platform != "local" or template.provenance not in {
        Provenance.LOCAL_SELFPLAY,
        Provenance.SYNTHETIC,
        Provenance.BENCHMARK,
    }:
        raise ValueError("optimizer requires local outcome evidence")
    if (
        "outcome" not in template.metrics
        or template.policy_revision != value["policy_revision"]
    ):
        raise ValueError("campaign template policy/metrics mismatch")
    if template.platform_config.get("stack_mode") != "reset":
        raise ValueError("optimizer requires the complete fixed hand horizon")
    # Validate the declared immutable timestamp using the normal spec contract.
    check = template.to_dict()
    check["created_at"] = value["created_at"]
    ExperimentSpec.from_dict(check)
    return template


def fixed_horizon(value, template):
    rule = StoppingRule(**exact(value, set(template.to_dict()["stopping"])))
    if (
        rule.min_matches < 2
        or rule.min_matches != rule.max_matches
        or rule.min_hands != rule.max_hands
        or rule.max_hands != rule.max_matches * rule.hands_per_match
    ):
        raise ValueError("optimizer requires an exact fixed horizon")
    return rule


def _load_pattern(path: Path, value) -> Campaign:
    exact(
        value,
        {
            "schema_version",
            "id",
            "governance",
            "purpose",
            "campaign_seed",
            "template",
            "policy_revision",
            "initial",
            "parameters",
            "algorithm",
            "search_stopping",
            "confirmation_stopping",
            "max_generations",
            "max_generated_vectors",
            "objective",
            "resources",
            "depends_on",
            "created_at",
            "fixed_policy",
            "seed_derivation",
        },
    )
    template = common(path, value, VERSION)
    exact(value["algorithm"], {"kind", "version"})
    if (
        type(value["algorithm"]["version"]) is not int
        or value["algorithm"] != ALGORITHM
    ):
        raise ValueError("unsupported optimizer algorithm")
    if (
        value["objective"] != OBJECTIVE
        or value["seed_derivation"] != "sha256-domain-separated-v1"
    ):
        raise ValueError("unsupported objective or seed derivation")
    for key, ceiling in (("max_generations", 4), ("max_generated_vectors", 45)):
        if type(value[key]) is not int or not 1 <= value[key] <= ceiling:
            raise ValueError("invalid campaign budget")
    deps = value["depends_on"]
    if (
        type(deps) is not list
        or any(type(d) is not str for d in deps)
        or len(deps) != len(set(deps))
    ):
        raise ValueError("invalid campaign dependencies")
    for key in ("search_stopping", "confirmation_stopping"):
        fixed_horizon(value[key], template)
    exact(value["initial"], set(PARAM_NAMES))
    for key, v in value["initial"].items():
        if not in_domain(key, number(v)):
            raise ValueError("initial policy vector outside policy domain")
    parameters = value["parameters"]
    if type(parameters) is not list or not parameters:
        raise ValueError("explicit tunable parameters required")
    seen = set()
    for p in parameters:
        exact(p, {"parameter", "min", "max", "initial_step", "min_step"})
        key = p["parameter"]
        if key not in PARAM_NAMES or key in seen:
            raise ValueError("unsupported or duplicate policy parameter")
        seen.add(key)
        lo, hi, step, minimum = (
            number(p[k]) for k in ("min", "max", "initial_step", "min_step")
        )
        domain = (
            (Decimal("0.5"), Decimal(30))
            if key == "steepness"
            else (Decimal(0), Decimal(1))
        )
        if not domain[0] <= lo < hi <= domain[1] or not 0 < minimum <= step <= hi - lo:
            raise ValueError("invalid bounds or steps")
    result = Campaign(canonical(value).decode("ascii"), template)
    validate_vector(result, value["initial"])
    seed_blocks(result)  # reject collisions before any execution
    return result


ROLES = ("soh", "opponent", "deck")


def horizon(campaign, phase):
    """The reviewed fixed StoppingRule mapping for a batch phase."""
    c = campaign.config
    if phase == "confirmation":
        return c["confirmation_stopping"]
    return c["screening_stopping" if campaign.kind == SCREENING else "search_stopping"]


def streams(campaign):
    """Every stream seed any batch of this campaign may ever consume."""
    return {
        SeedPolicy(seed).derive(role, match)
        for key, seed in seed_blocks(campaign).items()
        for match in range(horizon(campaign, key.rsplit("-", 1)[0])["max_matches"])
        for role in ROLES
    }


def seed_blocks(campaign):
    """Predeclare all blocks and verify actual control-plane stream disjointness."""
    if campaign.kind == SCREENING:
        from . import screening

        return screening.seed_blocks(campaign)
    c = campaign.config
    result = {}
    # Historical template evidence cannot become held-out or search evidence.
    streams = {
        campaign.template.seed_policy.derive(role, match)
        for match in range(campaign.template.stopping.max_matches)
        for role in ("soh", "opponent", "deck")
    }
    for phase, generation in [("search", n) for n in range(c["max_generations"])] + [
        ("confirmation", 0)
    ]:
        material = canonical([c["id"], c["campaign_seed"], generation, phase])
        seed = int(sha256(material)[:16], 16) & ((1 << 63) - 1)
        key = f"{phase}-{generation}"
        if seed == campaign.template.seed_policy.base_seed or seed in result.values():
            raise ValueError("seed block overlap")
        result[key] = seed
        horizon = c["search_stopping" if phase == "search" else "confirmation_stopping"]
        for match in range(horizon["max_matches"]):
            for role in ("soh", "opponent", "deck"):
                stream = SeedPolicy(seed).derive(role, match)
                if stream in streams:
                    raise ValueError("search/confirmation stream overlap")
                streams.add(stream)
    return result


def neighbors(campaign, incumbent, steps):
    validate_vector(campaign, incumbent)
    result = {}
    for p in sorted(campaign.config["parameters"], key=lambda p: p["parameter"]):
        key = p["parameter"]
        step = number(steps[key])
        if not number(p["min_step"]) <= step <= number(p["initial_step"]):
            raise ValueError("step outside approved bounds")
        for sign in (-1, 1):
            v = number(incumbent[key]) + sign * step
            if number(p["min"]) <= v <= number(p["max"]):
                vector = {**incumbent, key: float(v)}
                validate_vector(campaign, vector)
                result[canonical(vector)] = vector
    return [result[k] for k in sorted(result)]


def shrink(campaign, steps):
    return {
        p["parameter"]: float(
            max(number(p["min_step"]), number(steps[p["parameter"]]) / 2)
        )
        for p in campaign.config["parameters"]
    }
