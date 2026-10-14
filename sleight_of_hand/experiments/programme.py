"""Versioned, reviewed authority envelope; never discover specs by globbing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .campaign import Campaign, ReviewRequired, streams
from .campaign import load as load_campaign
from .model import Provenance, assert_no_secrets, canonical, exact, name, sha256
from .spec import ExperimentSpec
from .spec import load as load_spec

LOCAL = frozenset(
    {
        Provenance.LOCAL_SELFPLAY,
        Provenance.SYNTHETIC,
        Provenance.SCRIPTED_PROBE,
        Provenance.BENCHMARK,
    }
)
ONLINE = frozenset({Provenance.LIVE_UNRATED, Provenance.SCRIPTED_PROBE})
GOVERNANCE = frozenset({"LOCAL_APPROVED", "UNRATED_APPROVED", "REQUIRES_REVIEW"})
AUTHORITY_VERSION = 1


@dataclass(frozen=True)
class Item:
    experiment_id: str
    governance: str
    reason: str
    specs: tuple[ExperimentSpec, ...]
    depends_on: tuple[str, ...]


@dataclass(frozen=True)
class Programme:
    items: tuple[Item, ...]
    workers: int
    min_free_mib: float
    digest: str
    campaigns: tuple[Campaign, ...] = ()
    #: (campaign ID, digest of the programme truncated after that campaign).
    prefix_digests: tuple[tuple[str, str], ...] = ()

    def item(self, experiment_id: str) -> Item:
        for item in self.items:
            if item.experiment_id == experiment_id:
                return item
        raise ValueError("experiment is not in the approved programme")

    def campaign(self, identifier: str) -> Campaign:
        for campaign in self.campaigns:
            if campaign.identifier == identifier:
                return campaign
        raise ValueError("campaign is not in the approved programme")

    def prefix_digest(self, identifier: str) -> str | None:
        return dict(self.prefix_digests).get(identifier)

    def authority(self, identifier: str) -> str:
        """Per-campaign authority: its own hash plus its dependencies' identities.

        Unrelated reviewed programme entries do not participate, so appending
        reviewed work cannot invalidate an unchanged campaign's history.
        """
        campaign = self.campaign(identifier)
        value = {
            "authority_version": AUTHORITY_VERSION,
            "campaign_hash": campaign.digest,
            "dependencies": [self._identity(d) for d in campaign.dependencies],
        }
        return "sha256:" + sha256(canonical(value))

    def _identity(self, condition: dict) -> dict:
        identifier = condition["id"]
        if any(c.identifier == identifier for c in self.campaigns):
            return {**condition, "campaign_authority": self.authority(identifier)}
        item = self.item(identifier)
        return {
            **condition,
            "experiment": {
                "governance": item.governance,
                "specs": sorted(s.spec_hash for s in item.specs),
                "depends_on": [
                    self._identity(
                        {"id": d, "status": "COMPLETE", "classification": None}
                    )
                    for d in item.depends_on
                ],
            },
        }


def _unique_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate programme field")
        value[key] = item
    return value


def load(path: Path) -> Programme:
    path = Path(path).resolve()
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_fields
    )
    version = value.get("schema_version")
    exact(
        value,
        {"schema_version", "resources", "experiments"}
        | ({"campaigns"} if version == 2 else set()),
    )
    assert_no_secrets(value, "research programme")
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("unsupported programme schema")
    resources = exact(value["resources"], {"workers", "min_free_mib"})
    workers, free = resources["workers"], resources["min_free_mib"]
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("workers must be 1..32")
    if type(free) not in (int, float) or not 0 <= free <= 10**9:
        raise ValueError("invalid min_free_mib")
    if type(value["experiments"]) is not list:
        raise ValueError("experiments must be a list")
    items, seen, paths = [], set(), set()
    for entry in value["experiments"]:
        exact(entry, {"id", "governance", "reason", "specs", "depends_on"})
        identifier = name(entry["id"], "experiment ID")
        if identifier in seen:
            raise ValueError("duplicate experiment ID")
        if (
            type(entry["governance"]) is not str
            or entry["governance"] not in GOVERNANCE
        ):
            raise ValueError("unknown governance classification")
        if type(entry["reason"]) is not str or not entry["reason"].strip():
            raise ValueError("governance needs a reason")
        deps = entry["depends_on"]
        if type(deps) is not list or any(type(d) is not str for d in deps):
            raise ValueError("depends_on must list experiment IDs")
        if len(set(deps)) != len(deps) or not set(deps) <= seen:
            raise ValueError("dependencies must be unique and precede the experiment")
        if type(entry["specs"]) is not list or not entry["specs"]:
            raise ValueError("each experiment needs an explicit spec set")
        specs, arms = [], set()
        for reference in entry["specs"]:
            exact(reference, {"path", "spec_hash"})
            if type(reference["path"]) is not str:
                raise ValueError("spec path must be a relative string")
            relative = Path(reference["path"])
            source = (path.parent / relative).resolve()
            if relative.is_absolute() or not source.is_relative_to(path.parent):
                raise ValueError("spec path must stay inside the programme directory")
            if source in paths:
                raise ValueError("duplicate spec path")
            paths.add(source)
            spec = load_spec(source)
            if spec.spec_hash != reference["spec_hash"]:
                raise ValueError("spec differs from the programme's approved hash")
            if spec.experiment_id != identifier or spec.arm in arms:
                raise ValueError("ambiguous experiment/arm assignment")
            arms.add(spec.arm)
            if spec.platform not in {"local", "chipzen"}:
                raise ValueError("unsupported research platform")
            allowed = LOCAL if spec.platform == "local" else ONLINE
            if spec.provenance not in allowed:
                raise ValueError(
                    "provenance is outside the supervisor authority envelope"
                )
            expected = (
                "LOCAL_APPROVED" if spec.platform == "local" else "UNRATED_APPROVED"
            )
            if entry["governance"] not in {expected, "REQUIRES_REVIEW"}:
                raise ValueError("governance does not match the platform")
            specs.append(spec)
        if len({(s.platform, s.provenance) for s in specs}) != 1:
            raise ValueError("an experiment must use one platform/provenance stratum")
        items.append(
            Item(
                identifier,
                entry["governance"],
                entry["reason"],
                tuple(specs),
                tuple(deps),
            )
        )
        seen.add(identifier)
    campaigns, prefixes = [], []
    if version == 2:
        if type(value["campaigns"]) is not list:
            raise ValueError("campaigns must be an explicit list")
        for position, ref in enumerate(value["campaigns"]):
            exact(ref, {"path", "campaign_hash"})
            relative = Path(ref["path"])
            source = (path.parent / relative).resolve()
            if (
                relative.is_absolute()
                or not source.is_relative_to(path.parent)
                or source in paths
            ):
                raise ValueError("invalid campaign path")
            paths.add(source)
            campaign = load_campaign(source)
            if campaign.digest != ref["campaign_hash"]:
                raise ReviewRequired("campaign differs from programme hash")
            c = campaign.config
            if campaign.identifier in seen:
                raise ValueError("duplicate campaign ID")
            if (
                c["resources"]["workers"] > workers
                or c["resources"]["min_free_mib"] < free
            ):
                raise ValueError("campaign resources exceed programme authority")
            seen.add(campaign.identifier)
            campaigns.append(campaign)
            prefix = {**value, "campaigns": value["campaigns"][: position + 1]}
            prefixes.append(
                (campaign.identifier, "sha256:" + sha256(canonical(prefix)))
            )
        _check_dependencies(items, campaigns)
        _check_streams(items, campaigns)
    return Programme(
        tuple(items),
        workers,
        float(free),
        "sha256:" + sha256(canonical(value)),
        tuple(campaigns),
        tuple(prefixes),
    )


def _check_dependencies(items, campaigns):
    """Campaigns may depend on fixed experiments or strictly earlier campaigns.

    Requiring earlier declaration makes cycles impossible; a forward reference
    (including self-reference) is reported as a cycle risk, not reordered.
    """
    experiments = {i.experiment_id for i in items}
    order = [c.identifier for c in campaigns]
    for position, campaign in enumerate(campaigns):
        for condition in campaign.dependencies:
            identifier = condition["id"]
            if identifier in order[position:]:
                raise ValueError(
                    "campaign dependency cycle or forward reference: "
                    f"{campaign.identifier} -> {identifier}"
                )
            if identifier not in experiments and identifier not in order:
                raise ValueError(f"unknown campaign dependency: {identifier}")
            if identifier in experiments and condition["classification"] is not None:
                raise ValueError(
                    "impossible dependency requirement: fixed experiments have no "
                    "campaign classification"
                )


def _check_streams(items, campaigns):
    """Every campaign stream is fresh: disjoint from fixed and earlier campaigns."""
    from .runner import match_seeds

    claimed = {
        seed
        for item in items
        for spec in item.specs
        for match in range(spec.stopping.max_matches)
        for seed in match_seeds(spec, match).values()
    }
    for campaign in campaigns:
        declared = streams(campaign)
        if declared & claimed:
            raise ValueError(
                f"{campaign.identifier} seed streams overlap reviewed evidence"
            )
        claimed |= declared
