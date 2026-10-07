"""Local-only historical priors for synthetic validation of OpponentMemory.

A local scripted opponent is the same program every match, so its identity
is persistent *by construction*; the evidence string says exactly that. The
identity lives in the ``local`` provider namespace and can never collide
with, or be linked to, a Chipzen (or any other platform) identity. Priors
only reach the policy-inert memory; no policy code reads them.
"""

from __future__ import annotations

from pathlib import Path

from sleight_of_hand.holdem.memory import OpponentMemory
from sleight_of_hand.holdem.profile_store import BundledProfileStore
from sleight_of_hand.holdem.profiles import IdentityResolution, OpponentIdentity

from ..model import sha256

LOCAL_EVIDENCE = "local_policy_registry"


def local_identity(opponent_key: str) -> OpponentIdentity:
    return OpponentIdentity(
        provider="local",
        key=opponent_key,
        key_kind="external_bot_id",
        scope="persistent",
        persistence_evidence=LOCAL_EVIDENCE,
    )


def local_resolver(opponent_key: str):
    identity = local_identity(opponent_key)

    def resolve(provider, match_key, seats) -> IdentityResolution:
        # The local dealer, not the transport, vouches for this identity.
        return IdentityResolution(identity, "local_registry", 1)

    return resolve


def memory_for(policy_config: dict) -> OpponentMemory | None:
    """``None`` keeps the adapter's default fresh, match-scoped memory."""
    prior = policy_config.get("historical_prior")
    if not prior:
        return None
    if set(prior) != {"bundle_path", "bundle_sha256", "opponent_key"}:
        raise ValueError(
            "historical_prior needs bundle_path, bundle_sha256, opponent_key"
        )
    data = Path(prior["bundle_path"]).read_bytes()
    if "sha256:" + sha256(data) != prior["bundle_sha256"]:
        raise ValueError("prior bundle does not match its recorded hash")
    store = BundledProfileStore(data)
    if not store.valid:
        raise ValueError("prior bundle failed validation")
    return OpponentMemory(
        store=store, identity_resolver=local_resolver(prior["opponent_key"])
    )


def observation_dataset(run_dirs, opponent: str, data_version: str) -> dict:
    """Prior-compiler input from completed local runs against ``opponent``.

    ``opponent`` is the local ``kind.key`` (e.g. ``script.tag_simple``). Each
    hand is parsed with the runtime's own ``preflop_observations`` from SOH's
    point of view, tagged with the run's raw-journal checksum as its source.
    """
    import json

    from sleight_of_hand.holdem.observations import (
        NormalizedObservation,
        preflop_observations,
    )

    from ..journal import read_raw

    kind, key = opponent.split(".", 1)
    qualified = f"local/{kind}/{key}"
    identity = local_identity(opponent)
    rows = []
    for run_dir in sorted(Path(d) for d in run_dirs):
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest["status"] != "COMPLETED":
            continue
        source = manifest["checksums"]["raw_sha256"]
        matches: dict[int, list[dict]] = {}
        for event in read_raw(run_dir):
            matches.setdefault(event.get("match"), []).append(event)
        for match, events in sorted(matches.items()):
            meta = next((e for e in events if e.get("kind") == "match_meta"), None)
            if meta is None or meta.get("opponent") != qualified:
                continue
            hero, starts = meta["soh_seat"], {}
            for event in events:
                message = event.get("message") or {}
                if event.get("kind") != "message" or hero not in event["to"]:
                    continue
                if message.get("type") == "round_start":
                    starts[message["round_id"]] = message["state"]
                elif message.get("type") == "round_result":
                    rid = message["round_id"]
                    stats = preflop_observations(
                        starts.get(rid, {}), message["result"], rid, hero
                    )
                    if stats:
                        rows.append(
                            NormalizedObservation(
                                identity,
                                f"{manifest['run_id']}.m{match}"[-160:],
                                rid,
                                manifest["run_id"][-160:],
                                source,
                                "synthetic",
                                stats,
                            ).to_dict()
                        )
    return {"schema_version": 1, "data_version": data_version, "observations": rows}
