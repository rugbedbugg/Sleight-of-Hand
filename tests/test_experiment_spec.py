"""Immutable specs, deterministic hashing, identity and secret firewalls."""

import dataclasses
import json

import pytest

from sleight_of_hand.experiments.model import (
    OpponentRef,
    Provenance,
    Redactor,
    assert_no_secrets,
)
from sleight_of_hand.experiments.spec import (
    ExperimentSpec,
    SeedPolicy,
    StoppingRule,
    dump,
    load,
)

LOCAL = {
    "small_blind": 50,
    "big_blind": 100,
    "starting_stack": 10000,
    "stack_mode": "reset",
    "pot_convention": "committed",
    "to_call_convention": "owed",
    "cap_bets_to_effective": False,
    "accounting_observer": False,
}


def make(**changes) -> ExperimentSpec:
    values = {
        "experiment_id": "E9999",
        "arm": "canonical",
        "description": "unit test spec",
        "source_sha": "4a8603e5930d25b496dca296e8e3e5b41355dcde",
        "policy_revision": "sha256:" + "ab" * 32,
        "policy_config": {"opponent_memory": True},
        "platform": "local",
        "platform_config": LOCAL,
        "game_variant": "nlhe_hu",
        "opponent_cohort": (
            OpponentRef("local", "script", "calling_station"),
            OpponentRef("local", "script", "tag_simple"),
        ),
        "seed_policy": SeedPolicy(11),
        "stopping": StoppingRule(
            min_matches=2,
            max_matches=4,
            min_hands=20,
            max_hands=40,
            hands_per_match=10,
            matches_per_shard=3,
            evaluation_interval_matches=2,
        ),
        "metrics": ("behavior", "model", "outcome"),
        "provenance": Provenance.LOCAL_SELFPLAY,
        "created_at": "2026-10-08T00:00:00Z",
    }
    values.update(changes)
    return ExperimentSpec.create(**values)


def test_spec_is_frozen_and_config_copies_are_detached():
    spec = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.arm = "other"
    config = spec.policy_config
    config["opponent_memory"] = False
    assert spec.policy_config == {"opponent_memory": True}


def test_hash_is_deterministic_and_pinned():
    first, second = make(), make()
    assert first.spec_hash == second.spec_hash
    # Pinned: a change here means the canonical encoding changed.
    assert first.spec_hash == (
        "26e42cb33f48c986c2c30c6c975e077933df6aefdccaddd395a8fc9c990c6e27"
    )
    assert make(arm="other").spec_hash != first.spec_hash
    assert first.policy_config_hash == make(arm="x").policy_config_hash


def test_round_trip_and_tamper_detection(tmp_path):
    spec = make()
    path = tmp_path / "spec.json"
    path.write_bytes(dump(spec))
    assert load(path) == spec and load(path).spec_hash == spec.spec_hash
    value = json.loads(path.read_text())
    value["spec"]["stopping"]["max_matches"] = 3
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash mismatch"):
        load(path)


def test_unknown_fields_are_rejected():
    value = make().to_dict()
    value["surprise"] = 1
    with pytest.raises(ValueError, match="unknown"):
        ExperimentSpec.from_dict(value)
    with pytest.raises(ValueError):
        make(policy_config={"opponent_memory": True, "pricing_fix": True})
    with pytest.raises(ValueError):
        make(policy_config={"params": {"not_a_param": 1.0}})


def test_configuration_change_requires_a_new_arm():
    spec = make()
    derived = spec.derive_arm(
        "memory-off", "2026-10-08T01:00:00Z", policy_config={"opponent_memory": False}
    )
    assert derived.arm == "memory-off" and derived.spec_hash != spec.spec_hash
    assert spec.policy_config == {"opponent_memory": True}  # original untouched
    with pytest.raises(ValueError, match="new arm"):
        spec.derive_arm("canonical", "2026-10-08T01:00:00Z")
    with pytest.raises(ValueError, match="one experiment"):
        spec.derive_arm("x", "2026-10-08T01:00:00Z", experiment_id="E1")


def test_stopping_rule_is_fixed_horizon_and_validated():
    rule = make().stopping
    assert rule.shards() == [(0, 3), (3, 4)]
    assert rule.early_stopping == "none"
    with pytest.raises(ValueError, match="exceed"):
        StoppingRule(1, 5, 1, 10, 10, 1, 1)
    with pytest.raises(ValueError, match="fixed-horizon"):
        StoppingRule(1, 1, 1, 10, 10, 1, 1, early_stopping="sequential")
    with pytest.raises(ValueError):
        StoppingRule(5, 4, 1, 100, 10, 1, 1)  # min above max


def test_seed_derivation_is_deterministic_and_role_specific():
    seeds = SeedPolicy(11)
    assert seeds.derive("deck", 3) == SeedPolicy(11).derive("deck", 3)
    assert seeds.derive("deck", 3) != seeds.derive("soh", 3)
    assert seeds.derive("deck", 3) != SeedPolicy(12).derive("deck", 3)


def test_provenance_is_preserved_and_required():
    spec = make(provenance=Provenance.SYNTHETIC)
    assert ExperimentSpec.from_dict(spec.to_dict()).provenance is Provenance.SYNTHETIC
    value = spec.to_dict()
    value["provenance"] = "MIXED"
    with pytest.raises(ValueError):
        ExperimentSpec.from_dict(value)


def test_identities_never_cross_platforms():
    chipzen = OpponentRef("chipzen", "bot_uuid", "X")
    other = OpponentRef("other", "player_id", "X")
    assert chipzen != other and len({chipzen, other}) == 2
    named = OpponentRef("chipzen", "bot_uuid", "X", display_name="Same Name")
    assert named == chipzen  # display names are metadata only
    assert OpponentRef("chipzen", "bot_uuid", "Y", "Same Name") != named
    with pytest.raises(ValueError, match="cohort"):
        make(opponent_cohort=(OpponentRef("chipzen", "house_bot", "h1"),))


@pytest.mark.parametrize(
    "secret",
    [
        {"description": "use cz_extbot_abcdef123456 please"},
        {"platform_config": {**LOCAL, "api_token": "x"}},
        {"description": "wss://host/ws?token=abcdefgh"},
    ],
)
def test_secrets_are_rejected_not_serialized(secret):
    with pytest.raises(ValueError):
        make(**secret)


def test_environment_secrets_are_redacted(monkeypatch):
    monkeypatch.setenv("CHIPZEN_RESEARCH_TOKEN", "research-secret-value")
    redact = Redactor()
    scrubbed = redact(
        {
            "note": "echo research-secret-value back",
            "headers": {"Authorization": "abc"},
            "url": "wss://x/y?ticket=zzzzzzz&a=1",
        }
    )
    text = json.dumps(scrubbed)
    assert "research-secret-value" not in text and "zzzzzzz" not in text
    assert scrubbed["headers"]["Authorization"] == "[REDACTED]"
    with pytest.raises(ValueError):
        assert_no_secrets({"note": "research-secret-value"})
