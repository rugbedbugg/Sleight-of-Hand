"""Local NLHE dealer, seats, canonical-policy compatibility and the probe."""

import random

import pytest

from sleight_of_hand.experiments import metrics
from sleight_of_hand.experiments.model import OpponentRef, Provenance
from sleight_of_hand.experiments.normalization import normalize
from sleight_of_hand.experiments.platforms.base import MatchContext
from sleight_of_hand.experiments.platforms.local import (
    Hand,
    LocalPlatform,
    check_config,
    full_deck,
)
from sleight_of_hand.experiments.spec import StoppingRule
from tests.test_experiment_spec import LOCAL, make


def deck(seed=0):
    cards = full_deck()
    random.Random(seed).shuffle(cards)
    return cards


def random_hand(seed, config=LOCAL, stacks=(10000, 10000)):
    rng = random.Random(seed)
    hand = Hand(1, seed % 2, list(stacks), deck(seed), config, "r")
    first = hand.button
    while True:
        seat = first
        while hand.folded is None:
            if not hand.needs_action(seat):
                if not hand.needs_action(1 - seat):
                    break
                seat = 1 - seat
                continue
            legal = hand.legal(seat)
            action = rng.choice(legal["valid_actions"])
            amount = (
                rng.randint(legal["min_raise"], legal["max_raise"])
                if action == "raise"
                else 0
            )
            assert hand.apply(seat, action, amount)["rejected"] is None
            seat = 1 - seat
        if hand.folded is not None or not hand.next_street():
            break
        first = 1 - hand.button
    return hand, hand.settle()


@pytest.mark.parametrize("seed", range(300))
def test_random_hands_conserve_chips_and_stay_legal(seed):
    stacks = (10000, random.Random(seed).choice([150, 2500, 10000, 30000]))
    hand, result = random_hand(seed, stacks=stacks)
    assert sum(result["stacks"]) == sum(stacks)
    assert all(s >= 0 for s in result["stacks"])
    assert sum(p["amount"] for p in result["payouts"]) == hand.total[0] + hand.total[1]


def test_min_raise_short_all_in_and_reopening():
    hand = Hand(1, 0, [10000, 10000], deck(), LOCAL, "r")
    legal = hand.legal(0)  # button facing the big blind
    assert legal["valid_actions"] == ["fold", "call", "raise"]
    assert (legal["to_call"], legal["min_raise"], legal["max_raise"]) == (
        50,
        200,
        10000,
    )
    hand.apply(0, "raise", 300)  # increment 200
    assert hand.legal(1)["min_raise"] == 500
    short = Hand(1, 0, [10000, 600], deck(), LOCAL, "r")
    short.apply(0, "raise", 500)
    short.apply(1, "raise", 600)  # all-in for less than a full raise
    legal = short.legal(0)
    assert "raise" not in legal["valid_actions"]  # opponent has nothing behind
    illegal = Hand(1, 0, [10000, 10000], deck(), LOCAL, "r")
    record = illegal.apply(0, "raise", 150)  # below the minimum
    assert record["rejected"] == {"action": "raise", "amount": 150}
    assert record["action"] == "fold"  # SDK fallback: check, else fold


@pytest.mark.parametrize(
    ("conventions", "pot", "to_call", "max_raise"),
    [
        ({}, 100 + 100 + 9900, 9900, 9900),
        ({"pot_convention": "contestable"}, 100 + 100 + 900, 9900, 9900),
        ({"to_call_convention": "capped"}, 10100, 900, 9900),
        ({"cap_bets_to_effective": True}, None, None, 900),
    ],
)
def test_covering_bet_conventions_are_explicit(conventions, pot, to_call, max_raise):
    config = {**LOCAL, **conventions}
    hand = Hand(1, 1, [10000, 1000], deck(), config, "r")  # seat 0 is big blind
    hand.apply(1, "call", 0)
    hand.apply(0, "check", 0)
    hand.next_street()  # flop: big blind (seat 0) acts first
    assert hand.legal(0)["max_raise"] == max_raise
    if pot is None:
        return
    hand.apply(0, "raise", 9900)  # covers seat 1's 900 behind
    assert hand.pot() == pot
    assert hand.legal(1)["to_call"] == to_call
    assert hand.unmatched() == 9900 - 900


def test_config_requires_every_convention():
    with pytest.raises(ValueError, match="exactly"):
        check_config({k: v for k, v in LOCAL.items() if k != "pot_convention"})


def play(spec, seeds=None, match_index=0, opponent=None):
    events = []
    summary = LocalPlatform().play_match(
        MatchContext(
            spec,
            match_index,
            opponent or spec.opponent_cohort[0],
            seeds or {"soh": 1, "opponent": 2, "deck": 3},
        ),
        lambda e: events.append({"match": match_index, **e}),
    )
    return summary, events


def short_spec(**changes):
    values = {
        "opponent_cohort": (OpponentRef("local", "script", "random_legal"),),
        "stopping": StoppingRule(1, 1, 1, 30, 30, 1, 1),
    }
    values.update(changes)
    return make(**values)


def test_matches_are_reproducible_from_seeds():
    spec = short_spec()
    assert play(spec)[1] == play(spec)[1]
    assert play(spec)[1] != play(spec, {"soh": 1, "opponent": 2, "deck": 4})[1]


@pytest.mark.parametrize("opponent", ["random_legal", "tag_simple", "calling_station"])
def test_canonical_policy_accepts_every_local_state(opponent):
    spec = short_spec(opponent_cohort=(OpponentRef("local", "script", opponent),))
    _, events = play(spec)
    outputs = [e for e in events if e["kind"] == "seat_output"]
    assert outputs == []  # no "Preflop context unavailable", no state errors
    assert not [e for e in events if e["kind"] == "rejected"]
    normalized = normalize(events)
    assert len(normalized["hands"]) == 30
    preflop = [d for d in normalized["decisions"] if d["phase"] == "preflop"]
    assert preflop and all(d["action"] in d["valid"] for d in normalized["decisions"])


def test_memory_arm_cannot_change_decisions_or_rng():
    spec = short_spec(opponent_cohort=(OpponentRef("local", "script", "tag_simple"),))
    off = spec.derive_arm(
        "memory-off", spec.created_at, policy_config={"opponent_memory": False}
    )
    on_events, off_events = play(spec)[1], play(off)[1]
    strip = {"match_meta"}
    assert [e for e in on_events if e["kind"] not in strip] == [
        e for e in off_events if e["kind"] not in strip
    ]


def test_self_play_runs_two_independent_soh_instances():
    spec = short_spec(opponent_cohort=(OpponentRef("local", "soh", "canonical"),))
    summary, events = play(spec)
    assert summary.hands == 30
    seats = {e["seat"] for e in events if e["kind"] == "action"}
    assert seats == {0, 1}


def probe_spec(**conventions):
    return make(
        platform_config={
            **LOCAL,
            "stack_mode": "carry",
            "accounting_observer": True,
            **conventions,
        },
        opponent_cohort=(OpponentRef("local", "probe", "accounting_cover"),),
        provenance=Provenance.SCRIPTED_PROBE,
        metrics=("accounting",),
        stopping=StoppingRule(1, 1, 1, 400, 400, 1, 1),
    )


@pytest.mark.parametrize(
    ("conventions", "expected"),
    [
        ({}, "CONFIRMED_EXCESS_INCLUDED"),
        ({"pot_convention": "contestable"}, "RECONCILED_BEFORE_DECISION"),
        ({"cap_bets_to_effective": True}, "SERVER_EFFECTIVE_STACK_CAP"),
    ],
)
def test_probe_pipeline_classifies_each_dealer_convention(conventions, expected):
    """Validates the probe + observer + classifier pipeline, not Chipzen."""
    _, events = play(
        probe_spec(**conventions), {"soh": 2, "opponent": 102, "deck": 202}
    )
    summary = metrics.accounting(events)
    assert summary["primary"] == expected and not summary["conflicting"]
    result = summary["per_match"]["0"]
    assert result["classification"] == expected
    assert result["checksum"].startswith("VALID")
    assert result["derived"]["P_before_source"] == "preceding turn_result pot"
    frames = "".join(e.get("stdout", "") for e in events if e["kind"] == "seat_output")
    assert frames.count("SHA256") == 1  # exactly one capture, then disarmed
    for leak in ("your_hole_cards", "participant_id", "round_id"):
        assert leak not in result["raw_capture"]


def test_probe_never_trips_the_observer_before_its_target():
    for seed in (1, 3, 4):
        _, events = play(
            probe_spec(), {"soh": seed, "opponent": seed + 100, "deck": seed + 200}
        )
        result = metrics.classify_match(events)
        if result["classification"] == "INCONCLUSIVE":
            assert result["why"] == "no qualifying capture"
        else:
            assert result["classification"] == "CONFIRMED_EXCESS_INCLUDED"
