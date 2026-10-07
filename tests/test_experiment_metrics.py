"""Normalization, metrics, accounting reassembly and analysis verdicts."""

import hashlib
import json

import pytest

from sleight_of_hand.experiments import accounting, analysis, metrics
from sleight_of_hand.experiments.normalization import normalize, tendencies

P = "post_small_blind", "post_big_blind"


def entry(seat, action, amount=0, phase="preflop"):
    return {"seat": seat, "action": action, "amount": amount, "phase": phase}


BLINDS = [entry(0, P[0], 50), entry(1, P[1], 100)]


def test_tendencies_from_a_complete_history():
    history = BLINDS + [
        entry(0, "raise", 250),
        entry(1, "call", 150),
        entry(1, "check", 0, "flop"),
        entry(0, "raise", 300, "flop"),
        entry(1, "fold", 0, "flop"),
    ]
    found = {(s, k): v for s, k, v in tendencies(history, 0, [10000, 10000])}
    assert found[(0, "btn_rfi")] and not found[(0, "btn_limp")]
    assert found[(1, "bb_call_open")] and not found[(1, "bb_3bet")]
    assert found[(0, "cbet")] and found[(1, "fold_to_cbet")]
    assert not found[(0, "preflop_shove")]
    shove = BLINDS + [entry(0, "raise", 10000), entry(1, "fold")]
    found = {(s, k): v for s, k, v in tendencies(shove, 0, [10000, 10000])}
    assert found[(0, "preflop_shove")] and found[(1, "bb_fold_to_open")]
    limp = BLINDS + [entry(0, "call", 50), entry(1, "raise", 400), entry(0, "fold")]
    found = {(s, k): v for s, k, v in tendencies(limp, 0, [10000, 10000])}
    assert found[(0, "btn_limp")] and found[(1, "bb_iso")]


def test_donk_bet_is_not_a_cbet_opportunity():
    history = BLINDS + [
        entry(0, "raise", 250),
        entry(1, "call", 150),
        entry(1, "raise", 300, "flop"),
        entry(0, "fold", 0, "flop"),
    ]
    stats = {k for _, k, _ in tendencies(history, 0, [10000, 10000])}
    assert "cbet" not in stats and "fold_to_cbet" not in stats


def match_events(net_by_hand, match=0):
    events = [{"match": match, "kind": "match_meta", "soh_seat": 0}]
    for number, net in enumerate(net_by_hand, 1):
        rid = f"m{match}-r{number}"
        events.append(
            {
                "match": match,
                "kind": "message",
                "to": [0],
                "message": {
                    "type": "round_start",
                    "round_id": rid,
                    "state": {
                        "hand_number": number,
                        "dealer_seat": 0,
                        "stacks": [1000, 1000],
                    },
                },
            }
        )
        events.append(
            {
                "match": match,
                "kind": "message",
                "to": [0, 1],
                "message": {
                    "type": "round_result",
                    "round_id": rid,
                    "result": {
                        "hand_number": number,
                        "stacks": [1000 + net, 1000 - net],
                        "action_history": BLINDS + [entry(0, "fold")],
                        "pot": 150,
                    },
                },
            }
        )
    return events


def test_normalization_is_deterministic_and_attributes_soh():
    events = match_events([-50, 100]) + match_events([100], match=1)
    first, second = normalize(events), normalize(json.loads(json.dumps(events)))
    assert first == second
    assert [h["soh_net"] for h in first["hands"]] == [-50, 100, 100]
    assert all(h["big_blind"] == 100 for h in first["hands"])
    assert normalize([{"match": 0, "kind": "message", "to": [0], "message": {}}]) == {
        "hands": [],
        "decisions": [],
        "observations": [],
    }  # no match_meta: nothing is attributed


def test_outcome_uses_match_clustered_uncertainty():
    hands = normalize(match_events([100, 100]) + match_events([-100, -100], 1))["hands"]
    result = metrics.outcome(hands)
    assert result["hands"] == 4 and result["matches"] == 2
    assert result["bb_per_100"]["mean"] == 0
    # Two perfectly opposed clusters: the clustered SE is large.
    assert result["chips_per_hand"]["se"] == pytest.approx(100.0)
    assert result["matches_won"] == 1 and result["matches_lost"] == 1


def test_wilson_interval():
    assert metrics.wilson(0, 0) is None
    low, high = metrics.wilson(5, 10)
    assert low < 0.5 < high and low == pytest.approx(0.236593, abs=1e-5)


def test_model_replay_is_prequential():
    history = BLINDS + [entry(0, "fold")]  # SOH (seat 0) folds; opponent idle
    events = match_events([-50] * 3)
    for e in events:
        if e.get("message", {}).get("type") == "round_result":
            e["message"]["result"]["action_history"] = history
    # Make the opponent the button so its btn tendencies are observed.
    for e in events:
        if e.get("message", {}).get("type") == "round_start":
            e["message"]["state"]["dealer_seat"] = 1
    for e in events:
        if e.get("message", {}).get("type") == "round_result":
            e["message"]["result"]["action_history"] = [
                entry(1, P[0], 50),
                entry(0, P[1], 100),
                entry(1, "fold"),
            ]
    events.insert(
        1,
        {
            "match": 0,
            "kind": "message",
            "to": [0],
            "message": {
                "type": "match_start",
                "match_id": "m0",
                "seats": [
                    {"seat": 0, "is_self": True},
                    {"seat": 1, "participant_id": "p"},
                ],
            },
        },
    )
    result = metrics.model(events, {})
    fold = result["by_tendency"]["btn_fold"]
    assert fold["n"] == 3
    # Predictions 1/2, 2/3, 3/4 for three folds under Beta(1, 1).
    expected = sum((1 - p) ** 2 for p in (1 / 2, 2 / 3, 3 / 4)) / 3
    assert fold["brier"] == pytest.approx(expected)


# --- accounting capture reassembly -----------------------------------------------


def framed(payload: str, chunk: int = 10) -> list[str]:
    pieces = [payload[i : i + chunk] for i in range(0, len(payload), chunk)]
    lines = [
        f"{accounting.PREFIX} {i}/{len(pieces)} {p}" for i, p in enumerate(pieces, 1)
    ]
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return [*lines, f"{accounting.PREFIX} SHA256 {digest}"]


def test_reassembly_validates_order_and_checksum():
    payload = json.dumps({"x": list(range(20))})
    lines = framed(payload)
    assert accounting.reassemble(lines)[0] == payload
    logged = "\n".join(f"2026-10-08T00:00:00Z stdout {line}" for line in lines)
    assert accounting.reassemble(accounting.frames(logged))[0] == payload
    single = f"{accounting.PREFIX}:{payload}"
    assert accounting.reassemble([single])[1].startswith("single-line")
    defects = {
        "missing": lines[:2] + lines[3:],
        "duplicate": [lines[0], *lines],
        "truncated": lines[:-1],
        "checksum": [*lines[:-1], lines[-1][:-4] + "0000"],
    }
    for name, broken in defects.items():
        with pytest.raises(accounting.Inconclusive):
            accounting.reassemble(broken)
        assert name


def capture(target_pot, delivered, to_call=9900, stack=900, previous_pot=200):
    return {
        "buffer_complete": True,
        "trigger": {"decision_alias": 2},
        "events": [
            {
                "event": "turn_result",
                "data": {
                    "seat": 1,
                    "action": "call",
                    "amount": 50,
                    "pot": previous_pot,
                },
            },
            {
                "event": "turn_result",
                "data": {
                    "seat": 1,
                    "action": "raise",
                    "amount": 9900,
                    "pot": target_pot,
                },
            },
            {
                "event": "decision_state",
                "data": {
                    "decision_alias": 2,
                    "phase": "flop",
                    "your_seat": 0,
                    "your_stack": stack,
                    "to_call": to_call,
                    "pot": delivered,
                    "stacks": {"0": stack, "1": 0},
                    "action_history": [entry(1, "raise", 9900, "flop")],
                },
            },
        ],
    }


@pytest.mark.parametrize(
    ("target_pot", "delivered", "expected"),
    [
        (10100, 10100, "CONFIRMED_EXCESS_INCLUDED"),
        (1100, 1100, "RECONCILED_BEFORE_DECISION"),
        (10100, 1100, "RECONCILED_BEFORE_DECISION"),
        (10100, 5000, "INCONCLUSIVE"),
    ],
)
def test_classification_uses_an_independent_pot_before(target_pot, delivered, expected):
    result = accounting.classify(capture(target_pot, delivered))
    assert result["classification"] == expected
    assert result["derived"]["U"] == 9000
    assert result["derived"]["P_contestable"] == 1100


def test_unclean_or_incomplete_captures_are_inconclusive():
    value = capture(10100, 10100)
    value["buffer_complete"] = False
    assert accounting.classify(value)["classification"] == "INCONCLUSIVE"
    value = capture(10100, 10100)
    value["events"][2]["data"]["action_history"].insert(
        0, entry(1, "raise", 100, "flop")
    )
    assert "another contribution" in accounting.classify(value)["why"]


def test_recommendations_never_promote():
    base = {"arms": ["a", "b"], "paired_difference": {"ci95": [0.5, 2.0]}}
    verdict = analysis.recommend(base, True)
    assert verdict["verdict"] == "DIFFERENCE_DETECTED" and verdict["higher_arm"] == "b"
    assert "explicit promotion" in verdict["note"]
    assert analysis.recommend(base, False)["verdict"] == "INSUFFICIENT_SAMPLE"
    base["paired_difference"]["ci95"] = [-1, 1]
    assert analysis.recommend(base, True)["verdict"] == "NO_DIFFERENCE_DETECTED"
