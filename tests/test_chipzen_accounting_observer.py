"""Observation-only adapter diagnostics; no live server accounting assertions."""

import copy
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import pytest
from chipzen import Action, Bot, Card, GameState

from bots.chipzen import accounting_observer as observer
from bots.chipzen.bot import SleightOfHandBot
from sleight_of_hand.holdem.agent import HoldemAgent
from tests.test_postflop_characterization import FIXTURES
from tests.test_preflop import FIRST_IN, sdk_state, shove
from tests.test_shove_model import report

FLAG = "SLEIGHT_CHIPZEN_ACCOUNTING_OBSERVER"
LOG = "SLEIGHT_CHIPZEN_ACCOUNTING_LOG"


def make_bot(monkeypatch, tmp_path, enabled=True):
    monkeypatch.setenv(FLAG, "1" if enabled else "0")
    monkeypatch.setenv(LOG, str(tmp_path / "capture.json"))
    return SleightOfHandBot(seed=17, samples=128)


def messages(number=1):
    start, result = report.hand_messages(
        number, 1, (700, 6000), [(1, "raise", 6000), (0, "fold", 0)]
    )
    start["state"].update(
        your_hole_cards=["Ah", "Kd"], pot=150, small_blind=50, big_blind=100
    )
    result["result"].update(
        pot=6100,
        stacks=[600, 6100],
        payouts=[{"seat": 1, "amount": 6100}],
        board=["Qs", "7d", "2c", "3s", "4h"],
    )
    return start, result


def state(case="F11", number=1):
    # SDK-independent fixture values, passed through the actual SDK state type.
    values = copy.deepcopy(FIXTURES[case]["state"])
    allowed = GameState.__dataclass_fields__
    value = GameState(**{k: v for k, v in values.items() if k in allowed})
    value.round_id = f"r-{number:05d}"
    value.hand_number = number
    return value


def begin(bot, number=1):
    start, result = messages(number)
    bot.on_match_start({"seats": [{"seat": 0, "is_self": True}, {"seat": 1}]})
    bot.on_round_start(start)
    return result


def captured(tmp_path):
    return json.loads((tmp_path / "capture.json").read_text())


def snapshot(bot):
    return copy.deepcopy(
        (
            bot.rng.getstate(),
            vars(bot.shove_model),
            list(bot.shove_model._starts.items()),
            bot._seat,
            bot._warned_preflop,
        )
    )


def outcome(bot, method, argument):
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        try:
            value = getattr(bot, method)(copy.deepcopy(argument))
            error = None
        except Exception as exc:  # noqa: BLE001 - compare policy exceptions too
            value, error = None, (type(exc), str(exc))
    return value, error, stderr.getvalue(), snapshot(bot)


def test_disabled_has_no_diagnostic_side_effects(monkeypatch, tmp_path):
    with patch.object(observer, "AccountingObserver", side_effect=AssertionError):
        bot = make_bot(monkeypatch, tmp_path, False)
    assert bot._accounting_observer is None
    with patch.object(Path, "open", side_effect=AssertionError("unexpected I/O")):
        result = begin(bot)
        bot.decide(state())
        bot.on_round_result(result)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "path_kind", ["missing", "empty", "directory", "missing_parent", "existing"]
)
def test_bad_configuration_disables_once(monkeypatch, tmp_path, capsys, path_kind):
    monkeypatch.setenv(FLAG, "1")
    path = tmp_path / "existing"
    path.write_text("keep")
    values = {
        "empty": "",
        "directory": str(tmp_path),
        "missing_parent": str(tmp_path / "absent" / "capture"),
        "existing": str(path),
    }
    if path_kind == "missing":
        monkeypatch.delenv(LOG, raising=False)
    else:
        monkeypatch.setenv(LOG, values[path_kind])
    bot = SleightOfHandBot(seed=17)
    result = begin(bot)
    bot.decide(state())
    bot.on_round_result(result)
    assert bot._accounting_observer is None
    assert capsys.readouterr().err.count("[accounting observer]") == 1
    assert path.read_text() == "keep"


@pytest.mark.parametrize(
    "changes, reasons",
    [
        ({}, ["to_call_at_least_hero_stack", "opponent_remaining_zero"]),
        ({"to_call": 600}, ["to_call_at_least_hero_stack", "opponent_remaining_zero"]),
        ({"opponent_stacks": [100]}, ["to_call_at_least_hero_stack"]),
        ({"to_call": 50}, ["opponent_remaining_zero"]),
        ({"to_call": 50, "opponent_stacks": [100]}, None),
        ({"opponent_stacks": [0, 0]}, None),
        ({"your_stack": 0}, None),
        ({"valid_actions": ["fold"]}, None),
    ],
)
def test_trigger_is_candidate_evidence_only(monkeypatch, tmp_path, changes, reasons):
    bot = make_bot(monkeypatch, tmp_path)
    result = begin(bot)
    value = state()
    for key, item in changes.items():
        setattr(value, key, item)
    # Observe without invoking policy on intentionally inconsistent geometries.
    bot._observe("decision_state", value)
    assert not list(tmp_path.iterdir())
    bot.on_round_result(result)
    if reasons is None:
        assert not list(tmp_path.iterdir())
    else:
        assert captured(tmp_path)["trigger"]["reasons"] == reasons


def test_exactly_one_policy_call_and_same_action_object(monkeypatch, tmp_path):
    bot = make_bot(monkeypatch, tmp_path)
    begin(bot)
    chosen = Action("call", 600)
    # Existing translation constructs the SDK action; return that exact instance.
    with (
        patch.object(HoldemAgent, "decide", return_value=chosen) as policy,
        patch("bots.chipzen.bot._sdk_action", return_value=chosen),
        patch.object(Path, "open", side_effect=AssertionError("critical-path I/O")),
    ):
        assert bot.decide(state()) is chosen
    policy.assert_called_once()


def test_allowlist_excludes_secrets_names_cards_and_unknown_fields(
    monkeypatch, tmp_path
):
    sentinels = [
        "SECRET_TOKEN_SENTINEL",
        "SECRET_TICKET_SENTINEL",
        "SECRET_URL_SENTINEL",
        "PLAYER_NAME_SENTINEL",
        "PRIVATE_CARD_SENTINEL",
        "UNKNOWN_PRIVATE_SENTINEL",
    ]
    for key, secret in zip(
        ("CHIPZEN_TOKEN", "CHIPZEN_TICKET", "CHIPZEN_WS_URL"), sentinels
    ):
        monkeypatch.setenv(key, secret)
    bot = make_bot(monkeypatch, tmp_path)
    start, result = messages()
    pollution = dict(
        zip(("token", "ticket", "url", "name", "hole_cards", "unknown"), sentinels)
    )
    for value in (start, start["state"], result, result["result"]):
        value.update(pollution)
    result["result"]["showdown"] = [{"hole_cards": ["Ah", "Kd"], **pollution}]
    result["result"]["deck_reveal"] = sentinels
    for entry in result["result"]["action_history"]:
        entry.update(pollution)
    result["result"]["payouts"][0].update(pollution)
    bot.on_round_start(start)
    value = state()
    value.hole_cards = [Card.from_str("Ah"), Card.from_str("Kd")]
    value.board = [Card.from_str(c) for c in ("Qs", "7d", "2c")]
    value.valid_actions += sentinels
    value.action_history += [pollution]
    bot.decide(value)
    bot.on_turn_result(
        {
            "round_id": start["round_id"],
            "is_timeout": False,
            **pollution,
            "details": {
                "seat": 0,
                "action": "call",
                "amount": 600,
                "pot": 6700,
                "stacks": [0, 0],
                "phase": "flop",
                **pollution,
            },
        }
    )
    bot.on_round_result(result)
    text = (tmp_path / "capture.json").read_text()
    for secret in sentinels + [
        "Ah",
        "Kd",
        "r-00001",
        "showdown",
        "deck_reveal",
        "hole_cards",
    ]:
        assert secret not in text
    doc = captured(tmp_path)
    turn = next(e["data"] for e in doc["events"] if e["event"] == "turn_result")
    assert turn == {
        "seat": 0,
        "action": "call",
        "amount": 600,
        "pot": 6700,
        "stacks": {"0": 0, "1": 0},
        "phase": "flop",
        "is_timeout": False,
    }
    assert doc["events"][1]["data"]["board"] == ["Qs", "7d", "2c"]
    assert doc["events"][-1]["data"]["payouts"] == [{"seat": 1, "amount": 6100}]


def test_modern_to_legacy_delegation_records_once(monkeypatch, tmp_path):
    bot = make_bot(monkeypatch, tmp_path)
    start, result = messages()
    with (
        patch.object(Bot, "on_hand_start") as legacy_start,
        patch.object(Bot, "on_hand_result") as legacy_result,
    ):
        bot.on_round_start(start)
        legacy_start.assert_called_once()
        bot.on_round_start(start)  # Replayed callback, also not another event.
        bot.decide(state())
        bot.on_round_result(result)
        legacy_result.assert_called_once()
        bot.on_round_result(result)
    events = [e["event"] for e in captured(tmp_path)["events"]]
    assert events == [
        "round_start",
        "decision_state",
        "selected_action",
        "round_result",
    ]
    # Direct legacy hooks are intentionally not a second observation surface.
    bot.on_hand_start(1, [])
    bot.on_hand_result(result["result"])
    assert events == [e["event"] for e in captured(tmp_path)["events"]]


def test_one_bundle_then_disarm(monkeypatch, tmp_path):
    bot = make_bot(monkeypatch, tmp_path)
    result = begin(bot)
    for _ in range(3):
        bot.decide(state())
    assert not list(tmp_path.iterdir())
    bot.on_round_result(result)
    before = (tmp_path / "capture.json").read_bytes()
    assert not bot._accounting_observer.armed
    assert (
        len([e for e in captured(tmp_path)["events"] if e["event"] == "decision_state"])
        == 3
    )
    for number in (2, 3):
        start, result = messages(number)
        bot.on_round_start(start)
        bot.decide(state(number=number))
        bot.on_round_result(result)
    assert (tmp_path / "capture.json").read_bytes() == before
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize(
    "failure", ["permission", "serialization", "write", "notify", "existing_race"]
)
def test_diagnostic_failure_preserves_gameplay(monkeypatch, tmp_path, capsys, failure):
    disabled = make_bot(monkeypatch, tmp_path, False)
    enabled = make_bot(monkeypatch, tmp_path)
    result = begin(disabled)
    begin(enabled)
    assert outcome(disabled, "decide", state()) == outcome(enabled, "decide", state())
    if failure == "permission":
        patcher = patch.object(
            Path, "open", side_effect=PermissionError("SECRET_TOKEN_SENTINEL")
        )
    elif failure == "serialization":
        patcher = patch.object(
            observer.json, "dumps", side_effect=ValueError("PRIVATE_CARD_SENTINEL")
        )
    elif failure == "write":
        patcher = patch.object(Path, "open")
    elif failure == "notify":
        patcher = patch.object(
            enabled._accounting_observer,
            "notify",
            side_effect=RuntimeError("SECRET_TOKEN_SENTINEL"),
        )
    else:
        (tmp_path / "capture.json").write_text("preserve existing file")
        patcher = patch.object(observer, "MAX_EVENTS", observer.MAX_EVENTS)
    with patcher as mocked:
        if failure == "write":
            mocked.return_value.__enter__.return_value.write.side_effect = OSError(
                "SECRET_TICKET_SENTINEL"
            )
        disabled.on_round_result(copy.deepcopy(result))
        enabled.on_round_result(copy.deepcopy(result))
    assert snapshot(disabled) == snapshot(enabled)
    assert enabled._accounting_observer is None
    assert outcome(disabled, "decide", state()) == outcome(enabled, "decide", state())
    enabled.on_round_result(result)  # No repeated write attempt or warning.
    err = capsys.readouterr().err
    assert err.count("[accounting observer]") == 1 and "SENTINEL" not in err
    if failure == "existing_race":
        assert (tmp_path / "capture.json").read_text() == "preserve existing file"


@pytest.mark.parametrize("bound", ["events", "bytes", "history"])
def test_bounded_capture_keeps_final_accounting(monkeypatch, tmp_path, bound):
    baseline = make_bot(monkeypatch, tmp_path, False)
    bot = make_bot(monkeypatch, tmp_path)
    begin(baseline)
    result = begin(bot)
    value = state()
    if bound == "events":
        monkeypatch.setattr(observer, "MAX_EVENTS", 6)
    elif bound == "bytes":
        monkeypatch.setattr(observer, "MAX_BYTES", observer.RESULT_RESERVE + 1800)
    else:
        value.action_history *= 10000
    for _ in range(5):
        assert outcome(bot, "decide", value) == outcome(baseline, "decide", value)
    baseline.on_round_result(copy.deepcopy(result))
    bot.on_round_result(result)
    assert snapshot(bot) == snapshot(baseline)
    doc = captured(tmp_path)
    assert not doc["buffer_complete"] and doc["dropped_events"] > 0
    assert len(doc["events"]) <= observer.MAX_EVENTS
    assert len((tmp_path / "capture.json").read_bytes()) < observer.MAX_BYTES + 1024
    assert doc["events"][-1]["event"] == "round_result"
    assert doc["events"][-1]["data"]["stacks"] == {"0": 600, "1": 6100}


def test_mixed_sequence_exact_behavior_rng_model_and_lifecycle_parity(
    monkeypatch, tmp_path
):
    disabled = make_bot(monkeypatch, tmp_path, False)
    enabled = make_bot(monkeypatch, tmp_path)
    match = {"seats": [{"seat": 0, "is_self": True}, {"seat": 1}]}
    steps = [("on_match_start", match)]
    # Adaptation receives real keyed start/result histories before decisions.
    for number in range(1, 5):
        start, result = report.hand_messages(
            number, 1, (5000, 5000), [(1, "raise", 5000), (0, "fold", 0)]
        )
        steps += [
            ("on_round_start", start),
            ("on_round_result", result),
            ("on_round_result", result),
        ]
    start, result = messages(5)
    steps += [("on_round_start", start)]
    ordinary = sdk_state(FIRST_IN, ["Ah", "Kd"])
    adaptive = sdk_state(shove(50), ["Ah", "Kd"])
    for value in (ordinary, adaptive):
        value.round_id, value.hand_number = "r-00005", 5
    steps += [("decide", ordinary), ("decide", adaptive)]
    # All 28 frozen fixtures: all streets, invalid cards, single legal action,
    # capped/raw shoves, line invariance, and sizing boundaries.
    steps += [("decide", state(case, 5)) for case in FIXTURES]
    malformed = state("F01", 5)
    malformed.action_history = [None]  # Preserve the actual policy exception.
    steps += [
        ("decide", malformed),
        ("on_reconnected", match),
        (
            "on_turn_result",
            {
                "round_id": "r-00005",
                "details": {"seat": 0, "action": "call", "amount": 600},
            },
        ),
        ("on_round_result", result),
        ("on_round_result", result),
    ]
    exceptions = 0
    for method, argument in steps:
        old = outcome(disabled, method, argument)
        new = outcome(enabled, method, argument)
        assert new == old, method
        exceptions += old[1] is not None
    assert len(steps) == 49
    assert exceptions >= 1
    assert disabled.shove_model.stats  # Adaptation was exercised, not an empty model.
    assert disabled.shove_model.processed
    assert captured(tmp_path)["buffer_complete"] is False  # Reconnect gap.


def test_observer_itself_never_changes_rng(monkeypatch, tmp_path):
    bot = make_bot(monkeypatch, tmp_path)
    result = begin(bot)
    before = snapshot(bot)
    for event, args in (
        ("decision_state", (state(),)),
        ("selected_action", (Action("call", 600),)),
        ("turn_result", ({"details": {"seat": 0, "action": "call"}},)),
        ("round_result", (result,)),
    ):
        bot._observe(event, *args)
        assert snapshot(bot) == before


def test_out_of_order_result_and_mismatched_state_cannot_close_capture(
    monkeypatch, tmp_path
):
    bot = make_bot(monkeypatch, tmp_path)
    result = begin(bot)
    bot.decide(state())
    event_count = len(bot._accounting_observer.capture["events"])
    bot.decide(state(number=2))
    assert len(bot._accounting_observer.capture["events"]) == event_count
    bot.on_round_result(messages(2)[1])
    assert not list(tmp_path.iterdir())
    bot.on_round_result(result)
    assert not captured(tmp_path)["buffer_complete"]


def test_missing_round_start_does_not_invent_hand_context(monkeypatch, tmp_path):
    bot = make_bot(monkeypatch, tmp_path)
    bot.decide(state())
    bot.on_round_result(messages()[1])
    assert not list(tmp_path.iterdir())


def test_untrusted_strings_in_allowlisted_slots_are_not_serialized(
    monkeypatch, tmp_path
):
    bot = make_bot(monkeypatch, tmp_path)
    result = begin(bot)
    bot.decide(state())
    bot.on_turn_result(
        {
            "details": {
                "phase": "SECRET_TOKEN_SENTINEL",
                "action": "PLAYER_NAME_SENTINEL",
                "pot": "SECRET_TICKET_SENTINEL",
                "seat": "PRIVATE_CARD_SENTINEL",
                "stacks": ["SECRET_URL_SENTINEL"],
                "board": ["PRIVATE_CARD_SENTINEL"],
            }
        }
    )
    bot.on_round_result(result)
    assert "SENTINEL" not in (tmp_path / "capture.json").read_text()


def test_unwritable_parent_disables_before_any_hand(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(observer.os, "access", lambda *args: False)
    bot = make_bot(monkeypatch, tmp_path)
    assert bot._accounting_observer is None
    assert capsys.readouterr().err.count("[accounting observer]") == 1
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("hero", [0, 1, None])
def test_seat_pending_eviction_and_malformed_lifecycle_parity(
    monkeypatch, tmp_path, hero
):
    disabled = make_bot(monkeypatch, tmp_path, False)
    enabled = make_bot(monkeypatch, tmp_path)
    match = {"seats": [{"seat": i, "is_self": i == hero} for i in (0, 1)]}
    assert outcome(disabled, "on_match_start", match) == outcome(
        enabled, "on_match_start", match
    )
    assert enabled._seat == hero
    for number in range(1, 21):
        start, _ = messages(number)
        assert outcome(disabled, "on_round_start", start) == outcome(
            enabled, "on_round_start", start
        )
    assert len(enabled.shove_model._starts) == 16
    for method, argument in (
        ("on_round_result", messages(1)[1]),  # Missing/evicted start.
        ("on_round_result", messages(19)[1]),  # Out of order.
        ("on_round_result", messages(19)[1]),  # Duplicate.
        (
            "on_round_result",
            {"round_id": "r-00020", "result": {"action_history": [None]}},
        ),
        ("on_reconnected", match),
        ("on_round_start", None),  # Original SDK exception still propagates.
    ):
        assert outcome(disabled, method, argument) == outcome(enabled, method, argument)
    assert not list(tmp_path.iterdir())
