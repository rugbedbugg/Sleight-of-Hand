"""Hold'em rankings, action legality and real SDK state/action contracts."""

import itertools
import random

import pytest

pytest.importorskip("chipzen", reason="run `uv sync` (chipzen group) for port tests")
from chipzen import Action, Card, GameState

from bots.chipzen.bot import (
    SleightOfHandBot,
    betting_params,
    live_opponents,
    raise_amount,
)
from sleight_of_hand.engine.actions import ActionType
from sleight_of_hand.holdem.equity import encode, estimate_equity, rank_hand
from sleight_of_hand.policy.heuristic import PolicyParams, action_probs_from_strength


def ranked(cards):
    return rank_hand([encode(c) for c in cards.split()])


@pytest.mark.parametrize(
    "cards, expected",
    [
        ("As Ks Qs Js Ts 2d 3c", (8, 14)),
        ("As 2s 3s 4s 5s Kd Qc", (8, 5)),
        ("Ac Ad Ah As Kd 2h 3h", (7, 14, 13)),
        ("Ac Ad Ah Kc Kd Kh 2h", (6, 14, 13)),
        ("Ac Ad Ah Kc Kd Qh Qs", (6, 14, 13)),
        ("As Js 8s 4s 2s Kd Qd", (5, 14, 11, 8, 4, 2)),
        ("As 2d 3h 4s 5c Kd Qc", (4, 5)),
        ("2c 3c 4d 5h 6s 7s 7c", (4, 7)),
        ("As Ad Ac Kh Qc 2h 3c", (3, 14, 13, 12)),
        ("As Ad Kh Kc Qc Qh 2c", (2, 14, 13, 12)),
        ("As Ad Kc Qh 9s 2c 3h", (1, 14, 13, 12, 9)),
        ("As Kd Qc 9s 8h 2c 3h", (0, 14, 13, 12, 9, 8)),
    ],
)
def test_rank_categories_and_kickers(cards, expected):
    assert ranked(cards) == expected


def test_seven_card_rank_matches_best_five():
    rng = random.Random(3)
    for _ in range(150):
        cards = rng.sample(range(52), 7)
        assert rank_hand(cards) == max(
            rank_hand(list(hand)) for hand in itertools.combinations(cards, 5)
        )


def test_equity_nuts_ties_and_multiway():
    rng = random.Random(2)
    assert estimate_equity(["As", "Ks"], ["Qs", "Js", "Ts", "2c", "3d"], 5, rng) == 1
    for opponents in (1, 2, 5):
        assert estimate_equity(
            ["2c", "3d"], ["As", "Ks", "Qs", "Js", "Ts"], opponents, rng
        ) == pytest.approx(1 / (opponents + 1))
    aa = estimate_equity(["As", "Ah"], [], 1, random.Random(1), samples=512)
    trash = estimate_equity(["7s", "2h"], [], 1, random.Random(1), samples=512)
    assert 0.75 < aa < 0.95
    assert 0.2 < trash < 0.5
    assert aa > trash


@pytest.mark.parametrize(
    "hole, board",
    [
        (["As"], []),
        (["As", "As"], []),
        (["As", "Kh"], ["As", "2d", "3d"]),
        (["As", "Kh"], ["2d"]),
        (["Xs", "Kh"], []),
    ],
)
def test_invalid_cards_rejected(hole, board):
    with pytest.raises(ValueError):
        estimate_equity(hole, board, 1, random.Random(1))


def state(**overrides):
    fields = {
        "hole_cards": [Card.from_str(c) for c in ("Ah", "Kd")],
        "pot": 150,
        "your_stack": 9900,
        "opponent_stacks": [9850],
        "to_call": 50,
        "min_raise": 200,
        "max_raise": 9900,
        "valid_actions": ["fold", "call", "raise"],
    }
    fields.update(overrides)
    return GameState(**fields)


@pytest.mark.parametrize(
    "valid, to_call, low, high",
    [
        (["check", "raise"], 0, 10, 1000),
        (["fold", "call", "raise"], 50, 200, 1000),
        (["fold", "call"], 500, 0, 0),
        (["check"], 0, 0, 0),
        (["fold"], 100, 0, 0),
        (["call"], 500, 0, 0),
        (["raise"], 0, 10, 1000),
        (["fold", "call", "raise"], 50, 200, 80),
        (["check", "all_in"], 0, 10, 1000),
    ],
)
def test_only_legal_actions_and_bounded_total_bets(valid, to_call, low, high):
    s = state(valid_actions=valid, to_call=to_call, min_raise=low, max_raise=high)
    bot = SleightOfHandBot(seed=1, samples=8)
    for _ in range(30):
        result = bot.decide(s)
        assert result.action in valid
        if result.action == "raise":
            assert min(low, high) <= result.amount <= high
            assert result.to_wire() == {
                "action": "raise",
                "params": {"amount": result.amount},
            }


def test_raise_size_aggression_and_short_stack():
    s = state(pot=100, to_call=20, min_raise=60, max_raise=1000)
    assert raise_amount(s, 0) == 60
    assert raise_amount(s, 1) == 180
    assert raise_amount(state(min_raise=200, max_raise=80), 1) == 80


def test_live_opponents_include_all_in_and_exclude_folds():
    s = state(
        your_seat=2,
        opponent_stacks=[0, 300, 0, 400, 500],
        action_history=[
            {"seat": 1, "action": "fold"},
            {"seat": 4, "action": "fold"},
        ],
    )
    assert live_opponents(s) == 3
    assert SleightOfHandBot(seed=4).decide(s).action in s.valid_actions


def test_missing_cards_safe_fallback_and_unknown_actions():
    bot = SleightOfHandBot(seed=2)
    assert bot.decide(state(hole_cards=[])).action == "fold"
    assert (
        bot.decide(
            state(hole_cards=[], valid_actions=["check", "raise"], to_call=0)
        ).action
        == "check"
    )
    with pytest.raises(ValueError, match="no supported"):
        bot.decide(state(valid_actions=["draw"]))
    with pytest.raises(ValueError, match="without legal"):
        bot.decide(state(valid_actions=[]))


def test_real_sdk_wire_input_and_deterministic_seed():
    message = {
        "type": "turn_request",
        "request_id": "turn-42",
        "round_id": "hand-3",
        "valid_actions": ["check", "raise"],
        "state": {
            "hand_number": 3,
            "phase": "river",
            "your_hole_cards": ["As", "Ks"],
            "board": ["Qs", "Js", "Ts", "2c", "3d"],
            "pot": 100,
            "your_stack": 900,
            "opponent_stacks": [900],
            "to_call": 0,
            "min_raise": 10,
            "max_raise": 900,
        },
    }
    s = GameState.from_turn_request(message)
    assert s.request_id == "turn-42"
    params = PolicyParams(aggression=1, steepness=30, value_bet_threshold=0)
    a = SleightOfHandBot(params, seed=9).decide(s)
    b = SleightOfHandBot(params, seed=9).decide(s)
    assert a == b == Action.raise_to(110)


def test_pot_odds_reference_and_free_checks():
    params = PolicyParams()
    # 100 in the pot, opponent bets 100: current pot is 200, call costs 100.
    assert betting_params(state(pot=200, to_call=100), params) == params
    assert betting_params(state(to_call=0), params) is params
    assert betting_params(state(valid_actions=["check", "raise"]), params) is params


def test_call_cost_uses_affordable_stack_and_preserves_risk_margin():
    params = PolicyParams(call_threshold=1 / 3)
    # A half-pot bet (50 into 100) requires 25% equity, not 33%.
    assert betting_params(
        state(pot=150, to_call=50), params
    ).call_threshold == pytest.approx(0.25)
    short = betting_params(state(pot=200, to_call=500, your_stack=20), params)
    assert short.call_threshold == pytest.approx(20 / 220)
    cautious = betting_params(
        state(pot=150, to_call=50), PolicyParams(call_threshold=0.5)
    )
    assert cautious.call_threshold > 0.25


def test_expensive_calls_raise_both_thresholds_without_changing_other_genes():
    params = PolicyParams()
    adjusted = betting_params(state(pot=110, to_call=100), params)
    assert adjusted.call_threshold > params.call_threshold
    assert adjusted.value_bet_threshold > params.value_bet_threshold
    assert adjusted.aggression == params.aggression
    assert adjusted.bluff_freq == params.bluff_freq
    assert adjusted.steepness == params.steepness
    assert (
        betting_params(
            state(pot=1, to_call=100), PolicyParams(call_threshold=1)
        ).call_threshold
        <= 1
    )


@pytest.mark.parametrize("pot, cost", [(100, 10), (110, 100)])
def test_known_equity_call_fold_ev_improves_at_both_prices(pot, cost):
    # Controlled terminal decision, no future betting or side pots. EV(call)
    # is equity * (current pot + call) - call; folding has zero incremental EV.
    # This proves the response to price, not a full-match win-rate improvement.
    equity = 0.4
    params = PolicyParams()
    s = state(pot=pot, to_call=cost, valid_actions=["fold", "call"])
    legal = [ActionType.FOLD, ActionType.CALL]
    old = action_probs_from_strength(equity, legal, cost, params)[ActionType.CALL]
    new = action_probs_from_strength(equity, legal, cost, betting_params(s, params))[
        ActionType.CALL
    ]
    call_ev = equity * (pot + cost) - cost
    assert new * call_ev > old * call_ev


def test_bot_uses_price_adjustment(monkeypatch):
    monkeypatch.setattr("bots.chipzen.bot.estimate_equity", lambda *args: 0.4)
    cheap = state(pot=100, to_call=10, valid_actions=["fold", "call"])
    expensive = state(pot=110, to_call=100, valid_actions=["fold", "call"])
    cheap_bot = SleightOfHandBot(seed=7)
    expensive_bot = SleightOfHandBot(seed=7)
    cheap_calls = sum(cheap_bot.decide(cheap).action == "call" for _ in range(500))
    expensive_calls = sum(
        expensive_bot.decide(expensive).action == "call" for _ in range(500)
    )
    assert cheap_calls > expensive_calls + 150


def _extraction_snapshot():
    """Same driver in isolated baseline/current processes; no shared modules."""
    import copy
    import hashlib
    import io
    from contextlib import redirect_stderr
    from dataclasses import asdict

    from sleight_of_hand.holdem.hands import CLASSES
    from sleight_of_hand.holdem.opponent import ShoveModel
    from tests.test_preflop import sdk_state, shove
    from tests.test_preflop_gate import MATRIX, fallback_states
    from tests.test_shove_model import MALFORMED, report

    records = []

    def model_state(model):
        return (
            [(key, asdict(value)) for key, value in model.stats.items()],
            sorted(model.processed),
            list(model._starts.items()),
        )

    def invoke(bot, method, *args):
        err = io.StringIO()
        with redirect_stderr(err):
            try:
                value = getattr(bot, method)(*args)
                outcome = value.to_wire() if isinstance(value, Action) else value
            except Exception as exc:  # noqa: BLE001 - exceptions are oracle outputs
                outcome = (type(exc).__name__, str(exc))
        records.append(
            (
                method,
                outcome,
                err.getvalue(),
                bot._seat,
                bot._warned_preflop,
                hashlib.sha256(repr(bot.rng.getstate()).encode()).hexdigest(),
                model_state(bot.shove_model),
            )
        )

    # Every class in every existing context, including capped/full call costs.
    bot = SleightOfHandBot(seed=734, samples=8, trace_preflop=True)
    for name, spot in MATRIX:
        for label in CLASSES:
            hole = [label[0] + "s", label[1] + ("s" if label.endswith("s") else "h")]
            invoke(bot, "decide", sdk_state(spot, hole))

    for seed in (0, 17, 91):
        for trace in (False, True):
            bot = SleightOfHandBot(seed=seed, samples=8, trace_preflop=trace)
            # Hooks and decisions intentionally disagree about the hero seat.
            invoke(bot, "on_match_start", {"seats": [{"seat": "1", "is_self": True}]})
            for spot in (
                sdk_state(shove(14), ["Qs", "Jh"]),
                *fallback_states().values(),
            ):
                for _ in range(4):
                    invoke(bot, "decide", spot)
            for phase, board in (
                ("flop", ["2c", "7d", "Th"]),
                ("turn", ["2c", "7d", "Th", "Js"]),
                ("river", ["2c", "7d", "Th", "Js", "Qc"]),
            ):
                for opponents in ([0], [0, 500, 1000]):
                    invoke(
                        bot,
                        "decide",
                        state(phase=phase, board=board, opponent_stacks=opponents),
                    )
            for valid in (
                [],
                ["draw"],
                ["check"],
                ["call"],
                ["fold"],
                ["raise"],
                ["all_in"],
                ["check", "all_in"],
                ["fold", "call", "raise"],
                ["fold", "check", "call", "raise", "all_in"],
            ):
                for low, high in ((200, 80), (200, 0), (200, 1000)):
                    for cards in ([], ["As", "As"], ["As", "Kh"]):
                        invoke(
                            bot,
                            "decide",
                            state(
                                valid_actions=valid,
                                hole_cards=cards,
                                min_raise=low,
                                max_raise=high,
                            ),
                        )
            for history in (
                None,
                [None],
                ["bad"],
                [{"action": "fold", "seat": []}],
                [{"action": "raise", "seat": "1", "amount": True}],
            ):
                for phase in ("preflop", "river"):
                    invoke(bot, "decide", state(action_history=history, phase=phase))
            invoke(bot, "on_reconnected", {"seats": [{"seat": 0, "is_self": True}]})
            for number in range(1, 25):
                button = number % 2
                voluntary = (
                    [(1, "raise", 5000), (0, "fold", 0)]
                    if button
                    else [(0, "raise", 250), (1, "raise", 5000), (0, "call", 0)]
                )
                start, result = report.hand_messages(
                    number, button, (5000, 5000), voluntary
                )
                if number % 3 == 0:
                    start.pop("round_id")
                    result.pop("round_id")
                invoke(bot, "on_round_start", start)
                invoke(bot, "on_round_result", result)
                invoke(bot, "on_round_result", copy.deepcopy(result))
                invoke(bot, "decide", sdk_state(shove(50), ["Kh", "Td"]))

    # Preserve model no-ops, rejected-hand eviction and parse-exception retention.
    model = ShoveModel()
    for number in range(20):
        model.record_round_start(
            {"round_id": str(number), "state": {"stacks": [5000, 5000]}}
        )
    records.append(model_state(model))
    for fault in MALFORMED.values():
        model = ShoveModel()
        start, result = report.hand_messages(
            7, 1, (5000, 5000), [(1, "raise", 5000), (0, "fold", 0)]
        )
        start, result = fault(start, result)
        if start is not None:
            model.record_round_start(start)
        observation = model.observe_round_result(result, 0)
        records.append((observation, model_state(model)))
    for seat in (None, 0, 1, True, "0", [], 8):
        for amount in (True, "5000", -1, 5000):
            model = ShoveModel()
            start, result = report.hand_messages(
                8, 1, (5000, 5000), [(1, "raise", amount), (0, "fold", 0)]
            )
            model.record_round_start(start)
            observation = model.observe_round_result(result, seat)
            records.append(
                (asdict(observation) if observation else None, model_state(model))
            )
    return records


def test_extraction_matches_d0e964f_oracle(tmp_path):
    """Archive the immutable oracle and compare wire/log/RNG/model snapshots."""
    import json
    import os
    import subprocess
    import sys
    import tarfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    archive = tmp_path / "baseline.tar"
    with archive.open("wb") as stream:
        subprocess.run(
            ["git", "archive", "d0e964f880cbd7c4a8eb2f14e61f3b8e9dadff44"],
            cwd=root,
            stdout=stream,
            check=True,
        )
    with tarfile.open(archive) as stream:
        stream.extractall(baseline, filter="data")
    driver = (
        "import json, runpy, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "namespace = runpy.run_path(sys.argv[2]); "
        "records = namespace['_extraction_snapshot'](); "
        "open(sys.argv[3], 'w').write(json.dumps(records))"
    )
    outputs = []
    for label, source in (("oracle", baseline), ("current", root)):
        output = tmp_path / f"{label}.json"
        subprocess.run(
            [
                sys.executable,
                "-c",
                driver,
                str(source),
                str(Path(__file__).resolve()),
                str(output),
            ],
            cwd=source,
            check=True,
            capture_output=True,
            env={**os.environ, "PYTHONPATH": str(source)},
        )
        outputs.append(json.loads(output.read_text()))
    assert len(outputs[0]) == len(outputs[1])
    for index, (expected, actual) in enumerate(zip(*outputs)):
        assert actual == expected, f"d0e964f observable mismatch at record {index}"


def test_extracted_agent_works_without_sdk():
    import subprocess
    import sys
    from pathlib import Path

    # -S omits site-packages: neither the SDK nor a third-party dependency
    # may be needed by the extracted decision path.
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            """
import random
import sys
from types import SimpleNamespace
from sleight_of_hand.holdem.agent import HoldemAgent
from sleight_of_hand.holdem.decision import Decision
rng = random.Random(17)
agent = HoldemAgent(rng, samples=8)
assert agent.rng is rng
state = SimpleNamespace(
    phase='river', hole_cards=['As', 'Ks'], board=['Qs', 'Js', 'Ts', '2c', '3d'],
    pot=100, your_stack=900, opponent_stacks=[900], your_seat=0,
    to_call=0, min_raise=10, max_raise=900,
    valid_actions=['check', 'raise'], action_history=[],
)
decision = agent.decide(state)
assert isinstance(decision, Decision)
assert decision.action in state.valid_actions
assert 'chipzen' not in sys.modules
""",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
