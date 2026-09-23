"""Pre-evaluation gate: state matrix, legality, invariants and fallbacks.

Systematic companion to test_preflop.py. It exercises legal heads-up
preflop state shapes across stack depths under both server readings of
``to_call`` (full amount owed, or capped at our stack), checks every
action the version 3 layer can emit for all 169 classes, and confirms the
deliberate Season 6 fallbacks.
"""

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from sleight_of_hand.holdem import preflop
from sleight_of_hand.holdem.hands import CLASSES, COMBOS, TOTAL_COMBOS
from sleight_of_hand.holdem.preflop import (
    FOLD,
    RAISE,
    Facing,
    Position,
    action_distribution,
    diagnose_context,
    weighted_frequencies,
)
from sleight_of_hand.holdem.preflop_tables import ORDER
from tests.test_preflop import sdk_state, spot

chipzen = pytest.importorskip(
    "chipzen", reason="install bots/chipzen/requirements.txt for port tests"
)
from bots.chipzen.bot import SleightOfHandBot

DEPTHS = (4, 5, 7, 8, 10, 12, 14, 16, 18, 20, 25, 40, 100)
BB = 100


def sequence(position, depth, actions, villain_bb=None):
    """Clamp raises to each player's stack; None if the line is not legal."""
    start = {"hero": depth * BB, "villain": (villain_bb or depth) * BB}
    button = "hero" if position == "button" else "villain"
    bets = {button: BB // 2, ("villain" if button == "hero" else "hero"): BB}
    out = []
    for who, action, amount in actions:
        level = max(bets.values())
        if action == "raise":
            amount = min(amount, start[who])
            if amount <= level:
                return None
            bets[who] = amount
        else:
            bets[who] = min(level, start[who])
        out.append((who, action, amount))
        if bets[who] == start[who] and who == "hero":
            return None  # hero all-in: hero has no further decision
    return out


def matrix():
    """(name, expected facing, state) over every depth and state shape."""
    shapes = [
        ("first_in", "button", [], Facing.FIRST_IN, None),
        ("vs_limp", "big_blind", [("villain", "call", BB)], Facing.LIMP, None),
        ("vs_open_2.0", "big_blind", [("villain", "raise", 200)], Facing.OPEN, None),
        ("vs_open_2.5", "big_blind", [("villain", "raise", 250)], Facing.OPEN, None),
        ("vs_open_3.0", "big_blind", [("villain", "raise", 300)], Facing.OPEN, None),
        (
            "limp_iso",
            "button",
            [("hero", "call", BB), ("villain", "raise", 400)],
            Facing.LIMP_RAISED,
            None,
        ),
        (
            "open_3bet",
            "button",
            [("hero", "raise", 250), ("villain", "raise", 1000)],
            Facing.THREE_BET,
            None,
        ),
        (
            "bb_vs_4bet",
            "big_blind",
            [
                ("villain", "raise", 250),
                ("hero", "raise", 1000),
                ("villain", "raise", 2300),
            ],
            Facing.FOUR_BET_PLUS,
            None,
        ),
        (
            "btn_vs_5bet",
            "button",
            [
                ("hero", "raise", 250),
                ("villain", "raise", 1000),
                ("hero", "raise", 2300),
                ("villain", "raise", 6000),
            ],
            Facing.FOUR_BET_PLUS,
            None,
        ),
        ("open_shove", "big_blind", [("villain", "raise", 10**9)], Facing.SHOVE, None),
        (
            "reshove",
            "button",
            [("hero", "raise", 200), ("villain", "raise", 10**9)],
            Facing.SHOVE,
            None,
        ),
        (
            "covering_shove",
            "big_blind",
            [("villain", "raise", 10**9)],
            Facing.SHOVE,
            "cover",
        ),
        (
            "covering_reshove",
            "button",
            [("hero", "raise", 200), ("villain", "raise", 10**9)],
            Facing.SHOVE,
            "cover",
        ),
    ]
    out = []
    for depth in DEPTHS:
        for name, position, actions, facing, cover in shapes:
            villain_bb = max(3 * depth, 60) if cover else None
            line = sequence(position, depth, actions, villain_bb)
            if line is None:
                continue
            for capped in (False, True):
                state = spot(
                    position,
                    line,
                    stack_bb=depth,
                    villain_bb=villain_bb,
                    cap_to_call=capped,
                )
                out.append((f"{name}@{depth}bb{'/capped' if capped else ''}", state))
    return out


MATRIX = matrix()
PAIRS = [
    (MATRIX[i][0], MATRIX[i][1], MATRIX[i + 1][1]) for i in range(0, len(MATRIX), 2)
]


def test_matrix_covers_every_shape_and_depth():
    names = {name.split("@")[0] for name, _ in MATRIX}
    assert len(names) == 13
    assert {int(n.split("@")[1].split("bb")[0]) for n, _ in MATRIX} == set(DEPTHS)
    assert len(MATRIX) >= 250


@pytest.mark.parametrize("name, full, capped", PAIRS, ids=[p[0] for p in PAIRS])
def test_both_to_call_readings_derive_identical_economics(name, full, capped):
    a, reason_a = diagnose_context(full)
    b, reason_b = diagnose_context(capped)
    assert reason_a == reason_b == "ok", name
    assert a == b  # position, facing, owed, cost, stacks, sequence: all fields
    assert preflop.required_equity(a) == preflop.required_equity(b)
    for label in CLASSES:
        assert action_distribution(a, label) == action_distribution(b, label)


@pytest.mark.parametrize("name, full, capped", PAIRS, ids=[p[0] for p in PAIRS])
def test_matrix_classification_and_economics(name, full, capped):
    c = preflop.derive_context(full)
    shape = name.split("@")[0]
    # Short depths turn ordinary raises into all-ins, which are shoves.
    expected = {
        "first_in": Facing.FIRST_IN,
        "vs_limp": Facing.LIMP,
        "vs_open_2.0": Facing.OPEN,
        "vs_open_2.5": Facing.OPEN,
        "vs_open_3.0": Facing.OPEN,
        "limp_iso": Facing.LIMP_RAISED,
        "open_3bet": Facing.THREE_BET,
        "bb_vs_4bet": Facing.FOUR_BET_PLUS,
        "btn_vs_5bet": Facing.FOUR_BET_PLUS,
    }.get(shape, Facing.SHOVE)
    commits = c.amount_owed > 0 and (
        c.villain_behind == 0 or c.amount_owed >= c.hero_behind
    )
    assert c.facing is (Facing.SHOVE if commits else expected)
    assert c.position is (
        Position.BUTTON
        if shape in ("first_in", "limp_iso", "open_3bet", "btn_vs_5bet")
        or "reshove" in shape
        else Position.BIG_BLIND
    )
    assert c.call_cost == min(c.amount_owed, c.hero_behind)
    assert c.pot == c.hero_bet + c.villain_bet
    assert c.effective_stack == min(
        c.hero_bet + c.hero_behind, c.villain_bet + c.villain_behind
    )
    assert c.big_blind == BB
    if c.call_cost:
        live = c.pot - c.unmatched
        assert preflop.required_equity(c) == pytest.approx(
            c.call_cost / (live + c.call_cost)
        )


# --- Malformed states fall back safely ------------------------------------

MALFORMED = {
    "missing_blind": (
        lambda s: s.action_history.pop(1),
        "missing_blinds",
    ),
    "duplicate_blind_post": (
        lambda s: s.action_history.insert(
            2, {"seat": 1, "action": "post_big_blind", "amount": 100}
        ),
        "duplicate_blind",
    ),
    "same_seat_posts_both": (
        lambda s: s.action_history[1].update(seat=s.action_history[0]["seat"]),
        None,
    ),
    "impossible_order": (
        # The big blind acting first preflop (a check keeps the pot intact).
        lambda s: s.action_history.insert(
            2, {"seat": s.action_history[1]["seat"], "action": "check", "amount": 0}
        ),
        "impossible_order",
    ),
    "pot_mismatch": (lambda s: setattr(s, "pot", s.pot + 30), "pot_mismatch"),
    "to_call_mismatch": (
        lambda s: setattr(s, "to_call", s.to_call + 30),
        "to_call_mismatch",
    ),
    "unknown_action": (
        lambda s: s.action_history.insert(
            2, {"seat": 0, "action": "straddle", "amount": 200}
        ),
        "unknown_action",
    ),
    "multiway": (
        lambda s: setattr(s, "opponent_stacks", s.opponent_stacks + [5000]),
        "not_heads_up",
    ),
    "postflop_board": (lambda s: setattr(s, "board", ["As", "Kd", "2c"]), None),
    "hero_not_in_blinds": (
        lambda s: setattr(s, "your_seat", 2),
        "hero_not_in_blinds",
    ),
    "history_entry_missing_fields": (
        lambda s: s.action_history.append({"action": "raise"}),
        "unreadable_history",
    ),
}
BASES = {
    "first_in": lambda: spot("button"),
    "vs_open": lambda: spot("big_blind", [("villain", "raise", 250)]),
    "vs_shove": lambda: spot("big_blind", [("villain", "raise", 1000)], stack_bb=10),
}


@pytest.mark.parametrize("base", sorted(BASES))
@pytest.mark.parametrize("fault", sorted(MALFORMED))
def test_malformed_states_decline_with_reason_and_decide_legally(base, fault):
    state = BASES[base]()
    mutate, reason = MALFORMED[fault]
    mutate(state)
    context, why = diagnose_context(state)
    assert context is None
    if reason is not None:
        assert why == reason
    s = sdk_state(state, ["Ks", "9h"])
    s.board = [chipzen.Card.from_str(c) for c in state.board]
    s.phase = "flop" if state.board else "preflop"
    bot = SleightOfHandBot(seed=4, samples=8)
    for _ in range(5):
        action = bot.decide(s)  # no traceback
        assert action.action in s.valid_actions
        if action.action == "raise":
            assert min(s.min_raise, s.max_raise) <= action.amount <= s.max_raise


# --- Every emittable action is legal -----------------------------------------


def emitted_actions(bot, state, context):
    """Every action the v3 layer can emit for this state, over all classes."""
    choices = {
        a
        for label in CLASSES
        for a, p in action_distribution(context, label).items()
        if p > 0
    }
    return {choice: bot._legal_preflop(state, context, choice) for choice in choices}


@pytest.mark.parametrize("name, state", MATRIX, ids=[m[0] for m in MATRIX])
def test_every_emittable_action_is_legal(name, state):
    bot = SleightOfHandBot(seed=1)
    context = preflop.derive_context(state)
    for choice, action in emitted_actions(bot, state, context).items():
        assert action is not None, (name, choice)
        assert action.action in state.valid_actions, (name, choice)
        if "check" in state.valid_actions:
            assert action.action != "fold"
        if action.action == "raise":
            assert min(state.min_raise, state.max_raise) <= action.amount
            assert action.amount <= state.max_raise


RAISE_SHAPES = [(n, s) for n, s in MATRIX if "raise" in s.valid_actions]


@pytest.mark.parametrize("name, state", RAISE_SHAPES, ids=[m[0] for m in RAISE_SHAPES])
def test_raise_sizing_respects_perturbed_server_bounds(name, state):
    c = preflop.derive_context(state)
    hi = state.max_raise
    bounds = [
        (state.min_raise, hi),
        (hi, hi),  # min_raise == max_raise
        (hi + 100, hi),  # short all-in form: minimum above the maximum
        (max(state.min_raise, hi - 1), hi),  # target far below the minimum
        (state.min_raise, max(state.min_raise, (state.min_raise + hi) // 2)),
        (1, 10**9),  # effectively unbounded: target reported unclamped
    ]
    for lo, top in bounds:
        amount = preflop.raise_to(c, lo, top)
        assert min(lo, top) <= amount <= top, (name, lo, top, amount)
        assert isinstance(amount, int)


def test_commit_fraction_turns_raises_into_all_ins():
    for facing_state in (
        spot("button", stack_bb=4),  # 2bb open is >= 40% of 4bb
        spot("big_blind", [("villain", "call", 100)], stack_bb=8),  # 4bb iso
        spot("big_blind", [("villain", "raise", 250)], stack_bb=20),  # 10bb 3-bet
        spot("button", [("hero", "raise", 250), ("villain", "raise", 1000)], 50),
    ):
        c = preflop.derive_context(facing_state)
        assert preflop.raise_to(c, facing_state.min_raise, facing_state.max_raise) == (
            facing_state.max_raise
        )


# --- Policy sanity invariants ------------------------------------------------


def test_button_raise_probability_follows_strength():
    c = preflop.derive_context(spot("button"))
    raise_p = [action_distribution(c, label)[RAISE] for label in ORDER]
    play_p = [1 - action_distribution(c, label)[FOLD] for label in ORDER]
    assert raise_p == sorted(raise_p, reverse=True)
    assert play_p == sorted(play_p, reverse=True)


def test_explicit_regions_are_exact_combo_shares():
    cfg = preflop.DEFAULT_PREFLOP
    cases = [
        (spot("button"), RAISE, cfg.btn_raise_top),
        (spot("big_blind", [("villain", "call", 100)]), RAISE, cfg.bb_iso_top),
        (spot("big_blind", [("villain", "raise", 250)]), RAISE, cfg.bb_3bet_top),
        (
            spot("button", [("hero", "raise", 250), ("villain", "raise", 1000)]),
            RAISE,
            cfg.btn_4bet_top,
        ),
    ]
    for state, action, percent in cases:
        f = weighted_frequencies(preflop.derive_context(state))
        assert f[action] == pytest.approx(percent / 100, abs=1e-12)
    first = weighted_frequencies(preflop.derive_context(spot("button")))
    assert first[FOLD] == pytest.approx(1 - cfg.btn_play_top / 100, abs=1e-12)


@pytest.mark.parametrize("depth", [40, 100])
def test_defence_never_loosens_as_open_size_grows(depth):
    folds = []
    for size in (200, 250, 300):
        c = preflop.derive_context(
            spot("big_blind", [("villain", "raise", size)], stack_bb=depth)
        )
        folds.append(weighted_frequencies(c)[FOLD])
        # Per class too: a hand never continues more often against a bigger open.
    assert folds == sorted(folds)
    contexts = [
        preflop.derive_context(
            spot("big_blind", [("villain", "raise", size)], stack_bb=depth)
        )
        for size in (200, 250, 300)
    ]
    for label in CLASSES:
        cont = [1 - action_distribution(c, label)[FOLD] for c in contexts]
        assert cont == sorted(cont, reverse=True), label


@pytest.mark.parametrize("depth", [5, 7, 10, 14, 18])
def test_covering_and_equal_shoves_share_contestable_pot(depth):
    equal = preflop.derive_context(
        spot("big_blind", [("villain", "raise", depth * BB)], depth)
    )
    covering = preflop.derive_context(
        spot("big_blind", [("villain", "raise", 6000)], depth, villain_bb=60)
    )
    assert equal.call_cost == covering.call_cost
    assert equal.pot - equal.unmatched == covering.pot - covering.unmatched
    assert preflop.required_equity(equal) == preflop.required_equity(covering)
    assert weighted_frequencies(equal) == weighted_frequencies(covering)


@pytest.mark.parametrize("name, state", MATRIX, ids=[m[0] for m in MATRIX])
def test_free_check_never_folds_and_extremes_route_correctly(name, state):
    c = preflop.derive_context(state)
    if "check" in state.valid_actions:
        assert all(action_distribution(c, x)[FOLD] == 0 for x in CLASSES)
    top = ("AA", "KK", "QQ", "AKs")
    if c.facing in (Facing.FIRST_IN, Facing.OPEN, Facing.LIMP, Facing.SHOVE):
        for label in top:
            assert action_distribution(c, label)[FOLD] == 0, (name, label)
    bottom = ("32o", "42o", "62o", "72o")
    for label in bottom:
        assert action_distribution(c, label)[RAISE] == 0, (name, label)


# --- Calibration uses the production policy ---------------------------------


def load_calibration():
    path = Path(__file__).resolve().parents[1] / "scripts" / "preflop_calibration.py"
    spec = importlib.util.spec_from_file_location("preflop_calibration", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module
    spec.loader.exec_module(module)
    return module


class RecordingRng(random.Random):
    """Captures the weights the bot samples from; picks the first option."""

    def choices(self, population, weights=None, **kwargs):
        self.seen = dict(zip(population, weights))
        return [population[0]]


def test_calibration_scenarios_match_the_bot_decision_path():
    calibration = load_calibration()
    bot = SleightOfHandBot(seed=1)
    bot.rng = RecordingRng()
    checked = 0
    for name, state, _ in calibration.scenarios():
        context = preflop.derive_context(state)
        assert context is not None, name
        totals = weighted_frequencies(context)
        assert sum(totals.values()) == pytest.approx(1, abs=1e-12)
        for label in CLASSES:
            high, low = label[0], label[1]
            hole = (
                [high + "s", low + "h"]
                if len(label) == 2 or label[2] == "o"
                else [high + "s", low + "s"]
            )
            s = sdk_state(state, hole)
            bot.preflop_action(s)
            expected = {
                a: p for a, p in action_distribution(context, label).items() if p > 0
            }
            assert bot.rng.seen == expected, (name, label)
            checked += 1
    assert checked == 169 * len(calibration.scenarios())


def test_combo_weights_total_exactly():
    assert TOTAL_COMBOS == 1326 == sum(COMBOS.values())
    for label, n in COMBOS.items():
        assert n == (6 if len(label) == 2 else 4 if label[2] == "s" else 12)


# --- Tracing ------------------------------------------------------------------


def test_trace_disabled_by_default_prints_nothing(capsys):
    bot = SleightOfHandBot(seed=3)
    bot.decide(sdk_state(spot("button"), ["As", "Kd"]))
    assert capsys.readouterr().err == ""


def test_trace_reports_context_and_does_not_change_actions(capsys):
    states = [
        spot("button"),
        spot("big_blind", [("villain", "raise", 250)]),
        spot("big_blind", [("villain", "raise", 1000)], stack_bb=10),
    ]
    holes = (["As", "Kd"], ["9s", "8s"], ["7c", "2d"])
    plain, traced = (
        SleightOfHandBot(seed=8),
        SleightOfHandBot(seed=8, trace_preflop=True),
    )
    for state in states:
        for hole in holes:
            s = sdk_state(state, hole)
            for _ in range(10):
                assert plain.decide(s) == traced.decide(s)
    lines = [json.loads(x) for x in capsys.readouterr().err.splitlines()]
    assert len(lines) == len(states) * len(holes) * 10
    first = lines[0]
    assert first["context_ok"] and first["reason"] == "ok"
    assert (first["position"], first["facing"], first["big_blind"]) == (
        "button",
        "first_in",
        100,
    )
    assert first["hand_class"] == "AKo" and first["effective_bb"] == 100
    assert first["amount_owed"] == first["call_cost"] == 50
    assert set(first["probs"]) == {"raise", "call", "fold"}
    assert first["action"] == "raise" and first["raise_to"] == 250
    shove_line = lines[-1]
    assert shove_line["facing"] == "shove" and "raise_to" not in shove_line
    text = "\n".join(json.dumps(x) for x in lines)
    for secret in ("Kd", "As", "token", "ticket", "ws://", "wss://"):
        assert secret not in text


def test_trace_reports_declined_reason(capsys):
    state = spot("big_blind", [("villain", "raise", 250)])
    state.pot += 30
    bot = SleightOfHandBot(seed=1, samples=8, trace_preflop=True)
    bot.decide(sdk_state(state, ["As", "Kd"]))
    err = capsys.readouterr().err.splitlines()
    assert err[0].startswith("Preflop context unavailable")
    record = json.loads(err[1])
    assert record["context_ok"] is False and record["reason"] == "pot_mismatch"
    assert record["history"][-1] == [1, "raise", 250]


# --- Season 6 differential ------------------------------------------------------


class Season6Only(SleightOfHandBot):
    def preflop_action(self, state):
        return None


def fallback_states():
    multiway = sdk_state(spot("button"), ["Ks", "9h"])
    multiway.opponent_stacks = [9900, 9900]
    malformed = sdk_state(spot("big_blind", [("villain", "raise", 250)]), ["Ks", "9h"])
    malformed.pot += 30
    no_history = sdk_state(spot("button"), ["Ks", "9h"])
    no_history.action_history = []
    return {"multiway": multiway, "malformed": malformed, "no_history": no_history}


@pytest.mark.parametrize("name", ["multiway", "malformed", "no_history"])
def test_fallback_states_are_season6_decision_for_decision(name):
    s = fallback_states()[name]
    a, b = SleightOfHandBot(seed=21), Season6Only(seed=21)
    assert [a.decide(s) for _ in range(30)] == [b.decide(s) for _ in range(30)]


@pytest.mark.parametrize("name, state", MATRIX[::7], ids=[m[0] for m in MATRIX[::7]])
def test_recognized_heads_up_preflop_never_falls_back(name, state, monkeypatch):
    def no_equity(*args):
        raise AssertionError(f"{name} fell back to Season 6")

    monkeypatch.setattr("bots.chipzen.bot.estimate_equity", no_equity)
    bot = SleightOfHandBot(seed=2)
    for hole in (["As", "Ah"], ["Ts", "9s"], ["7c", "2d"]):
        assert bot.decide(sdk_state(state, hole)).action in state.valid_actions


def test_config_is_frozen_defaults():
    # The evaluation candidate's policy constants; changing them is a new
    # candidate, not a validation fix.
    assert preflop.DEFAULT_PREFLOP == preflop.PreflopConfig()
    assert (
        preflop.DEFAULT_PREFLOP.btn_raise_top,
        preflop.DEFAULT_PREFLOP.btn_play_top,
        preflop.DEFAULT_PREFLOP.bb_iso_top,
        preflop.DEFAULT_PREFLOP.bb_3bet_top,
    ) == (70.0, 87.0, 30.0, 16.0)
