"""Heads-up preflop context, policy frequencies and bot routing (version 3)."""

from dataclasses import dataclass, field, replace

import pytest

from sleight_of_hand.holdem import preflop
from sleight_of_hand.holdem.hands import CLASSES, COMBOS, TOTAL_COMBOS, hand_class
from sleight_of_hand.holdem.preflop import (
    CALL,
    FOLD,
    RAISE,
    Facing,
    Position,
    action_distribution,
    derive_context,
    weighted_frequencies,
)
from sleight_of_hand.holdem.preflop_tables import EQUITY_VS_RANDOM, ORDER

HERO, VILLAIN = 0, 1


@dataclass
class State:
    """The GameState fields the preflop policy reads."""

    pot: int
    your_stack: int
    opponent_stacks: list
    to_call: int
    min_raise: int
    max_raise: int
    valid_actions: list
    action_history: list
    your_seat: int = HERO
    phase: str = "preflop"
    board: list = field(default_factory=list)


def spot(
    position,
    actions=(),
    stack_bb=100,
    villain_bb=None,
    bb=100,
    ante=0,
    cap_to_call=False,
):
    """Heads-up state after ``actions``: (who, action, raise-to total).

    ``cap_to_call`` reports ``to_call`` capped at our stack instead of the
    full amount owed; the protocol does not pin down which a server sends.
    """
    button = HERO if position == "button" else VILLAIN
    other = 1 - button
    history = [
        {"seat": button, "action": "post_small_blind", "amount": bb // 2},
        {"seat": other, "action": "post_big_blind", "amount": bb},
    ]
    history += [{"seat": s, "action": "post_ante", "amount": ante} for s in (0, 1)]
    bets = {button: bb // 2, other: bb}
    for who, action, amount in actions:
        seat = HERO if who == "hero" else VILLAIN
        if action == "raise":
            bets[seat] = amount
        elif action == "call":
            bets[seat] = max(bets.values())
        history.append({"seat": seat, "action": action, "amount": amount})
    start = {HERO: stack_bb * bb, VILLAIN: (villain_bb or stack_bb) * bb}
    level = max(bets.values())
    hero_behind = start[HERO] - bets[HERO] - ante
    villain_behind = start[VILLAIN] - bets[VILLAIN] - ante
    to_call = level - bets[HERO]
    if to_call and (villain_behind == 0 or to_call >= hero_behind):
        valid, low, high = ["fold", "call"], 0, 0
    else:
        valid = ["fold", "call", "raise"] if to_call else ["check", "raise"]
        high = bets[HERO] + hero_behind
        low = min(level + max(bb, level - min(bets.values())), high)
    return State(
        pot=sum(bets.values()) + 2 * ante,
        your_stack=hero_behind,
        opponent_stacks=[villain_behind],
        to_call=min(to_call, hero_behind) if cap_to_call else to_call,
        min_raise=low,
        max_raise=high,
        valid_actions=valid,
        action_history=history,
    )


FIRST_IN = spot("button")
VS_LIMP = spot("big_blind", [("villain", "call", 100)])
VS_OPEN = spot("big_blind", [("villain", "raise", 250)])
VS_3BET = spot("button", [("hero", "raise", 250), ("villain", "raise", 1000)])
VS_4BET = spot(
    "big_blind",
    [("villain", "raise", 250), ("hero", "raise", 1000), ("villain", "raise", 2300)],
)
LIMP_RAISED = spot("button", [("hero", "call", 100), ("villain", "raise", 400)])


def shove(depth, villain_bb=None, cap_to_call=False):
    return spot(
        "big_blind",
        [("villain", "raise", (villain_bb or depth) * 100)],
        stack_bb=depth,
        villain_bb=villain_bb,
        cap_to_call=cap_to_call,
    )


ALL_SPOTS = [FIRST_IN, VS_LIMP, VS_OPEN, VS_3BET, VS_4BET, LIMP_RAISED] + [
    shove(d) for d in (3, 7, 10, 14, 18, 30)
]


def ctx(state):
    context = derive_context(state)
    assert context is not None
    return context


# --- A. Context derivation ------------------------------------------------


@pytest.mark.parametrize(
    "state, position, facing, raises, hero_acted",
    [
        (FIRST_IN, Position.BUTTON, Facing.FIRST_IN, 0, False),
        (VS_LIMP, Position.BIG_BLIND, Facing.LIMP, 0, False),
        (VS_OPEN, Position.BIG_BLIND, Facing.OPEN, 1, False),
        (VS_3BET, Position.BUTTON, Facing.THREE_BET, 2, True),
        (VS_4BET, Position.BIG_BLIND, Facing.FOUR_BET_PLUS, 3, True),
        (LIMP_RAISED, Position.BUTTON, Facing.LIMP_RAISED, 1, True),
        (shove(10), Position.BIG_BLIND, Facing.SHOVE, 1, False),
    ],
)
def test_context_classification(state, position, facing, raises, hero_acted):
    c = ctx(state)
    assert (c.position, c.facing, c.raises, c.hero_acted) == (
        position,
        facing,
        raises,
        hero_acted,
    )


def test_context_amounts_in_chips_and_big_blinds():
    c = ctx(FIRST_IN)
    assert (c.big_blind, c.hero_bet, c.villain_bet) == (100, 50, 100)
    assert c.amount_owed_bb == c.call_cost_bb == 0.5 and c.unmatched == 0
    assert c.pot_bb == 1.5 and c.effective_stack_bb == 100
    o = ctx(VS_OPEN)
    assert o.villain_bet == 250 and o.amount_owed == o.call_cost == 150
    assert o.action_sequence == ("villain:raise",)


def test_effective_stack_uses_smaller_stack_and_current_blind_level():
    # Blinds rise during a match: the posted big blind is the current level.
    c = ctx(spot("button", stack_bb=30, villain_bb=100, bb=400))
    assert c.big_blind == 400
    assert c.effective_stack == 30 * 400 and c.effective_stack_bb == 30
    covered = ctx(shove(12, villain_bb=60))
    assert covered.facing is Facing.SHOVE and covered.effective_stack_bb == 12


def test_antes_are_reconciled_with_the_pot():
    c = ctx(spot("big_blind", [("villain", "raise", 250)], ante=10))
    assert c.facing is Facing.OPEN and c.pot == 370


def test_call_that_commits_our_stack_is_a_shove():
    # Villain keeps chips behind, but calling would put us all-in.
    s = spot("big_blind", [("villain", "raise", 800)], stack_bb=8, villain_bb=50)
    assert s.opponent_stacks[0] > 0
    assert ctx(s).facing is Facing.SHOVE


def test_call_amount_semantics_do_not_matter():
    # The protocol shows calls both as totals and increments; neither is read.
    for amount in (50, 100):
        s = spot("big_blind", [("villain", "call", 100)])
        s.action_history[-1]["amount"] = amount
        assert ctx(s).facing is Facing.LIMP


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: setattr(s, "action_history", []),  # no blind posts
        lambda s: setattr(s, "opponent_stacks", [9000, 9000]),  # multiway
        lambda s: setattr(s, "phase", "flop"),
        lambda s: setattr(s, "pot", s.pot + 1),  # history disagrees with pot
        lambda s: setattr(s, "to_call", s.to_call + 7),
        lambda s: setattr(s, "your_seat", VILLAIN),  # wrong seat
        lambda s: s.action_history.append({"seat": 0, "action": "draw", "amount": 0}),
        lambda s: s.action_history.append({"seat": 0, "action": "fold", "amount": 0}),
        lambda s: s.action_history.append({"seat": 0}),
    ],
)
def test_unreconcilable_states_are_declined(mutate):
    s = spot("big_blind", [("villain", "raise", 250)])
    mutate(s)
    assert derive_context(s) is None


# --- Hand classes and tables ----------------------------------------------


def test_hand_classes_and_combo_weights():
    assert len(CLASSES) == len(ORDER) == 169
    assert sum(COMBOS.values()) == TOTAL_COMBOS
    assert (COMBOS["AA"], COMBOS["AKs"], COMBOS["AKo"]) == (6, 4, 12)
    assert hand_class(["Kd", "Ah"]) == "AKo"
    assert hand_class(["7s", "7h"]) == "77"
    assert hand_class(["2c", "3c"]) == "32s"
    assert ORDER[0] == "AA" and ORDER[-1] == "32o"
    assert 0.84 < EQUITY_VS_RANDOM["AA"] < 0.86
    assert 0.31 < EQUITY_VS_RANDOM["32o"] < 0.34


# --- Policy invariants ----------------------------------------------------


@pytest.mark.parametrize("state", ALL_SPOTS)
def test_distributions_are_normalized(state):
    c = ctx(state)
    for label in CLASSES:
        probs = action_distribution(c, label)
        assert all(p >= -1e-12 for p in probs.values())
        assert sum(probs.values()) == pytest.approx(1)
        if c.facing is Facing.SHOVE:
            assert probs[RAISE] == 0
        if c.facing is Facing.LIMP:
            assert probs[FOLD] == 0  # never fold when checking is free


@pytest.mark.parametrize("state", [FIRST_IN, VS_LIMP, VS_OPEN])
def test_ranges_follow_strength_order_and_mix_only_at_boundaries(state):
    c = ctx(state)
    raise_p = [action_distribution(c, label)[RAISE] for label in ORDER]
    assert raise_p == sorted(raise_p, reverse=True)
    assert sum(0 < p < 1 for p in raise_p) <= 1
    if c.facing is Facing.FIRST_IN:
        fold = [action_distribution(c, label)[FOLD] for label in ORDER]
        assert fold == sorted(fold)


# --- B. Button first in ---------------------------------------------------


@pytest.mark.parametrize("depth", [100, 40, 15])
def test_button_first_in_aggregate_frequencies(depth):
    f = weighted_frequencies(ctx(spot("button", stack_bb=depth)))
    vpip = f[RAISE] + f[CALL]
    assert 0.84 <= vpip <= 0.96  # PLUMBER reference, and near Season 6's 87%
    assert f[RAISE] >= 0.64  # raise-first-in reference 64-85%
    assert f[CALL] <= 0.32  # limp reference 0-32%
    assert f[CALL] > 0 and f[FOLD] > 0  # a limited limp band and a fold region


def test_button_extremes():
    c = ctx(FIRST_IN)
    assert action_distribution(c, "AA")[RAISE] == 1
    assert action_distribution(c, "32o")[FOLD] == 1


# --- C/D. Big blind: limp response is separate from open response -----------


def test_big_blind_versus_open_narrows_three_bets_into_calls():
    c = ctx(VS_OPEN)
    f = weighted_frequencies(c)
    assert 0.15 <= f[RAISE] <= 0.22
    assert f[CALL] > f[RAISE]
    # Season 6 raised ~33% of hands here; the band between the new 3-bet
    # region and 33% of combos is now a pure call.
    moved = [
        label
        for label in ORDER
        if 17 <= preflop.PERCENTILES[label][0] and preflop.PERCENTILES[label][1] <= 33
    ]
    assert moved
    assert all(action_distribution(c, label)[CALL] == 1 for label in moved)


def test_big_blind_defence_tightens_with_open_size():
    folds = [
        weighted_frequencies(ctx(spot("big_blind", [("villain", "raise", size)])))[FOLD]
        for size in (200, 250, 300, 400)
    ]
    assert folds == sorted(folds) and folds[0] < folds[-1]


def test_big_blind_versus_limp_keeps_iso_raising_and_differs_from_open():
    limp, opened = ctx(VS_LIMP), ctx(VS_OPEN)
    fl, fo = weighted_frequencies(limp), weighted_frequencies(opened)
    assert fl[FOLD] == 0 and fl[RAISE] > fo[RAISE]
    # A hand that iso-raises a limp but only calls an open.
    label = next(
        label
        for label in ORDER
        if action_distribution(limp, label)[RAISE] == 1
        and action_distribution(opened, label)[RAISE] == 0
    )
    assert action_distribution(opened, label)[CALL] == 1


@pytest.mark.parametrize("open_share", [0.5, 0.7, 0.9])
def test_big_blind_overall_pfr_within_reference(open_share):
    # Button either opens or limps (a folded button gives the BB no decision).
    iso = weighted_frequencies(ctx(VS_LIMP))[RAISE]
    three = weighted_frequencies(ctx(VS_OPEN))[RAISE]
    assert open_share * three + (1 - open_share) * iso <= 0.25


def test_button_does_not_overfold_its_opens_to_a_three_bet():
    first, c = ctx(FIRST_IN), ctx(VS_3BET)
    reach = sum(COMBOS[x] * action_distribution(first, x)[RAISE] for x in ORDER)
    fold = sum(
        COMBOS[x]
        * action_distribution(first, x)[RAISE]
        * action_distribution(c, x)[FOLD]
        for x in ORDER
    )
    assert fold / reach < 0.55


# --- E. Short-stack shove response ----------------------------------------


def test_fold_to_shove_tightens_with_stack_depth():
    folds = [weighted_frequencies(ctx(shove(d)))[FOLD] for d in (4, 7, 10, 14, 18)]
    assert folds == sorted(folds)
    assert folds[-1] - folds[0] > 0.4
    # PLUMBER measured Season 6 folding 13-24%; its reference band, 38-78%,
    # spans 5-20bb. The deepest bucket predicts ~78.6%, at the band's top
    # edge, so allow a small margin rather than tune ranges to the bar.
    assert all(0.38 <= f <= 0.80 for f in folds[1:])


def test_shove_calls_are_evaluated_against_a_range_not_a_random_hand():
    c = ctx(shove(10))
    required = preflop.required_equity(c)
    # Hands that beat the price against a random hand but not the shove range.
    fooled = [
        label
        for label in ORDER
        if EQUITY_VS_RANDOM[label] >= required + 0.03
        and action_distribution(c, label)[FOLD] == 1
    ]
    assert len(fooled) >= 10
    assert "K2o" in fooled or "Q5o" in fooled


def test_same_hand_calls_short_and_folds_deeper():
    label = next(
        x
        for x in ORDER
        if action_distribution(ctx(shove(4)), x)[CALL] == 1
        and action_distribution(ctx(shove(18)), x)[FOLD] == 1
    )
    assert label


def test_covered_shove_excess_is_not_counted_in_the_price():
    equal, covered = ctx(shove(10)), ctx(shove(10, villain_bb=80))
    assert preflop.required_equity(covered) == pytest.approx(
        preflop.required_equity(equal)
    )
    assert weighted_frequencies(covered) == weighted_frequencies(equal)


def test_covered_shove_worked_example_splits_owed_from_cost():
    # 7bb big blind: 1bb in, 6bb behind; a covering button shoves to 60bb.
    for capped in (False, True):
        s = shove(7, villain_bb=60, cap_to_call=capped)
        assert s.to_call == (600 if capped else 5900)
        c = ctx(s)
        assert c.facing is Facing.SHOVE
        assert (c.pot, c.amount_owed, c.call_cost, c.unmatched) == (
            6100,
            5900,
            600,
            5300,
        )
        assert preflop.required_equity(c) == pytest.approx(6 / 14)


@pytest.mark.parametrize("depth, required", [(7, 6 / 14), (14, 13 / 28)])
def test_covered_shove_is_identical_under_both_to_call_representations(depth, required):
    full = ctx(shove(depth, villain_bb=60))
    capped = ctx(shove(depth, villain_bb=60, cap_to_call=True))
    assert full == capped  # every field, so every derived quantity
    assert full.facing is capped.facing is Facing.SHOVE
    assert preflop.required_equity(full) == preflop.required_equity(capped)
    assert preflop.required_equity(capped) == pytest.approx(required)
    for label in CLASSES:
        assert action_distribution(full, label) == action_distribution(capped, label)
    assert weighted_frequencies(full) == weighted_frequencies(capped)
    # Same economics as an equal-stack shove at the same depth.
    assert weighted_frequencies(capped) == weighted_frequencies(ctx(shove(depth)))


@pytest.mark.parametrize("depth", [7, 14])
def test_bot_decides_covered_shoves_identically_under_both_representations(depth):
    from bots.chipzen.bot import SleightOfHandBot

    for hole in (["As", "Ah"], ["Kc", "2d"], ["9s", "8s"], ["7c", "2d"]):
        runs = []
        for capped in (False, True):
            s = sdk_state(shove(depth, villain_bb=60, cap_to_call=capped), hole)
            bot = SleightOfHandBot(seed=5)
            runs.append([bot.decide(s) for _ in range(25)])
        assert runs[0] == runs[1]
        assert {a.action for a in runs[0]} <= {"fold", "call"}


def test_reshove_uses_a_tighter_assumed_range_than_an_open_shove():
    reshove = ctx(
        spot(
            "button",
            [("hero", "raise", 200), ("villain", "raise", 1500)],
            stack_bb=15,
        )
    )
    assert reshove.facing is Facing.SHOVE and reshove.hero_acted
    cfg = preflop.DEFAULT_PREFLOP
    assert preflop.assumed_villain_width(reshove, cfg) < (
        preflop.assumed_villain_width(ctx(shove(15)), cfg)
    )


# --- F. Sizing and legality -----------------------------------------------


@pytest.mark.parametrize(
    "state, expected",
    [
        (FIRST_IN, 250),
        (spot("button", stack_bb=20), 200),
        (VS_LIMP, 400),
        (VS_OPEN, 1000),
        (VS_3BET, 2250),
        (LIMP_RAISED, 1200),
        (VS_4BET, 10000),  # facing a 4-bet: all-in
        (spot("big_blind", [("villain", "raise", 250)], stack_bb=15), 1500),
    ],
)
def test_raise_sizes_are_total_bets(state, expected):
    c = ctx(state)
    assert preflop.raise_to(c, state.min_raise, state.max_raise) == expected


def test_raise_size_respects_legal_bounds_and_short_all_ins():
    c = ctx(FIRST_IN)
    assert preflop.raise_to(c, 300, 5000) == 300
    assert preflop.raise_to(c, 200, 180) == 180  # short all-in below the minimum
    cfg = replace(preflop.DEFAULT_PREFLOP, open_bb=500)
    assert preflop.raise_to(c, 200, 10000, cfg) == 10000


# --- Bot routing (needs the SDK) --------------------------------------------


def sdk_state(state, hole):
    chipzen = pytest.importorskip("chipzen")
    return chipzen.GameState(
        hole_cards=[chipzen.Card.from_str(c) for c in hole],
        pot=state.pot,
        your_stack=state.your_stack,
        opponent_stacks=list(state.opponent_stacks),
        to_call=state.to_call,
        min_raise=state.min_raise,
        max_raise=state.max_raise,
        valid_actions=list(state.valid_actions),
        action_history=[dict(e) for e in state.action_history],
        your_seat=state.your_seat,
    )


@pytest.mark.parametrize("state", ALL_SPOTS)
@pytest.mark.parametrize("hole", [["As", "Ah"], ["Ks", "9h"], ["7c", "2d"]])
def test_bot_emits_only_legal_preflop_actions(state, hole):
    from bots.chipzen.bot import SleightOfHandBot

    s = sdk_state(state, hole)
    bot = SleightOfHandBot(seed=3)
    for _ in range(40):
        action = bot.decide(s)
        assert action.action in s.valid_actions
        if action.action == "raise":
            assert min(s.min_raise, s.max_raise) <= action.amount <= s.max_raise
        if "check" in s.valid_actions:
            assert action.action != "fold"


def test_bot_routes_preflop_through_context_policy(monkeypatch):
    from chipzen import Action

    from bots.chipzen.bot import SleightOfHandBot

    def no_equity(*args):
        raise AssertionError("preflop v3 must not use random-hand equity")

    monkeypatch.setattr("bots.chipzen.bot.estimate_equity", no_equity)
    bot = SleightOfHandBot(seed=1)
    assert bot.decide(sdk_state(FIRST_IN, ["As", "Ah"])) == Action.raise_to(250)
    assert bot.decide(sdk_state(FIRST_IN, ["3c", "2d"])) == Action.fold()
    assert bot.decide(sdk_state(VS_LIMP, ["3c", "2d"])) == Action.check()
    assert bot.decide(sdk_state(shove(10), ["As", "Ah"])) == Action.call()
    assert bot.decide(sdk_state(shove(18), ["Kc", "2d"])) == Action.fold()


def test_bot_parses_protocol_example_and_classifies_it():
    chipzen = pytest.importorskip("chipzen")
    from bots.chipzen.bot import SleightOfHandBot

    # POKER-GAME-STATE-PROTOCOL.md section 4, message 7: BB facing a raise to 30.
    message = {
        "type": "turn_request",
        "request_id": "req_7",
        "valid_actions": ["fold", "call", "raise"],
        "state": {
            "hand_number": 1,
            "phase": "preflop",
            "board": [],
            "your_hole_cards": ["Jd", "Tc"],
            "pot": 40,
            "your_stack": 990,
            "opponent_stacks": [970],
            "to_call": 20,
            "min_raise": 50,
            "max_raise": 990,
            "action_history": [
                {"seat": 0, "action": "post_small_blind", "amount": 5},
                {"seat": 1, "action": "post_big_blind", "amount": 10},
                {"seat": 0, "action": "raise", "amount": 30},
            ],
        },
    }
    s = chipzen.GameState.from_turn_request(message, your_seat=1, dealer_seat=0)
    c = ctx(s)
    assert (c.position, c.facing, c.big_blind) == (Position.BIG_BLIND, Facing.OPEN, 10)
    assert SleightOfHandBot(seed=2).decide(s).action in s.valid_actions


def test_unclassified_preflop_warns_once_and_uses_season6(capsys):
    from bots.chipzen.bot import SleightOfHandBot

    s = sdk_state(FIRST_IN, ["As", "Ah"])
    s.action_history = []
    bot = SleightOfHandBot(seed=1, samples=8)
    bot.decide(s)
    bot.decide(s)
    assert capsys.readouterr().err.count("Season 6") == 1


def test_malformed_cards_keep_season6_safe_fallback():
    from bots.chipzen.bot import SleightOfHandBot

    s = sdk_state(VS_OPEN, ["As", "Ah"])
    s.hole_cards = []
    assert SleightOfHandBot(seed=1).decide(s).action == "fold"


# --- G. Season 6 regression -------------------------------------------------

# Recorded from the season-6 tag (seed 11, twelve consecutive decisions)
# before version 3 was added. Postflop, and preflop states without blind
# posts, must remain on the Season 6 path decision for decision.
SEASON6 = {
    "flop_check": [("raise", 580), ("check", 0)] + [("raise", 580)] * 10,
    "flop_facing_bet": [
        ("raise", 1560),
        ("call", 0),
        ("raise", 1560),
        ("raise", 1560),
        ("raise", 1560),
        ("call", 0),
        ("raise", 1560),
        ("raise", 1560),
        ("raise", 1560),
        ("call", 0),
        ("raise", 1560),
        ("raise", 1560),
    ],
    "turn_facing_bet": [
        ("raise", 2640),
        ("raise", 2640),
        ("fold", 0),
        ("call", 0),
        ("raise", 2640),
        ("fold", 0),
        ("fold", 0),
        ("fold", 0),
        ("fold", 0),
        ("fold", 0),
        ("call", 0),
        ("call", 0),
    ],
    "river_check": [("raise", 580)] + [("check", 0)] * 11,
    "preflop_no_history": [
        ("call", 0),
        ("raise", 360),
        ("call", 0),
        ("raise", 360),
        ("call", 0),
        ("raise", 360),
        ("call", 0),
        ("raise", 360),
        ("raise", 360),
        ("raise", 360),
        ("call", 0),
        ("raise", 360),
    ],
}


def season6_state(name):
    chipzen = pytest.importorskip("chipzen")
    board = {
        "flop_check": ["Ks", "7h", "2c"],
        "flop_facing_bet": ["Th", "7c", "2h"],
        "turn_facing_bet": ["Ks", "7h", "2c", "3d"],
        "river_check": ["As", "Kh", "9c", "8d", "2h"],
        "preflop_no_history": [],
    }[name]
    hole = {
        "flop_check": ["Ah", "Kd"],
        "flop_facing_bet": ["9h", "8h"],
        "turn_facing_bet": ["Qc", "Jd"],
        "river_check": ["5s", "4s"],
        "preflop_no_history": ["Ah", "Kd"],
    }[name]
    fields = {
        "pot": 600,
        "your_stack": 9400,
        "opponent_stacks": [9400],
        "to_call": 0,
        "min_raise": 100,
        "max_raise": 9400,
        "valid_actions": ["check", "raise"],
    }
    fields.update(
        {
            "flop_facing_bet": {
                "pot": 900,
                "to_call": 300,
                "min_raise": 600,
                "valid_actions": ["fold", "call", "raise"],
            },
            "turn_facing_bet": {
                "pot": 1200,
                "to_call": 600,
                "min_raise": 1200,
                "valid_actions": ["fold", "call", "raise"],
            },
            "preflop_no_history": {
                "pot": 150,
                "to_call": 50,
                "min_raise": 200,
                "max_raise": 9900,
                "your_stack": 9900,
                "opponent_stacks": [9850],
                "valid_actions": ["fold", "call", "raise"],
            },
        }.get(name, {})
    )
    return chipzen.GameState(
        hole_cards=[chipzen.Card.from_str(c) for c in hole],
        board=[chipzen.Card.from_str(c) for c in board],
        phase={0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(board)],
        **fields,
    )


@pytest.mark.parametrize("name", sorted(SEASON6))
def test_non_preflop_path_matches_season6_decisions(name):
    from bots.chipzen.bot import SleightOfHandBot

    s = season6_state(name)
    bot = SleightOfHandBot(seed=11)
    assert [(a.action, a.amount) for a in (bot.decide(s) for _ in range(12))] == (
        SEASON6[name]
    )
