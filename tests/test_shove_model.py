"""Match-local adaptive shove defence: estimator, parsing and bot wiring."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sleight_of_hand.holdem import preflop
from sleight_of_hand.holdem.hands import CLASSES
from sleight_of_hand.holdem.opponent import (
    OPEN,
    PRIOR_STRENGTH,
    RESHOVE,
    ShoveModel,
    parse_hand,
)


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


report = _load("shove_adaptation_report")
HERO, VILLAIN, BB = report.HERO, report.VILLAIN, report.BB
DEEP = preflop.stack_bucket(50)


def width(model, kind=OPEN, depth=50):
    return model.estimate(kind, preflop.stack_bucket(depth))["adaptive_width"]


def calls(kind, depth, w):
    return report.call_share(report.shove_context(kind, depth), w)


# --- A. No evidence is exactly version 3 --------------------------------------


def test_prior_strength_is_ten():
    assert PRIOR_STRENGTH == 10.0 and ShoveModel().prior_strength == 10.0


@pytest.mark.parametrize("kind", [OPEN, RESHOVE])
@pytest.mark.parametrize("bucket", range(len(preflop.STACK_BUCKETS)))
def test_empty_model_returns_v3_width_exactly(kind, bucket):
    cfg = preflop.DEFAULT_PREFLOP
    table = cfg.open_shove_width if kind == OPEN else cfg.reshove_width
    est = ShoveModel().estimate(kind, bucket)
    assert est["adaptive_width"] == table[bucket]  # identical object value
    assert (est["shoves"], est["opportunities"]) == (0, 0)


def shove_matrix():
    from tests.test_preflop_gate import MATRIX

    return [
        (n, s) for n, s in MATRIX if preflop.derive_context(s).facing.value == "shove"
    ]


@pytest.mark.parametrize(
    "name, state", shove_matrix(), ids=[n for n, _ in shove_matrix()]
)
def test_fresh_bot_shove_decisions_equal_v3_for_every_class(name, state):
    from bots.chipzen.bot import SleightOfHandBot

    bot = SleightOfHandBot(seed=1)
    ctx = preflop.derive_context(state)
    est = bot.shove_estimate(ctx)
    for label in CLASSES:
        assert preflop.action_distribution(
            ctx, label, villain_width=est["adaptive_width"]
        ) == preflop.action_distribution(ctx, label)


# --- B. One deep shove: no panic -----------------------------------------------


def test_one_deep_shove_moves_modestly():
    model = ShoveModel()
    report.feed_open_pattern(model, 1, 1, 50)
    w = width(model)
    assert w == pytest.approx(100 * (10 * 0.25 + 1) / 11)  # 31.8%
    assert calls(OPEN, 50, w) - calls(OPEN, 50, 25.0) <= 0.05


# --- C/D. Staged evidence widens progressively, per context --------------------


@pytest.mark.parametrize(
    "kind, feeder, p0",
    [
        (OPEN, report.feed_open_pattern, 0.25),
        (RESHOVE, report.feed_reshove_pattern, 0.15),
    ],
)
def test_staged_jams_widen_monotonically(kind, feeder, p0):
    model, first, seen, stages = ShoveModel(), 1, (0, 0), []
    for shoves, opps in ((1, 2), (3, 4), (5, 6)):
        # Feed only the new evidence of each cumulative stage.
        first = feeder(model, shoves - seen[0], opps - seen[1], 50, first)
        seen = (shoves, opps)
        est = model.estimate(kind, DEEP)
        assert (est["shoves"], est["opportunities"]) == (shoves, opps)
        assert est["posterior"] == pytest.approx((10 * p0 + shoves) / (10 + opps))
        stages.append(est["adaptive_width"])
    assert stages == sorted(stages) and stages[-1] > stages[0]
    other = RESHOVE if kind == OPEN else OPEN
    assert model.estimate(other, DEEP)["opportunities"] == 0  # never pooled


# --- E. 100% jammer at ~50bb ----------------------------------------------------


def test_relentless_jammer_widens_calls_but_price_still_rules():
    model = ShoveModel()
    report.feed_open_pattern(model, 20, 20, 50)
    w = width(model)
    ctx = report.shove_context(OPEN, 50)
    assert w == pytest.approx(75.0)
    assert calls(OPEN, 50, w) > calls(OPEN, 50, 25.0) + 0.20  # 10.9% -> 38.7%
    after = report.hand_calls(ctx, w, CLASSES)
    before = report.hand_calls(ctx, 25.0, CLASSES)
    for premium in ("AA", "KK", "QQ", "AKs", "AKo"):
        assert before[premium] == after[premium] == 1
    newly = [h for h in CLASSES if before[h] == 0 and after[h] == 1]
    assert {"QJs", "A5s", "KTo"} <= set(newly)
    for trash in ("72o", "32o", "83o", "J4o", "98s"):
        assert after[trash] == 0  # equity does not meet the 49% price


# --- F. A normal opponent stays near version 3 -----------------------------------


def test_sparse_realistic_shoving_stays_near_v3():
    model = ShoveModel()
    report.feed_open_pattern(model, 10, 40, 50)  # 25% jams, as the prior assumes
    w = width(model)
    assert w == 25.0
    model = ShoveModel()
    report.feed_open_pattern(model, 3, 40, 50)  # a tight opponent: floor holds
    assert width(model) == 25.0
    model = ShoveModel()
    report.feed_open_pattern(model, 12, 40, 50)  # slightly looser: small change
    assert abs(calls(OPEN, 50, width(model)) - calls(OPEN, 50, 25.0)) <= 0.02


# --- G. Short stacks keep their anti-overcall defence ------------------------------


@pytest.mark.parametrize("depth", [7, 10, 14, 18])
def test_short_stacks_unchanged_without_same_bucket_evidence(depth):
    model = ShoveModel()
    report.feed_open_pattern(model, 20, 20, 50)  # deep evidence only
    report.feed_reshove_pattern(model, 20, 20, 50, first=100)
    bucket = preflop.stack_bucket(depth)
    for kind in (OPEN, RESHOVE):
        est = model.estimate(kind, bucket)
        cfg = preflop.DEFAULT_PREFLOP
        table = cfg.open_shove_width if kind == OPEN else cfg.reshove_width
        assert est["adaptive_width"] == table[bucket]
    folds = preflop.weighted_frequencies(report.shove_context(OPEN, depth))
    assert folds[preflop.FOLD] >= 0.38  # PLUMBER's short-stack reference floor


def test_short_stack_evidence_adapts_only_its_bucket():
    model = ShoveModel()
    report.feed_open_pattern(model, 5, 5, 10)
    assert width(model, OPEN, 10) > 50.0
    assert width(model, OPEN, 50) == 25.0 and width(model, OPEN, 7) == 65.0


# --- H. Deduplication ------------------------------------------------------------


def test_duplicate_and_later_street_histories_count_once():
    model = ShoveModel()
    later = [(HERO, "check", 0, "flop"), (VILLAIN, "check", 0, "turn")]
    start, result = report.hand_messages(
        1, VILLAIN, (5000, 5000), [(VILLAIN, "raise", 5000), (HERO, "call", 0)], later
    )
    model.record_round_start(start)
    assert model.observe_round_result(result, HERO) is not None
    for _ in range(3):  # redelivery, e.g. after a reconnect
        model.record_round_start(start)
        assert model.observe_round_result(copy.deepcopy(result), HERO) is None
    est = model.estimate(OPEN, DEEP)
    assert (est["shoves"], est["opportunities"]) == (1, 1)


def test_turn_requests_never_count():
    from bots.chipzen.bot import SleightOfHandBot
    from tests.test_preflop import sdk_state, shove

    bot = SleightOfHandBot(seed=3)
    s = sdk_state(shove(50), ["As", "Kd"])
    for _ in range(5):
        bot.decide(s)
    assert bot.shove_model.stats == {} and bot.shove_model.processed == set()


def test_villain_fold_is_an_open_opportunity_without_a_shove():
    model = ShoveModel()
    report.feed(model, 1, VILLAIN, (5000, 5000), [(VILLAIN, "fold", 0)])
    est = model.estimate(OPEN, DEEP)
    assert (est["shoves"], est["opportunities"]) == (0, 1)


def test_partial_jam_is_not_a_shove_but_effective_stack_jam_is():
    model = ShoveModel()
    report.feed(model, 1, VILLAIN, (5000, 5000), [(VILLAIN, "raise", 4000)])
    assert model.estimate(OPEN, DEEP)["shoves"] == 0
    # Villain covers us: raising to our whole stack is a shove.
    report.feed(model, 2, VILLAIN, (5000, 20000), [(VILLAIN, "raise", 5000)])
    assert model.estimate(OPEN, DEEP)["shoves"] == 1


def test_our_own_all_in_is_not_a_reshove_opportunity():
    model = ShoveModel()
    report.feed(
        model, 1, HERO, (5000, 5000), [(HERO, "raise", 5000), (VILLAIN, "call", 0)]
    )
    assert model.estimate(RESHOVE, DEEP)["opportunities"] == 0


# --- I. Match reset ---------------------------------------------------------------


def test_fresh_bot_starts_from_the_prior():
    bot, _ = report.simulate_jammer(hands=10)
    assert width(bot.shove_model) > 25.0
    from bots.chipzen.bot import SleightOfHandBot

    fresh = SleightOfHandBot(seed=11)
    assert fresh.shove_model.stats == {} and width(fresh.shove_model) == 25.0


# --- J. Malformed input never poisons the model --------------------------------------


def _good():
    return report.hand_messages(
        7, VILLAIN, (5000, 5000), [(VILLAIN, "raise", 5000), (HERO, "fold", 0)]
    )


MALFORMED = {
    "no_round_start": lambda s, r: (None, r),
    "multiway_stacks": lambda s, r: (s["state"].update(stacks=[5000] * 3) or s, r),
    "missing_blind": lambda s, r: (s, r["result"]["action_history"].pop(1) and r),
    "unknown_action": lambda s, r: (
        s,
        r["result"]["action_history"].insert(2, report.entry(0, "straddle", 200)) or r,
    ),
    "bool_amount": lambda s, r: (
        s,
        r["result"]["action_history"][2].update(amount=True) or r,
    ),
    "bad_seat": lambda s, r: (s, r["result"]["action_history"][2].update(seat=5) or r),
    "wrong_first_actor": lambda s, r: (
        s,
        r["result"]["action_history"].insert(2, report.entry(HERO, "check", 0)) or r,
    ),
    "history_not_list": lambda s, r: (s, r["result"].update(action_history={}) or r),
    "result_missing": lambda s, r: (s, r.pop("result") and r),
    "no_round_key": lambda s, r: (
        s,
        (r.pop("round_id"), r["result"].pop("hand_number")) and r,
    ),
    "blinds_all_in": lambda s, r: (s["state"].update(stacks=[80, 5000]) or s, r),
}


@pytest.mark.parametrize("fault", sorted(MALFORMED))
def test_malformed_hands_leave_the_model_untouched(fault):
    model = ShoveModel()
    report.feed_open_pattern(model, 2, 3, 50)
    snapshot = (copy.deepcopy(model.stats), set(model.processed))
    start, result = _good()
    start, result = MALFORMED[fault](start, result)
    if start is not None:
        model.record_round_start(start)
    assert model.observe_round_result(result, HERO) is None
    assert (model.stats, model.processed) == snapshot


def test_unknown_seat_is_not_observed():
    model = ShoveModel()
    start, result = _good()
    model.record_round_start(start)
    assert model.observe_round_result(result, None) is None
    assert model.stats == {} and model.processed == set()


def test_parse_is_pure():
    start, result = _good()
    before = (copy.deepcopy(start), copy.deepcopy(result))
    obs = parse_hand(start["state"], result["result"], "k", HERO)
    assert obs.open_opportunity and obs.open_shove and obs.bucket == DEEP
    assert (start, result) == before


def test_bot_hooks_never_raise_on_garbage():
    from bots.chipzen.bot import SleightOfHandBot

    bot = SleightOfHandBot(seed=1)
    bot.on_match_start({"seats": [{"seat": 0, "is_self": True}]})
    # The SDK's own default on_round_start rejects a non-dict state before
    # our model is involved (Season 6 behaviour; the SDK contains hook
    # errors), so feed the model directly for that shape.
    bot.shove_model.record_round_start({"type": "round_start", "state": "nonsense"})
    bot.shove_model.record_round_start("not even a dict")
    bot.on_round_result({"type": "round_result", "result": {"action_history": 5}})
    bot.on_round_result({"round_id": "x", "result": {"hand_number": 1}})
    assert bot.shove_model.stats == {}


# --- Live exploit regression and recovery ----------------------------------------------


def test_live_exploit_replay_adapts_both_contexts_separately():
    _, records = report.simulate_jammer(hands=40)
    seen = [r for r in records if r["observed"]]
    opens = [r for r in seen if r["kind"] == OPEN]
    reshoves = [r for r in seen if r["kind"] == RESHOVE]
    assert opens and reshoves
    # Every villain-button hand is an open jam; reshoves need our open or limp.
    assert opens[-1]["opportunities"] == opens[-1]["shoves"] == 20
    assert reshoves[-1]["opportunities"] == reshoves[-1]["shoves"] == len(reshoves)
    for series in (opens, reshoves):
        widths = [r["adaptive_width"] for r in series]
        assert widths == sorted(widths)
    # No panic after the first jam; materially wider after sustained jams.
    first = opens[0]["adaptive_width"]
    assert first < 35.0
    assert calls(OPEN, 50, first) - calls(OPEN, 50, 25.0) <= 0.05
    assert calls(OPEN, 50, opens[-1]["adaptive_width"]) >= 0.35
    assert calls(RESHOVE, 50, reshoves[-1]["adaptive_width"]) >= 0.30
    ctx = report.shove_context(OPEN, 50)
    assert report.hand_calls(ctx, opens[-1]["adaptive_width"], ("72o",))["72o"] == 0


def test_bot_calls_wider_after_repeated_jams():
    from chipzen import Action

    from tests.test_preflop import sdk_state, shove

    bot, _ = report.simulate_jammer(hands=30)
    from bots.chipzen.bot import SleightOfHandBot

    fresh = SleightOfHandBot(seed=5)
    s = sdk_state(shove(50), ["Kh", "Td"])  # KTo: a v3 fold at 50bb
    assert fresh.decide(s) == Action.fold()
    assert bot.decide(s) == Action.call()


def test_recovery_returns_to_the_floor_without_decay():
    model = ShoveModel()
    first = report.feed_open_pattern(model, 5, 6, 50)
    assert width(model) == pytest.approx(100 * 7.5 / 16)  # 46.9%
    report.feed_open_pattern(model, 0, 14, 50, first=first)
    assert model.estimate(OPEN, DEEP)["posterior"] == pytest.approx(0.25)
    assert width(model) == 25.0
    report.feed_open_pattern(model, 0, 6, 50, first=first + 14)
    est = model.estimate(OPEN, DEEP)
    assert est["posterior"] == pytest.approx(7.5 / 36)  # 20.8% < floor
    assert est["adaptive_width"] == 25.0


# --- Trace ------------------------------------------------------------------------------


def test_trace_reports_shove_model_and_stays_decision_neutral(capsys):
    from bots.chipzen.bot import SleightOfHandBot
    from tests.test_preflop import sdk_state, shove

    def trained(trace):
        bot = SleightOfHandBot(seed=9, trace_preflop=trace)
        bot.on_match_start({"seats": [{"seat": HERO, "is_self": True}]})
        for i in range(6):
            start, result = report.hand_messages(
                i + 1,
                VILLAIN,
                (5000, 5000),
                [(VILLAIN, "raise", 5000), (HERO, "fold", 0)],
            )
            bot.on_round_start(start)
            bot.on_round_result(result)
        return bot

    quiet, loud = trained(False), trained(True)
    s = sdk_state(shove(50), ["Qs", "Jh"])
    assert [quiet.decide(s) for _ in range(10)] == [loud.decide(s) for _ in range(10)]
    lines = [json.loads(x) for x in capsys.readouterr().err.splitlines()]
    model = lines[-1]["shove_model"]
    assert model["kind"] == OPEN and model["baseline_width"] == 25.0
    assert (model["shoves"], model["opportunities"]) == (6, 6)
    assert model["adaptive_width"] == pytest.approx(100 * 8.5 / 16, abs=1e-3)
    assert model["prior_strength"] == 10.0
    assert '"Qs"' not in json.dumps(lines) and '"Jh"' not in json.dumps(lines)
