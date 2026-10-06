"""Belief math, reliable opportunities, scoped identities and lifecycle isolation."""

import copy
import io
import random
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from sleight_of_hand.holdem.agent import HoldemAgent
from sleight_of_hand.holdem.memory import OpponentBelief, OpponentMemory
from sleight_of_hand.holdem.observations import preflop_observations
from sleight_of_hand.holdem.opponent import parse_hand
from sleight_of_hand.holdem.profile_store import InMemoryProfileStore
from sleight_of_hand.holdem.profiles import (
    BeliefConfig,
    Counts,
    IdentityResolution,
    encode,
)
from tests.test_opponent_profiles import identity, profile


def hand(actions, button=1, number=1, postflop=False):
    start = {"stacks": [5000, 5000]}
    history = [
        {
            "seat": button,
            "action": "post_small_blind",
            "amount": 50,
            "phase": "preflop",
        },
        {
            "seat": 1 - button,
            "action": "post_big_blind",
            "amount": 100,
            "phase": "preflop",
        },
    ]
    history += [
        {"seat": s, "action": a, "amount": n, "phase": "preflop"} for s, a, n in actions
    ]
    if postflop:
        history.append(
            {"seat": 1 - button, "action": "check", "amount": 0, "phase": "flop"}
        )
    return start, {"action_history": history, "hand_number": number}


def begin(memory, match="m", key="p-1", name="name"):
    memory.notify(
        "match_start",
        {
            "match_id": match,
            "seats": [
                {"seat": 0, "is_self": True},
                {"seat": 1, "participant_id": key, "display_name": name},
            ],
        },
        0,
    )


def feed(memory, number, pattern="open"):
    patterns = {
        "open": ([(1, "raise", 300), (0, "fold", 0)], False),
        "limp": ([(1, "call", 50), (0, "check", 0)], True),
        "tight": ([(1, "fold", 0)], False),
        "shove": ([(1, "raise", 5000), (0, "fold", 0)], False),
        "aggressive": (
            [(1, "raise", 300), (0, "raise", 900), (1, "raise", 5000), (0, "fold", 0)],
            False,
        ),
    }
    actions, postflop = patterns[pattern]
    start, result = hand(actions, number=number, postflop=postflop)
    key = f"h{number}"
    memory.record_start(key, start)
    return memory.observe_result(key, result, 0)


@pytest.mark.parametrize("n,expected", [(0, 0), (5, 5), (10, 10), (100000, 10)])
def test_historical_cap_boundaries(n, expected):
    belief = OpponentBelief(Counts(n, n), Counts(0, 0), BeliefConfig(1, 1, 10))
    assert belief.historical_ess == expected
    assert belief.alpha == 1 + expected and belief.beta == 1
    assert belief.effective_sample_size == expected


def test_posterior_formula_and_intervals():
    b = OpponentBelief(Counts(70, 100), Counts(2, 8), BeliefConfig(1, 1, 10))
    assert b.alpha == 10 and b.beta == 10 and b.mean == 0.5
    assert b.variance == 100 / (400 * 21)
    lo, hi = b.interval()
    assert lo <= 0.5 <= hi
    assert OpponentBelief(Counts(), Counts(), BeliefConfig()).interval() == (0, 1)
    for value in (0, 1, float("nan")):
        with pytest.raises(ValueError):
            b.interval(value)


def test_current_evidence_dominates_contradictory_history():
    historical = Counts(100000, 100000)
    weak = OpponentBelief(historical, Counts(0, 1), BeliefConfig())
    strong = OpponentBelief(historical, Counts(0, 1000), BeliefConfig())
    assert weak.mean > 0.8 and strong.mean < 0.02
    assert strong.historical == historical and strong.current == Counts(0, 1000)
    assert strong.interval()[1] < weak.interval()[0]


@pytest.mark.parametrize(
    "pattern,metric,expected",
    [
        ("limp", "btn_limp", True),
        ("open", "btn_open", True),
        ("tight", "btn_open", False),
        ("shove", "open_shove/5", True),
        ("aggressive", "reshove/5", True),
    ],
)
def test_synthetic_tendencies_move_without_policy(pattern, metric, expected):
    memory = OpponentMemory()
    begin(memory)
    before = memory.belief(metric).mean
    for i in range(40):
        assert feed(memory, i, pattern)
    belief = memory.belief(metric)
    assert (belief.mean > before) == expected
    assert belief.current.opportunities == 40
    assert memory.prior is None


def test_strategy_switch_changes_counts_without_fake_change_probability():
    m = OpponentMemory()
    begin(m)
    for i in range(20):
        feed(m, i, "open")
    old = m.belief("btn_open").mean
    for i in range(20, 100):
        feed(m, i, "tight")
    assert m.belief("btn_open").mean < old
    assert m.current["btn_open"] == Counts(20, 100)
    assert "change_probability" not in m.snapshot()


@pytest.mark.parametrize(
    "actions,postflop,want",
    [
        ([(0, "call", 50), (1, "check", 0)], True, {"bb_iso": 0, "bb_check_limp": 1}),
        (
            [(0, "call", 50), (1, "raise", 300), (0, "fold", 0)],
            False,
            {"bb_iso": 1, "bb_check_limp": 0},
        ),
        (
            [(0, "raise", 300), (1, "fold", 0)],
            False,
            {"bb_3bet": 0, "bb_fold_open": 1, "bb_call_open": 0},
        ),
        (
            [(0, "raise", 300), (1, "call", 200)],
            True,
            {"bb_3bet": 0, "bb_fold_open": 0, "bb_call_open": 1},
        ),
        (
            [(0, "raise", 300), (1, "raise", 900), (0, "fold", 0)],
            False,
            {"bb_3bet": 1, "bb_fold_open": 0, "bb_call_open": 0},
        ),
    ],
)
def test_bb_opportunity_predicates(actions, postflop, want):
    start, result = hand(actions, button=0, postflop=postflop)
    observations = dict(preflop_observations(start, result, "h", 0))
    for key, value in want.items():
        assert observations[key] == Counts(value, 1)
    assert "btn_open" not in observations


def test_covering_open_is_not_ordinary_bb_3bet_opportunity():
    start, result = hand([(0, "raise", 5000), (1, "call", 4900)], button=0)
    observations = dict(preflop_observations(start, result, "h", 0))
    assert not any(k.startswith("bb_") for k in observations)


@pytest.mark.parametrize("pattern", ["open", "limp", "tight", "shove", "aggressive"])
def test_parallel_shove_observations_equal_existing_pure_parser(pattern):
    m = OpponentMemory()
    begin(m)
    actions = {
        "open": [(1, "raise", 300), (0, "fold", 0)],
        "limp": [(1, "call", 50), (0, "check", 0)],
        "tight": [(1, "fold", 0)],
        "shove": [(1, "raise", 5000), (0, "fold", 0)],
        "aggressive": [
            (1, "raise", 300),
            (0, "raise", 900),
            (1, "raise", 5000),
            (0, "fold", 0),
        ],
    }[pattern]
    start, result = hand(actions, postflop=pattern == "limp")
    frozen = copy.deepcopy((start, result))
    legacy = parse_hand(start, result, "h", 0)
    obs = dict(preflop_observations(start, result, "h", 0))
    for kind, opportunity, success in [
        ("open_shove", legacy.open_opportunity, legacy.open_shove),
        ("reshove", legacy.reshove_opportunity, legacy.reshove),
    ]:
        assert obs.get(f"{kind}/{legacy.bucket}") == (
            Counts(int(success), 1) if opportunity else None
        )
    assert (start, result) == frozen


@pytest.mark.parametrize(
    "fault",
    [
        "bool_seat",
        "timeout",
        "amount",
        "missing_blind",
        "order",
        "forced",
        "partial",
        "unknown",
        "multiway",
        "unknown_phase",
        "too_long",
    ],
)
def test_malformed_hands_do_not_partially_update(fault):
    m = OpponentMemory()
    begin(m)
    feed(m, 0)
    start, result = hand([(1, "raise", 300), (0, "fold", 0)])
    h = result["action_history"]
    if fault == "bool_seat":
        h[-1]["seat"] = False
    if fault == "timeout":
        h[-1]["is_timeout"] = True
    if fault == "amount":
        h[-2]["amount"] = 90
    if fault == "missing_blind":
        h.pop(0)
    if fault == "order":
        h[-1]["seat"] = 1
    if fault == "forced":
        start["stacks"][0] = 100
    if fault == "partial":
        h.pop()
    if fault == "unknown":
        h[-1]["action"] = "draw"
    if fault == "multiway":
        start["stacks"].append(5000)
    if fault == "unknown_phase":
        h[-1]["phase"] = "unknown"
    if fault == "too_long":
        result["action_history"] = h * 100
    m.record_start("bad", start)
    before = copy.deepcopy((m.current, m.processed, m.starts))
    assert not m.observe_result("bad", result, 0)
    assert (m.current, m.processed, m.starts) == before


def test_round_dedup_reconnect_new_match_end_and_pending_bound():
    m = OpponentMemory()
    begin(m)
    assert feed(m, 1)
    before = m.snapshot()
    begin(m, name="renamed")
    m.notify("reconnected", {"match_id": "m"}, 0)
    assert m.snapshot() == before
    assert not feed(m, 1)
    for i in range(30):
        m.record_start(str(i), {"stacks": [5000, 5000]})
    assert list(m.starts) == list(map(str, range(14, 30)))
    m.notify("match_end", {"match_id": "m"}, 0)
    assert not m.starts and not feed(m, 40)
    begin(m, match="new")
    assert not m.current and not m.processed and feed(m, 1)


def test_cross_match_participant_never_loads_store():
    store = Mock()
    store.load.side_effect = AssertionError("must not load")
    m = OpponentMemory(store)
    for name in ["first", "renamed"]:
        begin(m, match=name, key="same-participant", name=name)
        assert feed(m, 1)
        assert m.prior is None
    store.load.assert_not_called()
    store.save.assert_not_called()


def test_persistent_prior_load_is_explicit_and_once_per_match():
    store = InMemoryProfileStore()
    store.save(profile())
    spy = Mock(wraps=store)
    m = OpponentMemory(spy)
    resolution = IdentityResolution(identity(), "fixture-proven", 1)
    m.begin_match("m", resolution)
    assert m.prior == profile()
    m.begin_match("m", replace(resolution, identity=identity(name="new")))
    assert spy.load.call_count == 1
    feed(m, 1)
    assert m.belief("btn_open").historical == Counts(7, 10)
    assert m.belief("btn_open").current == Counts(1, 1)
    m.begin_match("n", resolution)
    assert spy.load.call_count == 2 and not m.current
    spy.save.assert_not_called()


@pytest.mark.parametrize(
    "loaded", [None, "malformed", RuntimeError("SECRET_TOKEN_SENTINEL")]
)
def test_failed_or_missing_prior_continues_generic_match_learning(loaded):
    store = Mock()
    if isinstance(loaded, Exception):
        store.load.side_effect = loaded
    else:
        store.load.return_value = loaded
    m = OpponentMemory(store)
    m.begin_match("m", IdentityResolution(identity(), "trusted"))
    assert m.prior is None and feed(m, 1)
    assert b"SECRET" not in encode(m.snapshot())


def test_identity_conflict_fails_closed_without_reassigning_evidence():
    m = OpponentMemory()
    begin(m)
    feed(m, 1)
    begin(m, key="different")
    assert m.identity_conflict and m.prior is None and not m.current
    assert not feed(m, 2)


def test_anonymous_opponent_still_learns_locally():
    m = OpponentMemory()
    begin(m, key=None)
    assert m.identity.key is None and feed(m, 1)
    assert m.current["btn_open"] == Counts(1, 1)


def test_constructor_new_dependency_has_no_effects_or_singleton():
    rng = random.Random(17)
    before = rng.getstate()
    store = Mock()
    resolver = Mock(side_effect=AssertionError("resolver at construction"))
    forbidden = AssertionError("constructor side effect")
    with (
        patch("builtins.open", side_effect=forbidden),
        patch("pathlib.Path.open", side_effect=forbidden),
        patch("time.time", side_effect=forbidden),
        patch("random.Random.random", side_effect=forbidden),
        patch("socket.create_connection", side_effect=forbidden),
        patch("os.getenv", side_effect=forbidden),
        redirect_stderr(io.StringIO()) as err,
        redirect_stdout(io.StringIO()) as out,
    ):
        memory = OpponentMemory(store, identity_resolver=resolver)
        agent = HoldemAgent(rng, opponent_memory=memory)
        a, b = HoldemAgent(rng), HoldemAgent(rng)
    assert agent.opponent_memory is memory
    assert a.opponent_memory is not b.opponent_memory
    assert before == rng.getstate() and agent.shove_model.stats == {}
    assert not err.getvalue() and not out.getvalue()
    assert not store.mock_calls and not resolver.mock_calls


def test_diagnostics_allowlist_excludes_private_fields():
    m = OpponentMemory()
    begin(m, name="PLAYER_NAME_SENTINEL")
    start, result = hand([(1, "raise", 5000), (0, "fold", 0)])
    for payload in [start, result, *result["action_history"]]:
        payload.update(
            token="SECRET_TOKEN_SENTINEL",
            ticket="SECRET_TICKET_SENTINEL",
            url="SECRET_URL_SENTINEL",
            hole_cards=["PRIVATE_CARD_SENTINEL"],
            unknown="UNKNOWN_PRIVATE_SENTINEL",
        )
    m.record_start("h", start)
    m.observe_result("h", result, 0)
    output = encode(m.snapshot())
    assert b"SENTINEL" not in output and b"hole_cards" not in output
    assert m.starts == {}


def test_default_agent_memory_is_fresh_inert_and_never_persists():
    from sleight_of_hand.holdem.profile_store import NullProfileStore

    rng = random.Random(5)
    before = rng.getstate()
    a, b = HoldemAgent(rng), HoldemAgent(rng)
    assert rng.getstate() == before
    assert isinstance(a.opponent_memory, OpponentMemory)
    assert a.opponent_memory is not b.opponent_memory
    assert isinstance(a.opponent_memory.store, NullProfileStore)
    forbidden = AssertionError("default memory I/O")
    with (
        patch.object(NullProfileStore, "load", side_effect=forbidden),
        patch.object(NullProfileStore, "save", side_effect=forbidden),
        patch(
            "sleight_of_hand.holdem.profile_store.atomic_write", side_effect=forbidden
        ),
        patch("builtins.open", side_effect=forbidden),
        patch("pathlib.Path.open", side_effect=forbidden),
    ):
        # Match-scoped participant identity: learning without any store lookup.
        begin(a.opponent_memory)
        for number in range(1, 6):
            assert feed(a.opponent_memory, number)
        a.opponent_memory.notify("match_end", {"match_id": "m"}, 0)
    assert a.opponent_memory.prior is None
    assert a.opponent_memory.current["btn_open"] == Counts(5, 5)
    assert rng.getstate() == before


def test_store_lookup_requires_persistent_identity():
    store = Mock()
    store.load.return_value = None
    m = OpponentMemory(store)
    begin(m)  # participant_id -> match scope
    assert not store.load.called
    m = OpponentMemory(
        store, identity_resolver=lambda *a: IdentityResolution(identity(), "map", 1)
    )
    begin(m)
    store.load.assert_called_once_with(identity())
    assert not store.save.called
