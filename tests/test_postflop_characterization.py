"""SDK-independent snapshots: invariants and explicitly named current limitations.

These are characterization tests, not claims of optimal poker play. Covering
shove snapshots live in test_postflop_known_defects.py so a future pricing repair
can change those expectations without weakening unrelated regression coverage.
"""

import copy
import hashlib
import io
import json
import random
import subprocess
import sys
from contextlib import redirect_stderr
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sleight_of_hand.holdem import agent as core
from sleight_of_hand.holdem.decision import Decision
from sleight_of_hand.holdem.equity import estimate_equity
from sleight_of_hand.policy.heuristic import DEFAULT_PARAMS

BASELINE = json.loads(
    (Path(__file__).parent / "fixtures" / "postflop_baseline.json").read_text()
)
FIXTURES = {case["id"]: case for case in BASELINE["fixtures"]}
FLOAT_ABS = 1e-14  # Explicit absolute-only tolerance; never pytest's loose default.


def state_for(case_id):
    return SimpleNamespace(**copy.deepcopy(FIXTURES[case_id]["state"]))


def rng_hash(rng):
    return hashlib.sha256(repr(rng.getstate()).encode("utf-8")).hexdigest()


def observe(state, agent=None):
    """Observe actual core calls without substituting equity or probabilities."""
    if agent is None:
        agent = core.HoldemAgent(
            random.Random(BASELINE["seed"]), samples=BASELINE["samples"]
        )
    observed = {"equity": None, "params": None, "probs": None, "mapped_raise_to": None}
    work = {"equity_calls": 0, "probability_calls": 0}
    real_probs = core.action_probs_from_strength
    real_raise = core.raise_amount

    def equity(*args):
        work["equity_calls"] += 1
        assert args[-1] == BASELINE["samples"]
        observed["equity"] = estimate_equity(*args)
        return observed["equity"]

    def probabilities(strength, legal, to_call, params):
        work["probability_calls"] += 1
        observed["params"] = asdict(params)
        result = real_probs(strength, legal, to_call, params)
        observed["probs"] = {action.name: value for action, value in result.items()}
        return result

    def raise_to(state, aggression):
        observed["mapped_raise_to"] = real_raise(state, aggression)
        return observed["mapped_raise_to"]

    stderr = io.StringIO()
    with (
        patch.object(core, "action_probs_from_strength", side_effect=probabilities),
        patch.object(core, "raise_amount", side_effect=raise_to),
        redirect_stderr(stderr),
    ):
        action = agent.decide(state, equity_estimator=equity)
    observed.update(
        action=asdict(action), stderr=stderr.getvalue(), rng_hash=rng_hash(agent.rng)
    )
    return observed, work


def assert_expected(actual, expected):
    assert actual.keys() == expected.keys()
    for key in ("equity", "action", "stderr", "rng_hash", "mapped_raise_to"):
        assert actual[key] == expected[key], key
    for key in ("params", "probs"):
        if expected[key] is None:
            assert actual[key] is None
        else:
            # Probability iteration order also affects the seeded action choice.
            assert list(actual[key]) == list(expected[key]), key
            assert actual[key] == pytest.approx(expected[key], rel=0, abs=FLOAT_ABS)


def test_fixture_contract_and_default_configuration():
    assert list(FIXTURES) == [f"F{i:02}" for i in range(1, 29)]
    assert (BASELINE["seed"], BASELINE["samples"]) == (17, 128)
    assert asdict(DEFAULT_PARAMS) == BASELINE["params"]
    assert rng_hash(random.Random(17)) == BASELINE["initial_rng_hash"]
    assert len(BASELINE["mixed_sequence"]) == 7


@pytest.mark.parametrize(
    "case_id",
    [key for key, case in FIXTURES.items() if case["category"] == "invariant"],
)
def test_regression_fixture(case_id):
    actual, _ = observe(state_for(case_id))
    assert_expected(actual, FIXTURES[case_id]["expected"])


@pytest.mark.parametrize(
    "case_id",
    [key for key, case in FIXTURES.items() if case["category"] == "current_limitation"],
)
def test_current_limitation_fixture(case_id):
    """Freeze today's behavior, including limitations; not a permanent ideal."""
    actual, _ = observe(state_for(case_id))
    assert_expected(actual, FIXTURES[case_id]["expected"])


@pytest.mark.parametrize(
    "case_id, wager, price, call_threshold, value_threshold",
    [
        ("F07", 75, 1 / 6, 11 / 60, 11 / 20),
        ("F08", 150, 1 / 4, 4 / 15, 11 / 20),
        ("F09", 300, 1 / 3, 7 / 20, 11 / 20),
        ("F10", 600, 2 / 5, 5 / 12, 37 / 60),
    ],
)
def test_ordinary_price_ladder(case_id, wager, price, call_threshold, value_threshold):
    state = state_for(case_id)
    assert state.pot == 300 + wager
    assert state.to_call == wager < state.your_stack
    assert wager / (300 + 2 * wager) == price
    actual, _ = observe(state)
    assert actual["params"]["call_threshold"] == pytest.approx(
        call_threshold, rel=0, abs=FLOAT_ABS
    )
    assert actual["params"]["value_bet_threshold"] == pytest.approx(
        value_threshold, rel=0, abs=FLOAT_ABS
    )


@pytest.mark.parametrize(
    "case_id, candidate", [("F03", 200), ("F10", 2000), ("F27", 80), ("F28", None)]
)
def test_raise_mapping_boundaries(case_id, candidate):
    actual, _ = observe(state_for(case_id))
    assert actual["mapped_raise_to"] == candidate
    if candidate is None:
        assert "RAISE" not in actual["probs"]


def test_raise_to_includes_existing_street_contribution():
    state = state_for("F04")
    # A settled pot of 400, hero's 100 and villain's raise-to 300 give pot 800.
    # Hero owes 200; a raise-to of 1300 adds 1200, not another 1300 or 1400.
    state.pot, state.to_call = 800, 200
    state.your_stack, state.opponent_stacks = 1900, [1700]
    state.min_raise, state.max_raise = 500, 2000
    state.valid_actions = ["fold", "call", "raise"]
    state.action_history = [
        {"seat": 0, "action": "raise", "amount": 100, "phase": "turn"},
        {"seat": 1, "action": "raise", "amount": 300, "phase": "turn"},
    ]
    actual, _ = observe(state)
    assert actual["mapped_raise_to"] == 1300
    assert actual["action"] == {"action": "raise", "amount": 1300}


def test_single_supported_action_skips_equity_and_rng():
    agent = core.HoldemAgent(random.Random(17), samples=128)
    before = agent.rng.getstate()

    def forbidden_equity(*args):
        pytest.fail("single supported action must not estimate equity")

    assert agent.decide(
        state_for("F25"), equity_estimator=forbidden_equity
    ) == Decision("check")
    assert agent.rng.getstate() == before


def test_invalid_card_fallback_preserves_stderr_action_and_rng():
    agent = core.HoldemAgent(random.Random(17), samples=128)
    before = agent.rng.getstate()
    actual, work = observe(state_for("F26"), agent)
    assert_expected(actual, FIXTURES["F26"]["expected"])
    assert work == {"equity_calls": 1, "probability_calls": 0}
    assert agent.rng.getstate() == before


@pytest.mark.parametrize("other", ["F20", "F21", "F22"])
def test_current_history_invariance_characterization(other):
    """Raiser/caller, limp and 3-bet lines currently collapse, not necessarily forever."""
    first, _ = observe(state_for("F19"))
    second, _ = observe(state_for(other))
    assert first == second


@pytest.mark.parametrize("first, second", [("F01", "F02"), ("F05", "F06")])
def test_current_equity_compression_characterization(first, second):
    """Seeded estimates coincide; this does not claim identical true equity."""
    first_result, _ = observe(state_for(first))
    second_result, _ = observe(state_for(second))
    assert first_result == second_result


@pytest.mark.parametrize("other", ["F23", "F24"])
def test_current_position_and_street_label_invariance_characterization(other):
    """Intentionally inconsistent probes isolate fields, not reachable states."""
    first, _ = observe(state_for("F19"))
    second, _ = observe(state_for(other))
    assert first == second


def test_mixed_sequence_preserves_rng_continuity_after_every_decision():
    agent = core.HoldemAgent(random.Random(17), samples=128)
    for step in BASELINE["mixed_sequence"]:
        actual, _ = observe(state_for(step["fixture"]), agent)
        assert_expected(actual, step["expected"])


def test_fixture_replay_without_chipzen_sdk():
    # Fresh interpreter: no SDK cached by the existing adapter test modules.
    program = """
import sys
sys.modules['chipzen'] = None
sys.modules['bots.chipzen'] = None
from tests.test_postflop_characterization import FIXTURES, observe, state_for, assert_expected
for key, case in FIXTURES.items():
    actual, _ = observe(state_for(key))
    assert_expected(actual, case['expected'])
assert not any(name.startswith('chipzen.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
