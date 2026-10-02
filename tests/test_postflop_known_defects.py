"""Synthetic state assuming unmatched villain chips remain included in pot.

MODEL-ONLY CONFIRMED; live Chipzen reachability is INCONCLUSIVE. These passing
tests deliberately freeze the legacy defect, not correct poker or server
behavior. An approved pricing repair must update the affected expectations here
and in the six known-defect fixtures, preserving ordinary-pot/equity/RNG tests.
"""

import pytest

from tests.test_postflop_characterization import (
    BASELINE,
    FIXTURES,
    FLOAT_ABS,
    assert_expected,
    observe,
    state_for,
)


@pytest.mark.parametrize(
    "case_id",
    [key for key, case in FIXTURES.items() if case["category"] == "known_defect"],
)
def test_legacy_untrimmed_covering_shove_characterization(case_id):
    """Synthetic state assuming unmatched villain chips remain included in pot."""
    state = state_for(case_id)
    geometry = BASELINE["covering_shove"]
    assert state.phase in ("flop", "turn", "river")
    assert state.pot == geometry["supplied_pot"] == 6100
    assert state.your_stack == 600
    assert state.opponent_stacks == [0]
    assert state.to_call in (5900, 600)
    assert state.action_history[-1]["amount"] == geometry["villain_wager"] == 5900
    assert state.valid_actions == ["fold", "call"]

    affordable = min(state.to_call, state.your_stack)
    assert affordable == geometry["affordable_call"] == 600
    legacy_price = affordable / (state.pot + affordable)
    assert legacy_price == geometry["legacy_modeled_price"] == 6 / 67

    # Independent reference: settled pot 200 plus only 600 matchable villain
    # chips, plus hero's 600 call. Do not derive this via the production helper.
    contestable_pot = geometry["pot_before_wager"] + affordable
    assert contestable_pot == geometry["contestable_pot"] == 800
    assert state.pot - contestable_pot == geometry["unmatched"] == 5300
    reference_price = affordable / (geometry["pot_before_wager"] + 2 * affordable)
    assert reference_price == geometry["contestable_reference_price"] == 3 / 7
    assert legacy_price < reference_price

    actual, _ = observe(state)
    assert_expected(actual, FIXTURES[case_id]["expected"])
    # The observed threshold exposes the erroneous denominator today. A future
    # repair changes this assertion and these fixtures, not the reference price.
    assert actual["params"]["call_threshold"] == pytest.approx(
        legacy_price + 1 / 60, rel=0, abs=FLOAT_ABS
    )


@pytest.mark.parametrize(
    "raw, capped", [("F11", "F12"), ("F13", "F14"), ("F15", "F16")]
)
def test_capping_call_alone_leaves_legacy_denominator_defect(raw, capped):
    raw_state, capped_state = state_for(raw), state_for(capped)
    assert raw_state.to_call == 5900 and capped_state.to_call == 600
    assert raw_state.pot == capped_state.pot == 6100
    raw_result, _ = observe(raw_state)
    capped_result, _ = observe(capped_state)
    assert raw_result == capped_result
