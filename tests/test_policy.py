import pytest

from sleight_of_hand.engine.actions import ActionType
from sleight_of_hand.policy.heuristic import DEFAULT_PARAMS, action_probs, hand_strength

FOLD, CALL, RAISE = ActionType.FOLD, ActionType.CALL, ActionType.RAISE


def test_strength_ordering():
    assert hand_strength(0, -1) < hand_strength(1, -1) < hand_strength(2, -1)
    assert hand_strength(2, 2) > hand_strength(1, -1)  # K pair beats bare high card
    assert hand_strength(1, 2) < hand_strength(1, -1)  # bare Q worth less once a card is public

    # pairs strictly beat non-pairs of the same private rank
    assert hand_strength(2, 2) > hand_strength(2, 0)


def test_action_probs_sum_to_one_and_respect_legality():
    legal = [CALL, RAISE]
    probs = action_probs(2, -1, legal, to_call=0, params=DEFAULT_PARAMS)
    assert set(probs) <= set(legal)
    assert sum(probs.values()) == pytest.approx(1.0)

    legal2 = [CALL, FOLD, RAISE]
    probs2 = action_probs(0, -1, legal2, to_call=2, params=DEFAULT_PARAMS)
    assert set(probs2) <= set(legal2)
    assert sum(probs2.values()) == pytest.approx(1.0)


def test_strong_hand_bets_more_than_weak_hand():
    legal = [CALL, RAISE]
    p_strong = action_probs(2, -1, legal, to_call=0, params=DEFAULT_PARAMS)
    p_weak = action_probs(0, -1, legal, to_call=0, params=DEFAULT_PARAMS)
    assert p_strong.get(RAISE, 0) > p_weak.get(RAISE, 0)


def test_weak_hand_folds_more_than_strong_hand_facing_a_bet():
    legal = [CALL, FOLD, RAISE]
    p_strong = action_probs(2, -1, legal, to_call=2, params=DEFAULT_PARAMS)
    p_weak = action_probs(0, -1, legal, to_call=2, params=DEFAULT_PARAMS)
    assert p_weak.get(FOLD, 0) > p_strong.get(FOLD, 0)


def test_no_raise_when_illegal():
    legal = [CALL, FOLD]
    probs = action_probs(2, -1, legal, to_call=2, params=DEFAULT_PARAMS)
    assert RAISE not in probs
    assert sum(probs.values()) == pytest.approx(1.0)
