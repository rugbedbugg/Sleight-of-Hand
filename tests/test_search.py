import random

import pytest

from sleight_of_hand.agents.baselines import AlwaysCallAgent, RandomAgent, RuleBasedAgent
from sleight_of_hand.agents.bayes_search_agent import BayesSearchAgent
from sleight_of_hand.engine.actions import ActionType
from sleight_of_hand.engine.game import LeducGame
from sleight_of_hand.eval.harness import play_match
from sleight_of_hand.policy.heuristic import PolicyParams
from sleight_of_hand.search.expectiminimax import choose_action, node_value

FOLD, CALL, RAISE = ActionType.FOLD, ActionType.CALL, ActionType.RAISE


def test_node_value_matches_known_showdown():
    # Both players check both rounds; I hold K, opponent (fixed hypothesis)
    # holds J, public will be enumerated. My EV should be positive since K
    # beats J outright whenever there's no J on board pairing them, and even
    # then I never lose (worst case is a split only if board pairs *me*,
    # which can't happen here since I hold K and J is opponent's).
    state = LeducGame.deal_hand_with_cards(2, 0)
    state = LeducGame.apply_action(state, CALL)
    state = LeducGame.apply_action(state, CALL)
    assert state.awaiting_community
    val = node_value(state, my_player=0)
    assert val > 0


def test_choose_action_folds_weak_hand_facing_max_pressure():
    # I hold J, opponent (believed to be a K with certainty) has bet twice
    # (capped raises) on round 2 -- folding should dominate calling here.
    state = LeducGame.deal_hand_with_cards(0, 2)
    state = LeducGame.apply_action(state, CALL)
    state = LeducGame.apply_action(state, CALL)
    state = LeducGame.deal_community(state, 1)  # community = Q, doesn't pair either hand
    state = LeducGame.apply_action(state, RAISE)  # opponent bets round 2
    state = LeducGame.apply_action(state, RAISE)  # opponent raises again (I'm to act)
    legal = LeducGame.legal_actions(state)
    belief = {2: 1.0}  # certain the opponent holds K
    action, values = choose_action(state, my_player=0, belief=belief)
    assert action == FOLD
    assert values[FOLD] == pytest.approx(-5.0)  # ante(1) + round1 call(2) + round2 call(2)


def test_choose_action_value_bets_the_nuts():
    # I hold K, public is K (I have the nuts -- a pair of kings). With a
    # weak/uncertain opponent I should prefer betting over checking.
    state = LeducGame.deal_hand_with_cards(2, 0)
    state = LeducGame.apply_action(state, CALL)
    state = LeducGame.apply_action(state, CALL)
    state = LeducGame.deal_community(state, 2)
    legal = LeducGame.legal_actions(state)
    belief = {0: 0.4, 1: 0.4, 2: 0.2}
    action, values = choose_action(state, my_player=0, belief=belief)
    assert action == RAISE
    assert values[RAISE] >= values[CALL]


def test_bayes_search_agent_beats_always_call():
    rng = random.Random(5)
    search_agent = BayesSearchAgent(rng=random.Random(1))
    ac = AlwaysCallAgent()
    result = play_match(search_agent, ac, 600, rng)
    assert result.mean_a > 0


def test_bayes_search_agent_does_not_lose_badly_to_random():
    rng = random.Random(6)
    search_agent = BayesSearchAgent(rng=random.Random(2))
    r = RandomAgent(random.Random(3))
    result = play_match(search_agent, r, 600, rng)
    assert result.mean_a > 0


def test_minimax_value_never_exceeds_expectiminimax():
    # Against a fixed opponent policy model, assuming the worst-case reply
    # can only lower (never raise) our expected value relative to
    # best-responding to that policy. The inequality holds at every node,
    # so it must hold at the root. Here player 0 (the opponent, from
    # player 1's view) is first to act, giving an opponent node at the root.
    state = LeducGame.deal_hand_with_cards(2, 0)  # player 1 holds J, opponent holds K
    v_exp = node_value(state, my_player=1, mode="expectiminimax")
    v_min = node_value(state, my_player=1, mode="minimax")
    assert v_min <= v_exp + 1e-9


def test_expectiminimax_handles_extreme_opponent_params():
    # With extreme params the shared policy can drive an action's probability
    # to ~0 and omit it from the returned dict; expectiminimax must still sum
    # correctly over the opponent node rather than KeyError on a legal action.
    extreme = PolicyParams(
        value_bet_threshold=0.0, call_threshold=0.0, bluff_freq=1.0, aggression=1.0, steepness=1.0
    )
    state = LeducGame.deal_hand_with_cards(2, 0)
    v = node_value(state, my_player=1, opponent_params=extreme, mode="expectiminimax")
    assert isinstance(v, float)


def test_minimax_agent_plays_a_full_match():
    # The paranoid variant should still be a competent, non-crashing player
    # that beats a purely passive opponent.
    rng = random.Random(7)
    minimax_agent = BayesSearchAgent(rng=random.Random(4), search_mode="minimax")
    ac = AlwaysCallAgent()
    result = play_match(minimax_agent, ac, 600, rng)
    assert result.mean_a > 0
