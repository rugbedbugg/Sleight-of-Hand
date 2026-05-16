import random

import pytest

from sleight_of_hand.bayes.opponent_model import BayesianOpponentModel
from sleight_of_hand.engine.actions import ActionType
from sleight_of_hand.engine.game import LeducGame
from sleight_of_hand.policy.heuristic import DEFAULT_PARAMS, action_probs, sample_action

CALL, RAISE = ActionType.CALL, ActionType.RAISE


def test_prior_matches_deck_composition():
    model = BayesianOpponentModel(my_card=2)  # I hold a K
    # opponent's unseen cards: J,J,Q,Q,K (5 cards) -> P(K)=1/5, P(J)=P(Q)=2/5
    assert model.prob(0) == pytest.approx(0.4)
    assert model.prob(1) == pytest.approx(0.4)
    assert model.prob(2) == pytest.approx(0.2)
    assert sum(model.belief.values()) == pytest.approx(1.0)


def test_belief_always_normalized_after_updates():
    model = BayesianOpponentModel(my_card=0)
    state = LeducGame.deal_hand_with_cards(0, 2)
    model.update_on_action(state, opponent=1, action=RAISE)
    assert sum(model.belief.values()) == pytest.approx(1.0)
    model.update_on_public_card(1)
    assert sum(model.belief.values()) == pytest.approx(1.0)


def test_raise_shifts_belief_toward_stronger_cards():
    # Under DEFAULT_PARAMS, a K raises more often than a J. Observing a
    # raise should therefore increase our belief in K relative to J.
    model = BayesianOpponentModel(my_card=1)  # I hold a Q, opponent has J or K unseen (2 copies each)
    state = LeducGame.deal_hand_with_cards(1, -1)
    prior_k = model.prob(2)
    model.update_on_action(state, opponent=1, action=RAISE)
    assert model.prob(2) > prior_k


def test_public_card_update_reduces_belief_in_matching_hypothesis():
    # If I hold J and the public card comes out Q, the hypothesis "opponent
    # holds the other Q" should become less likely than it was, relative to
    # hypotheses that don't compete for that last Q copy.
    model = BayesianOpponentModel(my_card=0)
    prior_q = model.prob(1)
    model.update_on_public_card(1)
    assert model.prob(1) < prior_q


def test_belief_convergence_toward_true_card_on_average():
    # Self-consistency check: if the opponent truly plays according to the
    # same policy used as the likelihood model, repeated observations
    # should, on average, sharpen belief toward the opponent's true card.
    rng = random.Random(0)
    true_card = 2  # opponent secretly holds K
    n_trials = 400
    prior_true, post_true = [], []
    for _ in range(n_trials):
        my_card = rng.choice([0, 1])
        model = BayesianOpponentModel(my_card=my_card)
        prior_true.append(model.prob(true_card))
        state = LeducGame.deal_hand_with_cards(my_card, true_card)
        legal = LeducGame.legal_actions(state)
        to_call = state.round_state.to_call(1)
        probs = action_probs(true_card, -1, legal, to_call, params=DEFAULT_PARAMS)
        action = sample_action(rng, probs)
        model.update_on_action(state, opponent=1, action=action)
        post_true.append(model.prob(true_card))

    assert sum(post_true) / n_trials > sum(prior_true) / n_trials
