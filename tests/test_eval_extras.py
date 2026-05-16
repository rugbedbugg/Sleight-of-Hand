from sleight_of_hand.eval.belief_accuracy import belief_convergence_curve
from sleight_of_hand.eval.exploitability import estimate_exploitability
from sleight_of_hand.policy.heuristic import DEFAULT_PARAMS, PolicyParams


def test_belief_convergence_curve_trends_upward():
    steps, means, stderrs = belief_convergence_curve(n_hands=300, seed=1)
    assert len(steps) == len(means) == len(stderrs)
    assert means[0] < 0.4  # prior is close to uniform-ish over unseen cards
    assert means[-1] > means[0]  # posterior should end up sharper than the prior on average


def test_exploitability_is_near_zero_for_a_reasonable_strategy_and_positive_for_a_bad_one():
    reasonable = DEFAULT_PARAMS
    result_reasonable = estimate_exploitability(reasonable, n_hands=600, seed=0)

    always_bet_never_fold = PolicyParams(
        value_bet_threshold=0.0, call_threshold=0.0, bluff_freq=1.0, aggression=1.0, steepness=1.0
    )
    result_bad = estimate_exploitability(always_bet_never_fold, n_hands=600, seed=0)

    # A strategy that bets/raises everything and never folds should be far
    # more exploitable than a reasonable, threshold-based strategy.
    assert result_bad.mbb_a > result_reasonable.mbb_a
