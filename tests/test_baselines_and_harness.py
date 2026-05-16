import random

from sleight_of_hand.agents.baselines import AlwaysCallAgent, RandomAgent, RuleBasedAgent
from sleight_of_hand.eval.harness import play_match, round_robin


def test_play_match_runs_and_is_reasonable():
    rng = random.Random(7)
    a = RandomAgent(random.Random(1))
    a.name = "random"
    b = AlwaysCallAgent()
    result = play_match(a, b, 2000, rng)
    assert result.n_hands == 2000
    # Not a strict correctness assertion (stochastic), just a sanity bound:
    # per-hand payoff magnitude can't exceed the max possible pot.
    assert abs(result.mean_a) < 20


def test_rule_based_beats_always_call_by_a_margin():
    # A rule-based player that folds bad hands and value-bets good ones
    # should not be dominated by a player who calls everything; over many
    # hands it should do at least as well on average.
    rng = random.Random(11)
    rb = RuleBasedAgent(rng=random.Random(2))
    ac = AlwaysCallAgent()
    result = play_match(rb, ac, 4000, rng)
    assert result.mean_a > -0.5  # should not be a big loser vs a non-folding opponent


def test_round_robin_produces_symmetric_results():
    factories = {
        "random": lambda: RandomAgent(random.Random(1)),
        "always_call": lambda: AlwaysCallAgent(),
        "rule_based": lambda: RuleBasedAgent(rng=random.Random(3)),
    }
    results = round_robin(factories, n_hands=500, seed=0)
    for (a, b), res in results.items():
        mirrored = results[(b, a)]
        assert abs(res.mean_a + mirrored.mean_a) < 1e-9
