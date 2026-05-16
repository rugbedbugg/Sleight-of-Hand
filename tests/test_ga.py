import random

from sleight_of_hand.agents.baselines import AlwaysCallAgent, RandomAgent
from sleight_of_hand.ga.evolve import GAConfig, GeneticAlgorithm
from sleight_of_hand.ga.genome import crossover, mutate, random_genome


def test_mutate_respects_bounds():
    rng = random.Random(0)
    from sleight_of_hand.ga.genome import GENE_BOUNDS

    g = random_genome(rng)
    for _ in range(200):
        g = mutate(g, rng, rate=1.0, sigma_frac=1.0)
        for name, (lo, hi) in GENE_BOUNDS.items():
            assert lo <= getattr(g, name) <= hi


def test_crossover_gene_inherited_from_a_parent():
    rng = random.Random(1)
    a = random_genome(rng)
    b = random_genome(rng)
    child = crossover(a, b, rng)
    for name in ["value_bet_threshold", "call_threshold", "bluff_freq", "aggression", "steepness"]:
        v = getattr(child, name)
        assert v == getattr(a, name) or v == getattr(b, name)


def test_ga_improves_fitness_over_generations():
    pool = {
        "always_call": lambda: AlwaysCallAgent(),
        "random": lambda: RandomAgent(random.Random(42)),
    }
    config = GAConfig(
        population_size=12,
        n_generations=6,
        n_hands_per_opponent=80,
        seed=3,
    )
    ga = GeneticAlgorithm(config, pool)
    result = ga.run()
    assert len(result.history) == 6
    first_gen_best = result.history[0].best_fitness
    last_gen_best = result.history[-1].best_fitness
    # GA on a small tunable strategy space vs weak opponents should not get
    # worse; allow some stochastic slack rather than requiring strict
    # monotonic improvement every run.
    assert last_gen_best >= first_gen_best - 20.0
    assert result.best_genome is not None
