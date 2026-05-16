"""The GA genome is exactly the parametrized heuristic policy's five
strategy parameters (local search operators act over a real-valued
genome; the phenotype is a full Leduc betting strategy)."""

from __future__ import annotations

import random

from ..policy.heuristic import PolicyParams

GENE_BOUNDS: dict[str, tuple[float, float]] = {
    "value_bet_threshold": (0.0, 1.0),
    "call_threshold": (0.0, 1.0),
    "bluff_freq": (0.0, 0.6),
    "aggression": (0.0, 1.0),
    "steepness": (1.0, 20.0),
}
GENE_NAMES = list(GENE_BOUNDS)


def random_genome(rng: random.Random) -> PolicyParams:
    return PolicyParams(**{name: rng.uniform(lo, hi) for name, (lo, hi) in GENE_BOUNDS.items()})


def mutate(genome: PolicyParams, rng: random.Random, rate: float = 0.25, sigma_frac: float = 0.15) -> PolicyParams:
    """Gaussian creep mutation: each gene independently has probability
    `rate` of being perturbed by noise scaled to its own range."""
    values = {}
    for name, (lo, hi) in GENE_BOUNDS.items():
        v = getattr(genome, name)
        if rng.random() < rate:
            v = v + rng.gauss(0.0, sigma_frac * (hi - lo))
            v = max(lo, min(hi, v))
        values[name] = v
    return PolicyParams(**values)


def crossover(a: PolicyParams, b: PolicyParams, rng: random.Random) -> PolicyParams:
    """Uniform crossover: each gene independently inherited from one
    parent or the other."""
    values = {name: (getattr(a, name) if rng.random() < 0.5 else getattr(b, name)) for name in GENE_NAMES}
    return PolicyParams(**values)
