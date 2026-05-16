"""Genetic algorithm that evolves betting-strategy parameters.

Fitness of a genome is its mean win rate (mbb/hand) over simulated hands
against a *fixed* pool of opponents (the baseline agents). Keeping the
opponent pool fixed across generations -- rather than pure self-play --
keeps the fitness curve directly comparable generation-to-generation,
which is what makes "plot fitness over generations" a meaningful diagnostic
rather than a moving target. `coevolve_frac` optionally mixes in matches
against the previous generation's champion, adding a mild self-play
pressure without making fitness fully non-stationary.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from ..agents.base import Agent
from ..agents.baselines import RuleBasedAgent
from ..eval.harness import play_match
from ..policy.heuristic import PolicyParams
from .genome import crossover, mutate, random_genome


@dataclass
class GAConfig:
    population_size: int = 40
    n_generations: int = 30
    n_hands_per_opponent: int = 300
    elite_frac: float = 0.10
    tournament_size: int = 4
    mutation_rate: float = 0.25
    mutation_sigma_frac: float = 0.15
    crossover_rate: float = 0.7
    coevolve_frac: float = 0.25  # weight given to facing last gen's champion
    seed: int = 0


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    best_genome: PolicyParams


@dataclass
class GAResult:
    history: list[GenerationStats] = field(default_factory=list)
    best_genome: PolicyParams | None = None
    best_fitness: float = float("-inf")


def _tournament_select(
    population: list[PolicyParams], fitnesses: list[float], rng: random.Random, k: int
) -> PolicyParams:
    idxs = rng.sample(range(len(population)), min(k, len(population)))
    best_idx = max(idxs, key=lambda i: fitnesses[i])
    return population[best_idx]


class GeneticAlgorithm:
    def __init__(self, config: GAConfig, opponent_pool: dict[str, Callable[[], Agent]]):
        self.config = config
        self.opponent_pool = opponent_pool

    def evaluate_fitness(self, genome: PolicyParams, rng: random.Random, champion: PolicyParams | None) -> float:
        cfg = self.config
        contender = RuleBasedAgent(params=genome, rng=rng)
        scores = []
        for factory in self.opponent_pool.values():
            opponent = factory()
            result = play_match(contender, opponent, cfg.n_hands_per_opponent, rng)
            scores.append(result.mbb_a)
        baseline_score = sum(scores) / len(scores)

        if champion is not None and cfg.coevolve_frac > 0:
            champ_agent = RuleBasedAgent(params=champion, rng=rng)
            champ_result = play_match(contender, champ_agent, cfg.n_hands_per_opponent, rng)
            return (1 - cfg.coevolve_frac) * baseline_score + cfg.coevolve_frac * champ_result.mbb_a
        return baseline_score

    def run(self, callback: Callable[[GenerationStats], None] | None = None) -> GAResult:
        cfg = self.config
        rng = random.Random(cfg.seed)
        population = [random_genome(rng) for _ in range(cfg.population_size)]
        result = GAResult()
        champion: PolicyParams | None = None

        for gen in range(cfg.n_generations):
            fitnesses = [self.evaluate_fitness(g, rng, champion) for g in population]
            order = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)

            gen_best_genome = population[order[0]]
            gen_stats = GenerationStats(
                generation=gen,
                best_fitness=fitnesses[order[0]],
                mean_fitness=sum(fitnesses) / len(fitnesses),
                worst_fitness=fitnesses[order[-1]],
                best_genome=gen_best_genome,
            )
            result.history.append(gen_stats)
            if gen_stats.best_fitness > result.best_fitness:
                result.best_fitness = gen_stats.best_fitness
                result.best_genome = gen_best_genome
            champion = gen_best_genome
            if callback:
                callback(gen_stats)

            n_elite = max(1, int(cfg.elite_frac * cfg.population_size))
            elites = [population[i] for i in order[:n_elite]]
            new_population = list(elites)
            while len(new_population) < cfg.population_size:
                p1 = _tournament_select(population, fitnesses, rng, cfg.tournament_size)
                p2 = _tournament_select(population, fitnesses, rng, cfg.tournament_size)
                child = crossover(p1, p2, rng) if rng.random() < cfg.crossover_rate else p1
                child = mutate(child, rng, rate=cfg.mutation_rate, sigma_frac=cfg.mutation_sigma_frac)
                new_population.append(child)
            population = new_population

        return result
