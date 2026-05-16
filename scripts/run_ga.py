"""Run the genetic algorithm, save the fitness curve plot and
the best-evolved genome (results/ga_best_genome.json), for use by the
other experiment scripts and the demo.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sleight_of_hand.agents.baselines import AlwaysCallAgent, RandomAgent, RuleBasedAgent
from sleight_of_hand.eval.plotting import CATEGORICAL, new_figure
from sleight_of_hand.ga.evolve import GAConfig, GeneticAlgorithm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--hands-per-opponent", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    opponent_pool = {
        "always_call": lambda: AlwaysCallAgent(),
        "random": lambda: RandomAgent(random.Random(123)),
        "rule_based": lambda: RuleBasedAgent(rng=random.Random(456)),
    }
    config = GAConfig(
        population_size=args.population,
        n_generations=args.generations,
        n_hands_per_opponent=args.hands_per_opponent,
        seed=args.seed,
    )
    ga = GeneticAlgorithm(config, opponent_pool)

    def report(stats):
        print(
            f"gen {stats.generation:3d}  best={stats.best_fitness:+8.1f} mbb/hand  "
            f"mean={stats.mean_fitness:+8.1f}  worst={stats.worst_fitness:+8.1f}"
        )

    result = ga.run(callback=report)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    genome_path = os.path.join(RESULTS_DIR, "ga_best_genome.json")
    with open(genome_path, "w") as f:
        json.dump({"params": dataclasses.asdict(result.best_genome), "fitness_mbb_per_hand": result.best_fitness}, f, indent=2)
    print(f"\nBest genome: {result.best_genome}")
    print(f"Wrote {genome_path}")

    gens = [s.generation for s in result.history]
    best = [s.best_fitness for s in result.history]
    mean = [s.mean_fitness for s in result.history]
    worst = [s.worst_fitness for s in result.history]

    fig, ax = new_figure()
    ax.fill_between(gens, worst, best, color=CATEGORICAL[0], alpha=0.12, label="population range")
    ax.plot(gens, best, color=CATEGORICAL[0], linewidth=2, label="best")
    ax.plot(gens, mean, color=CATEGORICAL[2], linewidth=2, linestyle="--", label="mean")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness (mbb/hand vs baseline pool)")
    ax.set_title("GA fitness over generations")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "ga_fitness_curve.png")
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
