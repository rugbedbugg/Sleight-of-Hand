"""Round-robin win-rate matrix.

Plays every agent against every other agent (random, always-call,
rule-based, the GA-tuned strategy if a saved genome is available, and the
full Bayes+search agent) and reports mbb/hand, headline result for the
report. Saves a CSV table and a heatmap figure to results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sleight_of_hand.agents.baselines import AlwaysCallAgent, RandomAgent, RuleBasedAgent
from sleight_of_hand.agents.bayes_search_agent import BayesSearchAgent
from sleight_of_hand.eval.harness import round_robin
from sleight_of_hand.eval.plotting import diverging_cmap, new_figure
from sleight_of_hand.policy.heuristic import PolicyParams

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def build_agent_factories(ga_genome: PolicyParams | None):
    factories = {
        "random": lambda: RandomAgent(random.Random()),
        "always_call": lambda: AlwaysCallAgent(),
        "rule_based": lambda: RuleBasedAgent(rng=random.Random()),
        "bayes_search": lambda: BayesSearchAgent(rng=random.Random()),
        "bayes_minimax": lambda: BayesSearchAgent(
            rng=random.Random(), name="bayes_minimax", search_mode="minimax"
        ),
    }
    if ga_genome is not None:
        factories["ga_tuned"] = lambda: RuleBasedAgent(params=ga_genome, rng=random.Random(), name="ga_tuned")
    return factories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--genome", type=str, default=os.path.join(RESULTS_DIR, "ga_best_genome.json"))
    args = parser.parse_args()

    ga_genome = None
    if os.path.exists(args.genome):
        with open(args.genome) as f:
            ga_genome = PolicyParams(**json.load(f)["params"])
        print(f"Loaded GA-tuned genome from {args.genome}")
    else:
        print(f"No saved GA genome at {args.genome}; run scripts/run_ga.py first for the full matrix.")

    factories = build_agent_factories(ga_genome)
    names = list(factories.keys())
    print(f"Round-robin: {names}, {args.hands} hands per pairing")
    results = round_robin(factories, n_hands=args.hands, seed=args.seed)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "win_rate_matrix.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent"] + names)
        for a in names:
            row = [a]
            for b in names:
                if a == b:
                    row.append("")
                else:
                    row.append(f"{results[(a, b)].mbb_a:.1f}")
            writer.writerow(row)
    print(f"Wrote {csv_path}")

    matrix = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            matrix[i, j] = np.nan if a == b else results[(a, b)].mbb_a

    fig, ax = new_figure(figsize=(1.4 * len(names) + 2, 1.2 * len(names) + 2))
    vmax = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, cmap=diverging_cmap(), vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("Round-robin win rate: row's mbb/hand vs column")
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=8, color="#0b0b0b")
    fig.colorbar(im, ax=ax, label="mbb/hand (row vs column)", shrink=0.8)
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "win_rate_matrix.png")
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Wrote {fig_path}")

    print("\nHeadline mbb/hand (mean vs the rest of the field):")
    for a in names:
        others = [results[(a, b)].mbb_a for b in names if b != a]
        print(f"  {a:14s} {sum(others) / len(others):+8.1f} mbb/hand")


if __name__ == "__main__":
    main()
