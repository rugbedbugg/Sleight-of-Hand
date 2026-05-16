"""Belief convergence plot -- the Bayesian posterior's probability mass on
the opponent's *true* card, averaged over many hands, at each successive
observation.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sleight_of_hand.eval.belief_accuracy import belief_convergence_curve, play_hand_with_belief_tracking
from sleight_of_hand.eval.plotting import CATEGORICAL, new_figure
from sleight_of_hand.policy.heuristic import DEFAULT_PARAMS
import random

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    steps, means, stderrs = belief_convergence_curve(n_hands=3000, seed=0)

    fig, ax = new_figure()
    lower = [m - 1.96 * s for m, s in zip(means, stderrs)]
    upper = [m + 1.96 * s for m, s in zip(means, stderrs)]
    ax.fill_between(steps, lower, upper, color=CATEGORICAL[0], alpha=0.15)
    ax.plot(steps, means, color=CATEGORICAL[0], linewidth=2, marker="o", markersize=4)
    ax.axhline(1 / 3, color=CATEGORICAL[5], linestyle=":", linewidth=1.5, label="uniform guess (1/3)")
    ax.set_xlabel("observation index within the hand (0 = prior)")
    ax.set_ylabel("P(true opponent card)  [mean over 3000 hands]")
    ax.set_title("Belief convergence: posterior mass on the opponent's true card")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig_path = os.path.join(RESULTS_DIR, "belief_convergence.png")
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Wrote {fig_path}")
    print(f"prior mean P(true card) = {means[0]:.3f}, final mean = {means[-1]:.3f}")

    print("\nExample single-hand trace:")
    rng = random.Random(42)
    for _ in range(20):
        trace = play_hand_with_belief_tracking(my_player=0, params=DEFAULT_PARAMS, rng=rng)
        if len(trace.labels) >= 3:
            break
    print(f"  true opponent card rank = {trace.true_card}")
    for label, p, h in zip(trace.labels, trace.prob_true_card, trace.entropy):
        print(f"    after {label:14s}  P(true card)={p:.3f}  belief entropy={h:.3f} bits")


if __name__ == "__main__":
    main()
