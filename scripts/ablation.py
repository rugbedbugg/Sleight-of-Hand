"""Ablation of the Bayesian opponent model.

Compares the full pipeline (search using belief updated from observed
opponent actions) against the same search using only the deck-combinatoric
prior (no action-based updates) -- quantifying what the Bayesian layer is
actually worth, in mbb/hand, against several opponents.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sleight_of_hand.agents.baselines import RULE_BASED_PARAMS, RuleBasedAgent
from sleight_of_hand.agents.bayes_search_agent import BayesSearchAgent
from sleight_of_hand.eval.harness import play_match
from sleight_of_hand.eval.plotting import CATEGORICAL, new_figure

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main(n_hands: int = 4000, seed: int = 0):
    opponents = {
        "vs rule_based (bluffs 12%)": lambda: RuleBasedAgent(params=RULE_BASED_PARAMS, rng=random.Random(1)),
    }

    labels, with_bayes, without_bayes = [], [], []
    for label, opp_factory in opponents.items():
        rng = random.Random(seed)
        with_agent = BayesSearchAgent(use_bayes=True, rng=random.Random(1), name="search+bayes")
        result_with = play_match(with_agent, opp_factory(), n_hands, rng)

        rng = random.Random(seed)
        without_agent = BayesSearchAgent(use_bayes=False, rng=random.Random(1), name="search-only")
        result_without = play_match(without_agent, opp_factory(), n_hands, rng)

        print(f"{label}")
        print(f"  search + Bayes updates : {result_with}")
        print(f"  search, prior only     : {result_without}")
        print(f"  Bayes layer is worth   : {result_with.mbb_a - result_without.mbb_a:+.1f} mbb/hand\n")

        labels.append(label)
        with_bayes.append(result_with.mbb_a)
        without_bayes.append(result_without.mbb_a)

    import numpy as np

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = new_figure(figsize=(6, 4.5))
    ax.bar(x - width / 2, with_bayes, width, color=CATEGORICAL[0], label="search + Bayes updates")
    ax.bar(x + width / 2, without_bayes, width, color=CATEGORICAL[2], label="search, prior only (no Bayes)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mbb/hand")
    ax.set_title("Ablation: value of the Bayesian opponent model")
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig_path = os.path.join(RESULTS_DIR, "ablation.png")
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
