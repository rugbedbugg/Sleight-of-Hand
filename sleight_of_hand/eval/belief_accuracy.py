"""Belief-accuracy experiment: does the Bayesian posterior actually
sharpen toward the opponent's true card as a hand progresses?

We play hands where both seats use the same rule-based policy (so the
Bayes model's likelihood assumption exactly matches the generative
process -- the best case for calibration), track a fresh
`BayesianOpponentModel` for one designated "observer" seat, and record the
probability it assigns to the *true* opponent card after every
observation (each opponent action, and the community-card reveal).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..agents.baselines import RULE_BASED_PARAMS, RuleBasedAgent
from ..bayes.opponent_model import BayesianOpponentModel
from ..engine.game import LeducGame
from ..policy.heuristic import PolicyParams


@dataclass
class BeliefTrace:
    true_card: int
    labels: list[str]
    prob_true_card: list[float]
    entropy: list[float]


def play_hand_with_belief_tracking(
    my_player: int, params: PolicyParams, rng: random.Random
) -> BeliefTrace:
    state = LeducGame.new_hand(rng)
    opponent = 1 - my_player
    true_card = state.private[opponent]
    model = BayesianOpponentModel(state.private[my_player], params=params)

    agents = [None, None]
    agents[my_player] = RuleBasedAgent(params=params, rng=rng)
    agents[opponent] = RuleBasedAgent(params=params, rng=rng)

    while not state.done:
        player = state.to_act
        legal = LeducGame.legal_actions(state)
        action = agents[player].act(state, legal)
        if player == opponent:
            model.update_on_action(state, opponent=opponent, action=action)
        prev_public = state.public
        state = LeducGame.apply_action(state, action, rng=rng)
        if state.public != prev_public and state.public != -1:
            model.update_on_public_card(state.public)

    labels = [label for label, _ in model.trace]
    prob_true = [belief.get(true_card, 0.0) for _, belief in model.trace]
    entropy = [-sum(p * math.log2(p) for p in belief.values() if p > 0) for _, belief in model.trace]
    return BeliefTrace(true_card=true_card, labels=labels, prob_true_card=prob_true, entropy=entropy)


def belief_convergence_curve(
    n_hands: int,
    params: PolicyParams = RULE_BASED_PARAMS,
    seed: int = 0,
) -> tuple[list[int], list[float], list[float]]:
    """Aggregate P(true opponent card) at each observation index across
    many hands. Returns (step_indices, mean_prob, stderr_prob)."""
    rng = random.Random(seed)
    traces = []
    for i in range(n_hands):
        my_player = i % 2
        traces.append(play_hand_with_belief_tracking(my_player, params, rng).prob_true_card)

    max_len = max(len(t) for t in traces)
    steps, means, stderrs = [], [], []
    for step in range(max_len):
        values = [t[step] for t in traces if len(t) > step]
        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
        steps.append(step)
        means.append(mean)
        stderrs.append((var / n) ** 0.5)
    return steps, means, stderrs
