"""A parametrized heuristic betting policy.

This single function is the load-bearing piece that ties the whole project
together:

  * As-is (with hand-picked parameters) it is the "rule-based" baseline
    agent.
  * It is the opponent *action model* `P(action | card, context)` used as
    the likelihood term in the Bayesian opponent model.
  * It models the opponent's replies at opponent-decision nodes inside the
    expectiminimax search.
  * Its five parameters are exactly the genome the genetic algorithm
    evolves — thresholds for value-betting and calling, a bluff frequency,
    an aggression level, and a decision "sharpness".

Everything downstream is agnostic to *how* the parameters were chosen; it
just calls `action_probs(...)`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..engine.actions import ActionType

FOLD, CALL, RAISE = ActionType.FOLD, ActionType.CALL, ActionType.RAISE


@dataclass(frozen=True)
class PolicyParams:
    """Genome / strategy parameters. All thresholds live on hand-strength's
    [0, 1] scale; see `hand_strength`."""

    value_bet_threshold: float = 0.55  # strength above which we like to bet/raise for value
    call_threshold: float = 0.35  # strength below which we lean toward folding to a bet
    bluff_freq: float = 0.15  # baseline probability of betting/raising a weak hand
    aggression: float = 0.80  # probability of betting/raising a strong hand when able to
    steepness: float = 8.0  # how sharply probability transitions around the thresholds

    def clipped(self) -> "PolicyParams":
        def c(x, lo=0.0, hi=1.0):
            return max(lo, min(hi, x))

        return PolicyParams(
            value_bet_threshold=c(self.value_bet_threshold),
            call_threshold=c(self.call_threshold),
            bluff_freq=c(self.bluff_freq),
            aggression=c(self.aggression),
            steepness=max(0.5, min(30.0, self.steepness)),
        )

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (
            self.value_bet_threshold,
            self.call_threshold,
            self.bluff_freq,
            self.aggression,
            self.steepness,
        )


DEFAULT_PARAMS = PolicyParams()


def hand_strength(private: int, public: int) -> float:
    """Map a hand to a [0, 1] strength score.

    Pairs (private rank == public rank) are always strongest and ranked
    among themselves (K-pair > Q-pair > J-pair). Non-pairs are ranked by
    private card rank alone, and discounted once a public card is showing
    (a bare high card is worth less once an opponent could plausibly hold
    the pairing card).
    """
    if public != -1 and private == public:
        return 0.8 + 0.1 * private  # J=0.8, Q=0.9, K=1.0
    base = private / 2.0  # J=0.0, Q=0.5, K=1.0
    if public == -1:
        return base
    return base * 0.6


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def action_probs(
    private: int,
    public: int,
    legal_actions: list[ActionType],
    to_call: int,
    params: PolicyParams = DEFAULT_PARAMS,
) -> dict[ActionType, float]:
    """P(action | card, context) under the heuristic policy, restricted to
    `legal_actions` and normalized to sum to 1."""
    s = hand_strength(private, public)
    can_raise = RAISE in legal_actions
    z_strong = _sigmoid(params.steepness * (s - params.value_bet_threshold))

    if to_call == 0:
        p_bet = params.bluff_freq + (params.aggression - params.bluff_freq) * z_strong
        p_bet = max(0.0, min(1.0, p_bet)) if can_raise else 0.0
        probs = {CALL: 1.0 - p_bet}
        if p_bet > 0:
            probs[RAISE] = p_bet
    else:
        p_raise = max(0.0, min(1.0, params.aggression * z_strong)) if can_raise else 0.0
        z_fold = _sigmoid(params.steepness * (params.call_threshold - s))
        remaining = 1.0 - p_raise
        p_fold = remaining * z_fold if FOLD in legal_actions else 0.0
        p_call = remaining - p_fold
        probs = {CALL: max(0.0, p_call)}
        if p_fold > 0:
            probs[FOLD] = p_fold
        if p_raise > 0:
            probs[RAISE] = p_raise

    total = sum(probs.values())
    return {a: p / total for a, p in probs.items() if p > 0}


def sample_action(rng: random.Random, probs: dict[ActionType, float]) -> ActionType:
    actions = list(probs.keys())
    weights = list(probs.values())
    return rng.choices(actions, weights=weights, k=1)[0]
