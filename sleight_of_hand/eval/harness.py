"""Evaluation harness: play many hands between agents and report results
in milli-big-blinds per hand (mbb/hand), the standard poker-AI win-rate
unit.

Leduc hold'em has no literal blinds (players ante instead), so we adopt the
common convention of treating the round-1 fixed bet size as one "big
blind": BIG_BLIND = 2 chips. mbb/hand = 1000 * (mean chips won per hand) /
BIG_BLIND.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from ..agents.base import Agent
from ..engine.game import LeducGame
from ..engine.state import BET_SIZE

BIG_BLIND = BET_SIZE[1]


@dataclass
class MatchResult:
    name_a: str
    name_b: str
    n_hands: int
    mean_a: float
    stderr_a: float

    @property
    def mbb_a(self) -> float:
        return 1000.0 * self.mean_a / BIG_BLIND

    @property
    def mbb_stderr_a(self) -> float:
        return 1000.0 * self.stderr_a / BIG_BLIND

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.name_a} vs {self.name_b}: "
            f"{self.mbb_a:+.1f} +/- {1.96 * self.mbb_stderr_a:.1f} mbb/hand "
            f"({self.n_hands} hands)"
        )


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    n_hands: int,
    rng: random.Random,
    swap_seats: bool = True,
) -> MatchResult:
    """Play `n_hands` hands between two agents, alternating who sits in
    seat 0 (to cancel any positional first-to-act asymmetry), and return
    agent_a's per-hand payoff statistics."""
    payoffs_a = []
    for i in range(n_hands):
        a_is_seat0 = (i % 2 == 0) or not swap_seats
        seats = (agent_a, agent_b) if a_is_seat0 else (agent_b, agent_a)
        p0, p1 = LeducGame.play_hand(seats, rng)
        payoffs_a.append(p0 if a_is_seat0 else p1)

    n = len(payoffs_a)
    mean = sum(payoffs_a) / n
    var = sum((x - mean) ** 2 for x in payoffs_a) / max(1, n - 1)
    stderr = math.sqrt(var / n)
    return MatchResult(name_a=agent_a.name, name_b=agent_b.name, n_hands=n, mean_a=mean, stderr_a=stderr)


def round_robin(
    agent_factories: dict[str, Callable[[], Agent]],
    n_hands: int,
    seed: int = 0,
) -> dict[tuple[str, str], MatchResult]:
    """Play every ordered pair of distinct agents against each other.
    Returns a dict keyed by (name_a, name_b) -> MatchResult (agent_a's
    perspective); (name_b, name_a) is also populated with the mirrored
    result so callers can look up either order."""
    names = list(agent_factories.keys())
    results: dict[tuple[str, str], MatchResult] = {}
    rng = random.Random(seed)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            agent_a = agent_factories[name_a]()
            agent_b = agent_factories[name_b]()
            res = play_match(agent_a, agent_b, n_hands, rng)
            results[(name_a, name_b)] = res
            results[(name_b, name_a)] = MatchResult(
                name_a=name_b, name_b=name_a, n_hands=res.n_hands, mean_a=-res.mean_a, stderr_a=res.stderr_a
            )
    return results
