"""Evaluation harness: play many hands between agents and report results
in milli-big-blinds per hand (mbb/hand), the standard poker-AI win-rate
unit.

The denominator comes from the game's `GameSpec.big_blind`, so results are
scaled correctly whichever `--gamemode` is in play. Leduc has no literal
blinds (players ante instead), so it adopts the common convention of
treating the round-1 fixed bet size as one big blind (2 chips);
2-7 uses its actual big blind. mbb/hand = 1000 * (mean chips won per hand)
/ big_blind.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from ..agents.base import Agent
from ..engine.protocol import Game
from ..engine.registry import get_game

# The default game's denominator, kept as a module constant for callers
# that predate `--gamemode`. Per-match results carry their own (see
# `MatchResult.big_blind`), so a non-Leduc match is scaled correctly even
# though this constant is Leduc's.
BIG_BLIND = get_game().spec.big_blind


@dataclass
class MatchResult:
    name_a: str
    name_b: str
    n_hands: int
    mean_a: float
    stderr_a: float
    big_blind: int = BIG_BLIND

    @property
    def mbb_a(self) -> float:
        return 1000.0 * self.mean_a / self.big_blind

    @property
    def mbb_stderr_a(self) -> float:
        return 1000.0 * self.stderr_a / self.big_blind

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
    game: Game | None = None,
) -> MatchResult:
    """Play `n_hands` hands between two agents, alternating who sits in
    seat 0 (to cancel any positional first-to-act asymmetry), and return
    agent_a's per-hand payoff statistics.

    `game` defaults to the registry's default mode (Leduc)."""
    game = game or get_game()
    payoffs_a = []
    for i in range(n_hands):
        a_is_seat0 = (i % 2 == 0) or not swap_seats
        seats = (agent_a, agent_b) if a_is_seat0 else (agent_b, agent_a)
        p0, p1 = game.play_hand(seats, rng)
        payoffs_a.append(p0 if a_is_seat0 else p1)

    n = len(payoffs_a)
    mean = sum(payoffs_a) / n
    var = sum((x - mean) ** 2 for x in payoffs_a) / max(1, n - 1)
    stderr = math.sqrt(var / n)
    return MatchResult(
        name_a=agent_a.name,
        name_b=agent_b.name,
        n_hands=n,
        mean_a=mean,
        stderr_a=stderr,
        big_blind=game.spec.big_blind,
    )


def round_robin(
    agent_factories: dict[str, Callable[[], Agent]],
    n_hands: int,
    seed: int = 0,
    game: Game | None = None,
) -> dict[tuple[str, str], MatchResult]:
    """Play every ordered pair of distinct agents against each other.
    Returns a dict keyed by (name_a, name_b) -> MatchResult (agent_a's
    perspective); (name_b, name_a) is also populated with the mirrored
    result so callers can look up either order."""
    game = game or get_game()
    names = list(agent_factories.keys())
    results: dict[tuple[str, str], MatchResult] = {}
    rng = random.Random(seed)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            agent_a = agent_factories[name_a]()
            agent_b = agent_factories[name_b]()
            res = play_match(agent_a, agent_b, n_hands, rng, game=game)
            results[(name_a, name_b)] = res
            results[(name_b, name_a)] = MatchResult(
                name_a=name_b,
                name_b=name_a,
                n_hands=res.n_hands,
                mean_a=-res.mean_a,
                stderr_a=res.stderr_a,
                big_blind=res.big_blind,
            )
    return results
