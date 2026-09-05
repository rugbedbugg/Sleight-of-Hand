"""The game-agnostic interface every rules engine implements.

The belief, search, evaluation and GA layers are written against `Game`
rather than against a concrete engine, so that adding a variant means
adding one module and one registry entry rather than touching every
consumer. `--gamemode` selects the implementation (see `registry.py`).

Two generalizations of Leduc's vocabulary live here:

  * **Chance nodes.** Leduc's single community card and 2-7's draw
    replacements are the same kind of thing -- a stochastic transition the
    search must enumerate rather than choose. `awaiting_chance` /
    `chance_outcomes` / `apply_chance` replace the community-card-specific
    trio.
  * **Draws as actions.** A draw is an `Action` carrying a discard mask,
    not a separate agent method. Decision nodes in the tree stay uniform,
    `action_probs` needs no special case, and `Agent.act` is unchanged.
    Leduc only ever produces `ActionType`, so nothing in the existing
    engine is affected.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .actions import ActionType


@dataclass(frozen=True)
class DrawAction:
    """Discard `mask` (indices into the actor's hand) and draw replacements.

    Unused by Leduc; the type exists so `Action` is stable across variants.
    """

    mask: tuple[int, ...] = ()

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"DRAW({len(self.mask)})"


# A decision is either a betting action or a draw.
Action = ActionType | DrawAction


@dataclass(frozen=True)
class GameSpec:
    """Static description of a variant: everything the shared layers need
    to know without importing a concrete engine."""

    name: str
    num_players: int
    num_rounds: int
    bet_size: dict[int, int]  # round number -> fixed bet size
    max_raises: int  # raise cap per betting round
    big_blind: int  # the mbb/hand denominator (see eval/harness.py)
    ante: int = 0
    blinds: tuple[int, ...] = field(default_factory=tuple)  # (small, big), empty if antes
    has_draws: bool = False

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name} ({self.num_players}p, {self.num_rounds} rounds)"


@runtime_checkable
class Game(Protocol):
    """Structural interface for a rules engine.

    Implementations are registered as instances, but may (as `LeducGame`
    does) define every method as a `staticmethod` so that both
    `LeducGame.legal_actions(s)` and `game.legal_actions(s)` work. That
    keeps existing call sites valid while new code goes through the
    protocol.
    """

    spec: GameSpec

    def new_hand(self, rng: random.Random) -> Any:
        """Deal a fresh hand."""
        ...

    def legal_actions(self, state: Any) -> list[Action]:
        """Actions available to `state.to_act`; empty at terminal or
        chance-pending states."""
        ...

    def apply_action(self, state: Any, action: Action, rng: random.Random | None = None) -> Any:
        """Apply `action`. With `rng`, any chance event this triggers is
        resolved immediately (real play). Without it, the returned state
        may report `awaiting_chance` for the caller to branch on (search)."""
        ...

    def awaiting_chance(self, state: Any) -> bool:
        """True when a stochastic transition is pending."""
        ...

    def chance_outcomes(self, state: Any) -> list[tuple[Any, float]]:
        """(outcome, probability) pairs for the pending chance node."""
        ...

    def apply_chance(self, state: Any, outcome: Any) -> Any:
        """Resolve the pending chance node with a specific outcome."""
        ...

    def payoffs(self, state: Any) -> tuple[float, ...]:
        """Per-player payoffs at a terminal state, in chips."""
        ...

    def hand_strength(self, private: Any, public: Any) -> float:
        """Map a holding to a [0, 1] strength score, the scale every
        threshold in `policy/heuristic.py` is expressed on."""
        ...

    def play_hand(self, agents, rng: random.Random) -> tuple[float, ...]:
        """Play one hand to completion and return payoffs."""
        ...
