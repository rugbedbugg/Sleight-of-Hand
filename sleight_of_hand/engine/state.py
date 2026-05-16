from __future__ import annotations

from dataclasses import dataclass, field, replace

ANTE = 1
BET_SIZE = {1: 2, 2: 4}
MAX_RAISES = 2


@dataclass(frozen=True)
class RoundState:
    """Betting state within a single street (round 1 or round 2)."""

    contrib: tuple[int, int] = (0, 0)  # chips put in *this round* by each player
    n_actions: int = 0  # actions taken this round (fold excluded, it ends the hand)
    raises: int = 0  # number of bets/raises so far this round

    def to_call(self, player: int) -> int:
        other = 1 - player
        return max(0, self.contrib[other] - self.contrib[player])

    def is_closed(self) -> bool:
        return self.contrib[0] == self.contrib[1] and self.n_actions >= 2


@dataclass(frozen=True)
class GameState:
    """Full state of a Leduc hand.

    `private` holds both players' hole cards. When a `GameState` is used to
    represent one branch of a hypothetical search (see `search/`), both
    entries are filled in for that branch's hypothesis; the *agent* using
    the state is still responsible for not looking at bits it shouldn't
    know about (the engine itself never hides information).
    """

    private: tuple[int, int]
    public: int  # -1 until the community card is revealed
    round_no: int  # 1 or 2
    to_act: int  # 0 or 1
    round_state: RoundState
    round1_contrib: tuple[int, int] = (0, 0)  # frozen once round 2 starts
    history: tuple[tuple[int, int], ...] = field(default_factory=tuple)  # (player, action)
    done: bool = False
    folded: int = -1  # player index who folded, else -1
    awaiting_community: bool = False  # round 1 closed, community card not yet dealt

    def contrib_total(self, player: int) -> int:
        """Total chips `player` has committed so far (ante + both streets)."""
        r1 = self.round1_contrib[player]
        r2 = self.round_state.contrib[player] if self.round_no >= 2 else 0
        return ANTE + r1 + r2

    def clone(self, **changes) -> "GameState":
        return replace(self, **changes)
