"""The Leduc hold'em rules engine.

Rules (fixed, per the project spec):
  * Deck: {J, J, Q, Q, K, K}.
  * Each player antes 1 chip and is dealt 1 private card.
  * Round 1: bet size 2, at most 2 raises. Actions: fold / check-call / bet-raise.
  * One public (community) card is revealed.
  * Round 2: bet size 4, at most 2 raises.
  * Showdown: a private card matching the public card's rank wins (pair);
    otherwise the higher private card wins; equal ranks split the pot.
  * Player 0 acts first in both betting rounds (a standard simplification
    used by e.g. OpenSpiel's Leduc implementation, since heads-up Leduc has
    no separate "button"/"blind" seating rule specified beyond the ante).

This module is written as a set of pure functions over an immutable
`GameState` so the exact same engine can drive real play (via `rng`) and
hypothetical search (via explicit chance-node enumeration).
"""

from __future__ import annotations

import random

from .actions import ActionType
from .cards import full_deck, remaining_counts
from .state import BET_SIZE, MAX_RAISES, GameState, RoundState


class LeducGame:
    num_players = 2

    # --- dealing -----------------------------------------------------
    @staticmethod
    def new_hand(rng: random.Random) -> GameState:
        deck = full_deck()
        rng.shuffle(deck)
        private = (deck[0], deck[1])
        return GameState(
            private=private,
            public=-1,
            round_no=1,
            to_act=0,
            round_state=RoundState(),
        )

    @staticmethod
    def deal_hand_with_cards(p0: int, p1: int) -> GameState:
        """Construct a fresh hand with specific hole cards (used by search
        / tests to explore a hypothesis without touching a real deck)."""
        return GameState(
            private=(p0, p1),
            public=-1,
            round_no=1,
            to_act=0,
            round_state=RoundState(),
        )

    # --- legality ------------------------------------------------------
    @staticmethod
    def legal_actions(state: GameState) -> list[ActionType]:
        if state.done or state.awaiting_community:
            return []
        rs = state.round_state
        to_call = rs.to_call(state.to_act)
        actions = [ActionType.CALL]
        if to_call > 0:
            actions.append(ActionType.FOLD)
        if rs.raises < MAX_RAISES:
            actions.append(ActionType.RAISE)
        return actions

    # --- transitions -----------------------------------------------------
    @staticmethod
    def apply_action(state: GameState, action: ActionType, rng: random.Random | None = None) -> GameState:
        """Apply `action` for the current player. If this closes round 1,
        the community card is drawn immediately using `rng` (real play). In
        search contexts, don't pass `rng`; instead check
        `state.awaiting_community` on the returned state and use
        `possible_community_cards` / `deal_community` to branch explicitly."""
        if state.done or state.awaiting_community:
            raise ValueError("cannot act on a terminal or awaiting-community state")
        legal = LeducGame.legal_actions(state)
        if action not in legal:
            raise ValueError(f"illegal action {action} in state with legal={legal}")

        player = state.to_act
        rs = state.round_state
        history = state.history + ((player, int(action)),)

        if action == ActionType.FOLD:
            return state.clone(
                round_state=rs,
                history=history,
                done=True,
                folded=player,
            )

        if action == ActionType.CALL:
            to_call = rs.to_call(player)
            contrib = list(rs.contrib)
            contrib[player] += to_call
            new_rs = RoundState(contrib=tuple(contrib), n_actions=rs.n_actions + 1, raises=rs.raises)
            new_state = state.clone(round_state=new_rs, history=history, to_act=1 - player)
            if new_rs.is_closed():
                return LeducGame._close_round(new_state, rng=rng)
            return new_state

        if action == ActionType.RAISE:
            to_call = rs.to_call(player)
            bet_size = BET_SIZE[state.round_no]
            contrib = list(rs.contrib)
            contrib[player] += to_call + bet_size
            new_rs = RoundState(contrib=tuple(contrib), n_actions=rs.n_actions + 1, raises=rs.raises + 1)
            return state.clone(round_state=new_rs, history=history, to_act=1 - player)

        raise AssertionError("unreachable")

    @staticmethod
    def _close_round(state: GameState, rng: random.Random | None) -> GameState:
        """A betting round has just closed (contributions equal, action
        settled). Advance to round 2 (dealing the community card) or to
        showdown."""
        if state.round_no == 1:
            frozen = state.clone(
                round1_contrib=state.round_state.contrib,
                awaiting_community=True,
            )
            if rng is not None:
                card = LeducGame._draw_community(frozen, rng)
                return LeducGame.deal_community(frozen, card)
            return frozen
        # round 2 closed -> showdown
        return state.clone(done=True)

    @staticmethod
    def _draw_community(state: GameState, rng: random.Random) -> int:
        counts = remaining_counts(list(state.private))
        pool = [r for r, n in counts.items() for _ in range(n)]
        return rng.choice(pool)

    @staticmethod
    def possible_community_cards(state: GameState, known_private: tuple[int, int]) -> list[tuple[int, float]]:
        """Weighted outcomes for the community-card chance node, given both
        hole cards are known (used by search, where a hypothesis for the
        opponent's card has already been fixed)."""
        counts = remaining_counts(list(known_private))
        total = sum(counts.values())
        return [(r, n / total) for r, n in counts.items() if n > 0]

    @staticmethod
    def deal_community(state: GameState, card: int) -> GameState:
        if not state.awaiting_community:
            raise ValueError("state is not awaiting a community card")
        return state.clone(
            public=card,
            round_no=2,
            to_act=0,
            round_state=RoundState(),
            awaiting_community=False,
        )

    # --- showdown / payoffs ---------------------------------------------
    @staticmethod
    def _hand_rank(private: int, public: int) -> tuple[int, int]:
        """Higher tuple wins. (1, rank) = paired, (0, rank) = high card."""
        if private == public:
            return (1, private)
        return (0, private)

    @staticmethod
    def payoffs(state: GameState) -> tuple[float, float]:
        if not state.done:
            raise ValueError("hand is not finished")
        c0, c1 = state.contrib_total(0), state.contrib_total(1)
        pot = c0 + c1

        if state.folded != -1:
            winner = 1 - state.folded
            result = [0.0, 0.0]
            result[winner] = pot - [c0, c1][winner]
            result[state.folded] = -[c0, c1][state.folded]
            return (result[0], result[1])

        r0 = LeducGame._hand_rank(state.private[0], state.public)
        r1 = LeducGame._hand_rank(state.private[1], state.public)
        if r0 > r1:
            return (pot - c0, -c1)
        if r1 > r0:
            return (-c0, pot - c1)
        # split pot
        half = pot / 2.0
        return (half - c0, half - c1)

    @staticmethod
    def play_hand(agents, rng: random.Random) -> tuple[float, float]:
        """Play one full hand with `agents` = (agent0, agent1), each
        exposing `.act(state, legal_actions) -> ActionType`. Returns payoffs."""
        state = LeducGame.new_hand(rng)
        while not state.done:
            player = state.to_act
            legal = LeducGame.legal_actions(state)
            action = agents[player].act(state, legal)
            state = LeducGame.apply_action(state, action, rng=rng)
        return LeducGame.payoffs(state)
