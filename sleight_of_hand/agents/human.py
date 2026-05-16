"""A human-controlled agent: reads actions from the terminal so a person
can sit at the table against any of the AI agents.

It only ever renders information the acting player is entitled to see (its
own hole card, the public card, the pot, the amount to call) -- never the
opponent's hidden card -- so playing against it is a fair test.
"""

from __future__ import annotations

from ..engine.actions import ActionType
from ..engine.cards import rank_name
from ..engine.state import GameState
from .base import Agent

# Friendly labels: CALL means "check" when nothing is owed, RAISE means the
# opening "bet"; both change name once there is a live bet to answer.
_SYNONYMS = {
    "f": ActionType.FOLD,
    "fold": ActionType.FOLD,
    "c": ActionType.CALL,
    "call": ActionType.CALL,
    "check": ActionType.CALL,
    "k": ActionType.CALL,
    "r": ActionType.RAISE,
    "raise": ActionType.RAISE,
    "b": ActionType.RAISE,
    "bet": ActionType.RAISE,
}


def _label(action: ActionType, to_call: int) -> str:
    if action == ActionType.CALL:
        return "check" if to_call == 0 else f"call {to_call}"
    if action == ActionType.RAISE:
        return "bet" if to_call == 0 else "raise"
    return "fold"


class HumanAgent(Agent):
    """Prompts a person for each decision on stdin."""

    name = "human"

    def __init__(self, name: str | None = None, input_fn=input, print_fn=print):
        if name:
            self.name = name
        self._input = input_fn
        self._print = print_fn

    def act(self, state: GameState, legal_actions: list[ActionType]) -> ActionType:
        player = state.to_act
        to_call = state.round_state.to_call(player)
        pot = state.contrib_total(0) + state.contrib_total(1)
        public = "?" if state.public == -1 else rank_name(state.public)

        self._print(
            f"    your card: {rank_name(state.private[player])} | "
            f"board: {public} | round {state.round_no} | pot: {pot} | to call: {to_call}"
        )
        menu = "  ".join(
            f"[{_key(a)}] {_label(a, to_call)}" for a in legal_actions
        )
        while True:
            raw = self._input(f"    your move ({menu}): ").strip().lower()
            action = _SYNONYMS.get(raw)
            if action is None and raw in {str(int(a)) for a in legal_actions}:
                action = ActionType(int(raw))
            if action in legal_actions:
                return action
            self._print(f"    '{raw}' is not a legal move here. Choose one of: {menu}")


def _key(action: ActionType) -> str:
    return {ActionType.FOLD: "f", ActionType.CALL: "c", ActionType.RAISE: "r"}[action]
