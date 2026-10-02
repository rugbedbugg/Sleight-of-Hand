"""The existing Hold'em state attributes and an SDK-independent decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class DecisionState(Protocol):
    """Structural input only: no copying, coercion or runtime validation.

    Cards need only support ``str(card)``. Histories remain plain mappings;
    each consumer retains its own treatment of malformed entries.
    """

    phase: str
    hole_cards: list
    board: list
    pot: int
    your_stack: int
    opponent_stacks: list[int]
    your_seat: int
    to_call: int
    min_raise: int
    max_raise: int
    valid_actions: list[str]
    action_history: list[dict]
    hand_number: int
    round_id: str


@dataclass(frozen=True)
class Decision:
    """An action name and raise-to total; other actions keep amount zero."""

    action: str
    amount: int = 0
