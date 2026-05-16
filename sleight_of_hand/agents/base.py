from __future__ import annotations

from abc import ABC, abstractmethod

from ..engine.actions import ActionType
from ..engine.state import GameState


class Agent(ABC):
    """Interface every player in the engine must implement."""

    name: str = "agent"

    @abstractmethod
    def act(self, state: GameState, legal_actions: list[ActionType]) -> ActionType: ...
