from __future__ import annotations

from enum import IntEnum


class ActionType(IntEnum):
    FOLD = 0
    CALL = 1  # also represents "check" when there is nothing to call
    RAISE = 2  # also represents the opening "bet" when to_call == 0

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name
