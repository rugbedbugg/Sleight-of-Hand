"""The 169 heads-up starting-hand classes and their 1,326 card combinations.

A class is labelled ``"AA"``, ``"AKs"`` or ``"AKo"``, higher rank first.
Pairs have 6 combinations, suited non-pairs 4 and offsuit non-pairs 12,
so aggregate frequencies must weight each class by :data:`COMBOS`.
"""

from __future__ import annotations

from itertools import combinations

from .equity import RANKS, SUITS, encode

TOTAL_COMBOS = 1326


def _labels() -> tuple[str, ...]:
    labels = []
    for high in reversed(RANKS):
        for low in reversed(RANKS[: RANKS.index(high) + 1]):
            if high == low:
                labels.append(high + low)
            else:
                labels += [high + low + "s", high + low + "o"]
    return tuple(labels)


CLASSES: tuple[str, ...] = _labels()


def class_combos(label: str) -> list[tuple[int, int]]:
    """Every encoded two-card combination belonging to ``label``."""
    high, low = label[0], label[1]
    if high == low:
        return [(encode(high + a), encode(low + b)) for a, b in combinations(SUITS, 2)]
    suited = label[2] == "s"
    return [
        (encode(high + a), encode(low + b))
        for a in SUITS
        for b in SUITS
        if (a == b) == suited
    ]


COMBOS: dict[str, int] = {label: len(class_combos(label)) for label in CLASSES}


def hand_class(hole: list[str]) -> str:
    """Class label for two validated hole-card strings such as ``["Kd", "Ah"]``."""
    if len(hole) != 2:
        raise ValueError("expected two hole cards")
    a, b = sorted((encode(c) for c in hole), reverse=True)
    if a == b:
        raise ValueError("duplicate hole cards")
    high, low = RANKS[a // 4], RANKS[b // 4]
    if high == low:
        return high + low
    return high + low + ("s" if a % 4 == b % 4 else "o")
