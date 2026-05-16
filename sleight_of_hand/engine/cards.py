"""Card representation for Leduc hold'em.

Ranks are represented as small integers: 0 = J, 1 = Q, 2 = K (higher is
better). The deck has two copies of each rank, six cards total.
"""

from __future__ import annotations

RANK_NAMES = ("J", "Q", "K")
RANKS = (0, 1, 2)
COPIES_PER_RANK = 2


def rank_name(rank: int) -> str:
    return RANK_NAMES[rank]


def full_deck() -> list[int]:
    """Return the 6-card Leduc deck as a list of ranks."""
    return [r for r in RANKS for _ in range(COPIES_PER_RANK)]


def remaining_counts(excluded: list[int]) -> dict[int, int]:
    """Count of each rank left in the deck after removing `excluded` cards
    (each entry in `excluded` removes exactly one copy of that rank)."""
    counts = {r: COPIES_PER_RANK for r in RANKS}
    for c in excluded:
        counts[c] -= 1
    return counts
