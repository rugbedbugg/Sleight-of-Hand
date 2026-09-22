"""Bounded Monte Carlo showdown equity; standard library only.

Opponents are sampled uniformly from unseen cards. This is a first-port
strength estimate, not the Leduc Bayesian opponent model or search.
"""

from __future__ import annotations

import random
from collections import Counter

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = tuple(range(52))


def encode(card: str) -> int:
    if len(card) != 2 or card[0] not in RANKS or card[1] not in SUITS:
        raise ValueError(f"invalid card: {card!r}")
    return RANKS.index(card[0]) * 4 + SUITS.index(card[1])


def _straight(ranks: set[int]) -> int:
    if 14 in ranks:
        ranks = ranks | {1}
    for high in range(14, 5 - 1, -1):
        if all(rank in ranks for rank in range(high - 4, high + 1)):
            return high
    return 0


def rank_hand(cards: list[int]) -> tuple[int, ...]:
    """Rank the best five of 5–7 cards, with larger tuples winning.

    Includes wheel straights, board-playing ties and all category kickers.
    Input cards are encoded with :func:`encode` and already validated.
    """
    ranks = [card // 4 + 2 for card in cards]
    counts = Counter(ranks)
    groups = sorted(((n, rank) for rank, n in counts.items()), reverse=True)
    flush = []
    for suit in range(4):
        suited = sorted(
            (card // 4 + 2 for card in cards if card % 4 == suit), reverse=True
        )
        if len(suited) >= 5:
            high = _straight(set(suited))
            if high:
                return (8, high)
            flush = suited[:5]
            break
    if groups[0][0] == 4:
        quad = groups[0][1]
        return (7, quad, max(rank for rank in ranks if rank != quad))
    trips = sorted((r for r, n in counts.items() if n >= 3), reverse=True)
    if trips:
        pairs = sorted(
            (r for r, n in counts.items() if n >= 2 and r != trips[0]), reverse=True
        )
        if pairs:
            return (6, trips[0], pairs[0])
    if flush:
        return (5, *flush)
    high = _straight(set(ranks))
    if high:
        return (4, high)
    if trips:
        kickers = sorted((r for r in ranks if r != trips[0]), reverse=True)[:2]
        return (3, trips[0], *kickers)
    pairs = sorted((r for r, n in counts.items() if n == 2), reverse=True)
    if len(pairs) >= 2:
        return (2, *pairs[:2], max(r for r in ranks if r not in pairs[:2]))
    if pairs:
        return (
            1,
            pairs[0],
            *sorted((r for r in ranks if r != pairs[0]), reverse=True)[:3],
        )
    return (0, *sorted(ranks, reverse=True)[:5])


def estimate_equity(
    hole: list[str],
    board: list[str],
    opponents: int,
    rng: random.Random,
    samples: int = 128,
) -> float:
    """Expected share of a showdown against 1–5 random live opponents.

    Folded opponents should be excluded; all-in opponents still count.
    Sampling is capped to keep work bounded even with user configuration.
    Side-pot eligibility and opponents' betting ranges are not modeled.
    """
    if len(hole) != 2 or len(board) not in (0, 3, 4, 5):
        raise ValueError("expected two hole cards and a 0/3/4/5-card Hold'em board")
    if not 1 <= opponents <= 5 or not 1 <= samples <= 512:
        raise ValueError("expected 1–5 opponents and 1–512 equity samples")
    own, public = [encode(c) for c in hole], [encode(c) for c in board]
    known = set(own + public)
    if len(known) != len(own) + len(public):
        raise ValueError("duplicate known cards")
    deck = [card for card in DECK if card not in known]
    missing = 5 - len(public)
    share = 0.0
    for _ in range(samples):
        dealt = rng.sample(deck, missing + opponents * 2)
        completed = public + dealt[:missing]
        ours = rank_hand(own + completed)
        ties = 1
        for seat in range(opponents):
            start = missing + seat * 2
            theirs = rank_hand(completed + dealt[start : start + 2])
            if theirs > ours:
                break
            if theirs == ours:
                ties += 1
        else:
            share += 1.0 / ties
    return share / samples
