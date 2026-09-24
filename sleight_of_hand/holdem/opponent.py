"""Match-local adaptation of the assumed heads-up preflop shove range.

Version 3 evaluates a call against a villain all-in by assuming the shover
holds the top X% of hands, with X fixed per stack bucket. An opponent who
jams far wider than that can fold us out of every pot. This module counts,
per match, how often the villain actually shoves when they have the chance,
and widens X accordingly:

    p     = (n0 * p0 + shoves) / (n0 + opportunities)
    width = max(baseline, 100 * p)

``p0`` is the version 3 width (the prior) and ``n0`` its weight in
pseudo-opportunities. The floor means the model only ever widens: with no
evidence, or against tight opponents, it returns exactly the version 3
width. Linking a shove *frequency* to a top-X% *range* is the same linear
range assumption version 3 already makes.

Evidence comes only from the canonical ``round_result.action_history``,
which the server sends once per hand even when we never act. Each hand is
parsed into a :class:`ShoveObservation` first; the model changes only after
the whole hand validates, and each round is counted at most once. Open
shoves (villain's first action as button) and reshoves (villain responding
after we limped or raised) are separate contexts, each split by the
existing effective-stack buckets. Nothing is persisted; a new bot instance
(a new match) starts from the prior. Showdown cards are never read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .preflop import DEFAULT_PREFLOP, PreflopConfig, stack_bucket

OPEN, RESHOVE = "open", "reshove"
PRIOR_STRENGTH = 10.0
_VOLUNTARY = {"fold", "check", "call", "raise", "all_in"}
_MAX_PENDING_STARTS = 16


@dataclass(frozen=True)
class ShoveObservation:
    """What one completed hand says about the villain's preflop shoving."""

    round_key: str
    bucket: int
    effective_bb: float
    open_opportunity: bool
    open_shove: bool
    reshove_opportunity: bool
    reshove: bool


@dataclass
class ShoveStats:
    opportunities: int = 0
    shoves: int = 0


def _round_key(message: dict) -> str | None:
    round_id = message.get("round_id")
    if isinstance(round_id, str) and round_id:
        return round_id
    result = message.get("result") or message.get("state") or {}
    hand = result.get("hand_number") if isinstance(result, dict) else None
    if isinstance(hand, int) and not isinstance(hand, bool) and hand > 0:
        return f"hand:{hand}"
    return None


def parse_hand(start: dict, result: dict, round_key: str, hero_seat: int):
    """Pure parse of one heads-up hand; ``None`` if anything is unusable.

    ``start`` is ``round_start.state`` (stacks by seat before blinds, per the
    protocol) and ``result`` is ``round_result.result``.
    """
    stacks = start.get("stacks")
    history = result.get("action_history")
    if (
        not isinstance(stacks, list)
        or len(stacks) != 2
        or not all(isinstance(s, int) and not isinstance(s, bool) for s in stacks)
        or not isinstance(history, list)
        or hero_seat not in (0, 1)
    ):
        return None
    villain = 1 - hero_seat
    posts: dict[str, tuple[int, int]] = {}
    antes = [0, 0]
    voluntary: list[tuple[int, str, int]] = []
    for entry in history:
        if not isinstance(entry, dict):
            return None
        action, seat, amount = (
            entry.get("action"),
            entry.get("seat"),
            entry.get("amount", 0),
        )
        if (
            not isinstance(action, str)
            or seat not in (0, 1)
            or not isinstance(amount, int)
            or isinstance(amount, bool)
            or amount < 0
        ):
            return None
        if entry.get("phase", "preflop") != "preflop":
            break  # later streets carry no preflop evidence
        if action in ("post_small_blind", "post_big_blind"):
            if action in posts:
                return None
            posts[action] = (seat, amount)
        elif action == "post_ante":
            antes[seat] += amount
        elif action in _VOLUNTARY:
            voluntary.append((seat, action, amount))
        else:
            return None
    if set(posts) != {"post_small_blind", "post_big_blind"}:
        return None
    button, sb = posts["post_small_blind"]
    big, bb = posts["post_big_blind"]
    big_blind = max(sb, bb)
    if button == big or big_blind <= 0:
        return None
    if voluntary and voluntary[0][0] != button:
        return None  # the button acts first preflop
    effective = min(stacks[hero_seat], stacks[villain])
    if effective <= big_blind:
        return None  # someone is all-in from the blinds; no real choice

    def all_in(seat: int, amount: int) -> bool:
        # A raise to the effective stack puts the shorter player all-in.
        return amount >= effective - antes[seat]

    open_opportunity = open_shove = False
    if button == villain and voluntary:
        open_opportunity = True
        _, action, amount = voluntary[0]
        open_shove = action in ("raise", "all_in") and all_in(villain, amount)

    reshove_opportunity = reshove = False
    hero_first = next(
        (
            i
            for i, (seat, action, _) in enumerate(voluntary)
            if seat == hero_seat and action in ("call", "raise", "all_in")
        ),
        None,
    )
    if hero_first is not None:
        _, action, amount = voluntary[hero_first]
        hero_committed = amount if action in ("raise", "all_in") else 0
        later = [v for v in voluntary[hero_first + 1 :] if v[0] == villain]
        if later and not (hero_committed and all_in(hero_seat, hero_committed)):
            reshove_opportunity = True
            reshove = any(
                a in ("raise", "all_in") and all_in(villain, amt) for _, a, amt in later
            )

    if not (open_opportunity or reshove_opportunity):
        return None
    return ShoveObservation(
        round_key=round_key,
        bucket=stack_bucket(effective / big_blind),
        effective_bb=effective / big_blind,
        open_opportunity=open_opportunity,
        open_shove=open_shove,
        reshove_opportunity=reshove_opportunity,
        reshove=reshove,
    )


@dataclass
class ShoveModel:
    """Per-match open-shove and reshove frequencies by stack bucket."""

    config: PreflopConfig = DEFAULT_PREFLOP
    prior_strength: float = PRIOR_STRENGTH
    stats: dict[tuple[str, int], ShoveStats] = field(default_factory=dict)
    processed: set[str] = field(default_factory=set)
    _starts: dict[str, dict] = field(default_factory=dict)

    def record_round_start(self, message: dict) -> None:
        """Remember the hand's starting stacks until its result arrives."""
        if not isinstance(message, dict):
            return
        state = message.get("state")
        if not isinstance(state, dict):
            return
        key = _round_key(message)
        if key is None:
            return
        self._starts[key] = state
        while len(self._starts) > _MAX_PENDING_STARTS:
            self._starts.pop(next(iter(self._starts)))

    def observe_round_result(
        self, message: dict, hero_seat: int | None
    ) -> ShoveObservation | None:
        """Count one completed hand, at most once; never partially.

        Returns the accepted observation, or ``None`` (model unchanged) for
        duplicates, unknown seats, missing starts or unusable histories.
        """
        try:
            key = _round_key(message)
            result = message.get("result")
            if key is None or key in self.processed or not isinstance(result, dict):
                return None
            start = self._starts.get(key)
            if start is None or hero_seat is None:
                return None
            observation = parse_hand(start, result, key, hero_seat)
        except (AttributeError, TypeError, ValueError):
            return None
        if observation is None:
            self._starts.pop(key, None)
            return None
        # Accepted: apply the whole delta.
        self._starts.pop(key, None)
        self.processed.add(key)
        if observation.open_opportunity:
            s = self.stats.setdefault((OPEN, observation.bucket), ShoveStats())
            s.opportunities += 1
            s.shoves += observation.open_shove
        if observation.reshove_opportunity:
            s = self.stats.setdefault((RESHOVE, observation.bucket), ShoveStats())
            s.opportunities += 1
            s.shoves += observation.reshove
        return observation

    def estimate(self, kind: str, bucket: int) -> dict:
        """Prior, evidence and adaptive width for one context and bucket."""
        table = (
            self.config.open_shove_width if kind == OPEN else self.config.reshove_width
        )
        baseline = table[bucket]
        s = self.stats.get((kind, bucket), ShoveStats())
        n0, p0 = self.prior_strength, baseline / 100
        posterior = (n0 * p0 + s.shoves) / (n0 + s.opportunities)
        width = baseline if s.opportunities == 0 else max(baseline, 100 * posterior)
        return {
            "kind": kind,
            "bucket": bucket,
            "baseline_width": baseline,
            "adaptive_width": width,
            "posterior": posterior,
            "shoves": s.shoves,
            "opportunities": s.opportunities,
            "prior_strength": n0,
        }
