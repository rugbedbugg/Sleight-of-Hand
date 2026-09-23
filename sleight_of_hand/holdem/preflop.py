"""Context-aware heads-up preflop policy (ChipZen version 3).

Season 6 sent every preflop decision through one position-agnostic path:
equity against a uniformly random hand, the price, and five global
parameters. That path cannot tell the button's first action from the big
blind facing a limp, an open or a shove. This module derives that context
from public state and applies an explicit, interpretable policy per context.

Everything here is an empirical baseline, not a solver or Nash strategy.
The strength order and equities are measured by the repo's own evaluator
(:mod:`.preflop_tables`); region sizes, assumed opponent ranges and sizes
below are documented, provisional choices intended to be replaced by
measured or sourced values.

State is read by attribute from a ChipZen ``GameState`` (or any object
with the same fields); this module does not import the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .hands import COMBOS, TOTAL_COMBOS
from .preflop_tables import EQUITY_VS_TOP, ORDER, WIDTHS

FOLD, CALL, RAISE = "fold", "call", "raise"  # CALL means check when free
POSTS = ("post_small_blind", "post_big_blind", "post_ante")


class Position(str, Enum):
    BUTTON = "button"  # posts the small blind, acts first preflop
    BIG_BLIND = "big_blind"


class Facing(str, Enum):
    FIRST_IN = "first_in"  # button, nothing voluntary yet
    LIMP = "limp"  # big blind after the button completed
    OPEN = "open"  # big blind facing the button's first raise
    LIMP_RAISED = "limp_raised"  # button limped, big blind raised
    THREE_BET = "three_bet"  # the second raise of the hand
    FOUR_BET_PLUS = "four_bet_plus"
    SHOVE = "shove"  # villain is all-in, or calling commits our stack


@dataclass(frozen=True)
class PreflopContext:
    """Heads-up preflop information state, in chips, derived from public data.

    ``hero_bet``/``villain_bet`` are each player's chips committed this hand
    (antes excluded); ``*_behind`` are chips not yet in the pot.

    ``amount_owed`` is the full difference between the two bets and
    ``call_cost`` is what calling actually costs (``amount_owed`` capped at
    our stack). Both come from the rebuilt bets, never from the server's
    ``to_call``, which may use either representation.
    """

    position: Position
    facing: Facing
    big_blind: int
    pot: int
    amount_owed: int
    call_cost: int
    hero_bet: int
    villain_bet: int
    hero_behind: int
    villain_behind: int
    raises: int
    hero_acted: bool
    action_sequence: tuple[str, ...]

    @property
    def effective_stack(self) -> int:
        """Chips at risk this hand: the smaller starting stack."""
        return min(
            self.hero_bet + self.hero_behind, self.villain_bet + self.villain_behind
        )

    @property
    def effective_stack_bb(self) -> float:
        return self.effective_stack / self.big_blind

    @property
    def unmatched(self) -> int:
        """Chips of the villain's bet we cannot match; returned, never won."""
        return self.amount_owed - self.call_cost

    @property
    def amount_owed_bb(self) -> float:
        return self.amount_owed / self.big_blind

    @property
    def call_cost_bb(self) -> float:
        return self.call_cost / self.big_blind

    @property
    def pot_bb(self) -> float:
        return self.pot / self.big_blind


def derive_context(state) -> PreflopContext | None:
    """Classify a heads-up preflop decision, or ``None`` if unsupported.

    See :func:`diagnose_context`, which also names the reason for ``None``.
    """
    return diagnose_context(state)[0]


def diagnose_context(state) -> tuple[PreflopContext | None, str]:
    """Derive the context, or ``(None, reason)`` naming why it was declined.

    Uses the synthetic blind posts in ``action_history`` (blinds rise during
    a match, so the posted big blind is the current level) and raise-to
    totals. Call amounts are never read: the protocol's examples disagree on
    whether they are totals or increments. Each player's bet is rebuilt
    from posts and raise totals, then checked against ``pot`` and
    ``to_call``; any disagreement is declined so the caller falls back
    rather than acting on a misread state. ``to_call`` may be the full amount
    owed or that amount capped at our stack; both are accepted and yield the
    same context, because only the rebuilt bets feed it. Multiway tables
    are declined. The reason is ``"ok"`` on success.
    """
    if state.phase != "preflop" or state.board:
        return None, "not_preflop"
    if len(state.opponent_stacks) != 1:
        return None, "not_heads_up"
    hero = state.your_seat
    bets: dict[int, int] = {}
    blinds: dict[str, tuple[int, int]] = {}
    antes = level = raises = 0
    voluntary: list[tuple[int, str]] = []
    try:
        for entry in state.action_history:
            action, seat = entry["action"], int(entry["seat"])
            amount = int(entry.get("amount", 0))
            if entry.get("phase", "preflop") != "preflop" or amount < 0:
                return None, "bad_history_entry"
            if action == "post_ante":
                antes += amount
                continue
            if action in POSTS:
                if action in blinds:
                    return None, "duplicate_blind"
                blinds[action] = (seat, amount)
                bets[seat] = bets.get(seat, 0) + amount
                level = max(level, bets[seat])
                continue
            voluntary.append((seat, action))
            if action in ("raise", "all_in"):
                # Raise amounts are totals; a legacy all_in is read the same.
                if amount <= level:
                    return None, "raise_not_above_bet"
                bets[seat], level = amount, amount
                raises += 1
            elif action == "call":
                bets[seat] = level
            elif action == "fold":
                return None, "hand_folded"
            elif action != "check":
                return None, "unknown_action"
    except (KeyError, TypeError, ValueError, AttributeError):
        return None, "unreadable_history"

    if set(blinds) != {"post_small_blind", "post_big_blind"}:
        return None, "missing_blinds"
    sb_seat, sb = blinds["post_small_blind"]
    bb_seat, bb = blinds["post_big_blind"]
    if sb_seat == bb_seat or max(sb, bb) <= 0:
        return None, "bad_blinds"
    if hero not in (sb_seat, bb_seat):
        return None, "hero_not_in_blinds"
    villain = bb_seat if hero == sb_seat else sb_seat
    if any(seat not in (sb_seat, bb_seat) for seat, _ in voluntary):
        return None, "unexpected_seat"
    position = Position.BUTTON if hero == sb_seat else Position.BIG_BLIND
    hero_bet, villain_bet = bets.get(hero, 0), bets.get(villain, 0)
    owed = max(0, villain_bet - hero_bet)
    cost = min(owed, state.your_stack)
    if state.pot != hero_bet + villain_bet + antes:
        return None, "pot_mismatch"
    if state.to_call not in (owed, cost):
        return None, "to_call_mismatch"
    if voluntary and (voluntary[0][0] != sb_seat or voluntary[-1][0] == hero):
        # The button acts first preflop; hero acting last is not our turn.
        return None, "impossible_order"
    hero_acted = any(seat == hero for seat, _ in voluntary)
    if not voluntary and position is not Position.BUTTON:
        return None, "impossible_order"

    villain_behind = state.opponent_stacks[0]
    if owed > 0 and (villain_behind == 0 or owed >= state.your_stack):
        facing = Facing.SHOVE
    elif raises == 0:
        facing = Facing.FIRST_IN if position is Position.BUTTON else Facing.LIMP
        if (facing is Facing.FIRST_IN) != (owed > 0):
            return None, "inconsistent_amount_owed"
    else:
        if owed <= 0:
            return None, "inconsistent_amount_owed"
        facing = (
            (Facing.LIMP_RAISED if hero_acted else Facing.OPEN)
            if raises == 1
            else Facing.THREE_BET
            if raises == 2
            else Facing.FOUR_BET_PLUS
        )
    return PreflopContext(
        position=position,
        facing=facing,
        big_blind=max(sb, bb),  # a short big blind may post less than a full blind
        pot=state.pot,
        amount_owed=owed,
        call_cost=cost,
        hero_bet=hero_bet,
        villain_bet=villain_bet,
        hero_behind=state.your_stack,
        villain_behind=villain_behind,
        raises=raises,
        hero_acted=hero_acted,
        action_sequence=tuple(
            f"{'hero' if seat == hero else 'villain'}:{action}"
            for seat, action in voluntary
        ),
    ), "ok"


# ---------------------------------------------------------------------------
# Strength order and equity lookups
# ---------------------------------------------------------------------------

#: Combo-weighted percentile interval ``[start, end)`` each class occupies in
#: the strength order (strongest first), in percent of all 1,326 combos.
PERCENTILES: dict[str, tuple[float, float]] = {}
_seen = 0
for _label in ORDER:
    PERCENTILES[_label] = (
        100 * _seen / TOTAL_COMBOS,
        100 * (_seen + COMBOS[_label]) / TOTAL_COMBOS,
    )
    _seen += COMBOS[_label]
del _seen, _label


def share_in_top(label: str, percent: float) -> float:
    """Fraction of ``label``'s combos inside the strongest ``percent`` of combos.

    Classes wholly inside return 1 and wholly outside 0. The one boundary
    class mixes by its overlap, so the weighted region is exactly
    ``percent`` of all combos and every other class is played purely.
    """
    start, end = PERCENTILES[label]
    return max(0.0, min(1.0, (percent - start) / (end - start)))


def equity_vs_top(label: str, width: float) -> float:
    """All-in equity against the strongest ``width`` percent, interpolated."""
    row = EQUITY_VS_TOP[label]
    width = max(WIDTHS[0], min(WIDTHS[-1], width))
    step = WIDTHS[1] - WIDTHS[0]
    i = min(int((width - WIDTHS[0]) // step), len(WIDTHS) - 2)
    frac = (width - WIDTHS[i]) / step
    return row[i] + (row[i + 1] - row[i]) * frac


# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------

#: Effective-stack bucket upper bounds in big blinds; the last is open-ended.
STACK_BUCKETS: tuple[float, ...] = (5.0, 8.0, 12.0, 16.0, 20.0, float("inf"))


@dataclass(frozen=True)
class PreflopConfig:
    """Provisional region sizes, assumed ranges and raise sizes.

    Percent values are shares of all 1,326 combos in the strength order.
    Assumed widths describe the opponent range we evaluate a call against;
    they are modelling assumptions, not observations or Nash ranges.
    """

    # Button first in: raise the strongest region, complete a limited band
    # beneath it and fold the rest. Participation stays near Season 6's
    # measured ~87% VPIP; the raise/limp split follows PLUMBER's finding
    # that most playable button hands should enter by raising.
    btn_raise_top: float = 70.0
    btn_play_top: float = 87.0

    # Big blind facing a limp: iso-raise a value region, check the rest.
    # Checking is free, so there is no fold region.
    bb_iso_top: float = 30.0

    # Re-raise regions against a raise (strongest combos only). Opening
    # ranges are wide heads-up, so the 3-bet region is kept narrow and the
    # remaining continues are decided by price.
    bb_3bet_top: float = 16.0
    limp_reraise_top: float = 4.0
    btn_4bet_top: float = 5.0
    jam_top: float = 3.0  # facing a 4-bet or more

    # Assumed width (percent) of the range behind each villain raise. These
    # are linear (strongest-first) stand-ins for ranges that also contain
    # bluffs. At 20% the button folded ~62% of its opens to a 3-bet, which
    # any frequent 3-bettor exploits; 30% keeps that near half.
    open_width: float = 80.0
    iso_width: float = 40.0
    three_bet_width: float = 30.0
    four_bet_width: float = 10.0

    # Equity-realization factors for calls that are not all-in. Out of
    # position the big blind realizes less than its showdown equity; 0.75
    # defends roughly 93% / 68% / 55% of hands against 2 / 2.5 / 3bb opens.
    realization_ip: float = 1.0
    realization_oop: float = 0.75
    # Half-width (equity units) of the linear mix around a call threshold.
    call_band: float = 0.01

    # Assumed shove range width by effective-stack bucket (STACK_BUCKETS):
    # an open-shove, and a shove over our own limp or raise (tighter).
    open_shove_width: tuple[float, ...] = (80.0, 65.0, 50.0, 45.0, 38.0, 25.0)
    reshove_width: tuple[float, ...] = (60.0, 45.0, 35.0, 28.0, 22.0, 15.0)

    # Raise-to sizes. Opens and isos are in big blinds; re-raises multiply
    # the villain's total bet. Any raise committing at least
    # ``commit_fraction`` of the effective stack becomes an all-in.
    open_bb: float = 2.5
    short_open_bb: float = 2.0
    short_open_below_bb: float = 25.0
    iso_bb: float = 4.0
    three_bet_mult: float = 4.0
    limp_reraise_mult: float = 3.0
    four_bet_mult: float = 2.25
    commit_fraction: float = 0.4


DEFAULT_PREFLOP = PreflopConfig()


def stack_bucket(effective_bb: float) -> int:
    return next(i for i, top in enumerate(STACK_BUCKETS) if effective_bb <= top)


def required_equity(ctx: PreflopContext) -> float:
    """Break-even showdown share for calling, capped at our stack.

    Chips the villain bet beyond what we can match are returned, so they
    are removed from the pot we are playing for. For example, a 7bb big
    blind (1bb in, 6bb behind) facing a covering 60bb shove sees a 61bb pot
    but can contest only 8bb of it: 6 / (8 + 6) = 42.9%.
    """
    if ctx.call_cost <= 0:
        return 0.0
    live_pot = ctx.pot - ctx.unmatched
    return ctx.call_cost / (live_pot + ctx.call_cost)


def _smooth_call(value: float, threshold: float, band: float) -> float:
    if band <= 0:
        return 1.0 if value >= threshold else 0.0
    return max(0.0, min(1.0, 0.5 + (value - threshold) / (2 * band)))


def assumed_villain_width(ctx: PreflopContext, config: PreflopConfig) -> float:
    """Width of the range we assume the villain's last raise represents."""
    if ctx.facing is Facing.SHOVE:
        table = config.reshove_width if ctx.hero_acted else config.open_shove_width
        return table[stack_bucket(ctx.effective_stack_bb)]
    return {
        Facing.OPEN: config.open_width,
        Facing.LIMP_RAISED: config.iso_width,
        Facing.THREE_BET: config.three_bet_width,
        Facing.FOUR_BET_PLUS: config.four_bet_width,
    }[ctx.facing]


def action_distribution(
    ctx: PreflopContext, label: str, config: PreflopConfig = DEFAULT_PREFLOP
) -> dict[str, float]:
    """Probabilities of fold / call (or check) / raise for one hand class.

    Raise regions are combo-weighted tops of the strength order. Calls
    compare equity against the assumed villain range, discounted for
    realization unless all-in, with the exact price of this decision.
    """
    if ctx.facing is Facing.FIRST_IN:
        raise_p = share_in_top(label, config.btn_raise_top)
        play_p = share_in_top(label, config.btn_play_top)
        return {RAISE: raise_p, CALL: play_p - raise_p, FOLD: 1.0 - play_p}
    if ctx.facing is Facing.LIMP:
        raise_p = share_in_top(label, config.bb_iso_top)
        return {RAISE: raise_p, CALL: 1.0 - raise_p, FOLD: 0.0}

    width = assumed_villain_width(ctx, config)
    equity = equity_vs_top(label, width)
    if ctx.facing is Facing.SHOVE:
        raise_p = 0.0  # calling already puts one of us all-in
    else:
        top = {
            Facing.OPEN: config.bb_3bet_top,
            Facing.LIMP_RAISED: config.limp_reraise_top,
            Facing.THREE_BET: config.btn_4bet_top,
            Facing.FOUR_BET_PLUS: config.jam_top,
        }[ctx.facing]
        raise_p = share_in_top(label, top)
        realization = (
            config.realization_ip
            if ctx.position is Position.BUTTON
            else config.realization_oop
        )
        equity *= realization
    call_p = (1.0 - raise_p) * _smooth_call(
        equity, required_equity(ctx), config.call_band
    )
    return {RAISE: raise_p, CALL: call_p, FOLD: 1.0 - raise_p - call_p}


def weighted_frequencies(
    ctx: PreflopContext, config: PreflopConfig = DEFAULT_PREFLOP
) -> dict[str, float]:
    """Expected fold / call / raise shares over all 1,326 starting hands.

    Each class is weighted by its combinations (6 pairs, 4 suited, 12
    offsuit), so the result is what an observer would measure over many
    hands dealt into this exact context.
    """
    totals = dict.fromkeys((FOLD, CALL, RAISE), 0.0)
    for label, combos in COMBOS.items():
        for action, p in action_distribution(ctx, label, config).items():
            totals[action] += combos * p / TOTAL_COMBOS
    return totals


def raise_to(
    ctx: PreflopContext,
    min_raise: int,
    max_raise: int,
    config: PreflopConfig = DEFAULT_PREFLOP,
) -> int:
    """Raise-to total in chips for this context, within the legal bounds.

    ``min_raise``/``max_raise`` are the server's legal totals; a short
    stack's ``min_raise`` can exceed ``max_raise``, which is then an all-in.
    """
    bb = ctx.big_blind
    if ctx.facing is Facing.FIRST_IN:
        deep = ctx.effective_stack_bb >= config.short_open_below_bb
        target = (config.open_bb if deep else config.short_open_bb) * bb
    elif ctx.facing is Facing.LIMP:
        target = config.iso_bb * bb
    elif ctx.facing is Facing.OPEN:
        target = config.three_bet_mult * ctx.villain_bet
    elif ctx.facing is Facing.LIMP_RAISED:
        target = config.limp_reraise_mult * ctx.villain_bet
    elif ctx.facing is Facing.THREE_BET:
        target = config.four_bet_mult * ctx.villain_bet
    else:
        target = max_raise
    if target >= config.commit_fraction * ctx.effective_stack:
        target = max_raise
    lower = min(min_raise, max_raise)
    return max(lower, min(max_raise, round(target)))
