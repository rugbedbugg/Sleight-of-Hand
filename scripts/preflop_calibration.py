"""Print combo-weighted heads-up preflop frequencies, offline.

For each scenario, builds the exact turn state the bot would receive and
reports the expected share of all 1,326 starting hands that raise, call
(or limp/check) and fold under the version 3 preflop policy. With the
ChipZen SDK installed it also reports the Season 6 policy on the same
states, so a change can be predicted before another bot is submitted.

Season 6 values are expectations of its ``decide()`` branch: the same
pot-odds adjustment and sigmoid policy, averaged over repeated 128-sample
random-hand equity estimates per class (seeded, so output is repeatable).
These are local predictions, not measurements against real opponents.

    uv run --no-project python scripts/preflop_calibration.py
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sleight_of_hand.holdem import preflop
from sleight_of_hand.holdem.hands import (
    CLASSES,
    COMBOS,
    TOTAL_COMBOS,
    class_combos,
)

BB = 100
HERO, VILLAIN = 0, 1


@dataclass
class TurnState:
    """The GameState fields the preflop policy reads (SDK-free)."""

    pot: int
    your_stack: int
    opponent_stacks: list[int]
    to_call: int
    min_raise: int
    max_raise: int
    valid_actions: list[str]
    action_history: list[dict]
    your_seat: int = HERO
    phase: str = "preflop"
    board: list = field(default_factory=list)


def scenario(
    position: str,
    stack_bb: float,
    actions: list[tuple[str, str, int]],
    villain_bb: float | None = None,
    cap_to_call: bool = False,
):
    """A heads-up state after ``actions`` (who, action, raise-to total).

    Hero starts with ``stack_bb`` big blinds at 50/100 blinds; the villain
    with ``villain_bb`` (default: the same). The protocol does not say
    whether ``to_call`` is the full amount owed or capped at our stack;
    ``cap_to_call`` builds the capped representation.
    """
    start = round(stack_bb * BB)
    villain_start = round((villain_bb or stack_bb) * BB)
    button = HERO if position == "button" else VILLAIN
    seat = {"hero": HERO, "villain": VILLAIN}
    other = 1 - button
    history = [
        {"seat": button, "action": "post_small_blind", "amount": BB // 2},
        {"seat": other, "action": "post_big_blind", "amount": BB},
    ]
    bets = {button: BB // 2, other: BB}
    for who, action, amount in actions:
        s = seat[who]
        if action == "raise":
            bets[s] = amount
        elif action == "call":
            bets[s] = max(bets.values())
        history.append({"seat": s, "action": action, "amount": amount})
    level = max(bets.values())
    to_call = level - bets[HERO]
    villain_behind = villain_start - bets[VILLAIN]
    hero_behind = start - bets[HERO]
    if to_call and (villain_behind == 0 or to_call >= hero_behind):
        valid, min_raise, max_raise = ["fold", "call"], 0, 0
    else:
        valid = ["fold", "call", "raise"] if to_call else ["check", "raise"]
        last_raise = max(BB, level - min(bets.values()))
        max_raise = bets[HERO] + hero_behind
        min_raise = min(level + last_raise, max_raise)
    return TurnState(
        pot=sum(bets.values()),
        your_stack=hero_behind,
        opponent_stacks=[villain_behind],
        to_call=min(to_call, hero_behind) if cap_to_call else to_call,
        min_raise=min_raise,
        max_raise=max_raise,
        valid_actions=valid,
        action_history=history,
    )


def scenarios() -> list[tuple[str, TurnState, TurnState | None]]:
    """(name, state, prior) triples; ``prior`` is the state whose raise
    region reaches this node, used to also report frequencies conditional
    on the hands that actually get here (e.g. fold to 3-bet)."""
    out = []
    for depth in (100, 40, 15, 8):
        out.append((f"BTN first-in, {depth}bb", scenario("button", depth, []), None))
    out.append(
        (
            "BB vs BTN limp, 100bb",
            scenario("big_blind", 100, [("villain", "call", 0)]),
            None,
        )
    )
    for depth, size in ((100, 2.0), (100, 2.5), (100, 3.0), (40, 2.5)):
        out.append(
            (
                f"BB vs BTN open to {size}bb, {depth}bb",
                scenario("big_blind", depth, [("villain", "raise", round(size * BB))]),
                None,
            )
        )
    out.append(
        (
            "BTN open 2.5bb vs BB 3-bet to 10bb, 100bb",
            scenario(
                "button", 100, [("hero", "raise", 250), ("villain", "raise", 1000)]
            ),
            scenario("button", 100, []),
        )
    )
    for depth in (4, 7, 10, 14, 18, 25):
        out.append(
            (
                f"BB vs BTN shove, {depth}bb",
                scenario("big_blind", depth, [("villain", "raise", depth * BB)]),
                None,
            )
        )
    for depth in (7, 14):
        for capped in (False, True):
            out.append(
                (
                    f"BB {depth}bb vs covering BTN shove for 60bb"
                    + (", to_call capped at stack" if capped else ""),
                    scenario(
                        "big_blind",
                        depth,
                        [("villain", "raise", 6000)],
                        villain_bb=60,
                        cap_to_call=capped,
                    ),
                    None,
                )
            )
    for depth in (10, 15, 20):
        out.append(
            (
                f"BTN 2bb open vs BB reshove, {depth}bb",
                scenario(
                    "button",
                    depth,
                    [("hero", "raise", 200), ("villain", "raise", depth * BB)],
                ),
                scenario("button", depth, []),
            )
        )
    return out


def season6_frequencies(state: TurnState, strengths: dict[str, list[float]]):
    """Expected Season 6 fold/call/raise shares on ``state``."""
    from chipzen import GameState

    from bots.chipzen.bot import betting_params
    from sleight_of_hand.engine.actions import ActionType
    from sleight_of_hand.policy.heuristic import (
        DEFAULT_PARAMS,
        action_probs_from_strength,
    )

    game = GameState(
        pot=state.pot,
        your_stack=state.your_stack,
        opponent_stacks=state.opponent_stacks,
        to_call=state.to_call,
        min_raise=state.min_raise,
        max_raise=state.max_raise,
        valid_actions=state.valid_actions,
        action_history=state.action_history,
    )
    valid = set(state.valid_actions)
    legal = [ActionType.CALL]
    if "fold" in valid:
        legal.append(ActionType.FOLD)
    if "raise" in valid and state.max_raise > 0:
        legal.append(ActionType.RAISE)
    to_call = 0 if "check" in valid else state.to_call
    params = betting_params(game, DEFAULT_PARAMS.clipped())
    names = {
        ActionType.FOLD: preflop.FOLD,
        ActionType.CALL: preflop.CALL,
        ActionType.RAISE: preflop.RAISE,
    }
    totals = dict.fromkeys((preflop.FOLD, preflop.CALL, preflop.RAISE), 0.0)
    for label in CLASSES:
        for strength in strengths[label]:
            probs = action_probs_from_strength(strength, legal, to_call, params)
            for action, p in probs.items():
                totals[names[action]] += (
                    COMBOS[label] * p / (TOTAL_COMBOS * len(strengths[label]))
                )
    return totals


def season6_strengths(draws: int, samples: int = 128) -> dict[str, list[float]]:
    """Repeated Season 6 equity estimates per class (uniform random villain)."""
    from sleight_of_hand.holdem.equity import RANKS, SUITS, estimate_equity

    def text(card: int) -> str:
        return RANKS[card // 4] + SUITS[card % 4]

    rng = random.Random(6)
    out = {}
    for label in CLASSES:
        combos = class_combos(label)
        out[label] = [
            estimate_equity(
                [text(c) for c in combos[i % len(combos)]], [], 1, rng, samples
            )
            for i in range(draws)
        ]
    return out


def conditional(state: TurnState, prior: TurnState) -> dict[str, float]:
    """Version 3 shares among only the hands whose prior action was a raise."""
    ctx, before = preflop.derive_context(state), preflop.derive_context(prior)
    totals = dict.fromkeys((preflop.FOLD, preflop.CALL, preflop.RAISE), 0.0)
    reach = 0.0
    for label, combos in COMBOS.items():
        weight = combos * preflop.action_distribution(before, label)[preflop.RAISE]
        reach += weight
        for action, p in preflop.action_distribution(ctx, label).items():
            totals[action] += weight * p
    return {action: value / reach for action, value in totals.items()}


def labels(state: TurnState) -> tuple[str, str]:
    if "check" in state.valid_actions:
        return "raise", "check"
    ctx = preflop.derive_context(state)
    if ctx is not None and ctx.facing is preflop.Facing.FIRST_IN:
        return "raise", "limp"
    if ctx is not None and ctx.facing is preflop.Facing.OPEN:
        return "3-bet", "call"
    if ctx is not None and ctx.facing is preflop.Facing.THREE_BET:
        return "4-bet", "call"
    return "raise", "call"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-season6", action="store_true", help="skip the Season 6 comparison"
    )
    parser.add_argument(
        "--draws", type=int, default=16, help="Season 6 equity draws per class"
    )
    args = parser.parse_args()

    strengths = None
    if not args.no_season6:
        try:
            import chipzen  # noqa: F401
        except ImportError:
            print("chipzen SDK not installed; skipping the Season 6 comparison.\n")
        else:
            strengths = season6_strengths(args.draws)

    for name, state, prior in scenarios():
        ctx = preflop.derive_context(state)
        assert ctx is not None, name
        v3 = preflop.weighted_frequencies(ctx)
        aggressive, passive = labels(state)
        print(
            f"Scenario: {name}  [{ctx.facing.value}, "
            f"{ctx.effective_stack_bb:g}bb effective, "
            f"required equity {preflop.required_equity(ctx):.1%}]"
        )
        rows = [("v3", v3)]
        if prior is not None:
            rows.append(("v3|opened", conditional(state, prior)))
        if strengths is not None:
            rows.append(("season-6", season6_frequencies(state, strengths)))
        for tag, f in rows:
            vpip = f[preflop.RAISE] + (0 if passive == "check" else f[preflop.CALL])
            print(
                f"  {tag:<10} {aggressive}: {f[preflop.RAISE]:6.1%}  "
                f"{passive}: {f[preflop.CALL]:6.1%}  "
                f"fold: {f[preflop.FOLD]:6.1%}  VPIP: {vpip:6.1%}"
            )
        print()

    # Both to_call representations must yield identical version 3 economics.
    for depth in (7, 14):
        states = [
            scenario(
                "big_blind",
                depth,
                [("villain", "raise", 6000)],
                villain_bb=60,
                cap_to_call=capped,
            )
            for capped in (False, True)
        ]
        full, capped = (preflop.derive_context(s) for s in states)
        same = (
            full is not None
            and capped is not None
            and preflop.required_equity(full) == preflop.required_equity(capped)
            and preflop.weighted_frequencies(full)
            == preflop.weighted_frequencies(capped)
        )
        print(
            f"to_call representation check, {depth}bb covered shove: "
            + ("identical" if same else "MISMATCH")
        )
        if not same:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
