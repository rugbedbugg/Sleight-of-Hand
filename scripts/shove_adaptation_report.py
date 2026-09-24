"""Report how the match-local shove model widens the assumed shove range.

Report-only diagnostics (no tuning). Feeds synthetic but protocol-shaped
``round_start`` / ``round_result`` messages into
:class:`sleight_of_hand.holdem.opponent.ShoveModel`, and replays the live
exploit (an opponent who jams every chance at ~50bb) through the real bot's
hooks and ``decide()``.

    uv run --no-project python scripts/shove_adaptation_report.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sleight_of_hand.holdem import preflop
from sleight_of_hand.holdem.equity import RANKS, SUITS
from sleight_of_hand.holdem.hands import CLASSES, COMBOS, class_combos
from sleight_of_hand.holdem.opponent import OPEN, RESHOVE, ShoveModel

BB = 100
HERO, VILLAIN = 0, 1
DEPTHS = (14, 25, 50, 100)
PATTERNS = ((1, 1), (2, 3), (3, 4), (5, 6), (8, 10))
HANDS = (
    "AA", "AKs", "AQs", "AJo", "KQs", "99", "77", "55",
    "QJs", "JTs", "98s", "A5s", "KTo", "72o",
)  # fmt: skip


# --- protocol-shaped messages ------------------------------------------------


def entry(seat, action, amount, phase="preflop"):
    return {
        "seat": seat,
        "action": action,
        "amount": amount,
        "phase": phase,
        "is_timeout": False,
    }


def hand_messages(number, button, stacks, voluntary, later=()):
    """(round_start, round_result) for one heads-up hand at 50/100.

    ``voluntary`` holds preflop (seat, action, raise-to total) entries;
    ``later`` holds postflop entries, which carry no preflop evidence.
    """
    round_id = f"r-{number:05d}"
    history = [
        entry(button, "post_small_blind", BB // 2),
        entry(1 - button, "post_big_blind", BB),
    ]
    history += [entry(s, a, amt) for s, a, amt in voluntary]
    history += [entry(s, a, amt, phase) for s, a, amt, phase in later]
    start = {
        "type": "round_start",
        "round_id": round_id,
        "round_number": number,
        "state": {
            "hand_number": number,
            "dealer_seat": button,
            "stacks": list(stacks),
        },
    }
    result = {
        "type": "round_result",
        "round_id": round_id,
        "round_number": number,
        "result": {"hand_number": number, "action_history": history},
    }
    return start, result


def feed(model, number, button, stacks, voluntary, later=()):
    start, result = hand_messages(number, button, stacks, voluntary, later)
    model.record_round_start(start)
    return model.observe_round_result(result, HERO)


def feed_open_pattern(model, shoves, opportunities, depth, first=1):
    """Villain on the button: ``shoves`` jams, the rest are folds."""
    stacks = (depth * BB, depth * BB)
    for i in range(opportunities):
        jam = i < shoves
        feed(
            model,
            first + i,
            VILLAIN,
            stacks,
            [(VILLAIN, "raise", depth * BB), (HERO, "fold", 0)]
            if jam
            else [(VILLAIN, "fold", 0)],
        )
    return first + opportunities


def feed_reshove_pattern(model, shoves, opportunities, depth, first=1):
    """Hero opens to 2.5bb; the villain jams ``shoves`` times, else folds."""
    stacks = (depth * BB, depth * BB)
    for i in range(opportunities):
        jam = i < shoves
        tail = (
            [(VILLAIN, "raise", depth * BB), (HERO, "fold", 0)]
            if jam
            else [(VILLAIN, "fold", 0)]
        )
        feed(model, first + i, HERO, stacks, [(HERO, "raise", 250)] + tail)
    return first + opportunities


# --- decision diagnostics ----------------------------------------------------


def shove_context(kind, depth):
    calibration = _calibration()
    if kind == OPEN:
        state = calibration.scenario(
            "big_blind", depth, [("villain", "raise", depth * BB)]
        )
    else:
        state = calibration.scenario(
            "button", depth, [("hero", "raise", 250), ("villain", "raise", depth * BB)]
        )
    return preflop.derive_context(state)


def call_share(ctx, width):
    return preflop.weighted_frequencies(ctx, villain_width=width)[preflop.CALL]


def hand_calls(ctx, width, hands=HANDS):
    return {
        h: preflop.action_distribution(ctx, h, villain_width=width)[preflop.CALL]
        for h in hands
    }


def _calibration():
    import importlib.util

    name = "preflop_calibration"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- live exploit replay ------------------------------------------------------


def _hole(rng, label=None):
    if label is None:
        label = rng.choice(
            [c for c in CLASSES for _ in range(COMBOS[c])]
        )  # combo-weighted deal
    a, b = rng.choice(class_combos(label))
    text = [RANKS[c // 4] + SUITS[c % 4] for c in (a, b)]
    return label, text


def simulate_jammer(hands=40, hero_stack=10000, villain_stack=5000, seed=11):
    """Replay the live exploit through the real bot: the villain jams at every
    preflop chance, hero seat 0, button alternating, stacks reset per hand.

    Returns one record per hand; ``observed`` marks hands the model counted.
    """
    from chipzen import Card, GameState

    from bots.chipzen.bot import SleightOfHandBot

    bot = SleightOfHandBot(seed=seed)
    bot.on_match_start({"seats": [{"seat": HERO, "is_self": True}, {"seat": VILLAIN}]})
    rng = random.Random(seed)
    stacks = (hero_stack, villain_stack)
    records = []

    def decide(history, button, hole, pot, to_call, lo, hi, valid):
        bets = {HERO: 0, VILLAIN: 0}
        for e in history:
            bets[e["seat"]] = max(bets[e["seat"]], e["amount"])
        state = GameState(
            hand_number=number,
            phase="preflop",
            hole_cards=[Card.from_str(c) for c in hole],
            pot=pot,
            your_stack=hero_stack - bets[HERO],
            opponent_stacks=[villain_stack - bets[VILLAIN]],
            your_seat=HERO,
            dealer_seat=button,
            to_call=to_call,
            min_raise=lo,
            max_raise=hi,
            valid_actions=valid,
            action_history=history,
        )
        return bot.decide(state)

    for number in range(1, hands + 1):
        button = VILLAIN if number % 2 else HERO
        label, hole = _hole(rng)
        start, _ = hand_messages(number, button, stacks, [])
        bot.on_round_start(start)
        history = [
            entry(button, "post_small_blind", BB // 2),
            entry(1 - button, "post_big_blind", BB),
        ]
        jam = villain_stack
        if button == VILLAIN:
            history.append(entry(VILLAIN, "raise", jam))
            act = decide(
                history, button, hole, jam + BB, jam - BB, 0, 0, ["fold", "call"]
            )
            history.append(entry(HERO, act.action, 0))
        else:
            act = decide(
                history,
                button,
                hole,
                150,
                50,
                200,
                hero_stack,
                ["fold", "call", "raise"],
            )
            if act.action == "fold":
                history.append(entry(HERO, "fold", 0))
            else:
                hero_bet = act.amount if act.action == "raise" else BB
                history.append(entry(HERO, act.action, act.amount))
                history.append(entry(VILLAIN, "raise", jam))
                act2 = decide(
                    history,
                    button,
                    hole,
                    hero_bet + jam,
                    jam - hero_bet,
                    0,
                    0,
                    ["fold", "call"],
                )
                history.append(entry(HERO, act2.action, 0))
        _, result = hand_messages(number, button, stacks, [])
        result["result"]["action_history"] = history
        before = len(bot.shove_model.processed)
        bot.on_round_result(result)
        observed = len(bot.shove_model.processed) > before
        kind = OPEN if button == VILLAIN else RESHOVE
        est = bot.shove_model.estimate(kind, preflop.stack_bucket(50))
        records.append(
            {
                "hand": number,
                "hero_class": label,
                "hero_first_action": history[2]["action"],
                "observed": observed,
                **est,
            }
        )
    return bot, records


# --- report ---------------------------------------------------------------------


def pct(x):
    return f"{x:5.1%}"


def main() -> None:
    print("== Baseline v3 assumed shove widths")
    cfg = preflop.DEFAULT_PREFLOP
    for depth in DEPTHS:
        b = preflop.stack_bucket(depth)
        print(
            f"  {depth:>3}bb  open shove {cfg.open_shove_width[b]:.0f}%"
            f"  reshove {cfg.reshove_width[b]:.0f}%"
        )

    for kind in (OPEN, RESHOVE):
        feeder = feed_open_pattern if kind == OPEN else feed_reshove_pattern
        for depth in DEPTHS:
            ctx = shove_context(kind, depth)
            print(
                f"\n== {kind} shoves at {depth}bb  (required equity "
                f"{preflop.required_equity(ctx):.1%})"
            )
            base = preflop.assumed_villain_width(ctx, cfg)
            rows = [("v3 baseline", base)]
            for shoves, opps in PATTERNS:
                model = ShoveModel()
                feeder(model, shoves, opps, depth)
                est = model.estimate(kind, preflop.stack_bucket(depth))
                rows.append((f"{shoves}/{opps} seen", est["adaptive_width"]))
            header = "  ".join(f"{h:>4}" for h in HANDS)
            print(f"  {'pattern':<12} width  call   fold   | call prob: {header}")
            for name, width in rows:
                f = preflop.weighted_frequencies(ctx, villain_width=width)
                calls = hand_calls(ctx, width)
                cells = "  ".join(f"{calls[h]:4.2f}" for h in HANDS)
                print(
                    f"  {name:<12} {width:5.1f}  {pct(f[preflop.CALL])} "
                    f"{pct(f[preflop.FOLD])} | {'':10} {cells}"
                )

    print("\n== Recovery at 50bb: 5/6 open jams, then non-shove opportunities")
    for extra in (0, 5, 10, 14, 15, 20):
        m = ShoveModel()
        n = feed_open_pattern(m, 5, 6, 50)
        feed_open_pattern(m, 0, extra, 50, first=n)
        est = m.estimate(OPEN, preflop.stack_bucket(50))
        print(
            f"  +{extra:>2} folds: {est['shoves']}/{est['opportunities']}  "
            f"posterior {est['posterior']:.4f}  width {est['adaptive_width']:.2f}%"
        )

    print("\n== Live exploit replay: villain jams every chance, hero 100bb vs 50bb")
    bot, records = simulate_jammer()
    ctxs = {OPEN: shove_context(OPEN, 50), RESHOVE: shove_context(RESHOVE, 50)}
    print(
        f"  {'hand':>4} {'kind':<7} opp/shv  posterior  width  call%  | "
        "AA  AKs AJo  99  77  55  QJs A5s KTo 72o"
    )
    for r in records:
        if not r["observed"]:
            continue
        ctx = ctxs[r["kind"]]
        calls = hand_calls(
            ctx,
            r["adaptive_width"],
            ("AA", "AKs", "AJo", "99", "77", "55", "QJs", "A5s", "KTo", "72o"),
        )
        print(
            f"  {r['hand']:>4} {r['kind']:<7} {r['opportunities']:>3}/{r['shoves']:<3} "
            f"{r['posterior']:9.4f} {r['adaptive_width']:6.1f} "
            f"{pct(call_share(ctx, r['adaptive_width']))} | "
            + " ".join(f"{v:3.1f}" for v in calls.values())
        )
    for kind in (OPEN, RESHOVE):
        e = bot.shove_model.estimate(kind, preflop.stack_bucket(50))
        print(
            f"  final {kind}: {e['shoves']}/{e['opportunities']} width {e['adaptive_width']:.1f}%"
        )


if __name__ == "__main__":
    main()
