"""Audit the version 3 preflop policy and its generated equity tables.

Report-only; it never rewrites the tables or the policy.

``--policy`` prints frequencies across open sizes and on both sides of every
shove stack bucket boundary, so bucketed discontinuities can be inspected.

``--equity`` re-estimates, with independent seeds and larger samples than
generation used, the equities that decide policy boundaries:

* the strength order near each raise-region boundary (3, 4, 5, 16, 30, 70
  and 87% of combos): does the order hold beyond Monte Carlo noise?
* equities of classes near the call/fold threshold in representative
  spots: would re-estimated values move aggregate frequencies?
* every EQUITY_VS_TOP row: are there drops between adjacent widths larger
  than sampling noise explains?

    uv run --no-project python scripts/audit_preflop.py --policy
    uv run --no-project python scripts/audit_preflop.py --equity  # minutes
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sleight_of_hand.holdem import preflop
from sleight_of_hand.holdem.hands import COMBOS, TOTAL_COMBOS
from sleight_of_hand.holdem.preflop_tables import EQUITY_VS_TOP, ORDER, WIDTHS


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


calibration = _load("preflop_calibration")
generator = _load("generate_preflop_tables")
scenario = calibration.scenario
BB = calibration.BB

GEN_RANDOM, GEN_RANGE = 40000, 6000  # samples used by the committed tables


def fmt(f: dict[str, float]) -> str:
    return (
        f"raise {f[preflop.RAISE]:6.1%}  call {f[preflop.CALL]:6.1%}  "
        f"fold {f[preflop.FOLD]:6.1%}"
    )


def policy_report() -> None:
    print("== Big blind vs open size (defence must not loosen as size grows)")
    for depth in (100, 40, 20):
        for size in (2.0, 2.5, 3.0):
            state = scenario(
                "big_blind", depth, [("villain", "raise", round(size * BB))]
            )
            ctx = preflop.derive_context(state)
            print(
                f"  {depth:>3}bb, open {size}bb  req {preflop.required_equity(ctx):5.1%}  "
                + fmt(preflop.weighted_frequencies(ctx))
            )
    print("\n== Big blind vs open-shove at each bucket boundary")
    for edge in preflop.STACK_BUCKETS[:-1]:
        for depth in (edge - 0.5, edge, edge + 0.05, edge + 0.5):
            chips = round(depth * BB)
            state = scenario("big_blind", depth, [("villain", "raise", chips)])
            ctx = preflop.derive_context(state)
            width = preflop.assumed_villain_width(ctx, preflop.DEFAULT_PREFLOP)
            print(
                f"  {depth:5.2f}bb  bucket {preflop.stack_bucket(ctx.effective_stack_bb)}"
                f"  assumed range {width:4.0f}%  req {preflop.required_equity(ctx):5.1%}"
                f"  {fmt(preflop.weighted_frequencies(ctx))}"
            )
        print()


def _cell(job):
    label, width, samples, tag = job
    villain = None if width == 100 else generator.top_range(list(ORDER), width)
    return label, width, generator.equity(label, villain, samples, tag)


def se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-9) / n)


def near_boundaries(margin: float = 1.5) -> dict[float, list[str]]:
    out = {}
    for edge in (3, 4, 5, 16, 30, 70, 87):
        out[edge] = [
            label
            for label in ORDER
            if preflop.PERCENTILES[label][0] - margin
            <= edge
            <= preflop.PERCENTILES[label][1] + margin
        ]
    return out


def threshold_spots():
    return {
        "BB vs 2.5bb open": scenario("big_blind", 100, [("villain", "raise", 250)]),
        "BB vs 3.0bb open": scenario("big_blind", 100, [("villain", "raise", 300)]),
        **{
            f"BB vs {d}bb shove": scenario(
                "big_blind", d, [("villain", "raise", d * BB)]
            )
            for d in (7, 10, 14, 18)
        },
    }


def equity_report(samples_random: int, samples_range: int, workers) -> None:
    boundary = near_boundaries()
    spots = threshold_spots()
    jobs = {(label, 100) for labels in boundary.values() for label in labels}
    marginal = {}
    for name, state in spots.items():
        ctx = preflop.derive_context(state)
        width = preflop.assumed_villain_width(ctx, preflop.DEFAULT_PREFLOP)
        realization = (
            1.0
            if ctx.facing is preflop.Facing.SHOVE
            else preflop.DEFAULT_PREFLOP.realization_oop
        )
        need = preflop.required_equity(ctx)
        marginal[name] = (ctx, width, realization, need, [])
        for label in ORDER:
            eq = preflop.equity_vs_top(label, width) * realization
            if abs(eq - need) <= 0.02:
                marginal[name][4].append(label)
                jobs.add((label, round(width)))
    with ProcessPoolExecutor(workers) as pool:
        results = {
            (label, width): value
            for label, width, value in pool.map(
                _cell,
                [
                    (
                        label,
                        width,
                        samples_random if width == 100 else samples_range,
                        f"audit{width}",
                    )
                    for label, width in sorted(jobs)
                ],
                chunksize=2,
            )
        }

    def table(label: str, width: int) -> float:
        return preflop.equity_vs_top(label, width)

    print(
        f"== Strength order near raise-region boundaries "
        f"(audit {samples_random} samples/class vs committed {GEN_RANDOM})"
    )
    for edge, labels in boundary.items():
        print(f"  boundary {edge}%:")
        audited = sorted(labels, key=lambda x: -results[(x, 100)])
        for label in labels:
            a, t = results[(label, 100)], table(label, 100)
            noise = math.hypot(se(a, samples_random), se(t, GEN_RANDOM))
            start, end = preflop.PERCENTILES[label]
            print(
                f"    {label:<4} pct {start:5.1f}-{end:5.1f}  table {t:.4f}  "
                f"audit {a:.4f}  diff {a - t:+.4f} ({(a - t) / noise:+.1f} sd)"
            )
        swaps = [
            (x, y)
            for i, x in enumerate(labels)
            for y in labels[i + 1 :]
            if audited.index(x) > audited.index(y)
        ]
        print(f"    order swaps under audit: {swaps or 'none'}")

    print(
        f"\n== Call/fold-threshold classes (audit {samples_range} samples/cell "
        f"vs committed {GEN_RANGE}); aggregate impact of re-estimates"
    )
    for name, (ctx, width, realization, need, labels) in marginal.items():
        w = round(width)
        flips, delta = [], 0.0
        for label in labels:
            t = table(label, w) * realization
            a = results[(label, w)] * realization
            before = preflop._smooth_call(t, need, preflop.DEFAULT_PREFLOP.call_band)
            after = preflop._smooth_call(a, need, preflop.DEFAULT_PREFLOP.call_band)
            share = 1 - preflop.share_in_top(
                label,
                0
                if ctx.facing is preflop.Facing.SHOVE
                else preflop.DEFAULT_PREFLOP.bb_3bet_top,
            )
            delta += COMBOS[label] * share * (after - before) / TOTAL_COMBOS
            if abs(after - before) > 0.5:
                flips.append(f"{label}({t:.3f}->{a:.3f})")
        print(
            f"  {name:<18} width {w}%  need {need:5.1%}  marginal classes "
            f"{len(labels):>2}  call-share change {delta:+.2%}  flips: "
            f"{', '.join(flips) or 'none'}"
        )

    print("\n== EQUITY_VS_TOP rows: drops between adjacent widths")
    noise = 2 * se(0.5, GEN_RANGE) * math.sqrt(2)
    drops = []
    for label, row in EQUITY_VS_TOP.items():
        for i in range(len(WIDTHS) - 1):
            if row[i + 1] < row[i] - noise:
                drops.append((row[i] - row[i + 1], label, WIDTHS[i], WIDTHS[i + 1]))
    drops.sort(reverse=True)
    print(
        f"  drops larger than 2 sd ({noise:.4f}): {len(drops)} of "
        f"{len(EQUITY_VS_TOP) * (len(WIDTHS) - 1)} adjacent pairs"
    )
    for size, label, a, b in drops[:10]:
        print(f"    {label:<4} {a}%->{b}%  drop {size:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", action="store_true")
    parser.add_argument("--equity", action="store_true")
    parser.add_argument("--random-samples", type=int, default=400000)
    parser.add_argument("--range-samples", type=int, default=60000)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    if not (args.policy or args.equity):
        parser.error("choose --policy and/or --equity")
    if args.policy:
        policy_report()
    if args.equity:
        equity_report(args.random_samples, args.range_samples, args.workers)


if __name__ == "__main__":
    main()
