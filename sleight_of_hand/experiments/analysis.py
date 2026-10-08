"""Compare arms of one experiment; produce research recommendations only.

Arms are compared only within one (platform, provenance) stratum. Arms that
share seeds under common random numbers are compared match-by-match (paired
differences) and decision-by-decision. Nothing here can promote a result:
every recommendation is labelled research evidence requiring review.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import NormalDist

from . import metrics
from .journal import read_raw, verify
from .model import RunStatus
from .storage import Index

Z95 = metrics.Z95


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_arm(index: Index, spec, check_raw: bool = True) -> dict:
    runs = index.runs(spec.spec_hash)
    completed = [r for r in runs if r["status"] == RunStatus.COMPLETED.value]
    data = {"hands": [], "decisions": [], "observations": [], "raw": []}
    problems = []
    seen_shards = set()
    for run in completed:
        shard = (run["shard_start"], run["shard_end"])
        if shard in seen_shards:
            continue  # a shard re-run after a failure counts once
        seen_shards.add(shard)
        run_dir = Path(run["run_dir"])
        if check_raw and not verify(run_dir)["ok"]:
            problems.append(f"raw checksum failed: {run['run_id']}")
            continue
        for name in ("hands", "decisions", "observations"):
            data[name] += _rows(run_dir / "normalized" / f"{name}.jsonl")
        if "model" in spec.metrics or "accounting" in spec.metrics:
            data["raw"] += read_raw(run_dir)
    return {
        "spec": spec,
        "runs": {
            status.value: sum(r["status"] == status.value for r in runs)
            for status in RunStatus
        },
        "problems": problems,
        **data,
    }


def paired(a: dict, b: dict) -> dict:
    """Per-match differences (b - a) in big blinds per 100 hands."""

    def per_match(arm):
        out = defaultdict(lambda: [0.0, 0])
        for hand in arm["hands"]:
            cell = out[hand["match"]]
            cell[0] += hand["soh_net"] / (hand["big_blind"] or 1)
            cell[1] += 1
        return out

    left, right = per_match(a), per_match(b)
    common = sorted(set(left) & set(right))
    diffs = [
        100 * (right[m][0] / right[m][1] - left[m][0] / left[m][1])
        for m in common
        if left[m][1] and right[m][1]
    ]
    if len(diffs) < 2:
        return {"matches": len(diffs), "mean_bb_per_100": None, "ci95": None}
    mean = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
    se = sd / math.sqrt(len(diffs))
    return {
        "matches": len(diffs),
        "mean_bb_per_100": mean,
        "se": se,
        "ci95": [mean - Z95 * se, mean + Z95 * se],
        "identical_match_results": sum(abs(d) < 1e-12 for d in diffs),
    }


def decision_identity(a: dict, b: dict) -> dict:
    def keyed(arm):
        return {
            (d["match"], d["hand"], d["seq"]): (d["phase"], d["action"], d["amount"])
            for d in arm["decisions"]
        }

    left, right = keyed(a), keyed(b)
    common = sorted(set(left) & set(right))
    differing = [k for k in common if left[k] != right[k]]
    return {
        "compared": len(common),
        "only_in_first": len(set(left) - set(right)),
        "only_in_second": len(set(right) - set(left)),
        "identical": len(common) - len(differing),
        "first_divergence": list(differing[0]) if differing else None,
    }


def _comparable(a, b) -> bool:
    sa, sb = a["spec"], b["spec"]
    return (
        sa.seed_policy == sb.seed_policy
        and sa.seed_policy.common_random_numbers
        and sa.opponent_cohort == sb.opponent_cohort
        and sa.platform_config_json == sb.platform_config_json
        and sa.stopping.hands_per_match == sb.stopping.hands_per_match
    )


def analyze(
    root: Path, experiment_id: str, check_raw: bool = True, *, familywise: bool = False
) -> dict:
    index = Index(Path(root) / "index.sqlite")
    specs = index.specs(experiment_id)
    if not specs:
        raise ValueError(f"no specs registered for {experiment_id}")
    strata = defaultdict(list)
    for spec in specs:
        strata[(spec.platform, spec.provenance.value)].append(
            load_arm(index, spec, check_raw)
        )
    report = {"experiment_id": experiment_id, "strata": []}
    for (platform, provenance), arms in sorted(strata.items()):
        stratum = {
            "platform": platform,
            "provenance": provenance,
            "arms": {},
            "comparisons": [],
        }
        for arm in arms:
            spec = arm["spec"]
            hands, matches = len(arm["hands"]), len({h["match"] for h in arm["hands"]})
            measured = {
                "outcome": metrics.outcome(arm["hands"]),
                "behavior": metrics.behavior(arm["observations"]),
            }
            if "model" in spec.metrics:
                measured["model"] = metrics.model(arm["raw"], spec.policy_config)
            if "accounting" in spec.metrics:
                measured["accounting"] = metrics.accounting(arm["raw"])
            stratum["arms"][spec.arm] = {
                "spec_hash": spec.spec_hash,
                "policy_config_hash": spec.policy_config_hash,
                "runs": arm["runs"],
                "problems": arm["problems"],
                "hands": hands,
                "matches": matches,
                "planned_matches": spec.stopping.max_matches,
                "sufficient_sample": hands >= spec.stopping.min_hands
                and matches >= spec.stopping.min_matches,
                "metrics": measured,
            }
        pairs = list(combinations(sorted(arms, key=lambda x: x["spec"].arm), 2))
        family_size = sum(_comparable(a, b) for a, b in pairs) if familywise else None
        for a, b in pairs:
            if not _comparable(a, b):
                continue
            comparison = {
                "arms": [a["spec"].arm, b["spec"].arm],
                "paired_difference": paired(a, b),
                "decisions": decision_identity(a, b),
            }
            comparison["recommendation"] = recommend(
                comparison,
                stratum["arms"][a["spec"].arm]["sufficient_sample"]
                and stratum["arms"][b["spec"].arm]["sufficient_sample"],
                family_size=family_size,
            )
            stratum["comparisons"].append(comparison)
        report["strata"].append(stratum)
    return report


def recommend(
    comparison: dict, sufficient: bool, *, family_size: int | None = None
) -> dict:
    """Existing paired normal model, optionally Bonferroni-adjusted as one family.

    The family is predeclared by the comparable arm set, never by observed
    results. No new estimator or test is introduced; only its critical value
    increases. Historical callers retain their original unadjusted reports.
    """
    if family_size is not None and (type(family_size) is not int or family_size < 1):
        raise ValueError("comparison family must be positive")
    note = "Research evidence only; any change requires review, regression and explicit promotion."
    diff = comparison["paired_difference"]
    if not sufficient or diff["ci95"] is None:
        return {"verdict": "INSUFFICIENT_SAMPLE", "note": note}
    low, high = diff["ci95"]
    if family_size is not None and family_size > 1:
        z = NormalDist().inv_cdf(1 - 0.05 / (2 * family_size))
        low = diff["mean_bb_per_100"] - z * diff["se"]
        high = diff["mean_bb_per_100"] + z * diff["se"]
    if family_size is not None:
        comparison["multiplicity"] = {
            "method": "bonferroni",
            "family_size": family_size,
            "family_alpha": 0.05,
            "interval": [low, high],
        }
    if low <= 0 <= high:
        return {"verdict": "NO_DIFFERENCE_DETECTED", "note": note}
    better = comparison["arms"][1] if low > 0 else comparison["arms"][0]
    return {"verdict": "DIFFERENCE_DETECTED", "higher_arm": better, "note": note}


def markdown(report: dict) -> str:
    lines = [f"# {report['experiment_id']}", ""]
    for stratum in report["strata"]:
        lines += [f"## {stratum['platform']} / {stratum['provenance']}", ""]
        lines += [
            "| arm | hands | matches | bb/100 | 95% CI | runs |",
            "|---|---|---|---|---|---|",
        ]
        for arm, value in sorted(stratum["arms"].items()):
            bb = value["metrics"]["outcome"]["bb_per_100"]
            ci = bb.get("ci95")
            mean = "n/a" if bb.get("mean") is None else f"{bb['mean']:.1f}"
            interval = "n/a" if not ci else f"[{ci[0]:.1f}, {ci[1]:.1f}]"
            lines.append(
                f"| {arm} | {value['hands']} | {value['matches']} | {mean} | "
                f"{interval} | {value['runs']['COMPLETED']} done |"
            )
        for c in stratum["comparisons"]:
            d, ident = c["paired_difference"], c["decisions"]
            lines += [
                "",
                f"**{c['arms'][1]} - {c['arms'][0]}**: "
                + (
                    "n/a"
                    if d["mean_bb_per_100"] is None
                    else f"{d['mean_bb_per_100']:+.2f} bb/100, 95% CI "
                    f"[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}] over {d['matches']} paired matches"
                )
                + f"; decisions identical {ident['identical']}/{ident['compared']}"
                + f"; verdict {c['recommendation']['verdict']}",
            ]
        lines.append("")
    return "\n".join(lines)
