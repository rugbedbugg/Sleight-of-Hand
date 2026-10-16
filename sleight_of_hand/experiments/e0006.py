"""E0006: the predeclared, opponent-stratified SafeExploit v1 ablation analysis.

The general :func:`.analysis.analyze` pools every opponent of a spec into one
paired comparison. E0006 asks a stratified question instead, so this module
computes exactly the frozen plan in ``experiments/safe-exploit-v1/DESIGN.md``:

* unit: the match-level paired difference in bb/100, ``safe-exploit-v1 -
  canonical``, at the same match index (common random numbers);
* primary: the equal-weight mean of the three exploit-sensitive opponents'
  mean paired differences, one two-sided 95% normal interval, no multiplicity
  adjustment (a single primary hypothesis);
* secondary: each of the five opponents plus the equal-weight reference
  cohort, one Bonferroni family at familywise alpha 0.05;
* everything else (stability halves, divergence, evidence activation, ESS,
  tendency rates, rejected actions) is descriptive only.

It reads only verified sealed evidence through a read-only index view, fails
closed on any missing, duplicated, mismatched or corrupted block, and never
registers, migrates, recovers or rewrites anything except its own report. The
evidence-activation diagnostics replay ``OpponentMemory`` and the pure
``SafeExploitController`` over the messages SOH received; no strategy runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist

from chipzen import GameState

from sleight_of_hand.holdem.exploit import SafeExploitController
from sleight_of_hand.holdem.hands import hand_class
from sleight_of_hand.holdem.memory import OpponentMemory
from sleight_of_hand.holdem.preflop import diagnose_context

from . import journal
from . import programme as programmes
from .analysis import Z95
from .model import canonical, sha256
from .runner import ROOT, match_seeds
from .supervisor import Snapshot, _verify_evidence

EXPERIMENT = "E0006"
CONTROL = "canonical"
TREATMENT = "safe-exploit-v1"
PROGRAMME = ROOT / "experiments" / "safe-exploit-v1" / "programme.json"
HISTORICAL_ROOT = ROOT / "runs"
EXPLOIT_SENSITIVE = (
    "local/script/wide-opener",
    "local/script/tight-opener",
    "local/script/threebet-heavy",
)
REFERENCE = ("local/script/tag_simple", "local/soh/canonical")
FAMILY_ALPHA = 0.05
#: bb/100. Non-detection is called NO_DIFFERENCE_DETECTED only when the
#: interval also excludes effects beyond this; otherwise INCONCLUSIVE.
RESOLUTION = 30.0
HAND_BIN = 50
TENDENCIES = ("btn_open", "bb_iso", "bb_3bet")
SECONDARY = (
    *(f"opponent:{o}" for o in EXPLOIT_SENSITIVE + REFERENCE),
    "reference_equal_weight",
)

PLAN = {
    "experiment_id": EXPERIMENT,
    "contrast": f"{TREATMENT} - {CONTROL}",
    "unit": "match-level paired difference in bb/100 at the same match index",
    "primary": {
        "label": "exploit_sensitive_equal_weight",
        "weights": {o: 1 / len(EXPLOIT_SENSITIVE) for o in EXPLOIT_SENSITIVE},
        "alpha": FAMILY_ALPHA,
        "sides": 2,
        "interval": "normal, stratified standard error",
        "multiplicity": "none: a single primary hypothesis",
    },
    "secondary": {
        "members": list(SECONDARY),
        "reference_weights": {o: 1 / len(REFERENCE) for o in REFERENCE},
        "correction": "bonferroni",
        "family_alpha": FAMILY_ALPHA,
    },
    "resolution_bb_per_100": RESOLUTION,
    "classification": [
        "IMPROVEMENT_DETECTED: interval lower bound > 0",
        "REGRESSION_DETECTED: interval upper bound < 0",
        (
            "NO_DIFFERENCE_DETECTED: interval within [-resolution, +resolution];"
            " not proven equivalence"
        ),
        "INCONCLUSIVE: interval contains 0 and extends beyond the resolution",
        (
            "INCONCLUSIVE: any weighted stratum has zero sample variance"
            " (degenerate; no interval)"
        ),
    ],
    "degenerate_variance": (
        "a weighted stratum whose paired differences are all identical has zero"
        " sample variance, so its uncertainty cannot be estimated: the contrast"
        " reports estimate, SE and the affected strata with no interval and is"
        " INCONCLUSIVE, never an improvement or regression"
    ),
    "reference_regression": "REGRESSION_DETECTED on the corrected interval only;"
    " no non-inferiority margin is claimed",
    "descriptive": [
        "stability: primary contrast on match-index halves",
        "action divergence",
        "SafeExploit evaluations, eligibility, activation, ESS, progression",
        "observed btn_open / bb_iso / bb_3bet rates",
        "rejected actions",
    ],
}
PLAN_HASH = "sha256:" + sha256(canonical(PLAN))


class EvidenceError(RuntimeError):
    """E0006 evidence is missing, mismatched or unverifiable."""


# --- configuration ------------------------------------------------------------


def arms(item) -> tuple:
    """The (control, treatment) specs, after proving they differ only as declared."""
    by_arm = {s.arm: s for s in item.specs}
    if item.experiment_id != EXPERIMENT or set(by_arm) != {CONTROL, TREATMENT}:
        raise EvidenceError(
            "E0006 needs exactly the canonical and safe-exploit-v1 arms"
        )
    control, treatment = by_arm[CONTROL], by_arm[TREATMENT]
    check_arms(control, treatment)
    return control, treatment


def check_arms(control, treatment) -> None:
    a, b = control.to_dict(), treatment.to_dict()
    policy_a, policy_b = a.pop("policy_config"), b.pop("policy_config")
    a.pop("arm")
    b.pop("arm")
    if a != b:
        raise EvidenceError("arms differ outside safe_exploit")
    if policy_a != {"opponent_memory": True, "safe_exploit": False} or policy_b != {
        "opponent_memory": True,
        "safe_exploit": True,
    }:
        raise EvidenceError("arms must differ exactly in safe_exploit")
    cohort = tuple(o.qualified for o in control.opponent_cohort)
    if cohort != EXPLOIT_SENSITIVE + REFERENCE:
        raise EvidenceError("cohort differs from the predeclared E0006 cohort")
    if (
        not control.seed_policy.common_random_numbers
        or control.platform_config["stack_mode"] != "reset"
        or control.stopping.max_matches % len(cohort)
        or control.stopping.min_matches != control.stopping.max_matches
    ):
        raise EvidenceError("E0006 needs paired, balanced, fixed-horizon arms")


# --- evidence -----------------------------------------------------------------


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _policy_reached(state: GameState) -> bool:
    """Mirror of ``HoldemAgent.decide``: preflop policy runs with >1 choice."""
    valid = set(state.valid_actions)
    passive = "check" in valid or "call" in valid
    aggressive = ("raise" in valid and state.max_raise > 0) or "all_in" in valid
    return passive + ("fold" in valid) + aggressive > 1


def replay(events: list[dict], soh: int) -> tuple[list[dict], OpponentMemory]:
    """Every SafeExploit evaluation SOH's own memory would have produced.

    Memory receives exactly the bot's notifications in delivery order; each
    preflop request that reaches the preflop policy is evaluated by the pure
    controller. In the canonical arm these are counterfactual (unused).
    """
    memory, controller = OpponentMemory(), SafeExploitController()
    records = []
    for event in events:
        if event.get("kind") != "message" or soh not in event["to"]:
            continue
        message = event["message"]
        kind = message.get("type")
        if kind == "turn_request":
            if message.get("seat") != soh or message["state"]["phase"] != "preflop":
                continue
            state = GameState.from_turn_request(
                message, your_seat=soh, dealer_seat=message["state"]["dealer_seat"]
            )
            if not _policy_reached(state):
                continue
            context, _ = diagnose_context(state)
            if context is None:
                continue
            try:
                hand_class([str(c) for c in state.hole_cards])
            except ValueError:
                continue
            records.append(
                {
                    "hand": message["state"]["hand_number"],
                    **controller.adjust(context, memory).to_dict(),
                }
            )
        elif kind in {
            "match_start",
            "reconnected",
            "round_start",
            "round_result",
            "turn_result",
            "match_end",
        }:
            memory.notify(kind, message, soh)
    return records, memory


def _match(spec, index: int, events: list[dict], hands: list[dict], decisions):
    meta = [e for e in events if e.get("kind") == "match_meta"]
    summary = [e for e in events if e.get("kind") == "match_summary"]
    planned = spec.stopping.hands_per_match
    if len(meta) != 1 or len(summary) != 1:
        raise EvidenceError(f"match {index}: expected one meta and one summary")
    meta, summary = meta[0], summary[0]
    if meta["opponent"] != spec.opponent_for(index).qualified:
        raise EvidenceError(f"match {index}: opponent differs from the assignment")
    if meta["seeds"] != match_seeds(spec, index) or meta["soh_seat"] != 0:
        raise EvidenceError(f"match {index}: seeds or seat differ from the spec")
    # The journal envelope (``seq``) counts every event, so it diverges with play.
    deals = [
        {k: v for k, v in e.items() if k not in {"match", "seq"}}
        for e in events
        if e.get("kind") == "deal"
    ]
    numbers = list(range(1, planned + 1))
    if (
        summary["hands"] != planned
        or summary["ended"] != "complete"
        or [d["hand"] for d in deals] != numbers
        or [h["hand"] for h in hands] != numbers
    ):
        raise EvidenceError(f"match {index}: incomplete or reordered hands")
    records, memory = replay(events, meta["soh_seat"])
    return {
        "opponent": meta["opponent"],
        "seeds": meta["seeds"],
        "deals": sha256(canonical(deals)),
        "bb_per_100": 100
        * sum(h["soh_net"] / h["big_blind"] for h in hands)
        / len(hands),
        "decisions": {
            (d["hand"], d["seq"]): (d["phase"], d["action"], d["amount"])
            for d in decisions
        },
        "rejected": Counter(
            "soh" if e["seat"] == meta["soh_seat"] else "opponent"
            for e in events
            if e.get("kind") == "rejected"
        ),
        "exploit": records,
        "tendencies": {
            t: [c.successes, c.opportunities]
            for t, c in memory.current.items()
            if t in TENDENCIES
        },
    }


def load_arm(root: Path, snapshot: Snapshot, spec) -> tuple[dict, list]:
    """Per-match evidence of one arm: exactly the planned shards, verified."""
    completed = [r for r in snapshot.runs(spec.spec_hash) if r["status"] == "COMPLETED"]
    shards = sorted((r["shard_start"], r["shard_end"]) for r in completed)
    if shards != spec.stopping.shards():
        raise EvidenceError(f"{spec.arm}: completed shards differ from the horizon")
    matches, runs = {}, []
    for run in sorted(completed, key=lambda r: r["shard_start"]):
        directory = Path(run["run_dir"])
        raw = journal.read_raw(directory)
        hands = _rows(directory / "normalized" / "hands.jsonl")
        decisions = _rows(directory / "normalized" / "decisions.jsonl")
        planned = range(run["shard_start"], run["shard_end"])
        if {e.get("match") for e in raw if "match" in e} != set(planned):
            raise EvidenceError(f"{run['run_id']}: matches differ from its shard")
        for index in planned:
            matches[index] = _match(
                spec,
                index,
                [e for e in raw if e.get("match") == index],
                [h for h in hands if h["match"] == index],
                [d for d in decisions if d["match"] == index],
            )
        runs.append([run["run_id"], run["raw_sha256"]])
    return matches, runs


def verify(root: Path, item) -> tuple[dict, dict, dict]:
    """Deep verification, root isolation and exact match-index pairing."""
    root = Path(root)
    control, treatment = arms(item)
    snapshot = Snapshot(root)
    allowed = {control.spec_hash, treatment.spec_hash}
    if {s.spec_hash for s in snapshot.registered} - allowed or any(
        r["spec_hash"] not in allowed for r in snapshot.records
    ):
        raise EvidenceError("root holds evidence outside E0006; use its own root")
    _verify_evidence(root, item)  # raw + derived checksums against the index
    left, left_runs = load_arm(root, snapshot, control)
    right, right_runs = load_arm(root, snapshot, treatment)
    for index in range(control.stopping.max_matches):
        a, b = left.get(index), right.get(index)
        if a is None or b is None:
            raise EvidenceError(f"match {index} is missing from an arm")
        if a["opponent"] != b["opponent"] or a["seeds"] != b["seeds"]:
            raise EvidenceError(f"match {index}: arms are not paired")
        if a["deals"] != b["deals"]:
            raise EvidenceError(f"match {index}: pre-intervention deals differ")
    return left, right, {CONTROL: left_runs, TREATMENT: right_runs}


# --- statistics ----------------------------------------------------------------


def classify(low: float, high: float, resolution: float = RESOLUTION) -> str:
    if low > 0:
        return "IMPROVEMENT_DETECTED"
    if high < 0:
        return "REGRESSION_DETECTED"
    if -resolution <= low and high <= resolution:
        return "NO_DIFFERENCE_DETECTED"
    return "INCONCLUSIVE"


def bonferroni_z(family_size: int, alpha: float = FAMILY_ALPHA) -> float:
    if type(family_size) is not int or family_size < 1:
        raise ValueError("family must be positive")
    return NormalDist().inv_cdf(1 - alpha / (2 * family_size))


def stratified(diffs: dict[str, list[float]], weights: dict[str, float]) -> dict:
    """``sum(w * mean_o)`` with the independent-strata standard error."""
    if set(weights) - set(diffs):
        raise EvidenceError("a weighted opponent has no paired matches")
    if any(len(diffs[o]) < 2 for o in weights):
        raise EvidenceError("each weighted opponent needs at least two matches")
    mean = variance = 0.0
    strata, degenerate = {}, []
    for opponent, weight in sorted(weights.items()):
        values = diffs[opponent]
        n = len(values)
        m = sum(values) / n
        var = sum((v - m) ** 2 for v in values) / (n - 1)
        mean += weight * m
        variance += weight**2 * var / n
        strata[opponent] = {"weight": weight, "matches": n, "mean": m}
        # Exact identity, not a variance threshold: rounding cannot hide it.
        if len(set(values)) == 1:
            degenerate.append(opponent)
    se = math.sqrt(variance)
    return {"mean": mean, "se": se, "strata": strata, "degenerate_strata": degenerate}


def _interval(estimate: dict, z: float) -> list[float]:
    return [
        estimate["mean"] - z * estimate["se"],
        estimate["mean"] + z * estimate["se"],
    ]


def _judge(estimate: dict, corrected_z: float | None = None) -> dict:
    """Intervals and classification; degenerate variance is never a claim.

    The primary is classified on its 95% interval, a secondary member on its
    Bonferroni ``corrected`` interval. A zero-variance weighted stratum makes
    the standard error meaningless, so no interval is reported and the
    result is INCONCLUSIVE with the affected strata named.
    """
    keys = ["ci95"] + ([] if corrected_z is None else ["corrected"])
    if estimate["degenerate_strata"]:
        estimate.update(dict.fromkeys(keys))
        estimate.update(
            classification="INCONCLUSIVE",
            explanation=PLAN["degenerate_variance"],
        )
        return estimate
    estimate["ci95"] = _interval(estimate, Z95)
    if corrected_z is not None:
        estimate["corrected"] = _interval(estimate, corrected_z)
    estimate["classification"] = classify(*estimate[keys[-1]])
    return estimate


def contrasts(left: dict, right: dict, matches) -> dict:
    """Primary, secondary family and stability from paired per-match values."""
    diffs = defaultdict(list)
    for index in matches:
        diffs[left[index]["opponent"]].append(
            right[index]["bb_per_100"] - left[index]["bb_per_100"]
        )
    primary = _judge(stratified(diffs, PLAN["primary"]["weights"]))
    z = bonferroni_z(len(SECONDARY))
    members = {}
    for label in SECONDARY:
        if label == "reference_equal_weight":
            weights = PLAN["secondary"]["reference_weights"]
        else:
            weights = {label.split(":", 1)[1]: 1.0}
        members[label] = _judge(stratified(diffs, weights), z)
    return {
        "primary": primary,
        "secondary": {
            "family_size": len(SECONDARY),
            "family_alpha": FAMILY_ALPHA,
            "method": "bonferroni",
            "z": z,
            "members": members,
        },
    }


def stability(left: dict, right: dict, total: int) -> dict:
    """Descriptive primary contrast on independent match-index halves."""
    halves = {
        "first_half": range(total // 2),
        "second_half": range(total // 2, total),
    }
    out = {}
    for name, block in halves.items():
        result = contrasts(left, right, block)["primary"]
        out[name] = {k: result[k] for k in ("mean", "se", "ci95", "degenerate_strata")}
    out["same_sign"] = len({math.copysign(1, v["mean"]) for v in out.values()}) == 1
    out["binding"] = False
    return out


# --- diagnostics -----------------------------------------------------------------


def _quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)

    def at(q):
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p25": at(0.25),
        "median": at(0.5),
        "p75": at(0.75),
        "max": ordered[-1],
    }


def _exploit(matches: list[dict], hands: int) -> dict:
    reasons = Counter()
    eligible_ess, first_active = [], []
    activated = full = 0
    bins = defaultdict(lambda: [0, 0, 0.0])  # eligible, activated, weight sum
    for match in matches:
        first = None
        for record in match["exploit"]:
            reasons[(record["facing"], record["reason"])] += 1
            if record["reason"] in {"unsupported_context", "unsupported_line"}:
                continue
            eligible_ess.append(record["evidence_ess"])
            cell = bins[(record["hand"] - 1) // HAND_BIN]
            cell[0] += 1
            if record["evidence_weight"] > 0:
                activated += 1
                full += record["evidence_weight"] >= 1
                cell[1] += 1
                cell[2] += record["evidence_weight"]
                first = record["hand"] if first is None else first
        if first is not None:
            first_active.append(first)
    supported = {
        k: v
        for k, v in reasons.items()
        if k[1] not in {"unsupported_context", "unsupported_line"}
    }
    return {
        "evaluations": sum(reasons.values()),
        "by_context_reason": {f"{f}/{r}": n for (f, r), n in sorted(reasons.items())},
        "eligible": sum(supported.values()),
        "activated": activated,
        "full_weight": full,
        "changed_width": sum(n for (_, r), n in supported.items() if r == "adjusted"),
        "eligible_ess": _quantiles(eligible_ess),
        "matches_with_activation": len(first_active),
        "first_activation_hand": _quantiles(first_active),
        "progression": [
            {
                "hands": [b * HAND_BIN + 1, min(hands, (b + 1) * HAND_BIN)],
                "eligible": cell[0],
                "activated": cell[1],
                "mean_weight_when_active": cell[2] / cell[1] if cell[1] else None,
            }
            for b, cell in sorted(bins.items())
        ],
    }


def _divergence(left: list[dict], right: list[dict]) -> dict:
    compared = identical = 0
    firsts = []
    for a, b in zip(left, right):
        common = sorted(set(a["decisions"]) & set(b["decisions"]))
        same = [k for k in common if a["decisions"][k] == b["decisions"][k]]
        compared += len(common)
        identical += len(same)
        differing = [k for k in common if a["decisions"][k] != b["decisions"][k]]
        if differing or set(a["decisions"]) != set(b["decisions"]):
            keys = differing + sorted(set(a["decisions"]) ^ set(b["decisions"]))
            firsts.append(min(keys)[0])
    return {
        "compared": compared,
        "identical": identical,
        "matches_diverged": len(firsts),
        "first_divergence_hand": _quantiles(firsts),
    }


def diagnostics(left: dict, right: dict, hands: int) -> dict:
    out = {}
    for opponent in EXPLOIT_SENSITIVE + REFERENCE:
        indices = sorted(i for i in left if left[i]["opponent"] == opponent)
        a = [left[i] for i in indices]
        b = [right[i] for i in indices]
        per_arm = {}
        for arm, matches in ((CONTROL, a), (TREATMENT, b)):
            tendencies = {}
            for name in TENDENCIES:
                s = sum(m["tendencies"].get(name, [0, 0])[0] for m in matches)
                n = sum(m["tendencies"].get(name, [0, 0])[1] for m in matches)
                tendencies[name] = {
                    "successes": s,
                    "opportunities": n,
                    "rate": s / n if n else None,
                }
            per_arm[arm] = {
                "mean_bb_per_100": sum(m["bb_per_100"] for m in matches) / len(matches),
                "rejected": dict(sum((m["rejected"] for m in matches), Counter())),
                "tendencies": tendencies,
                "safe_exploit": _exploit(matches, hands),
            }
        out[opponent] = {
            "matches": len(indices),
            "arms": per_arm,
            "divergence": _divergence(a, b),
        }
    return out


# --- report ------------------------------------------------------------------------


def analyze(root: Path, programme_path: Path = PROGRAMME) -> dict:
    approved = programmes.load(programme_path)
    item = approved.item(EXPERIMENT)
    left, right, runs = verify(root, item)
    control, treatment = arms(item)
    total = control.stopping.max_matches
    result = contrasts(left, right, range(total))
    return {
        "experiment_id": EXPERIMENT,
        "plan": PLAN,
        "plan_hash": PLAN_HASH,
        "programme_hash": approved.digest,
        "specs": {CONTROL: control.spec_hash, TREATMENT: treatment.spec_hash},
        "policy_revision": control.policy_revision,
        "evidence": {
            "runs": runs,
            "matches_per_arm": total,
            "hands_per_match": control.stopping.hands_per_match,
            "paired_matches": total,
            "identical_seeds_and_deals": total,
        },
        **result,
        "stability": stability(left, right, total),
        "diagnostics": diagnostics(left, right, control.stopping.hands_per_match),
        "binding": False,
        "note": "Research evidence only; any policy change requires review, "
        "regression validation and explicit promotion.",
    }


def markdown(report: dict) -> str:
    def row(label, value, interval_key):
        if value[interval_key] is None:
            interval = (
                "n/a (zero variance: " + ", ".join(value["degenerate_strata"]) + ")"
            )
        else:
            low, high = value[interval_key]
            interval = f"[{low:+.2f}, {high:+.2f}]"
        return (
            f"| {label} | {value['mean']:+.2f} | {interval} | "
            f"{value['classification']} |"
        )

    p = report["primary"]
    lines = [
        f"# {report['experiment_id']} — SafeExploit v1 ablation (stratified)",
        "",
        (
            f"Plan {report['plan_hash']}; {report['evidence']['matches_per_arm']} "
            f"paired matches x {report['evidence']['hands_per_match']} hands per arm."
        ),
        "",
        "| endpoint | bb/100 | interval | classification |",
        "|---|---|---|---|",
        row("primary: exploit-sensitive equal weight (95%)", p, "ci95"),
    ]
    for label, value in report["secondary"]["members"].items():
        lines.append(row(f"secondary: {label} (Bonferroni)", value, "corrected"))
    s = report["stability"]
    lines += [
        "",
        (
            f"Stability (descriptive): first half {s['first_half']['mean']:+.2f}, "
            f"second half {s['second_half']['mean']:+.2f}; same sign {s['same_sign']}."
        ),
        "",
        "| opponent | eligible | activated | changed | decisions identical |",
        "|---|---|---|---|---|",
    ]
    for opponent, d in report["diagnostics"].items():
        e = d["arms"][TREATMENT]["safe_exploit"]
        v = d["divergence"]
        lines.append(
            f"| {opponent} | {e['eligible']} | {e['activated']} | "
            f"{e['changed_width']} | {v['identical']}/{v['compared']} |"
        )
    lines += ["", report["note"]]
    return "\n".join(lines)


class ReportConflict(EvidenceError):
    """An E0006 report differs from, or only partly matches, what exists."""


def _create_once(path: Path, data: bytes) -> None:
    """Publish ``data`` read-only at ``path``; never replace an existing file."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, 0o444)
        os.link(temporary, path)  # atomic, and fails if the path exists
    finally:
        temporary.unlink()


def write(root: Path, report: dict) -> Path:
    """Write the final report once; an identical rewrite changes nothing.

    A differing existing report, or a pair with only one file present, is
    rejected for review: an earlier research result is never replaced.
    """
    out = Path(root) / "analysis"
    path = out / f"{EXPERIMENT}.stratified.json"
    outputs = {
        path: canonical(report) + b"\n",
        out / f"{EXPERIMENT}.stratified.md": (markdown(report) + "\n").encode(),
    }
    present = [p for p in outputs if p.exists()]
    if present:
        if len(present) != len(outputs):
            raise ReportConflict("partial E0006 report pair; review required")
        if any(p.read_bytes() != data for p, data in outputs.items()):
            raise ReportConflict("an existing E0006 report differs; never replaced")
        return path
    out.mkdir(parents=True, exist_ok=True)
    for target, data in outputs.items():
        try:
            _create_once(target, data)
        except FileExistsError as exc:
            raise ReportConflict(
                "concurrent E0006 report write; review required"
            ) from exc
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--programme", type=Path, default=PROGRAMME)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.root.resolve() == HISTORICAL_ROOT.resolve():
        print("e0006: the historical runs root is reserved", file=sys.stderr)
        return 1
    try:
        report = analyze(args.root, args.programme)
        write(args.root, report)
    except (EvidenceError, OSError, ValueError, RuntimeError) as exc:
        print(f"e0006: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2) if args.json else markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
