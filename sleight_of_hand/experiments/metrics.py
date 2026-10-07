"""Outcome, behavioral, model and accounting metrics.

Only metrics the observations support are computed. Uncertainty is reported
with every estimate: outcome intervals are clustered by match (hands within
a match share a bot instance), behavioral rates carry Wilson intervals, and
model metrics are prequential (each prediction is made before its outcome).
"""

from __future__ import annotations

import math
from collections import defaultdict

from sleight_of_hand.holdem.memory import OpponentBelief, OpponentMemory
from sleight_of_hand.holdem.profiles import Counts

from . import accounting as accounting_capture
from .platforms.priors import memory_for

Z95 = 1.959963984540054


def wilson(successes: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + Z95**2 / n
    centre = (p + Z95**2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / denom
    return [round(centre - half, 6), round(centre + half, 6)]


def clustered(values_by_cluster: dict) -> dict:
    """Ratio estimator of the per-item mean with a cluster-robust SE."""
    totals = [(sum(v), len(v)) for v in values_by_cluster.values() if v]
    n = sum(k for _, k in totals)
    m = len(totals)
    if n == 0:
        return {"n": 0, "clusters": 0, "mean": None, "se": None, "ci95": None}
    mean = sum(t for t, _ in totals) / n
    if m < 2:
        return {"n": n, "clusters": m, "mean": mean, "se": None, "ci95": None}
    resid = sum((t - mean * k) ** 2 for t, k in totals)
    se = math.sqrt(m / (m - 1) * resid) / n
    return {
        "n": n,
        "clusters": m,
        "mean": mean,
        "se": se,
        "ci95": [mean - Z95 * se, mean + Z95 * se],
    }


def outcome(hands: list[dict]) -> dict:
    by_match_bb = defaultdict(list)
    by_match_chips = defaultdict(list)
    wins = losses = ties = 0
    for hand in hands:
        net, bb = hand["soh_net"], hand["big_blind"] or 1
        by_match_bb[hand["match"]].append(net / bb)
        by_match_chips[hand["match"]].append(net)
        wins += net > 0
        losses += net < 0
        ties += net == 0
    bb = clustered(by_match_bb)
    match_totals = [sum(v) for v in by_match_chips.values()]
    result = {
        "hands": len(hands),
        "matches": len(by_match_chips),
        "chips_per_hand": clustered(by_match_chips),
        "bb_per_100": {
            k: (
                None
                if v is None
                else (v * 100 if k != "ci95" else [x * 100 for x in v])
            )
            for k, v in bb.items()
            if k in ("mean", "se", "ci95")
        },
        "hands_won": wins,
        "hands_lost": losses,
        "hands_tied": ties,
        "matches_won": sum(t > 0 for t in match_totals),
        "matches_lost": sum(t < 0 for t in match_totals),
        "showdown_rate": (
            sum(h["showdown"] for h in hands) / len(hands) if hands else None
        ),
    }
    return result


def behavior(observations: list[dict]) -> dict:
    counts = defaultdict(lambda: [0, 0])
    for o in observations:
        cell = counts[(o["subject"], o["stat"])]
        cell[0] += o["success"]
        cell[1] += 1
    out: dict = defaultdict(dict)
    for (subject, stat), (s, n) in sorted(counts.items()):
        out[subject][stat] = {
            "successes": s,
            "opportunities": n,
            "rate": s / n,
            "ci95": wilson(s, n),
        }
    return dict(out)


def _score(pairs: list[tuple[float, int]]) -> dict:
    if not pairs:
        return {"n": 0}
    eps = 1e-6
    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    logloss = -sum(
        math.log(max(eps, p)) if y else math.log(max(eps, 1 - p)) for p, y in pairs
    ) / len(pairs)
    bins = defaultdict(list)
    for p, y in pairs:
        bins[min(9, int(p * 10))].append((p, y))
    ece = sum(
        len(b)
        / len(pairs)
        * abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b))
        for b in bins.values()
    )
    base = sum((0.5 - y) ** 2 for _, y in pairs) / len(pairs)
    return {
        "n": len(pairs),
        "brier": brier,
        "log_loss": logloss,
        "calibration_error": ece,
        "brier_uniform_prior": base,
        "brier_skill": None if base == 0 else 1 - brier / base,
    }


def model(raw_events: list[dict], policy_config: dict) -> dict:
    """Replay OpponentMemory over SOH-delivered messages, scoring predictions.

    Each prediction is the belief mean immediately before the round result
    that resolves it. The replay never touches the policy or the live bot.
    """
    pairs: dict[str, list[tuple[float, int]]] = defaultdict(list)
    halves = {"first_half": [], "second_half": []}
    final_ess: dict[str, list[float]] = defaultdict(list)
    matches = defaultdict(list)
    for event in raw_events:
        if "match" in event:
            matches[event["match"]].append(event)
    for stream in matches.values():
        soh = next(
            (e["soh_seat"] for e in stream if e.get("kind") == "match_meta"), None
        )
        if soh is None:
            continue
        memory = memory_for(policy_config) or OpponentMemory()
        messages = [
            e["message"]
            for e in stream
            if e.get("kind") == "message" and soh in e["to"]
        ]
        results = sum(1 for m in messages if m.get("type") == "round_result")
        seen = 0
        for message in messages:
            mtype = message.get("type")
            if mtype in ("match_start", "round_start", "match_end"):
                memory.notify(mtype, message, soh)
            elif mtype == "round_result":
                names = set(memory.current) | (
                    set(dict(memory.prior.stats)) if memory.prior else set()
                )
                before = {t: memory.current.get(t) for t in names}
                means = {t: memory.belief(t).mean for t in names}
                memory.notify("round_result", message, soh)
                seen += 1
                for tendency, counts in memory.current.items():
                    old = before.get(tendency)
                    d_n = counts.opportunities - (old.opportunities if old else 0)
                    d_s = counts.successes - (old.successes if old else 0)
                    if d_n != 1:
                        continue
                    p = means.get(tendency)
                    if p is None:  # first sighting: predict the prior mean
                        p = _prior_mean(memory, tendency)
                    pairs[tendency].append((p, d_s))
                    half = "first_half" if seen <= results / 2 else "second_half"
                    halves[half].append((p, d_s))
        for tendency in memory.current:
            final_ess[tendency].append(memory.belief(tendency).effective_sample_size)
    overall = [pair for values in pairs.values() for pair in values]
    return {
        "replayed_with_prior": bool(policy_config.get("historical_prior")),
        "overall": _score(overall),
        "by_tendency": {t: _score(v) for t, v in sorted(pairs.items())},
        "convergence": {k: _score(v) for k, v in halves.items()},
        "mean_final_ess": {
            t: sum(v) / len(v) for t, v in sorted(final_ess.items()) if v
        },
    }


def _prior_mean(memory: OpponentMemory, tendency: str) -> float:
    """The belief before any match-local evidence for ``tendency``."""
    prior = dict(memory.prior.stats) if memory.prior else {}
    return OpponentBelief(prior.get(tendency, Counts()), Counts(), memory.config).mean


def classify_match(raw_events: list[dict]) -> dict:
    """Classify one match's observer capture (one bot process, one capture)."""
    text = "".join(
        e.get("stdout", "") for e in raw_events if e.get("kind") == "seat_output"
    )
    if accounting_capture.PREFIX not in text:
        return {"classification": "INCONCLUSIVE", "why": "no qualifying capture"}
    return accounting_capture.analyze_text(text)


def accounting(raw_events: list[dict]) -> dict:
    """Per-match classifications and a primary verdict.

    The primary verdict is the conclusive class every conclusive match agrees
    on; with no conclusive match, or with disagreement, it is INCONCLUSIVE.
    """
    matches = defaultdict(list)
    for event in raw_events:
        matches[event.get("match")].append(event)
    per_match = {
        str(m): classify_match(events) for m, events in sorted(matches.items())
    }
    counts = defaultdict(int)
    for result in per_match.values():
        counts[result["classification"]] += 1
    conclusive = {c for c in counts if c != "INCONCLUSIVE"}
    primary = conclusive.pop() if len(conclusive) == 1 else "INCONCLUSIVE"
    return {
        "primary": primary,
        "conflicting": len(conclusive) > 0,
        "classifications": dict(sorted(counts.items())),
        "per_match": per_match,
    }


def compute(groups, normalized: dict, raw_events: list[dict], policy_config: dict):
    out = {}
    if "outcome" in groups:
        out["outcome"] = outcome(normalized["hands"])
    if "behavior" in groups:
        out["behavior"] = behavior(normalized["observations"])
    if "model" in groups:
        out["model"] = model(raw_events, policy_config)
    if "accounting" in groups:
        out["accounting"] = accounting(raw_events)
    return out
