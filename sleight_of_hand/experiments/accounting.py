"""Reassemble and classify SOH_ACCOUNTING_CAPTURE_V1 observer captures.

The observer itself (``bots/chipzen/accounting_observer.py``) is reused
unchanged; this module only reads its frames from a bot's captured stdout.
Frames are checksum-verified before any JSON is parsed. Classification is
measurement only and never feeds back into pricing.
"""

from __future__ import annotations

import hashlib
import json

PREFIX = "SOH_ACCOUNTING_CAPTURE_V1"
CLASSES = (
    "CONFIRMED_EXCESS_INCLUDED",
    "RECONCILED_BEFORE_DECISION",
    "SERVER_EFFECTIVE_STACK_CAP",
    "INCONCLUSIVE",
)


class Inconclusive(ValueError):
    """The capture cannot support a classification."""


def frames(text: str) -> list[str]:
    """Framed lines; log viewers may prepend timestamps or stream labels."""
    found = []
    for line in text.splitlines():
        at = line.find(PREFIX)
        if at >= 0:
            found.append(line[at:].rstrip("\r"))
    return found


def reassemble(lines: list[str]) -> tuple[str, str]:
    """Return (payload, checksum note); raise Inconclusive on any defect."""
    if not lines:
        raise Inconclusive("no capture frames")
    if lines[0].startswith(PREFIX + ":"):
        if len(lines) != 1:
            raise Inconclusive("extra frames after a single-line capture")
        return lines[0][len(PREFIX) + 1 :], "single-line (no checksum by design)"
    if not lines[-1].startswith(PREFIX + " SHA256 "):
        raise Inconclusive("missing terminal SHA256 frame (truncated?)")
    chunks: dict[int, str] = {}
    total = None
    for line in lines[:-1]:
        try:
            framing, piece = line[len(PREFIX) + 1 :].split(" ", 1)
            index, count = map(int, framing.split("/"))
        except ValueError as exc:
            raise Inconclusive("malformed chunk frame") from exc
        if total not in (None, count) or index in chunks:
            raise Inconclusive("inconsistent or duplicate chunk framing")
        total, chunks[index] = count, piece
    if sorted(chunks) != list(range(1, (total or 0) + 1)):
        raise Inconclusive("missing chunks")
    payload = "".join(chunks[i] for i in range(1, total + 1))
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
    if digest != lines[-1].split()[-1]:
        raise Inconclusive("checksum mismatch")
    return payload, f"VALID sha256={digest} chunks={total}"


def _inconclusive(out: dict, why: str) -> dict:
    return {**out, "classification": "INCONCLUSIVE", "why": why}


def classify(capture: dict) -> dict:
    """Derive A, P_after, H, T, P_delivered and classify the target street.

    Requires a clean street: the opponent's target action must be its only
    action on that street before SOH's decision, so its amount is both the
    raise-to total and its whole street contribution.
    """
    events = capture.get("events", [])
    trigger = capture.get("trigger") or {}
    alias = trigger.get("decision_alias")
    out = {"trigger": trigger, "buffer_complete": capture.get("buffer_complete")}
    position = next(
        (
            i
            for i, e in enumerate(events)
            if e.get("event") == "decision_state"
            and e.get("data", {}).get("decision_alias") == alias
        ),
        None,
    )
    if position is None or capture.get("buffer_complete") is not True:
        return _inconclusive(out, "no complete target decision")
    decision = events[position]["data"]
    street, hero = decision.get("phase"), decision.get("your_seat")
    out["street"] = street
    prior = [e["data"] for e in events[:position] if e.get("event") == "turn_result"]
    target = prior[-1] if prior else None
    out["target_turn_result"] = target
    if street in (None, "preflop"):
        return _inconclusive(out, "target is not a postflop street")
    if target is None or target.get("seat") in (None, hero):
        return _inconclusive(out, "no opponent action before the decision")
    history = [
        h for h in decision.get("action_history", []) if h.get("phase") == street
    ]
    mine = [h for h in history if h.get("seat") == target["seat"]]
    if len(mine) != 1 or history[-1].get("seat") != target["seat"]:
        return _inconclusive(out, "opponent had another contribution on the street")
    A, P_after = target.get("amount"), target.get("pot")
    H, T, P_delivered = (
        decision.get("your_stack"),
        decision.get("to_call"),
        decision.get("pot"),
    )
    out["raw"] = {
        "A": A,
        "P_after": P_after,
        "H": H,
        "T": T,
        "P_delivered": P_delivered,
    }
    if any(type(v) is not int for v in (A, H, T, P_delivered)):
        return _inconclusive(out, "missing raw values")
    if type(P_after) is not int:
        return _inconclusive(out, "turn_result carried no pot")
    # P_before must be independent of the target action's own report. The
    # pot after the preceding action has no unmatched chips under any
    # convention; P_after - A (the task's definition) is kept for comparison
    # and silently assumes the turn_result pot includes the whole bet.
    previous = prior[-2].get("pot") if len(prior) >= 2 else None
    P_before = previous if type(previous) is int else P_after - A
    C, U = min(A, H), max(0, A - H)
    out["derived"] = {
        "P_before": P_before,
        "P_before_source": (
            "preceding turn_result pot" if type(previous) is int else "P_after - A"
        ),
        "P_after_minus_A": P_after - A,
        "turn_result_pot_includes_bet": P_after == P_before + A,
        "C": C,
        "U": U,
        "P_contestable": P_before + C,
    }
    out["secondary_to_call"] = (
        "T >= H (to_call not capped)" if T >= H else "T < H (to_call capped)"
    )
    stacks = decision.get("stacks") or {}
    behind = stacks.get(str(target["seat"]))
    out["raw"]["opponent_behind"] = behind
    if A <= H:
        if A == H and type(behind) is int and behind > 0:
            return {
                **out,
                "classification": "SERVER_EFFECTIVE_STACK_CAP",
                "why": "the bet stopped exactly at SOH's stack with chips behind",
            }
        return _inconclusive(out, "geometry not reached: the bet did not cover SOH")
    if U == 0:
        return _inconclusive(out, "no unmatched excess")
    if P_delivered == P_before + A:
        return {**out, "classification": "CONFIRMED_EXCESS_INCLUDED"}
    if P_delivered == P_before + C:
        return {**out, "classification": "RECONCILED_BEFORE_DECISION"}
    return _inconclusive(out, "pot relationships match neither hypothesis")


def analyze_text(text: str) -> dict:
    """Full pipeline over captured stdout; never raises for bad evidence."""
    try:
        payload, checksum = reassemble(frames(text))
    except Inconclusive as exc:
        return {"classification": "INCONCLUSIVE", "why": str(exc), "checksum": None}
    capture = json.loads(payload)
    return {"checksum": checksum, "raw_capture": payload, **classify(capture)}
