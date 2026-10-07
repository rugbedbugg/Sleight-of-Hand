"""Raw platform events -> normalized SOH interpretation (regenerable).

Both platforms journal the same envelope, so one message-level parser serves
them: ``{"kind": "message", "to": [seats], "message": <as delivered>}``,
``{"kind": "action", "seat", "action", "amount"}`` and a per-match
``{"kind": "match_meta", "soh_seat"}``. Platform-only events (the local
dealer's ``deal`` truth, ``seat_output``) are ignored here. Outputs are
deterministic functions of the raw stream; bump ``VERSION`` on any change.
"""

from __future__ import annotations

from collections import defaultdict

VERSION = "normalize/1"
POSTS = {"post_small_blind", "post_big_blind", "post_ante"}


def _by_match(events: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        if "match" in event:
            grouped[event["match"]].append(event)
    return dict(sorted(grouped.items()))


def tendencies(history: list[dict], button: int, start_stacks: list[int]):
    """(seat, stat, success) opportunities one completed hand reveals."""
    out = []
    big = 1 - button
    preflop = [
        h for h in history if h.get("phase") == "preflop" and h["action"] not in POSTS
    ]
    if preflop and preflop[0]["seat"] == button:
        first = preflop[0]["action"]
        out += [
            (button, "btn_rfi", first == "raise"),
            (button, "btn_limp", first == "call"),
            (button, "btn_fold", first == "fold"),
        ]
        reply = next((h for h in preflop[1:] if h["seat"] == big), None)
        if reply is not None and first == "raise":
            out += [
                (big, "bb_3bet", reply["action"] == "raise"),
                (big, "bb_call_open", reply["action"] == "call"),
                (big, "bb_fold_to_open", reply["action"] == "fold"),
            ]
        if reply is not None and first == "call":
            out.append((big, "bb_iso", reply["action"] == "raise"))
    seen: set[int] = set()
    for h in preflop:
        if h["seat"] in seen:
            continue
        seen.add(h["seat"])
        shove = h["action"] == "raise" and h["amount"] >= start_stacks[h["seat"]]
        out.append((h["seat"], "preflop_shove", shove))
    for h in history:
        if h.get("phase") in ("flop", "turn", "river") and h["action"] in (
            "raise",
            "call",
            "check",
            "fold",
        ):
            out.append((h["seat"], "postflop_aggression", h["action"] == "raise"))
    raisers = [h["seat"] for h in preflop if h["action"] == "raise"]
    flop = [h for h in history if h.get("phase") == "flop"]
    if raisers and flop:
        aggressor = raisers[-1]
        before = []
        for h in flop:
            if h["seat"] == aggressor:
                if not any(b["action"] == "raise" for b in before):
                    cbet = h["action"] == "raise"
                    out.append((aggressor, "cbet", cbet))
                    if cbet:
                        after = flop[flop.index(h) + 1 :]
                        reply = next((a for a in after if a["seat"] != aggressor), None)
                        if reply is not None:
                            out.append(
                                (
                                    1 - aggressor,
                                    "fold_to_cbet",
                                    reply["action"] == "fold",
                                )
                            )
                break
            before.append(h)
    return out


def normalize(events: list[dict]) -> dict[str, list[dict]]:
    hands, decisions, observations = [], [], []
    for match, stream in _by_match(events).items():
        soh = next(
            (e["soh_seat"] for e in stream if e.get("kind") == "match_meta"), None
        )
        if soh is None:
            continue  # cannot attribute anything without knowing SOH's seat
        starts: dict[str, dict] = {}
        pending = None
        per_hand = defaultdict(int)
        for event in stream:
            kind = event.get("kind")
            message = event.get("message") or {}
            mtype = message.get("type")
            if kind == "message" and mtype == "round_start" and soh in event["to"]:
                starts[message["round_id"]] = message["state"]
            elif kind == "message" and mtype == "turn_request" and soh in event["to"]:
                pending = message
            elif kind == "action" and event.get("seat") == soh and pending is not None:
                state = pending["state"]
                hand = state.get("hand_number")
                decisions.append(
                    {
                        "match": match,
                        "hand": hand,
                        "seq": per_hand[hand],
                        "phase": state.get("phase"),
                        "pot": state.get("pot"),
                        "to_call": state.get("to_call"),
                        "your_stack": state.get("your_stack"),
                        "opponent_stack": (state.get("opponent_stacks") or [None])[0],
                        "min_raise": state.get("min_raise"),
                        "max_raise": state.get("max_raise"),
                        "valid": pending.get("valid_actions"),
                        "action": event.get("action"),
                        "amount": event.get("amount"),
                    }
                )
                per_hand[hand] += 1
                pending = None
            elif kind == "message" and mtype == "round_result" and soh in event["to"]:
                start = starts.get(message.get("round_id"))
                result = message.get("result") or {}
                if start is None or "stacks" not in result:
                    continue
                history = result.get("action_history") or []
                button = start["dealer_seat"]
                big_blind = next(
                    (h["amount"] for h in history if h["action"] == "post_big_blind"),
                    None,
                )
                phases = [h.get("phase") for h in history]
                last = max(
                    (
                        ("preflop", "flop", "turn", "river").index(p)
                        for p in phases
                        if p
                    ),
                    default=0,
                )
                hand = result.get("hand_number")
                hands.append(
                    {
                        "match": match,
                        "hand": hand,
                        "round_id": message.get("round_id"),
                        "soh_seat": soh,
                        "button": button,
                        "soh_is_button": button == soh,
                        "big_blind": big_blind,
                        "start_stacks": start["stacks"],
                        "end_stacks": result["stacks"],
                        "soh_net": result["stacks"][soh] - start["stacks"][soh],
                        "pot": result.get("pot"),
                        "showdown": bool(result.get("showdown")),
                        "last_street": ("preflop", "flop", "turn", "river")[last],
                        "winner_seats": result.get("winner_seats"),
                    }
                )
                for seat, stat, success in tendencies(history, button, start["stacks"]):
                    observations.append(
                        {
                            "match": match,
                            "hand": hand,
                            "subject": "soh" if seat == soh else "opponent",
                            "stat": stat,
                            "success": int(success),
                        }
                    )
    return {"hands": hands, "decisions": decisions, "observations": observations}
