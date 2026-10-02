"""One opt-in, public accounting capture. No policy or RNG dependencies.

Only modern SDK hooks feed this collector. All payloads are allowlisted; round
IDs stay in memory for correlation and are never written. This is evidence of
what was delivered, not an interpretation of contestable pots or contribution
semantics. A trigger does not establish that unmatched chips exist.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PHASES = frozenset(("preflop", "flop", "turn", "river", "showdown"))
ACTIONS = frozenset(
    (
        "fold",
        "check",
        "call",
        "bet",
        "raise",
        "all_in",
        "all-in",
        "post_small_blind",
        "post_big_blind",
        "post_ante",
        "return_uncalled",
        "uncalled_bet",
        "refund",
    )
)
MAX_EVENTS = 128
MAX_BYTES = 65536  # Serialized events; fixed document metadata is < 1 KiB.
RESULT_RESERVE = 16384
MAX_HISTORY = 64
MAX_SEATS = 10


def warning() -> None:
    # Never include paths, exception text, or payload values in diagnostics.
    try:
        print("[accounting observer] disabled: diagnostic unavailable", file=sys.stderr)
    except Exception:  # noqa: BLE001, S110 - even stderr may be unavailable
        pass


def from_environment():
    if os.environ.get("SLEIGHT_CHIPZEN_ACCOUNTING_OBSERVER") != "1":
        return None
    try:
        value = os.environ.get("SLEIGHT_CHIPZEN_ACCOUNTING_LOG", "")
        if not value or len(value) > 4096:
            raise ValueError
        path = Path(value)
        if (
            path.exists()
            or not path.parent.is_dir()
            or not os.access(path.parent, os.W_OK)
        ):
            raise ValueError
        return AccountingObserver(path)
    except Exception:  # noqa: BLE001 - diagnostics must fail open
        warning()
        return None


def number(value):
    return type(value) is int and -(2**63) < value < 2**63


def seat(value):
    return type(value) is int and 0 <= value < MAX_SEATS


def fields(source, numeric=(), enums=(), booleans=()):
    if not isinstance(source, dict):
        return {}
    result = {k: source[k] for k in numeric if number(source.get(k))}
    for key, allowed in enums:
        value = source.get(key)
        if type(value) is str and value in allowed:
            result[key] = value
    for key in booleans:
        if type(source.get(key)) is bool:
            result[key] = source[key]
    return result


def correlation(message, nested):
    """Bounded private correlation token; never part of the persisted schema."""
    value = message.get("round_id")
    if type(value) is str and 0 < len(value) <= 128:
        return ("round", value)
    state = message.get(nested, {})
    value = state.get("hand_number") if isinstance(state, dict) else None
    return ("hand", value) if number(value) else None


class AccountingObserver:
    def __init__(self, path: Path):
        self.path = path
        self.armed = True
        self.alias = 0
        self.key = None
        self.closed_key = None
        self.capture = None
        self.used_bytes = 0
        self.decision = 0
        self.observed_decision = False
        self.seat_count = None

    def notify(self, event, *args):
        if self.armed:
            getattr(self, "_" + event)(*args)

    def _incomplete(self):
        self.capture["buffer_complete"] = False
        # Saturation keeps even indefinitely long malformed hands bounded.
        self.capture["dropped_events"] = min(
            self.capture["dropped_events"] + 1, 2**31 - 1
        )

    def _append(self, event, data, final=False):
        entry = {"event": event, "data": data}
        size = len(json.dumps(entry, separators=(",", ":"), allow_nan=False)) + 1
        limit = MAX_BYTES if final else MAX_BYTES - RESULT_RESERVE
        count = MAX_EVENTS if final else MAX_EVENTS - 1
        if len(self.capture["events"]) >= count or self.used_bytes + size > limit:
            self._incomplete()
            return
        self.capture["events"].append(entry)
        self.used_bytes += size

    def _history(self, value):
        if not isinstance(value, list):
            self._incomplete()
            return []
        if len(value) > MAX_HISTORY:
            self._incomplete()
        return [
            fields(
                item,
                ("seat", "amount"),
                (("phase", PHASES), ("action", ACTIONS)),
                ("is_timeout",),
            )
            for item in value[:MAX_HISTORY]
        ]

    def _public(self, source):
        result = fields(
            source,
            (
                "pot",
                "your_seat",
                "dealer_seat",
                "your_stack",
                "to_call",
                "min_raise",
                "max_raise",
                "small_blind",
                "big_blind",
                "ante",
            ),
            (("phase", PHASES),),
        )
        if not isinstance(source, dict):
            self._incomplete()
            return result
        for key in ("stacks", "post_blind_stacks"):
            values = source.get(key)
            if isinstance(values, list):
                if len(values) > MAX_SEATS:
                    self._incomplete()
                result[key] = {
                    str(i): v for i, v in enumerate(values[:MAX_SEATS]) if number(v)
                }
        if "board" in source:
            cards = source["board"]
            result["board"] = []
            if isinstance(cards, list):
                if len(cards) > 5:
                    self._incomplete()
                for card in cards[:5]:
                    # SDK Card fields, or protocol strings; never serialize an
                    # arbitrary object's repr/str (which could contain secrets).
                    if type(card) is not str:
                        rank, suit = (
                            getattr(card, "rank", None),
                            getattr(card, "suit", None),
                        )
                        card = (
                            rank + suit
                            if type(rank) is str
                            and type(suit) is str
                            and len(rank) == len(suit) == 1
                            else None
                        )
                    if (
                        type(card) is str
                        and len(card) == 2
                        and card[0] in "23456789TJQKA"
                        and card[1] in "hdcs"
                    ):
                        result["board"].append(card)
                    else:
                        self._incomplete()
            else:
                self._incomplete()
        if "action_history" in source:
            result["action_history"] = self._history(source["action_history"])
        return result

    def _match_start(self):
        # New match, same one-capture process budget. Do not carry correlation
        # across matches; retain the monotonically increasing local alias.
        self.capture = None
        self.key = self.closed_key = None

    def _reconnected(self):
        if self.capture is not None:
            self._incomplete()  # Delivery gaps are unknown, never reconstructed.

    def _round_start(self, message, hero):
        key = correlation(message, "state")
        if key is None or key == self.key or key == self.closed_key:
            return
        self.key = key
        self.alias += 1
        self.decision = self.used_bytes = 0
        self.capture = {
            "schema_version": 1,
            "purpose": "chipzen_uncalled_chip_accounting",
            "classification": "observation_only",
            "hand_alias": self.alias,
            "trigger": None,
            "events": [],
            "buffer_complete": True,
            "dropped_events": 0,
        }
        state = message.get("state", {})
        stacks = state.get("stacks") if isinstance(state, dict) else None
        self.seat_count = len(stacks) if isinstance(stacks, list) else None
        data = self._public(state)
        if seat(hero):
            data["your_seat"] = hero
        self._append("round_start", data)

    def _decision_state(self, state):
        self.observed_decision = False
        if self.capture is None:
            return  # No reliable hand correlation: wait for a round start.
        key = correlation(
            {
                "round_id": getattr(state, "round_id", None),
                "state": {"hand_number": getattr(state, "hand_number", None)},
            },
            "state",
        )
        if key != self.key:
            self._incomplete()
            return
        self.observed_decision = True
        self.decision = min(self.decision + 1, 2**31 - 1)
        source = {
            k: getattr(state, k, None)
            for k in (
                "phase",
                "board",
                "your_seat",
                "dealer_seat",
                "pot",
                "your_stack",
                "to_call",
                "min_raise",
                "max_raise",
                "action_history",
            )
        }
        data = self._public(source)
        data["decision_alias"] = self.decision
        legal = getattr(state, "valid_actions", None)
        legal = (
            [a for a in legal[:16] if type(a) is str and a in ACTIONS]
            if isinstance(legal, list)
            else []
        )
        data["valid_actions"] = legal
        opponents = getattr(state, "opponent_stacks", None)
        hero, stack, call = source["your_seat"], source["your_stack"], source["to_call"]
        # Only HU lets the SDK's opponent_stacks list be mapped to a seat
        # without inventing a multiway ordering contract.
        hu = (
            isinstance(opponents, list)
            and len(opponents) == 1
            and self.seat_count == 2
            and type(hero) is int
            and hero in (0, 1)
        )
        if hu and number(stack) and number(opponents[0]):
            data["stacks"] = {str(hero): stack, str(1 - hero): opponents[0]}
            if "call" in legal and stack > 0 and number(call):
                reasons = []
                if call >= stack:
                    reasons.append("to_call_at_least_hero_stack")
                if opponents[0] == 0:
                    reasons.append("opponent_remaining_zero")
                if reasons and self.capture["trigger"] is None:
                    self.capture["trigger"] = {
                        "reasons": reasons,
                        "decision_alias": self.decision,
                        "hero_stack": stack,
                        "to_call": call,
                        **fields(source, enums=(("phase", PHASES),)),
                    }
        self._append("decision_state", data)

    def _selected_action(self, action):
        if self.capture is not None and self.observed_decision:
            data = fields(
                {
                    "action": getattr(action, "action", None),
                    "amount": getattr(action, "amount", None),
                },
                ("amount",),
                (("action", ACTIONS),),
            )
            data["decision_alias"] = self.decision
            self._append("selected_action", data)

    def _turn_result(self, message):
        if self.capture is None:
            return
        if "round_id" in message and correlation(message, "details") != self.key:
            self._incomplete()
            return
        details = message.get("details", {})
        data = self._public(details)
        data.update(fields(details, ("seat", "amount"), (("action", ACTIONS),)))
        data.update(fields(message, booleans=("is_timeout",)))
        # SDK TurnResult also accepts envelope seat when absent in details.
        if "seat" not in data:
            data.update(fields(message, ("seat",)))
        self._append("turn_result", data)

    def _round_result(self, message):
        if self.capture is None or correlation(message, "result") != self.key:
            return
        result = message.get("result", {})
        data = self._public(result)
        payouts = result.get("payouts")
        if isinstance(payouts, list):
            if len(payouts) > MAX_SEATS:
                self._incomplete()
            data["payouts"] = [
                fields(p, ("seat", "amount")) for p in payouts[:MAX_SEATS]
            ]
        self._append("round_result", data, final=True)
        self.closed_key, self.key = self.key, None
        capture, self.capture = self.capture, None
        if capture["trigger"] is None:
            return
        # One attempt, after the hand, never in decide. Exclusive creation
        # protects an existing capture. Failures are isolated by the adapter.
        self.armed = False
        encoded = (
            json.dumps(capture, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
