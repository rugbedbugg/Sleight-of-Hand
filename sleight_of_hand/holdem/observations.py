"""Card-free, versioned evidence; only validated HU preflop opportunities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .opponent import parse_hand
from .profiles import (
    Counts,
    OpponentIdentity,
    digest,
    fields,
    identifier,
    validate_stats,
)

MAX_HISTORY = 256


@dataclass(frozen=True)
class NormalizedObservation:
    identity: OpponentIdentity
    match_id: str
    hand_id: str
    source_id: str
    source_sha256: str
    source_kind: str
    stats: tuple[tuple[str, Counts], ...]
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported observation schema")
        for v in (self.match_id, self.hand_id, self.source_id):
            identifier(v)
        digest(self.source_sha256)
        if self.source_kind not in {"synthetic", "public_history", "delivered_match"}:
            raise ValueError("unapproved source kind")
        validate_stats(self.stats)
        if any(c.opportunities != 1 for _, c in self.stats):
            raise ValueError("one opportunity per context per hand required")

    @property
    def dedup_key(self) -> tuple:
        return (self.identity.persistent_key, self.match_id, self.hand_id)

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "identity": self.identity.to_dict(),
            "match_id": self.match_id,
            "hand_id": self.hand_id,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "source_kind": self.source_kind,
            "stats": {k: asdict(v) for k, v in self.stats},
        }

    @classmethod
    def from_dict(cls, value):
        fields(
            value,
            {
                "schema_version",
                "identity",
                "match_id",
                "hand_id",
                "source_id",
                "source_sha256",
                "source_kind",
                "stats",
            },
        )
        if type(value["stats"]) is not dict:
            raise ValueError("invalid statistics")
        stats = tuple(
            (k, Counts(**fields(v, {"successes", "opportunities"})))
            for k, v in sorted(value["stats"].items())
        )
        return cls(
            OpponentIdentity.from_dict(value["identity"]),
            value["match_id"],
            value["hand_id"],
            value["source_id"],
            value["source_sha256"],
            value["source_kind"],
            stats,
            value["schema_version"],
        )


def preflop_observations(
    start: dict, result: dict, key: str, hero: int
) -> tuple[tuple[str, Counts], ...] | None:
    """Pure transactional parse; never reconstruct pot, payouts or side pots.

    Use blind seats, first voluntary choices and monotone raise-to amounts.
    Exclude incomplete/ambiguous forced choices, timeouts, unknown actions,
    out-of-order streets/actors and missing stacks/blinds. Call amounts are
    deliberately not interpreted (canonical protocol ambiguity).
    Shove labels retain the existing parse_hand definition exactly on this
    validated subset: effective commitment, not a literal stack-zero label.
    """
    try:
        if type(hero) is not int or hero not in (0, 1):
            return None
        stacks = start.get("stacks")
        history = result.get("action_history")
        if (
            type(stacks) is not list
            or len(stacks) != 2
            or any(type(v) is not int or v <= 0 for v in stacks)
            or type(history) is not list
            or not 2 <= len(history) <= MAX_HISTORY
        ):
            return None
        posts, antes, voluntary = {}, [0, 0], []
        later_street = False
        last_street = 0
        level = 0
        folded = False
        for entry in history:
            if type(entry) is not dict:
                return None
            seat, action, amount = (
                entry.get("seat"),
                entry.get("action"),
                entry.get("amount", 0),
            )
            phase = entry.get("phase", "preflop")
            if (
                type(seat) is not int
                or seat not in (0, 1)
                or type(amount) is not int
                or amount < 0
                or phase not in ("preflop", "flop", "turn", "river")
                or action
                not in {
                    "post_small_blind",
                    "post_big_blind",
                    "post_ante",
                    "fold",
                    "check",
                    "call",
                    "raise",
                    "all_in",
                }
                or entry.get("is_timeout", False) is not False
                or entry.get("timeout", False) is not False
            ):
                return None
            street = ("preflop", "flop", "turn", "river").index(phase)
            if street < last_street or folded:
                return None
            last_street = street
            if phase != "preflop":
                later_street = True
                continue
            if action.startswith("post_"):
                if voluntary or amount == 0:
                    return None
                if action == "post_ante":
                    antes[seat] += amount
                elif action in posts:
                    return None
                else:
                    posts[action] = (seat, amount)
                    level = max(level, amount)
            else:
                if voluntary and voluntary[-1][0] == seat:
                    return None
                if action in ("raise", "all_in"):
                    if amount <= level or amount > stacks[seat] - antes[seat]:
                        return None
                    level = amount
                voluntary.append((seat, action, amount))
                folded = action == "fold"
        if set(posts) != {"post_small_blind", "post_big_blind"} or not voluntary:
            return None
        button, sb = posts["post_small_blind"]
        big, bb = posts["post_big_blind"]
        if (
            button == big
            or sb >= bb
            or any(
                stacks[s] <= antes[s] + posts[k][1]
                for k, s in (("post_small_blind", button), ("post_big_blind", big))
            )
        ):
            return None
        if voluntary[0][0] != button:
            return None
        first = voluntary[0]
        if first[1] not in {"fold", "call", "raise", "all_in"}:
            return None
        # A final incomplete preflop prefix is not a completed-hand observation.
        if not folded and not later_street:
            if len(voluntary) < 2 or voluntary[-1][1] != "call":
                return None
            if level < min(stacks[s] - antes[s] for s in (0, 1)):
                return None
        if len(voluntary) >= 2:
            response = voluntary[1][1]
            if first[1] == "call" and response not in {"check", "raise", "all_in"}:
                return None
            if first[1] in {"raise", "all_in"} and response not in {
                "fold",
                "call",
                "raise",
                "all_in",
            }:
                return None
        villain = 1 - hero
        out = {}

        def add(metric, success):
            out[metric] = Counts(int(success), 1)

        if button == villain:
            add("btn_open", first[1] in {"raise", "all_in"})
            add("btn_limp", first[1] == "call")
            add("btn_fold", first[1] == "fold")
        if len(voluntary) >= 2 and big == villain:
            response = voluntary[1][1]
            effective = min(stacks[s] - antes[s] for s in (0, 1))
            if first[1] == "call" and response in {"check", "raise", "all_in"}:
                add("bb_iso", response in {"raise", "all_in"})
                add("bb_check_limp", response == "check")
            elif (
                first[1] in {"raise", "all_in"}
                and first[2] < effective
                and response in {"fold", "call", "raise", "all_in"}
            ):
                add("bb_3bet", response in {"raise", "all_in"})
                add("bb_call_open", response == "call")
                add("bb_fold_open", response == "fold")
        shove = parse_hand(start, result, key, hero)
        if shove is not None:
            if shove.open_opportunity:
                add(f"open_shove/{shove.bucket}", shove.open_shove)
            if shove.reshove_opportunity:
                add(f"reshove/{shove.bucket}", shove.reshove)
        return tuple(sorted(out.items()))
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
