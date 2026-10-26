"""Strict, bounded JSON contracts for the M1 river experiment."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = "6a10442877ffc8fd28af93e16e279b9bbdd97b2a"
SOURCE = "https://github.com/noambrown/poker_solver"
MAX_BYTES = 2_000_000
DECK = tuple(r + s for r in "23456789TJQKA" for s in "cdhs")


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numerically_equal(a, b, tolerance=1e-10):
    """Cross-interpreter diagnostic comparison; stored targets remain exact."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(
            numerically_equal(a[k], b[k], tolerance) for k in a
        )
    if isinstance(a, list):
        return len(a) == len(b) and all(
            numerically_equal(x, y, tolerance) for x, y in zip(a, b)
        )
    if type(a) is float:
        return math.isclose(a, b, rel_tol=0, abs_tol=tolerance)
    return a == b


def exact(value, keys):
    if type(value) is not dict or set(value) != set(keys.split()):
        raise ValueError(f"expected fields: {keys}")


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("integer outside supported bounds")


def finite(value, low=0, high=1):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError("non-finite or out-of-bounds number")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("JSON exceeds 2 MB cap")
    return json.loads(
        raw,
        object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
    )


def write_json(path, value):
    data = canonical(value) + b"\n"
    if len(data) > MAX_BYTES:
        raise ValueError("JSON exceeds 2 MB cap")
    # Refuse to overwrite evidence, including a partial earlier run.
    with Path(path).open("xb") as stream:
        stream.write(data)


def cards(value, count):
    if (
        type(value) is not list
        or len(value) != count
        or any(type(c) is not str or c not in DECK for c in value)
    ):
        raise ValueError("invalid cards")
    if len(set(value)) != count:
        raise ValueError("duplicate cards")


def combo(value):
    cards(value, 2)
    return tuple(sorted(value, key=DECK.index))


def all_combos():
    """1,326 physical combinations, never 169 equally weighted classes."""
    return tuple(itertools.combinations(DECK, 2))


def validate_game(game):
    exact(game, "id split variant board pot stacks big_blind ranges betting")
    if type(game["id"]) is not str or not 1 <= len(game["id"]) <= 64:
        raise ValueError("invalid game id")
    if (
        game["split"] not in ("train", "validation", "test")
        or game["variant"] != "nlhe_hu_river"
    ):
        raise ValueError("unsupported split or variant")
    cards(game["board"], 5)
    integer(game["pot"], 4, 10000)
    if game["pot"] % 2:
        raise ValueError("M1 requires an even settled pot")
    if type(game["stacks"]) is not list or len(game["stacks"]) != 2:
        raise ValueError("two stacks required")
    for stack in game["stacks"]:
        integer(stack, game["pot"], game["pot"])
    integer(game["big_blind"], 1, game["pot"] // 2)
    # This narrow family avoids upstream's missing minimum-raise and short-call rules.
    if canonical(game["betting"]) != canonical(
        {"bet_fractions": [0.5, 1.0], "include_all_in": True, "max_wagers": 2}
    ):
        raise ValueError("unsupported betting abstraction")
    if type(game["ranges"]) is not list or len(game["ranges"]) != 2:
        raise ValueError("two explicit ranges required")
    combos = []
    for entries in game["ranges"]:
        if type(entries) is not list or not 1 <= len(entries) <= 16:
            raise ValueError("M1 permits 1..16 combos per range")
        seen = set()
        for entry in entries:
            exact(entry, "cards weight")
            hand = combo(entry["cards"])
            if hand in seen or set(hand) & set(game["board"]):
                raise ValueError("duplicate or board-blocked combination")
            if list(hand) != entry["cards"]:
                raise ValueError("range combos must use canonical rank-major order")
            finite(entry["weight"], 1e-9, 1)
            seen.add(hand)
        combos.append(seen)
    for p in (0, 1):
        if any(not any(set(h).isdisjoint(o) for o in combos[1 - p]) for h in combos[p]):
            raise ValueError("range hand has no compatible opponent combination")


def validate_manifest(manifest):
    exact(manifest, "schema_version source algorithm checkpoints limits games")
    integer(manifest["schema_version"], 1, 1)
    if manifest["source"] != {"url": SOURCE, "revision": REVISION, "license": "MIT"}:
        raise ValueError("untraceable solver provenance")
    if manifest["algorithm"] != "vector_cfr_plus_alternating_linear_naive":
        raise ValueError("unsupported solver algorithm")
    points = manifest["checkpoints"]
    if type(points) is not list or not 2 <= len(points) <= 8:
        raise ValueError("invalid checkpoints")
    for n in points:
        integer(n, 1, 10000)
    if sorted(set(points)) != points:
        raise ValueError("checkpoints must increase")
    limits = manifest["limits"]
    exact(
        limits,
        "address_space_mb cpu_seconds wall_seconds cpus max_exploitability_pot_fraction",
    )
    integer(limits["address_space_mb"], 64, 512)
    integer(limits["cpu_seconds"], 1, 120)
    integer(limits["wall_seconds"], limits["cpu_seconds"], 180)
    integer(limits["cpus"], 1, 2)
    finite(limits["max_exploitability_pot_fraction"], 0, 0.001)
    games = manifest["games"]
    if type(games) is not list or not 1 <= len(games) <= 6:
        raise ValueError("M1 permits at most six trees")
    ids, boards = set(), set()
    for game in games:
        validate_game(game)
        # Stronger than just split disjointness: each board may occur only once.
        board = tuple(sorted(game["board"]))
        if game["id"] in ids or board in boards:
            raise ValueError("duplicate tree or board: partition leakage")
        ids.add(game["id"])
        boards.add(board)


def generator_hashes():
    hashes = {p.name: file_digest(p) for p in sorted(ROOT.glob("*.py"))}
    # The independent oracle imports this existing evaluator read-only.
    hashes["../holdem/equity.py"] = file_digest(ROOT.parent / "holdem/equity.py")
    return hashes
