"""Game registry: maps a `--gamemode` name to a rules engine.

Every entry point takes `--gamemode`; `DEFAULT_GAMEMODE` keeps existing
invocations (and the whole Leduc test suite) working unchanged.

Planned modes not yet implemented are listed in `PLANNED` so that
`--gamemode deuce27` fails with a useful message rather than an unknown-key
error. See `docs/SPEC.md` for what each one is.
"""

from __future__ import annotations

from .game import LeducGame
from .protocol import Game

DEFAULT_GAMEMODE = "leduc"

_GAMES: dict[str, Game] = {
    "leduc": LeducGame(),
}

# Registered in the spec, not yet built -- see docs/SPEC.md sections 3.2/3.3.
PLANNED: dict[str, str] = {
    "deuce27": "full-scale heads-up fixed-limit 2-7 triple draw (spec section 3.2)",
    "mini27": "scaled-down 2-7, exactly solvable, used as ground truth (spec section 3.3)",
}


def available_games() -> list[str]:
    """Names that can actually be played right now."""
    return sorted(_GAMES)


def all_gamemodes() -> list[str]:
    """Implemented modes plus those the spec reserves."""
    return sorted(set(_GAMES) | set(PLANNED))


def get_game(name: str = DEFAULT_GAMEMODE) -> Game:
    """Resolve a `--gamemode` name to its engine."""
    try:
        return _GAMES[name]
    except KeyError:
        pass
    if name in PLANNED:
        raise NotImplementedError(
            f"gamemode {name!r} is specified but not implemented yet: {PLANNED[name]}. "
            f"Available now: {', '.join(available_games())}."
        )
    raise ValueError(f"unknown gamemode {name!r}; available: {', '.join(available_games())}")


def register(name: str, game: Game) -> None:
    """Add an engine to the registry (used by variants as they land)."""
    if name in _GAMES:
        raise ValueError(f"gamemode {name!r} is already registered")
    _GAMES[name] = game
    PLANNED.pop(name, None)
