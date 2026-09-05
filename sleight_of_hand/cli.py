"""Shared `--gamemode` plumbing for the entry points.

Every script that plays hands accepts `--gamemode`, so switching variants
never means editing code. Modes reserved by `docs/SPEC.md` but not yet
implemented are accepted by the parser and rejected with an explanatory
message by `resolve_game`, rather than being silently unknown.
"""

from __future__ import annotations

import argparse

from .engine.registry import DEFAULT_GAMEMODE, all_gamemodes, available_games, get_game
from .engine.protocol import Game


def add_gamemode_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--gamemode",
        type=str,
        default=DEFAULT_GAMEMODE,
        choices=all_gamemodes(),
        help=f"which variant to play (implemented: {', '.join(available_games())}; default: {DEFAULT_GAMEMODE})",
    )
    return parser


def resolve_game(args: argparse.Namespace) -> Game:
    """Turn parsed args into an engine, exiting cleanly on a mode that is
    specified but not yet built."""
    name = getattr(args, "gamemode", DEFAULT_GAMEMODE)
    try:
        return get_game(name)
    except NotImplementedError as exc:
        raise SystemExit(f"error: {exc}")
