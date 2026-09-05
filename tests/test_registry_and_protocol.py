"""Tests for the game abstraction: the registry behind `--gamemode`, the
`Game` protocol, and the guarantee that routing Leduc through the protocol
produces exactly what calling the engine directly does."""

from __future__ import annotations

import random

import pytest

from sleight_of_hand.agents.baselines import AlwaysCallAgent, RandomAgent
from sleight_of_hand.engine.game import LeducGame
from sleight_of_hand.engine.protocol import DrawAction, Game, GameSpec
from sleight_of_hand.engine.registry import (
    DEFAULT_GAMEMODE,
    PLANNED,
    all_gamemodes,
    available_games,
    get_game,
    register,
)
from sleight_of_hand.engine.state import BET_SIZE, MAX_RAISES
from sleight_of_hand.eval.harness import MatchResult, play_match


# --- registry --------------------------------------------------------
def test_default_gamemode_is_leduc():
    assert DEFAULT_GAMEMODE == "leduc"
    assert available_games() == ["leduc"]
    assert get_game().spec.name == "leduc"


def test_planned_modes_are_listed_but_not_playable():
    assert {"deuce27", "mini27"} <= set(all_gamemodes())
    for name in PLANNED:
        assert name not in available_games()
        with pytest.raises(NotImplementedError, match=name):
            get_game(name)


def test_unknown_gamemode_raises_value_error():
    with pytest.raises(ValueError, match="unknown gamemode"):
        get_game("stud")


def test_register_rejects_duplicates():
    with pytest.raises(ValueError, match="already registered"):
        register("leduc", LeducGame())


# --- protocol conformance --------------------------------------------
def test_leduc_satisfies_the_game_protocol():
    assert isinstance(get_game(), Game)


def test_leduc_spec_matches_the_engine_constants():
    spec = LeducGame.spec
    assert isinstance(spec, GameSpec)
    assert spec.num_players == 2
    assert spec.num_rounds == 2
    assert spec.bet_size == BET_SIZE
    assert spec.max_raises == MAX_RAISES
    # Leduc antes rather than blinds; round-1 bet size is the mbb denominator.
    assert spec.big_blind == BET_SIZE[1]
    assert spec.has_draws is False


def test_chance_wrappers_agree_with_the_leduc_specific_methods():
    """`awaiting_chance` / `chance_outcomes` / `apply_chance` must be exact
    renames of the community-card trio, not a reimplementation."""
    game = get_game()
    state = LeducGame.deal_hand_with_cards(0, 1)

    # Close round 1 without an rng so the community card stays pending.
    state = LeducGame.apply_action(state, LeducGame.legal_actions(state)[0])
    state = LeducGame.apply_action(state, LeducGame.legal_actions(state)[0])

    assert state.awaiting_community
    assert game.awaiting_chance(state) is state.awaiting_community
    assert game.chance_outcomes(state) == LeducGame.possible_community_cards(state, state.private)

    total = sum(p for _, p in game.chance_outcomes(state))
    assert total == pytest.approx(1.0)

    outcome = game.chance_outcomes(state)[0][0]
    assert game.apply_chance(state, outcome) == LeducGame.deal_community(state, outcome)


def test_hand_strength_is_reachable_through_the_protocol():
    game = get_game()
    # A paired J (0.8) beats a bare J (0.0) on the shared [0, 1] scale.
    # Note a bare K pre-community already scores 1.0, so pick a rank where
    # pairing actually changes the score.
    assert game.hand_strength(0, 0) > game.hand_strength(0, -1)


def test_draw_action_is_defined_but_unused_by_leduc():
    """2-7 needs draws-as-actions; Leduc must never produce one."""
    assert DrawAction((0, 2)).mask == (0, 2)
    game = get_game()
    rng = random.Random(1)
    state = game.new_hand(rng)
    while not state.done:
        legal = game.legal_actions(state)
        assert all(not isinstance(a, DrawAction) for a in legal)
        state = game.apply_action(state, legal[0], rng=rng)


# --- harness integration ---------------------------------------------
def test_play_match_default_game_matches_explicit_game():
    # Fresh agents per call: RandomAgent carries its own RNG, so reusing
    # one instance would desynchronise the second match.
    def pair():
        return AlwaysCallAgent(), RandomAgent(random.Random(7))

    implicit = play_match(*pair(), 200, random.Random(3))
    explicit = play_match(*pair(), 200, random.Random(3), game=get_game())
    assert implicit.mean_a == explicit.mean_a
    assert implicit.big_blind == explicit.big_blind == LeducGame.spec.big_blind


def test_mbb_scales_by_the_result_s_own_big_blind():
    """A result carries its game's denominator, so a future variant with a
    different big blind is converted correctly."""
    res = MatchResult("a", "b", n_hands=10, mean_a=2.0, stderr_a=0.5, big_blind=4)
    assert res.mbb_a == pytest.approx(500.0)
    assert res.mbb_stderr_a == pytest.approx(125.0)
