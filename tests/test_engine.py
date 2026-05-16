import random

import pytest

from sleight_of_hand.engine import ActionType, GameState, LeducGame, RoundState

FOLD, CALL, RAISE = ActionType.FOLD, ActionType.CALL, ActionType.RAISE


def test_deal_new_hand_two_distinct_cards_from_deck():
    rng = random.Random(0)
    for _ in range(50):
        state = LeducGame.new_hand(rng)
        assert state.private[0] in (0, 1, 2)
        assert state.private[1] in (0, 1, 2)
        assert state.round_no == 1
        assert state.to_act == 0
        assert not state.done


def test_legal_actions_no_bet_yet():
    state = LeducGame.deal_hand_with_cards(0, 1)
    legal = LeducGame.legal_actions(state)
    assert FOLD not in legal
    assert CALL in legal
    assert RAISE in legal


def test_legal_actions_facing_bet():
    state = LeducGame.deal_hand_with_cards(0, 1)
    state = LeducGame.apply_action(state, RAISE)
    legal = LeducGame.legal_actions(state)
    assert FOLD in legal
    assert CALL in legal
    assert RAISE in legal  # 1 raise so far, cap is 2


def test_raise_cap_enforced():
    state = LeducGame.deal_hand_with_cards(0, 1)
    state = LeducGame.apply_action(state, RAISE)  # p0 bets, raises=1
    state = LeducGame.apply_action(state, RAISE)  # p1 raises, raises=2
    legal = LeducGame.legal_actions(state)
    assert RAISE not in legal
    assert set(legal) == {FOLD, CALL}


def test_check_check_closes_round_and_deals_community():
    rng = random.Random(1)
    state = LeducGame.deal_hand_with_cards(0, 1)
    state = LeducGame.apply_action(state, CALL, rng=rng)  # p0 checks
    assert not state.awaiting_community
    state = LeducGame.apply_action(state, CALL, rng=rng)  # p1 checks -> round closes
    assert state.round_no == 2
    assert state.public in (0, 1, 2)
    assert state.round_state.contrib == (0, 0)
    assert state.to_act == 0


def test_bet_call_closes_round():
    rng = random.Random(2)
    state = LeducGame.deal_hand_with_cards(0, 1)
    state = LeducGame.apply_action(state, RAISE, rng=rng)  # p0 bets 2
    assert state.round_state.contrib == (2, 0)
    state = LeducGame.apply_action(state, CALL, rng=rng)  # p1 calls -> round closes
    assert state.round_no == 2
    assert state.round_state.contrib == (0, 0)


def test_fold_ends_hand_immediately():
    rng = random.Random(3)
    state = LeducGame.deal_hand_with_cards(0, 1)
    state = LeducGame.apply_action(state, RAISE, rng=rng)
    state = LeducGame.apply_action(state, FOLD, rng=rng)
    assert state.done
    assert state.folded == 1
    p0, p1 = LeducGame.payoffs(state)
    # p0 anted 1 + bet 2 = 3; p1 anted 1 and folded, contributing only 1.
    assert p0 == pytest.approx(1.0)
    assert p1 == pytest.approx(-1.0)
    assert p0 + p1 == pytest.approx(0.0)


def test_showdown_pair_beats_high_card():
    # p0 has K, public is K (pair). p1 has Q. Both check both rounds.
    rng = random.Random(0)
    state = GameState(private=(2, 1), public=-1, round_no=1, to_act=0, round_state=RoundState())
    state = LeducGame.apply_action(state, CALL, rng=rng)
    state = LeducGame.apply_action(state, CALL, rng=rng)
    # force community to K by re-dealing deterministically
    state = state.clone(public=2)
    state = LeducGame.apply_action(state, CALL, rng=rng)
    state = LeducGame.apply_action(state, CALL, rng=rng)
    assert state.done
    p0, p1 = LeducGame.payoffs(state)
    assert p0 > 0 and p1 < 0
    assert p0 + p1 == pytest.approx(0.0)


def test_showdown_high_card_no_pair():
    rng = random.Random(0)
    state = GameState(private=(2, 1), public=-1, round_no=1, to_act=0, round_state=RoundState())
    state = LeducGame.apply_action(state, CALL, rng=rng)
    state = LeducGame.apply_action(state, CALL, rng=rng)
    state = state.clone(public=0)  # J on board, neither pairs
    state = LeducGame.apply_action(state, CALL, rng=rng)
    state = LeducGame.apply_action(state, CALL, rng=rng)
    p0, p1 = LeducGame.payoffs(state)
    assert p0 == pytest.approx(1.0)  # K beats Q, wins the ante-only pot
    assert p1 == pytest.approx(-1.0)


def test_showdown_split_pot_on_tie():
    rng = random.Random(0)
    state = GameState(private=(1, 1), public=-1, round_no=1, to_act=0, round_state=RoundState())
    state = LeducGame.apply_action(state, CALL, rng=rng)
    state = LeducGame.apply_action(state, CALL, rng=rng)
    state = state.clone(public=2)
    state = LeducGame.apply_action(state, CALL, rng=rng)
    state = LeducGame.apply_action(state, CALL, rng=rng)
    p0, p1 = LeducGame.payoffs(state)
    assert p0 == pytest.approx(0.0)
    assert p1 == pytest.approx(0.0)


def test_full_hand_zero_sum_random_play():
    rng = random.Random(42)

    class RandomPlayer:
        def act(self, state, legal):
            return rng.choice(legal)

    agents = [RandomPlayer(), RandomPlayer()]
    for _ in range(500):
        p0, p1 = LeducGame.play_hand(agents, rng)
        assert p0 + p1 == pytest.approx(0.0)


def test_possible_community_cards_excludes_dealt_and_sums_to_one():
    state = LeducGame.deal_hand_with_cards(0, 0)  # both hold J
    outcomes = LeducGame.possible_community_cards(state, (0, 0))
    probs = {r: p for r, p in outcomes}
    assert 0 not in probs  # no J's left in the deck once both hole cards are J
    assert sum(p for _, p in outcomes) == pytest.approx(1.0)
