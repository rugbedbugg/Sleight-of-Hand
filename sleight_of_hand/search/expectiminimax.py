"""Expectiminimax search over the Leduc betting tree.

The tree has three node kinds:
  * MAX nodes -- our own decision points; pick the action maximizing EV.
  * chance nodes -- the community-card deal; take the probability-weighted
    average over remaining deck outcomes.
  * opponent nodes -- the opponent's decision points, combined one of two
    ways selected by the ``mode`` argument:
      - ``"expectiminimax"`` (default): the probability-weighted average
        over the opponent's actions under the shared heuristic policy
        model, exactly as the Bayesian likelihood model does. That policy's
        parameters are the same ones the GA tunes. This best-responds to
        what the opponent *probably* does.
      - ``"minimax"``: the opponent picks the action that *minimizes* our
        EV (equivalently, since Leduc is zero-sum, maximizes their own) --
        the classical adversarial game-tree rule: a paranoid model that
        ignores the opponent policy and assumes the worst legal reply.

The one piece of imperfect information -- the opponent's hidden card -- is
resolved by fixing a hypothesis for it once, at the root, weighted by the
belief distribution supplied by the Bayesian opponent model. Everything
below that fixed hypothesis is a search over a *fully observed* game (both
hole cards known to the search, though never to the calling agent), which
is what makes plain expectiminimax applicable. This is exact given the
policy model of the opponent; it is not an approximation of Leduc's true
Nash equilibrium (that would require best-responding to a strategy, not a
fixed hypothesis of one -- see `eval/exploitability.py` for that piece).
"""

from __future__ import annotations

from ..engine.actions import ActionType
from ..engine.game import LeducGame
from ..engine.state import GameState
from ..policy.heuristic import DEFAULT_PARAMS, PolicyParams, action_probs


def node_value(
    state: GameState,
    my_player: int,
    opponent_params: PolicyParams = DEFAULT_PARAMS,
    mode: str = "expectiminimax",
) -> float:
    """Expected value to `my_player` of `state`, given both hole cards are
    already fixed in `state.private` (a single opponent-card hypothesis).

    `mode` selects how opponent nodes are combined: `"expectiminimax"`
    (policy-weighted average, the default) or `"minimax"` (worst-case min).
    """
    if state.done:
        return LeducGame.payoffs(state)[my_player]

    if state.awaiting_community:
        total = 0.0
        for card, p in LeducGame.possible_community_cards(state, state.private):
            ns = LeducGame.deal_community(state, card)
            total += p * node_value(ns, my_player, opponent_params, mode)
        return total

    legal = LeducGame.legal_actions(state)
    children = {a: node_value(LeducGame.apply_action(state, a), my_player, opponent_params, mode) for a in legal}

    if state.to_act == my_player:
        return max(children.values())

    if mode == "minimax":
        # Paranoid adversarial node: the opponent plays the reply that
        # minimizes our EV (== maximizes theirs, Leduc being zero-sum).
        return min(children.values())

    # expectiminimax: weight each opponent action by the shared policy model.
    opponent = state.to_act
    opp_card = state.private[opponent]
    to_call = state.round_state.to_call(opponent)
    probs = action_probs(opp_card, state.public, legal, to_call, params=opponent_params)
    # `action_probs` may omit near-zero-probability actions, so sum over its
    # keys (which are a subset of `legal`) rather than indexing it by every
    # legal action.
    return sum(p * children[a] for a, p in probs.items())


def choose_action(
    state: GameState,
    my_player: int,
    belief: dict[int, float],
    opponent_params: PolicyParams = DEFAULT_PARAMS,
    mode: str = "expectiminimax",
) -> tuple[ActionType, dict[ActionType, float]]:
    """Pick the EV-maximizing action for `my_player`, marginalizing over
    the opponent's possible hidden card weighted by `belief`.

    `mode` is forwarded to `node_value` to choose between expectiminimax
    (best-respond to the opponent policy) and minimax (worst-case) search.

    Returns (best_action, {action: expected_value}) so callers (and the
    demo) can show the full decision, not just the choice.
    """
    my_card = state.private[my_player]
    legal = LeducGame.legal_actions(state)
    action_values: dict[ActionType, float] = {}
    for a in legal:
        total = 0.0
        for opp_card, p in belief.items():
            if p <= 0:
                continue
            private = (my_card, opp_card) if my_player == 0 else (opp_card, my_card)
            hypothesis = state.clone(private=private)
            ns = LeducGame.apply_action(hypothesis, a)
            total += p * node_value(ns, my_player, opponent_params, mode)
        action_values[a] = total
    best = max(action_values, key=action_values.get)
    return best, action_values
