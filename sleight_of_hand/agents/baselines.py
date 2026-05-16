from __future__ import annotations

import random

from ..engine.actions import ActionType
from ..engine.state import GameState
from ..policy.heuristic import PolicyParams, action_probs, sample_action
from .base import Agent

FOLD, CALL, RAISE = ActionType.FOLD, ActionType.CALL, ActionType.RAISE


class RandomAgent(Agent):
    """Picks uniformly among legal actions."""

    name = "random"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def act(self, state: GameState, legal_actions: list[ActionType]) -> ActionType:
        return self.rng.choice(legal_actions)


class AlwaysCallAgent(Agent):
    """Never folds, never raises: calls/checks every time it can."""

    name = "always_call"

    def act(self, state: GameState, legal_actions: list[ActionType]) -> ActionType:
        return CALL


# Hand-tuned "reasonable poker player" parameters: bets/raises value hands,
# calls with medium strength, folds weak hands to a bet, bluffs occasionally.
RULE_BASED_PARAMS = PolicyParams(
    value_bet_threshold=0.55,
    call_threshold=0.35,
    bluff_freq=0.12,
    aggression=0.75,
    steepness=9.0,
)


class RuleBasedAgent(Agent):
    """Plays the parametrized heuristic policy with fixed, hand-tuned
    parameters. This is both a baseline opponent and the reference
    implementation the GA evolves away from."""

    name = "rule_based"

    def __init__(
        self,
        params: PolicyParams = RULE_BASED_PARAMS,
        rng: random.Random | None = None,
        name: str | None = None,
    ):
        self.params = params
        self.rng = rng or random.Random()
        if name:
            self.name = name

    def act(self, state: GameState, legal_actions: list[ActionType]) -> ActionType:
        player = state.to_act
        to_call = state.round_state.to_call(player)
        probs = action_probs(
            state.private[player], state.public, legal_actions, to_call, params=self.params
        )
        return sample_action(self.rng, probs)
