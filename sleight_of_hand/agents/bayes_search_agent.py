from __future__ import annotations

import random

from ..bayes.opponent_model import infer_belief
from ..engine.actions import ActionType
from ..engine.state import GameState
from ..policy.heuristic import DEFAULT_PARAMS, PolicyParams
from ..search.expectiminimax import choose_action
from .base import Agent


class BayesSearchAgent(Agent):
    """The full pipeline agent: infers a belief over the opponent's hidden
    card via Bayes' rule, then runs expectiminimax search over the
    belief-weighted betting tree to pick an action.

    `assumed_opponent_params` are the heuristic-policy parameters used both
    as the Bayesian likelihood model and as the opponent's simulated policy
    inside search -- set these to an opponent's true parameters for a
    "knows the opponent" evaluation, or leave at the default for a
    generic/robust assumption. `use_bayes=False` disables belief updates
    from observed actions (keeping only the deck-combinatorics prior/public
    card conditioning), which is the "search without an opponent model"
    ablation.
    """

    name = "bayes_search"

    def __init__(
        self,
        assumed_opponent_params: PolicyParams = DEFAULT_PARAMS,
        use_bayes: bool = True,
        rng: random.Random | None = None,
        name: str | None = None,
    ):
        self.assumed_opponent_params = assumed_opponent_params
        self.use_bayes = use_bayes
        self.rng = rng or random.Random()
        if name:
            self.name = name
        self.last_belief = None
        self.last_action_values = None

    def act(self, state: GameState, legal_actions: list[ActionType]) -> ActionType:
        my_player = state.to_act
        model = infer_belief(state, my_player, params=self.assumed_opponent_params, use_actions=self.use_bayes)
        action, values = choose_action(state, my_player, model.belief, opponent_params=self.assumed_opponent_params)
        self.last_belief = model.belief
        self.last_action_values = values
        return action
