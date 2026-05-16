"""The Bayesian opponent model: P(opponent's private card | history).

The model tracks a belief distribution over the three ranks {J, Q, K} and
updates it via Bayes' rule after every observation:

  1. Prior: uniform over the cards the agent can't see, weighted by how
     many copies of each rank remain in the deck once the agent's own card
     is removed (2 copies for ranks != my card, 1 copy for the rank == my
     card).
  2. Likelihood of an opponent action: `P(action | card, context)`, taken
     from the shared parametrized heuristic policy
     (`policy.heuristic.action_probs`). This is exactly the same function
     used to *play* the rule-based baseline and to model the opponent
     inside expectiminimax search — and it's the function the GA tunes,
     which is how the GA and the Bayes model connect.
  3. Update: `P(card | action) ∝ P(action | card) * P(card)`, renormalized
     over the remaining hypotheses after every observed opponent action.
  4. Board conditioning: when the public card is revealed, it is itself
     evidence about the opponent's hidden card (a public card of rank X is
     less likely to appear if the opponent is already holding one of the
     two X's) — handled with a genuine Bayesian update using the
     hypergeometric likelihood of drawing that public card from the
     4-card deck remaining after both hole cards.
"""

from __future__ import annotations

import math

from ..engine.actions import ActionType
from ..engine.cards import RANKS, remaining_counts
from ..engine.game import LeducGame
from ..engine.state import GameState, RoundState
from ..policy.heuristic import DEFAULT_PARAMS, PolicyParams, action_probs


class BayesianOpponentModel:
    def __init__(self, my_card: int, params: PolicyParams = DEFAULT_PARAMS):
        self.my_card = my_card
        self.params = params
        self.public = -1
        self.belief: dict[int, float] = self._prior(my_card)
        # Each entry: (label, belief_snapshot) for plotting belief convergence.
        self.trace: list[tuple[str, dict[int, float]]] = [("prior", dict(self.belief))]

    @staticmethod
    def _prior(my_card: int) -> dict[int, float]:
        counts = remaining_counts([my_card])
        total = sum(counts.values())
        return {r: n / total for r, n in counts.items() if n > 0}

    def reset(self, my_card: int) -> None:
        self.my_card = my_card
        self.public = -1
        self.belief = self._prior(my_card)
        self.trace = [("prior", dict(self.belief))]

    # --- updates -----------------------------------------------------
    def update_on_action(self, state_before: GameState, opponent: int, action: ActionType) -> None:
        """Bayes-update the belief after observing the opponent take
        `action` in `state_before` (the state prior to the action being
        applied)."""
        to_call = state_before.round_state.to_call(opponent)
        legal = LeducGame.legal_actions(state_before)

        new_belief: dict[int, float] = {}
        for card, prior_p in self.belief.items():
            probs = action_probs(card, self.public, legal, to_call, params=self.params)
            likelihood = probs.get(action, 0.0)
            new_belief[card] = prior_p * likelihood

        total = sum(new_belief.values())
        if total <= 0:
            # Every hypothesis assigned ~0 likelihood (shouldn't happen with
            # the smooth logistic policy; guards against float underflow).
            new_belief = {c: p for c, p in self.belief.items()}
            total = sum(new_belief.values())
        self.belief = {c: p / total for c, p in new_belief.items()}
        self.trace.append((f"P0{opponent}:{ActionType(action).name}", dict(self.belief)))

    def update_on_public_card(self, public: int) -> None:
        """Bayesian update from observing the community card: a card of
        rank `public` is less likely to have been drawable if the opponent
        is already holding one of its copies."""
        new_belief: dict[int, float] = {}
        for card, prior_p in self.belief.items():
            counts = remaining_counts([self.my_card, card])
            total_remaining = sum(counts.values())
            likelihood = counts.get(public, 0) / total_remaining
            new_belief[card] = prior_p * likelihood

        total = sum(new_belief.values())
        if total <= 0:
            new_belief = {c: p for c, p in self.belief.items()}
            total = sum(new_belief.values())
        self.public = public
        self.belief = {c: p / total for c, p in new_belief.items() if p > 0}
        self.trace.append((f"public:{public}", dict(self.belief)))

    # --- queries -----------------------------------------------------
    def prob(self, card: int) -> float:
        return self.belief.get(card, 0.0)

    def most_likely_card(self) -> int:
        return max(self.belief, key=self.belief.get)

    def entropy(self) -> float:
        return -sum(p * math.log2(p) for p in self.belief.values() if p > 0)

    def as_vector(self) -> list[float]:
        return [self.belief.get(r, 0.0) for r in RANKS]


def infer_belief(
    state: GameState,
    my_player: int,
    params: PolicyParams = DEFAULT_PARAMS,
    use_actions: bool = True,
) -> "BayesianOpponentModel":
    """Reconstruct the belief over the opponent's card from scratch by
    replaying `state.history` (this is a pure function of publicly known
    information: it only ever reads `state.private[my_player]`, never the
    opponent's hidden card). Recomputing from history each decision, rather
    than mutating persistent per-hand state, keeps the model trivially
    correct by construction.

    `use_actions=False` conditions only on the community card (pure deck
    combinatorics) and ignores the opponent's betting behaviour -- used for
    the "search without an opponent model" ablation.
    """
    my_card = state.private[my_player]
    model = BayesianOpponentModel(my_card, params=params)

    replay = GameState(
        private=(my_card, my_card),
        public=-1,
        round_no=1,
        to_act=0,
        round_state=RoundState(),
    )
    for player, action_int in state.history:
        action = ActionType(action_int)
        if player != my_player and use_actions:
            model.update_on_action(replay, opponent=player, action=action)
        replay = LeducGame.apply_action(replay, action)
        if replay.awaiting_community:
            model.update_on_public_card(state.public)
            replay = LeducGame.deal_community(replay, state.public)
    return model
