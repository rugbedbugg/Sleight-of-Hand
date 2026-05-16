"""Live demo: play hands between the full Bayes+search agent and a
rule-based opponent, printing the belief state and the search's expected
value for each candidate action at every decision point.

Usage:
    python demo.py                  # 3 hands, seat 0 = the search agent
    python demo.py --hands 5 --seed 7
"""

from __future__ import annotations

import argparse
import random

from sleight_of_hand.agents.baselines import RuleBasedAgent
from sleight_of_hand.agents.bayes_search_agent import BayesSearchAgent
from sleight_of_hand.bayes.opponent_model import infer_belief
from sleight_of_hand.engine.actions import ActionType
from sleight_of_hand.engine.cards import rank_name
from sleight_of_hand.engine.game import LeducGame
from sleight_of_hand.policy.heuristic import DEFAULT_PARAMS

SEARCH_SEAT = 0


def describe_state(state, seat_names):
    to_call = state.round_state.to_call(state.to_act)
    print(
        f"  round {state.round_no} | pot contrib this round {state.round_state.contrib} | "
        f"to_call={to_call} | public={'?' if state.public == -1 else rank_name(state.public)} | "
        f"{seat_names[state.to_act]} to act"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    search_agent = BayesSearchAgent(rng=random.Random(args.seed + 1))
    opponent = RuleBasedAgent(rng=random.Random(args.seed + 2))
    seat_names = {SEARCH_SEAT: "bayes_search (you)", 1 - SEARCH_SEAT: "rule_based (opponent)"}
    agents = [None, None]
    agents[SEARCH_SEAT] = search_agent
    agents[1 - SEARCH_SEAT] = opponent

    for hand_no in range(1, args.hands + 1):
        print(f"\n{'#' * 70}\nHand {hand_no}\n{'#' * 70}")
        state = LeducGame.new_hand(rng)
        print(f"  bayes_search holds {rank_name(state.private[SEARCH_SEAT])}")

        while not state.done:
            player = state.to_act
            legal = LeducGame.legal_actions(state)
            describe_state(state, seat_names)

            if player == SEARCH_SEAT:
                model = infer_belief(state, SEARCH_SEAT, params=DEFAULT_PARAMS)
                print(
                    "    belief over opponent card: "
                    + ", ".join(f"{rank_name(c)}={p:.2f}" for c, p in sorted(model.belief.items()))
                )
                action = search_agent.act(state, legal)
                values_str = ", ".join(f"{ActionType(a).name}={v:+.2f}" for a, v in search_agent.last_action_values.items())
                print(f"    search EVs: {values_str}")
                print(f"    -> bayes_search chooses {ActionType(action).name}")
            else:
                action = opponent.act(state, legal)
                print(f"    -> opponent chooses {ActionType(action).name}")

            prev_public = state.public
            state = LeducGame.apply_action(state, action, rng=rng)
            if state.public != prev_public and state.public != -1:
                print(f"  *** community card revealed: {rank_name(state.public)} ***")

        p0, p1 = LeducGame.payoffs(state)
        payoffs = {SEARCH_SEAT: p0, 1 - SEARCH_SEAT: p1}
        if state.folded != -1:
            print(f"  {seat_names[state.folded]} folded.")
        else:
            print(
                f"  Showdown: bayes_search had {rank_name(state.private[SEARCH_SEAT])}, "
                f"opponent had {rank_name(state.private[1 - SEARCH_SEAT])}, "
                f"public was {rank_name(state.public)}"
            )
        print(f"  Payoffs: bayes_search {payoffs[SEARCH_SEAT]:+.1f}, opponent {payoffs[1 - SEARCH_SEAT]:+.1f}")


if __name__ == "__main__":
    main()
