"""Play Leduc hold'em yourself against any of the AI agents.

This is the live, human-in-the-loop test of the trained model. You are dealt
a card, you bet, and the AI reasons about your hidden card and responds.

Examples:
    python scripts/play.py                       # vs the full Bayes+search agent
    python scripts/play.py --opponent ga_tuned   # vs the GA-evolved strategy
    python scripts/play.py --opponent rule_based --hands 20 --seed 3
    python scripts/play.py --reveal              # also print the AI's belief + EVs

By default the AI's reasoning is hidden while you play (a fair test). Pass
--reveal to watch its belief over your card and its search EVs each turn --
useful for the demo / understanding what it is doing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sleight_of_hand.agents.baselines import (  # noqa: E402
    AlwaysCallAgent,
    RandomAgent,
    RuleBasedAgent,
)
from sleight_of_hand.agents.bayes_search_agent import BayesSearchAgent  # noqa: E402
from sleight_of_hand.agents.human import HumanAgent  # noqa: E402
from sleight_of_hand.engine.actions import ActionType  # noqa: E402
from sleight_of_hand.engine.cards import rank_name  # noqa: E402
from sleight_of_hand.engine.game import LeducGame  # noqa: E402
from sleight_of_hand.policy.heuristic import PolicyParams  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_ga_genome(path: str) -> PolicyParams | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return PolicyParams(**json.load(f)["params"])


def build_opponent(name: str, rng: random.Random):
    if name == "random":
        return RandomAgent(rng=rng)
    if name == "always_call":
        return AlwaysCallAgent()
    if name == "rule_based":
        return RuleBasedAgent(rng=rng, name="rule_based")
    if name == "bayes_search":
        return BayesSearchAgent(rng=rng, name="bayes_search")
    if name == "ga_tuned":
        genome = load_ga_genome(os.path.join(RESULTS_DIR, "ga_best_genome.json"))
        if genome is None:
            raise SystemExit(
                "No GA genome found. Run `python scripts/run_ga.py` first, "
                "or pick a different --opponent."
            )
        return RuleBasedAgent(params=genome, rng=rng, name="ga_tuned")
    raise SystemExit(f"unknown opponent '{name}'")


def maybe_reveal(agent, human_seat: int, reveal: bool):
    """If the AI just acted and --reveal is on, print what it was thinking."""
    if not reveal or not isinstance(agent, BayesSearchAgent):
        return
    if agent.last_belief is not None:
        belief = ", ".join(
            f"{rank_name(c)}={p:.2f}" for c, p in sorted(agent.last_belief.items())
        )
        print(f"      [AI belief over your card: {belief}]")
    if agent.last_action_values is not None:
        evs = ", ".join(
            f"{ActionType(a).name}={v:+.2f}" for a, v in agent.last_action_values.items()
        )
        print(f"      [AI search EVs: {evs}]")


def play_hand(hand_no, human, opponent, human_seat, rng, reveal):
    ai_seat = 1 - human_seat
    seats = [None, None]
    seats[human_seat] = human
    seats[ai_seat] = opponent
    names = {human_seat: "You", ai_seat: f"AI ({opponent.name})"}

    print(f"\n{'=' * 66}\nHand {hand_no}   (you act {'first' if human_seat == 0 else 'second'})\n{'=' * 66}")

    state = LeducGame.new_hand(rng)
    print(f"  You are dealt: {rank_name(state.private[human_seat])}")

    while not state.done:
        player = state.to_act
        legal = LeducGame.legal_actions(state)
        prev_public = state.public
        action = seats[player].act(state, legal)

        if player == ai_seat:
            to_call = state.round_state.to_call(player)
            print(f"  {names[player]} chose: {_verb(action, to_call)}")
            maybe_reveal(opponent, human_seat, reveal)

        state = LeducGame.apply_action(state, action, rng=rng)
        if state.public != prev_public and state.public != -1:
            print(f"  *** community card revealed: {rank_name(state.public)} ***")

    p = LeducGame.payoffs(state)
    if state.folded != -1:
        print(f"  {names[state.folded]} folded.")
    else:
        print(
            f"  Showdown: you had {rank_name(state.private[human_seat])}, "
            f"AI had {rank_name(state.private[ai_seat])}, board was {rank_name(state.public)}"
        )
    human_payoff = p[human_seat]
    verdict = "you win" if human_payoff > 0 else ("you lose" if human_payoff < 0 else "split")
    print(f"  Result: {human_payoff:+.1f} chips ({verdict}).")
    return human_payoff


def _verb(action: ActionType, to_call: int) -> str:
    if action == ActionType.CALL:
        return "check" if to_call == 0 else "call"
    if action == ActionType.RAISE:
        return "bet" if to_call == 0 else "raise"
    return "fold"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--opponent",
        choices=["bayes_search", "ga_tuned", "rule_based", "random", "always_call"],
        default="bayes_search",
        help="which AI agent to play against (default: bayes_search)",
    )
    parser.add_argument("--hands", type=int, default=0, help="number of hands (0 = until you quit)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for a reproducible session")
    parser.add_argument("--reveal", action="store_true", help="show the AI's belief and search EVs each turn")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    human = HumanAgent(name="You")
    opponent = build_opponent(args.opponent, rng=random.Random(None if args.seed is None else args.seed + 1))

    print("#" * 66)
    print(f"#  Leduc hold'em -- You vs {opponent.name}")
    print("#  Actions: [f]old  [c]heck/call  [r]bet/raise. Round-1 bet=2, round-2 bet=4.")
    print("#  Pair with the board wins; else higher card wins; ties split.")
    print("#" * 66)

    bankroll = 0.0
    hand_no = 0
    try:
        while args.hands == 0 or hand_no < args.hands:
            hand_no += 1
            human_seat = (hand_no - 1) % 2  # alternate position each hand for fairness
            bankroll += play_hand(hand_no, human, opponent, human_seat, rng, args.reveal)
            print(f"  Running total: {bankroll:+.1f} chips over {hand_no} hand(s).")
            if args.hands == 0:
                cont = input("  Play another hand? [Y/n] ").strip().lower()
                if cont in {"n", "no", "q", "quit", "exit"}:
                    break
    except (EOFError, KeyboardInterrupt):
        print("\n  (session ended)")

    print(f"\nFinal: {bankroll:+.1f} chips over {hand_no} hand(s) "
          f"= {1000.0 * bankroll / max(1, hand_no) / 2:+.1f} mbb/hand vs {opponent.name}.")


if __name__ == "__main__":
    main()
