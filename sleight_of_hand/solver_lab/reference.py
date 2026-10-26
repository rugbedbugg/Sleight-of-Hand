"""Optional OpenSpiel checks; run only in the separate oracle environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import resource
import sys
import time
from pathlib import Path

from .contract import ROOT, file_digest, read_json, write_json
from .dataset import validate_dataset
from .tree import Node, actions, advance
from .worker import impose_limits, verify_source


def check_river(spec, solved):
    import pyspiel
    from open_spiel.python import policy
    from open_spiel.python.algorithms import exploitability

    from sleight_of_hand.holdem.equity import encode, rank_hand

    joint = [
        (i, j, a["weight"] * b["weight"])
        for i, a in enumerate(spec["ranges"][0])
        for j, b in enumerate(spec["ranges"][1])
        if set(a["cards"]).isdisjoint(b["cards"])
    ]
    mass = sum(w for _, _, w in joint)
    game_type = pyspiel.GameType(
        short_name="soh_m1_river",
        long_name="SOH restricted river",
        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
        information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
        utility=pyspiel.GameType.Utility.ZERO_SUM,
        reward_model=pyspiel.GameType.RewardModel.TERMINAL,
        max_num_players=2,
        min_num_players=2,
        provides_information_state_string=True,
        provides_information_state_tensor=False,
        provides_observation_string=False,
        provides_observation_tensor=False,
        parameter_specification={},
    )

    class Game(pyspiel.Game):
        def __init__(self):
            super().__init__(
                game_type,
                pyspiel.GameInfo(
                    num_distinct_actions=3,
                    max_chance_outcomes=len(joint),
                    num_players=2,
                    min_utility=-1.5 * spec["pot"],
                    max_utility=1.5 * spec["pot"],
                    utility_sum=0,
                    max_game_length=4,
                ),
                {},
            )

        def new_initial_state(self):
            return State(self)

    class State(pyspiel.State):
        def __init__(self, game):
            super().__init__(game)
            self.deal = None
            self.node = Node()

        def current_player(self):
            if self.deal is None:
                return pyspiel.PlayerId.CHANCE
            return pyspiel.PlayerId.TERMINAL if self.node.terminal else self.node.player

        def _legal_actions(self, player):
            return list(range(len(actions(spec, self.node))))

        def chance_outcomes(self):
            return [(k, w / mass) for k, (_, _, w) in enumerate(joint)]

        def _apply_action(self, action):
            if self.deal is None:
                self.deal = joint[action][:2]
            else:
                self.node = advance(
                    spec, self.node, actions(spec, self.node)[action]["id"]
                )

        def information_state_string(self, player=None):
            player = self.current_player() if player is None else player
            return f"{player}:{self.deal[player]}:{self.node.key}"

        def is_terminal(self):
            return self.node.terminal

        def returns(self):
            if not self.is_terminal():
                return [0.0, 0.0]
            if self.node.winner is None:
                ranks = [
                    rank_hand(
                        [
                            encode(c)
                            for c in spec["ranges"][p][self.deal[p]]["cards"]
                            + spec["board"]
                        ]
                    )
                    for p in (0, 1)
                ]
                share = (
                    1.0 if ranks[0] > ranks[1] else 0.0 if ranks[0] < ranks[1] else 0.5
                )
            else:
                share = float(self.node.winner == 0)
            value = (
                share * (spec["pot"] + sum(self.node.contributions))
                - self.node.contributions[0]
                - spec["pot"] / 2
            )
            return [value, -value]

        def _action_to_string(self, player, action):
            return str(action)

        def __str__(self):
            return f"{self.deal}:{self.node.key}"

    game = Game()

    class Teacher(policy.Policy):
        def action_probabilities(self, state, player_id=None):
            player = state.current_player()
            row = solved["teacher_profile"][str(player)][state.node.key]["strategy"][
                state.deal[player]
            ]
            return dict(enumerate(row))

    teacher = Teacher(game, [0, 1])
    # Python OpenSpiel BR is independent of the teacher's vector BR implementation.
    result = exploitability.nash_conv(game, teacher, use_cpp_br=False) / 2
    expected = solved["diagnostics"][-1]["solver_exploitability_chips"]
    if abs(result - expected) > 1e-8:
        raise ValueError("OpenSpiel river exploitability mismatch")
    return {
        "id": spec["id"],
        "openspiel_exploitability_chips": float(result),
        "teacher_exploitability_chips": expected,
        "absolute_error": abs(result - expected),
    }


def check_kuhn(source):
    import pyspiel
    from open_spiel.python import policy
    from open_spiel.python.algorithms import exploitability

    source, _ = verify_source(source)
    sys.path.insert(0, str(source / "python/src"))
    from algorithms.cfr import CFRConfig, CFRTrainer
    from algorithms.evaluation import expected_value
    from algorithms.evaluation import exploitability as teacher_exp
    from games.kuhn import KuhnPoker

    reference_game = KuhnPoker()
    trainer = CFRTrainer(
        reference_game,
        CFRConfig(use_plus=True, linear_weighting=True, alternating=True),
    )
    trainer.run(10000)
    profile = trainer.average_strategy_profile()
    game = pyspiel.load_game("kuhn_poker")

    class Teacher(policy.Policy):
        def action_probabilities(self, state, player_id=None):
            key = state.information_state_string()
            rank, history = "JQK"[int(key[0])], key[1:]
            # OpenSpiel pass=0 means check/fold; bet=1 means bet/call.
            original = history.replace("p", "c")
            probs = profile[f"{rank}|{original}"]
            return {
                0: probs["f" if history.endswith("b") else "c"],
                1: probs["c" if history.endswith("b") else "b"],
            }

    measured = exploitability.exploitability(game, Teacher(game, [0, 1]))
    own = teacher_exp(reference_game, profile)
    ev = expected_value(reference_game, profile)
    if abs(measured - own) > 1e-10 or measured > 0.001 or abs(ev + 1 / 18) > 0.001:
        raise ValueError("Kuhn reference check failed")
    return {
        "iterations": 10000,
        "teacher_exploitability": own,
        "openspiel_exploitability": measured,
        "ev_player0": ev,
        "known_value": -1 / 18,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    versions = {}
    for line in (ROOT / "oracle-requirements.in").read_text().splitlines():
        if line and not line.startswith("#"):
            name, expected = line.split("==")
            versions[name] = importlib.metadata.version(name)
            if versions[name] != expected:
                raise ValueError(f"oracle dependency drift: {name}")
    artifact = read_json(args.dataset)
    validate_dataset(artifact)
    limits = artifact["payload"]["manifest"]["limits"]
    impose_limits(limits)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    started = time.monotonic()
    rivers = [
        check_river(g, s)
        for g, s in zip(
            artifact["payload"]["manifest"]["games"], artifact["payload"]["games"]
        )
    ]
    kuhn = check_kuhn(args.source)
    import pyspiel

    write_json(
        args.output,
        {
            "dataset_sha256": artifact["sha256"],
            "open_spiel_version": importlib.metadata.version("open_spiel"),
            "dependency_versions": versions,
            "open_spiel_release_revision": "fcb98881bffd25a324d5505041c837ca9b7ed0bf",
            "pyspiel_binary_sha256": file_digest(pyspiel.__file__),
            "river_checks": rivers,
            "kuhn": kuhn,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )


if __name__ == "__main__":
    main()
