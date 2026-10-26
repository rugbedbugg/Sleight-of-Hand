"""Validate complete teacher artifacts, then expose exact research lookups."""

import copy
import math

from .contract import (
    ROOT,
    canonical,
    combo,
    digest,
    exact,
    finite,
    generator_hashes,
    integer,
    numerically_equal,
    read_json,
    validate_manifest,
)
from .tree import decision, nodes


def expected_samples(game, profile):
    result = []
    for node in nodes(game):
        if node.terminal:
            continue
        entry = profile[str(node.player)][node.key]
        for hand_index, hand in enumerate(game["ranges"][node.player]):
            state = decision(game, node, hand["cards"])
            result.append(
                {"state": state, "probabilities": entry["strategy"][hand_index]}
            )
    return result


def validate_profile(game, profile):
    exact(profile, "0 1")
    for p in (0, 1):
        relevant = [n for n in nodes(game) if not n.terminal and n.player == p]
        if type(profile[str(p)]) is not dict or set(profile[str(p)]) != {
            n.key for n in relevant
        }:
            raise ValueError("incomplete teacher profile")
        for node in relevant:
            entry = profile[str(p)][node.key]
            exact(entry, "actions strategy")
            state = decision(game, node, game["ranges"][p][0]["cards"])
            if entry["actions"] != [a["id"] for a in state["legal_actions"]]:
                raise ValueError("illegal teacher actions")
            matrix = entry["strategy"]
            if type(matrix) is not list or len(matrix) != len(game["ranges"][p]):
                raise ValueError("missing private-hand strategies")
            for row in matrix:
                if type(row) is not list or len(row) != len(entry["actions"]):
                    raise ValueError("incomplete action distribution")
                for probability in row:
                    finite(probability)
                if not math.isclose(sum(row), 1, rel_tol=0, abs_tol=1e-12):
                    raise ValueError("unnormalized probability distribution")


def validate_dataset(artifact):
    exact(artifact, "payload sha256")
    payload = artifact["payload"]
    if artifact["sha256"] != digest(payload):
        raise ValueError("output checksum mismatch")
    exact(payload, "manifest input_sha256 source_files generator_files python games")
    manifest = payload["manifest"]
    validate_manifest(manifest)
    if payload["input_sha256"] != digest(manifest):
        raise ValueError("input checksum mismatch")
    lock = read_json(ROOT / "source.lock.json")
    if payload["source_files"] != lock["files"]:
        raise ValueError("untraceable source file hashes")
    if payload["generator_files"] != generator_hashes():
        raise ValueError("generator source has changed; regenerate the artifact")
    if type(payload["python"]) is not str or not payload["python"].startswith(
        ("3.10.", "3.13.")
    ):
        raise ValueError("unsupported generation interpreter")
    if type(payload["games"]) is not list or len(payload["games"]) != len(
        manifest["games"]
    ):
        raise ValueError("missing solved trees")
    for game, solved in zip(manifest["games"], payload["games"]):
        exact(solved, "game_sha256 teacher_profile diagnostics samples")
        if solved["game_sha256"] != digest(game):
            raise ValueError("tree checksum mismatch")
        validate_profile(game, solved["teacher_profile"])
        if canonical(solved["samples"]) != canonical(
            expected_samples(game, solved["teacher_profile"])
        ):
            raise ValueError("samples disagree with teacher or legal state geometry")
        diagnostics = solved["diagnostics"]
        if type(diagnostics) is not list or len(diagnostics) != len(
            manifest["checkpoints"]
        ):
            raise ValueError("missing convergence diagnostics")
        for iteration, point in zip(manifest["checkpoints"], diagnostics):
            exact(point, "iterations solver_exploitability_chips oracle")
            if type(point["iterations"]) is not int or point["iterations"] != iteration:
                raise ValueError("iteration mismatch")
            finite(point["solver_exploitability_chips"], -1e-10, game["pot"] * 2)
            oracle = point["oracle"]
            exact(
                oracle,
                "ev_chips best_response_chips exploitability_chips compatible_deals joint_weight",
            )
            for key in ("ev_chips", "best_response_chips"):
                if type(oracle[key]) is not list or len(oracle[key]) != 2:
                    raise ValueError("invalid oracle values")
                for value in oracle[key]:
                    finite(value, -game["pot"], 3 * game["pot"])
            finite(oracle["exploitability_chips"], -1e-10, game["pot"] * 2)
            integer(oracle["compatible_deals"], 1, 256)
            finite(oracle["joint_weight"], 1e-18, 256)
            if any(
                br + 1e-8 < ev
                for br, ev in zip(oracle["best_response_chips"], oracle["ev_chips"])
            ):
                raise ValueError("best response below on-policy value")
            if (
                abs(
                    (sum(oracle["best_response_chips"]) - game["pot"]) / 2
                    - oracle["exploitability_chips"]
                )
                > 1e-8
            ):
                raise ValueError("inconsistent oracle exploitability")
            if (
                abs(
                    point["solver_exploitability_chips"]
                    - oracle["exploitability_chips"]
                )
                > 1e-8
            ):
                raise ValueError("solver/oracle exploitability disagreement")
            if abs(sum(oracle["ev_chips"]) - game["pot"]) > 1e-8:
                raise ValueError("non-conserving expected payouts")
        if (
            diagnostics[-1]["solver_exploitability_chips"]
            > game["pot"] * manifest["limits"]["max_exploitability_pot_fraction"]
        ):
            raise ValueError("predeclared convergence gate not met")
        # Recompute the final independent evidence instead of trusting claimed metrics.
        from .oracle import evaluate

        measured = evaluate(game, solved["teacher_profile"])
        if not numerically_equal(measured, diagnostics[-1]["oracle"]):
            raise ValueError("final oracle evidence mismatch")
    return artifact


class TeacherPolicy:
    """Exact lookup baseline, not a trained policy or a runtime bot adapter."""

    def __init__(self, artifact):
        validate_dataset(artifact)
        self._tables = {}
        payload = artifact["payload"]
        for game, solved in zip(payload["manifest"]["games"], payload["games"]):
            self._tables[canonical(game)] = {
                canonical(sample["state"]): copy.deepcopy(sample["probabilities"])
                for sample in solved["samples"]
            }

    def lookup(self, game, state):
        """Return a legal distribution or an explicit abstention; never interpolate."""
        try:
            table = self._tables.get(canonical(game))
            if table is None:
                return {"status": "abstain", "reason": "uncovered_game_or_ranges"}
            query = copy.deepcopy(state)
            query["hole_cards"] = list(combo(query["hole_cards"]))
            probabilities = table.get(canonical(query))
            if probabilities is None:
                return {"status": "abstain", "reason": "unsupported_or_uncovered_state"}
            return {
                "status": "supported",
                "actions": copy.deepcopy(query["legal_actions"]),
                "probabilities": list(probabilities),
            }
        except (KeyError, TypeError, ValueError):
            return {"status": "abstain", "reason": "invalid_state"}
