"""M1 contracts, independent accounting, isolation and optional real reproduction."""

import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sleight_of_hand.experiments.runner import policy_revision
from sleight_of_hand.solver_lab import __main__ as cli
from sleight_of_hand.solver_lab.contract import (
    ROOT,
    all_combos,
    canonical,
    digest,
    numerically_equal,
    read_json,
    validate_game,
    validate_manifest,
)
from sleight_of_hand.solver_lab.dataset import TeacherPolicy, validate_dataset
from sleight_of_hand.solver_lab.tree import Node, actions, advance, decision, nodes


@pytest.fixture
def manifest():
    return read_json(ROOT / "manifest.json")


@pytest.fixture
def artifact():
    return read_json(ROOT / "data/dataset.json")


def reseal(artifact):
    artifact["sha256"] = digest(artifact["payload"])
    return artifact


def test_cards_and_combination_fidelity(manifest):
    assert len(all_combos()) == len(set(all_combos())) == 1326
    board = set(manifest["games"][0]["board"])
    assert sum(board.isdisjoint(h) for h in all_combos()) == 1081
    assert sum(all(c[0] == "A" for c in h) for h in all_combos()) == 6
    assert sum({c[0] for c in h} == {"A", "K"} for h in all_combos()) == 16
    validate_manifest(manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("board", ["As"] * 5),
        ("board", ["As", "Kd", "Qh", "Jc"]),
        ("board", ["Xs", "Kd", "Qh", "Jc", "Ts"]),
        ("pot", -100),
        ("pot", True),
        ("pot", 99),
        ("stacks", [100, 99]),
        ("stacks", [True, 100]),
        ("big_blind", 51),
        ("variant", "nlhe_6max"),
        ("betting", {}),
    ],
)
def test_invalid_geometry(manifest, field, value):
    game = manifest["games"][0]
    game[field] = value
    with pytest.raises(ValueError):
        validate_game(game)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "blocked",
        "self_duplicate",
        "negative",
        "nan",
        "empty",
        "too_large",
        "dead",
    ],
)
def test_invalid_ranges(manifest, mutation):
    game = manifest["games"][0]
    entry = game["ranges"][0][0]
    if mutation == "duplicate":
        game["ranges"][0].append(copy.deepcopy(entry))
    elif mutation == "blocked":
        entry["cards"] = ["2s", "As"]
    elif mutation == "self_duplicate":
        entry["cards"] = ["As", "As"]
    elif mutation == "negative":
        entry["weight"] = -1
    elif mutation == "nan":
        entry["weight"] = float("nan")
    elif mutation == "empty":
        game["ranges"][0] = []
    elif mutation == "too_large":
        game["ranges"][0] *= 3
    else:
        game["ranges"] = [[{"cards": ["Ah", "As"], "weight": 1}]] * 2
    with pytest.raises(ValueError):
        validate_game(game)


def test_raise_to_call_and_all_in_geometry(manifest):
    game = manifest["games"][0]
    root = Node()
    assert [a["raise_to"] for a in actions(game, root)] == [None, 50, 100]
    bet = advance(game, root, "b50")
    state = decision(game, bet, game["ranges"][1][0]["cards"])
    assert state["pot"] == 150 and state["to_call"] == 50
    assert state["stacks"] == [50, 100]
    raised = advance(game, bet, "r50")
    state = decision(game, raised, game["ranges"][0][0]["cards"])
    assert raised.contributions == (50, 100)
    assert state["pot"] == 250 and state["to_call"] == 50
    assert state["stacks"] == [50, 0]
    assert [a["action"] for a in state["legal_actions"]] == ["call", "fold"]
    assert state["legal_actions"][0]["all_in"]
    called = advance(game, raised, "c")
    assert called.terminal and called.contributions == (100, 100)
    assert advance(game, raised, "f").winner == 1
    with pytest.raises(ValueError):
        advance(game, raised, "r50")


def test_every_tree_edge_matches_local_dealer(manifest):
    from sleight_of_hand.experiments.platforms.local import Hand, full_deck

    game = manifest["games"][0]
    config = {
        "small_blind": 1,
        "big_blind": 2,
        "starting_stack": 150,
        "stack_mode": "reset",
        "pot_convention": "committed",
        "to_call_convention": "owed",
        "cap_bets_to_effective": False,
        "accounting_observer": False,
    }
    for node in nodes(game):
        hand = Hand(1, 1, [150, 150], full_deck(), config, "solver-test")
        # Start at an explicitly settled 100-chip river, not a simulated preflop path.
        hand.phase, hand.stack, hand.street, hand.total = (
            "river",
            [100, 100],
            [0, 0],
            [50, 50],
        )
        prior = Node()
        for token in node.history:
            action = next(a for a in actions(game, prior) if a["id"] == token)
            result = hand.apply(prior.player, action["action"], action["raise_to"] or 0)
            assert result["rejected"] is None
            prior = advance(game, prior, token)
        assert hand.pot() == 100 + sum(node.contributions)
        assert hand.street == list(node.contributions)
        assert hand.stack == [100 - c for c in node.contributions]
        if not node.terminal:
            legal = hand.legal(node.player)
            for action in actions(game, node):
                assert action["action"] in legal["valid_actions"]
                if action["raise_to"] is not None:
                    assert (
                        legal["min_raise"] <= action["raise_to"] <= legal["max_raise"]
                    )


def test_complete_artifact_and_teacher_agreement(artifact):
    validate_dataset(artifact)
    assert "../holdem/equity.py" in artifact["payload"]["generator_files"]
    policy = TeacherPolicy(artifact)
    count = 0
    for game, solved in zip(
        artifact["payload"]["manifest"]["games"], artifact["payload"]["games"]
    ):
        for sample in solved["samples"]:
            result = policy.lookup(game, sample["state"])
            assert result["status"] == "supported"
            assert result["probabilities"] == sample["probabilities"]
            assert result["actions"] == sample["state"]["legal_actions"]
            # Only private-card ordering is normalized, with no suit/board interpolation.
            state = copy.deepcopy(sample["state"])
            state["hole_cards"].reverse()
            assert policy.lookup(game, state) == result
            result["probabilities"][0] = -1
            assert policy.lookup(game, state)["probabilities"][0] >= 0
            count += 1
    assert count == 144


@pytest.mark.parametrize(
    "change",
    [
        "pot",
        "range",
        "hole",
        "board",
        "position",
        "phase",
        "history",
        "actions",
        "missing",
        "nan",
    ],
)
def test_abstention(artifact, change):
    policy = TeacherPolicy(artifact)
    game = copy.deepcopy(artifact["payload"]["manifest"]["games"][0])
    state = copy.deepcopy(artifact["payload"]["games"][0]["samples"][0]["state"])
    if change == "range":
        game["ranges"][0][0]["weight"] = 0.123
    elif change == "pot":
        state["pot"] += 1
    elif change == "hole":
        state["hole_cards"] = ["2s", "As"]
    elif change == "board":
        state["board"][0] = "Kc"
    elif change == "position":
        state["hero_position"] = 1
    elif change == "phase":
        state["phase"] = "turn"
    elif change == "history":
        state["history"] = ["b49"]
    elif change == "actions":
        state["legal_actions"][1]["raise_to"] = 51
    elif change == "nan":
        state["pot"] = float("nan")
    else:
        del state["pot"]
    assert policy.lookup(game, state)["status"] == "abstain"


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), 1.1, True])
def test_bad_probabilities_rejected_even_with_new_checksum(artifact, value):
    artifact["payload"]["games"][0]["teacher_profile"]["0"]["root"]["strategy"][0][
        0
    ] = value
    with pytest.raises(ValueError):
        validate_dataset(reseal(artifact))


@pytest.mark.parametrize(
    "change",
    [
        "sum",
        "missing_tree",
        "missing_info",
        "missing_hand",
        "missing_sample",
        "amount",
        "source",
        "input",
        "iterations",
        "oracle",
        "generator",
    ],
)
def test_corrupt_data(artifact, change):
    payload = artifact["payload"]
    solved = payload["games"][0]
    if change == "sum":
        solved["teacher_profile"]["0"]["root"]["strategy"][0] = [0, 0, 0]
    elif change == "missing_tree":
        payload["games"].pop()
    elif change == "missing_info":
        del solved["teacher_profile"]["0"]["root"]
    elif change == "missing_hand":
        solved["teacher_profile"]["0"]["root"]["strategy"].pop()
    elif change == "missing_sample":
        solved["samples"].pop()
    elif change == "amount":
        solved["samples"][0]["state"]["legal_actions"][1]["raise_to"] = 49
    elif change == "source":
        payload["source_files"] = {}
    elif change == "input":
        payload["input_sha256"] = "sha256:" + "0" * 64
    elif change == "iterations":
        solved["diagnostics"][-1]["iterations"] -= 1
    elif change == "generator":
        payload["generator_files"] = {}
    else:
        solved["diagnostics"][-1]["oracle"]["best_response_chips"][0] += 1
    with pytest.raises(ValueError):
        validate_dataset(reseal(artifact))


def test_raw_corruption_and_bounded_json(artifact, tmp_path):
    artifact["sha256"] = "bad"
    with pytest.raises(ValueError):
        validate_dataset(artifact)
    path = tmp_path / "bad.json"
    for content in ('{"x":1,"x":2}', '{"x":NaN}', " " * 2_000_001):
        path.write_text(content)
        with pytest.raises(ValueError):
            read_json(path)


def test_partition_and_resource_caps(manifest):
    assert {g["split"] for g in manifest["games"]} == {"train", "validation", "test"}
    for field, value in (
        ("address_space_mb", 513),
        ("cpus", 3),
        ("cpu_seconds", 121),
        ("wall_seconds", 181),
    ):
        bad = copy.deepcopy(manifest)
        bad["limits"][field] = value
        with pytest.raises(ValueError):
            validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["games"][1] = copy.deepcopy(bad["games"][0])
    bad["games"][1]["id"], bad["games"][1]["split"] = "duplicate-board", "validation"
    bad["games"][1]["board"].reverse()
    with pytest.raises(ValueError, match="partition leakage"):
        validate_manifest(bad)


def test_memory_limit_is_enforced_in_child():
    code = """
from sleight_of_hand.solver_lab.worker import impose_limits
impose_limits({'address_space_mb':64,'cpu_seconds':5,'cpus':2})
try:
    bytearray(128 * 1024**2)
except MemoryError:
    print('bounded')
else:
    raise SystemExit('allocation unexpectedly succeeded')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.stdout.strip() == "bounded"


def test_wall_timeout_does_not_publish_partial_evidence(monkeypatch, tmp_path):
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 180
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(cli.subprocess, "run", timeout)
    output = tmp_path / "output"
    with pytest.raises(subprocess.TimeoutExpired):
        cli.run(tmp_path / "source", ROOT / "manifest.json", output)
    assert not output.exists()


def test_policy_and_staging_isolation():
    assert (
        policy_revision()
        == "sha256:eff445aa51a2b78871c294df085353b4f00b3b38a0df73a954f13f2270a0cfbf"
    )
    assert "solver_lab" not in Path("scripts/build_chipzen.py").read_text()


@pytest.mark.skipif(
    not os.environ.get("SOH_SOLVER_SOURCE"),
    reason="external pinned checkout is optional in normal CI",
)
def test_real_solver_deterministic_reproduction(tmp_path, artifact):
    source = Path(os.environ["SOH_SOLVER_SOURCE"])
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        cli.run(source, ROOT / "manifest.json", output)
    a, b = [read_json(p / "dataset.json") for p in outputs]
    assert canonical(a) == canonical(b)
    # Artifact records the interpreter; numeric outputs also agree across 3.10/3.13.
    assert numerically_equal(a["payload"]["games"], artifact["payload"]["games"])
    assert a["payload"]["source_files"] == artifact["payload"]["source_files"]
    with pytest.raises(ValueError, match="already exists"):
        cli.run(source, ROOT / "manifest.json", outputs[0])


@pytest.mark.skipif(
    not os.environ.get("SOH_SOLVER_SOURCE"), reason="requires external pinned source"
)
def test_reject_modified_external_source(tmp_path):
    from sleight_of_hand.solver_lab.worker import verify_source

    source = tmp_path / "solver"
    shutil.copytree(os.environ["SOH_SOLVER_SOURCE"], source)
    verify_source(source)
    target = source / "python/src/games/river_holdem.py"
    original = target.read_bytes()
    target.write_bytes(original + b"\n# altered\n")
    with pytest.raises(ValueError, match="differs"):
        verify_source(source)
    target.write_bytes(original)
    (source / "python/src/unpinned.py").write_text("pass\n")
    with pytest.raises(ValueError, match="differs"):
        verify_source(source)


@pytest.mark.skipif(
    not os.environ.get("SOH_SOLVER_SOURCE"), reason="requires external pinned source"
)
def test_external_encoding_and_sparse_fast_path_defect():
    code = """
import sys
from sleight_of_hand.solver_lab.contract import DECK, ROOT, read_json
from sleight_of_hand.solver_lab.worker import verify_source
from sleight_of_hand.holdem.equity import encode, rank_hand
source, _ = verify_source(sys.argv[1])
sys.path.insert(0, str(source / 'python/src'))
from games.river_holdem import card_id, card_str, evaluate_7, RiverHoldemGame, RiverHoldemConfig
from algorithms.vector_cfr import VectorCFRTrainer
for card in DECK:
    assert card_str(card_id(card)) == card
assert card_id('As') != encode('As') or card_id('2d') != encode('2d')
for spec in read_json(ROOT / 'manifest.json')['games']:
    for hand_range in spec['ranges']:
        for hand in hand_range:
            cards = hand['cards'] + spec['board']
            assert evaluate_7([card_id(c) for c in cards]) == rank_hand([encode(c) for c in cards])
spec = read_json(ROOT / 'manifest.json')['games'][0]
game = RiverHoldemGame(RiverHoldemConfig(board=spec['board'], pot=100, stacks=(100,100), bet_sizes=(0.5,1.0), max_raises=2,
    ranges=tuple([''.join(h['cards']) for h in r] for r in spec['ranges'])))
try:
    VectorCFRTrainer(game).run(1)
except KeyError as error:
    assert error.args[0] == (3, 13, 10, 7)
else:
    raise AssertionError('expected pinned fast-evaluator defect')
"""
    subprocess.run(
        [sys.executable, "-c", code, os.environ["SOH_SOLVER_SOURCE"]],
        check=True,
        timeout=10,
    )
