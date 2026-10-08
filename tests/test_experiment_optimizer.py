"""Bounded authority, deterministic proposals and real offline recovery."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from unittest.mock import Mock

import pytest

from sleight_of_hand.experiments import (
    analysis,
    campaign,
    cli,
    journal,
    optimizer,
    programme,
    runner,
    scheduler,
    supervisor,
)
from sleight_of_hand.experiments.model import canonical, sha256
from sleight_of_hand.experiments.spec import ExperimentSpec, SeedPolicy, dump
from sleight_of_hand.experiments.storage import Index
from sleight_of_hand.policy.heuristic import DEFAULT_PARAMS
from tests.test_experiment_chipzen import ENV, chipzen_spec
from tests.test_experiment_supervisor import write_programme

CONFIG = runner.ROOT / "experiments/O0001.json"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key in ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(scheduler, "available_mib", lambda: 4096)


def fixture(tmp_path, *, tiny=True, change=None, chipzen=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    template = campaign.load(CONFIG).template
    (tmp_path / "template.json").write_bytes(dump(template))
    config["template"]["path"] = "template.json"
    config["id"] = "O9999"
    config["depends_on"] = []
    config["resources"]["min_free_mib"] = 0
    if tiny:
        config["parameters"] = [
            {
                "parameter": "steepness",
                "min": 8.0,
                "max": 10.0,
                "initial_step": 2.0,
                "min_step": 1.0,
            }
        ]
        config["max_generations"] = 2
        for key in ("search_stopping", "confirmation_stopping"):
            config[key] = {
                "min_matches": 2,
                "max_matches": 2,
                "min_hands": 2,
                "max_hands": 2,
                "hands_per_match": 1,
                "matches_per_shard": 1,
                "evaluation_interval_matches": 1,
                "early_stopping": "none",
            }
    if change:
        change(config)
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(config))
    c = campaign.load(path)
    if chipzen:
        programme_path = write_programme(
            tmp_path,
            [[replace(chipzen_spec(), policy_revision=runner.policy_revision())]],
        )
        p = json.loads(programme_path.read_text())
    else:
        p = {"resources": {"workers": 2, "min_free_mib": 0}, "experiments": []}
        programme_path = tmp_path / "programme.json"
    p.update(
        schema_version=2, campaigns=[{"path": path.name, "campaign_hash": c.digest}]
    )
    programme_path.write_text(json.dumps(p))
    return c, programme.load(programme_path), programme_path


def prepared(c):
    state = optimizer.initial(c)
    state["pending"] = optimizer.batch(c, state)
    return state


def report_for(state, improvements=None):
    """Synthetic already-computed statistics for selection-unit tests only."""
    improvements = improvements or {}
    item = optimizer.item_for(state["pending"])
    arms = {
        s.arm: {
            "spec_hash": s.spec_hash,
            "problems": [],
            "hands": s.stopping.max_hands,
            "matches": s.stopping.max_matches,
            "sufficient_sample": True,
        }
        for s in item.specs
    }
    comparisons = []
    family_size = len(arms) * (len(arms) - 1) // 2
    for a, b in combinations(sorted(arms), 2):
        mean = improvements.get(b, 0) - improvements.get(a, 0)
        comparison = {
            "arms": [a, b],
            "paired_difference": {
                "matches": item.specs[0].stopping.max_matches,
                "mean_bb_per_100": mean,
                "se": 1.0,
                "ci95": [mean - analysis.Z95, mean + analysis.Z95],
            },
        }
        comparison["recommendation"] = analysis.recommend(
            comparison, True, family_size=family_size
        )
        comparisons.append(comparison)
    return {
        "experiment_id": item.experiment_id,
        "strata": [
            {
                "platform": "local",
                "provenance": item.specs[0].provenance.value,
                "arms": arms,
                "comparisons": comparisons,
            }
        ],
    }


def test_exact_canonical_authority():
    c = campaign.load(CONFIG)
    v = c.config
    assert v["id"] == "O0001" and v["depends_on"] == ["E0003"]
    assert v["initial"] == DEFAULT_PARAMS.__dict__
    assert v["max_generations"] == 4 and v["max_generated_vectors"] == 45
    assert v["resources"] == {"workers": 2, "min_free_mib": 1024}
    assert v["fixed_policy"] == {
        "samples": 128,
        "opponent_memory": True,
        "historical_prior": None,
    }
    assert v["search_stopping"]["max_hands"] == 2000
    assert v["confirmation_stopping"]["max_hands"] == 6000
    assert (
        c.digest
        == programme.load(runner.ROOT / "experiments/programme.json")
        .campaigns[0]
        .digest
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(unknown=True),
        lambda v: v.update(schema_version=2),
        lambda v: v.update(schema_version=True),
        lambda v: v.update(governance="UNRATED_APPROVED"),
        lambda v: v["parameters"].append(copy.deepcopy(v["parameters"][0])),
        lambda v: v["parameters"][0].update(parameter="samples"),
        lambda v: v["parameters"][0].update(unknown=True),
        lambda v: v["parameters"][0].update(min=12, max=4),
        lambda v: v["parameters"][0].update(initial_step=0),
        lambda v: v["parameters"][0].update(initial_step=-1),
        lambda v: v["parameters"][0].update(min_step=0),
        lambda v: v["parameters"][0].update(min_step=-1),
        lambda v: v["parameters"][0].update(min_step=3),
        lambda v: v["initial"].update(steepness=7),
        lambda v: v["initial"].update(steepness=float("nan")),
        lambda v: v.update(max_generations=5),
        lambda v: v.update(max_generated_vectors=46),
        lambda v: v["resources"].update(workers=3),
        lambda v: v["fixed_policy"].update(samples=2),
        lambda v: v["fixed_policy"].update(opponent_memory=False),
        lambda v: v["fixed_policy"].update(historical_prior={}),
        lambda v: v["search_stopping"].update(early_stopping="adaptive"),
        lambda v: v["search_stopping"].update(min_matches=1),
        lambda v: v["template"].update(spec_hash="0" * 64),
        lambda v: v.update(objective="fitness"),
        lambda v: v.update(depends_on=["E0003", "E0003"]),
    ],
)
def test_reject_invalid_campaign(tmp_path, change):
    with pytest.raises((ValueError, TypeError)):
        fixture(tmp_path, change=change)


def test_duplicate_json_parameter_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "campaign.json"
    path.write_text(
        path.read_text().replace(
            '"steepness": 8.0', '"steepness": 8.0, "steepness": 8.0'
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        campaign.load(path)


def test_duplicate_campaign_ids_rejected(tmp_path):
    _, _, path = fixture(tmp_path)
    other = tmp_path / "other.json"
    other.write_bytes((tmp_path / "campaign.json").read_bytes())
    p = json.loads(path.read_text())
    p["campaigns"].append({**p["campaigns"][0], "path": "other.json"})
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="duplicate campaign ID"):
        programme.load(path)


def test_aggression_point_six_forbidden():
    c = campaign.load(CONFIG)
    with pytest.raises(ValueError, match="outside approved"):
        campaign.validate_vector(c, {**c.config["initial"], "aggression": 0.6})


def test_first_generation_exact_neighbors_and_full_vectors():
    c = campaign.load(CONFIG)
    s = optimizer.initial(c)
    vectors = campaign.neighbors(c, s["incumbent"], s["steps"])
    assert len(vectors) == 10 and len({canonical(v) for v in vectors}) == 10
    for p in c.config["parameters"]:
        key = p["parameter"]
        values = [v[key] for v in vectors if v[key] != s["incumbent"][key]]
        assert sorted(values) == pytest.approx(
            [
                s["incumbent"][key] - p["initial_step"],
                s["incumbent"][key] + p["initial_step"],
            ]
        )
    b = optimizer.batch(c, s)
    assert len(b["specs"]) == 11
    assert all(
        set(e["spec"]["policy_config"]["params"]) == set(c.config["initial"])
        for e in b["specs"]
    )


def test_determinism_ignores_map_order_time_rng(tmp_path):
    c, _, _ = fixture(tmp_path, tiny=False)
    first = optimizer.initial(c)
    second = {
        **first,
        "incumbent": dict(reversed(list(first["incumbent"].items()))),
        "steps": dict(reversed(list(first["steps"].items()))),
    }
    assert optimizer.batch(c, first) == optimizer.batch(c, second)
    assert optimizer.batch(c, first) == optimizer.batch(c, first)
    assert all(
        ExperimentSpec.from_dict(s["spec"]).spec_hash == s["spec_hash"]
        for s in optimizer.batch(c, first)["specs"]
    )


def test_bounds_omit_without_clipping(tmp_path):
    c, _, _ = fixture(tmp_path)
    s = optimizer.initial(c)
    assert [
        v["steepness"] for v in campaign.neighbors(c, s["incumbent"], s["steps"])
    ] == [10.0]
    s["incumbent"]["steepness"] = 9.0
    assert campaign.neighbors(c, s["incumbent"], s["steps"]) == []


def test_search_and_confirmation_streams_disjoint_and_fresh():
    c = campaign.load(CONFIG)
    blocks = campaign.seed_blocks(c)
    assert blocks == campaign.seed_blocks(c) and len(set(blocks.values())) == 5
    seen = {
        SeedPolicy(c.template.seed_policy.base_seed).derive(role, m)
        for role in ("deck", "soh", "opponent")
        for m in range(60)
    }
    for block, seed in blocks.items():
        streams = {
            SeedPolicy(seed).derive(role, m)
            for role in ("deck", "soh", "opponent")
            for m in range(60 if block.startswith("confirmation") else 20)
        }
        assert seen.isdisjoint(streams)
        seen |= streams


@pytest.mark.parametrize(
    "used,generation,reason", [(40, 0, "budget"), (0, 4, "generations")]
)
def test_budget_and_generation_limits(used, generation, reason):
    c = campaign.load(CONFIG)
    s = optimizer.initial(c)
    s.update(used=used, generation=generation)
    assert reason in optimizer.stop_reason(c, s)


def test_no_qualifier_shrinks_and_never_below_minimum(tmp_path):
    c, _, _ = fixture(tmp_path)
    s = prepared(c)
    decision = optimizer.selection(c, s, report_for(s))
    assert decision["incumbent"] == c.config["initial"]
    assert decision["step_shrink"] and decision["steps"] == {"steepness": 1.0}
    assert campaign.shrink(c, decision["steps"]) == decision["steps"]


def test_raw_mean_alone_insufficient(tmp_path):
    c, _, _ = fixture(tmp_path)
    s = prepared(c)
    candidate = s["pending"]["specs"][1]["spec"]["arm"]
    decision = optimizer.selection(c, s, report_for(s, {candidate: 1.0}))
    assert decision["selected_arm"] == "baseline"


def test_qualifying_candidate_and_deterministic_tie_break(tmp_path):
    c, _, _ = fixture(tmp_path, tiny=False)
    s = prepared(c)
    candidates = sorted(
        e["spec"]["arm"]
        for e in s["pending"]["specs"]
        if e["spec"]["arm"] != "baseline"
    )
    r = report_for(s, {candidates[0]: 10.0, candidates[1]: 10.0})
    decision = optimizer.selection(c, s, r)
    assert decision["selected_arm"] == candidates[0] and not decision["step_shrink"]
    assert decision["incumbent"] != c.config["initial"]


def test_bonferroni_reuses_existing_se_and_is_more_conservative():
    base = {
        "arms": ["a", "b"],
        "paired_difference": {
            "mean_bb_per_100": 2.1,
            "se": 1.0,
            "ci95": [2.1 - analysis.Z95, 2.1 + analysis.Z95],
        },
    }
    assert analysis.recommend(base, True)["verdict"] == "DIFFERENCE_DETECTED"
    assert (
        analysis.recommend(base, True, family_size=55)["verdict"]
        == "NO_DIFFERENCE_DETECTED"
    )
    assert base["multiplicity"]["family_size"] == 55


@pytest.mark.parametrize(
    "defect", ["hands", "matches", "problems", "family", "paired", "pairs"]
)
def test_selection_needs_complete_valid_evidence(tmp_path, defect):
    c, _, _ = fixture(tmp_path)
    s = prepared(c)
    r = report_for(s)
    stratum = r["strata"][0]
    if defect in {"hands", "matches"}:
        stratum["arms"]["baseline"][defect] -= 1
    elif defect == "problems":
        stratum["arms"]["baseline"]["problems"] = ["checksum failed"]
    elif defect == "family":
        stratum["comparisons"][0]["multiplicity"]["method"] = "none"
    elif defect == "paired":
        stratum["comparisons"][0]["paired_difference"]["matches"] -= 1
    else:
        stratum["comparisons"] = []
    with pytest.raises(ValueError):
        optimizer.selection(c, s, r)


@pytest.mark.parametrize(
    "improvement,expected",
    [(0.0, "INCONCLUSIVE"), (10.0, "PROMISING"), (-10.0, "REJECTED")],
)
def test_confirmation_classification_and_firewall(tmp_path, improvement, expected):
    c, _, _ = fixture(tmp_path)
    s = optimizer.initial(c)
    s.update(
        stop_reason="maximum generations reached",
        incumbent={**c.config["initial"], "steepness": 10.0},
    )
    s["pending"] = optimizer.batch(c, s)
    candidate = s["pending"]["specs"][1]["spec"]["arm"]
    before = DEFAULT_PARAMS.__dict__.copy()
    pointer = subprocess.check_output(
        [
            "git",
            "for-each-ref",
            "refs/heads/deploy/chipzen",
            "refs/remotes/origin/deploy/chipzen",
        ],
        cwd=runner.ROOT,
    )
    assert (
        optimizer.selection(c, s, report_for(s, {candidate: improvement}))[
            "classification"
        ]
        == expected
    )
    assert DEFAULT_PARAMS.__dict__ == before
    assert (
        subprocess.check_output(
            [
                "git",
                "for-each-ref",
                "refs/heads/deploy/chipzen",
                "refs/remotes/origin/deploy/chipzen",
            ],
            cwd=runner.ROOT,
        )
        == pointer
    )


def test_plan_is_read_only_and_cli_shows_campaign(tmp_path, monkeypatch):
    c, p, _ = fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    monkeypatch.setattr(
        scheduler, "run", Mock(side_effect=AssertionError("planning executed"))
    )
    result = supervisor.plan(p, root)
    assert not root.exists()
    row = result["campaigns"][0]
    assert row["status"] == "PENDING" and row["incumbent"] == c.config["initial"]
    assert row["generation"] == 0 and row["candidate_budget_used"] == 0
    assert "O9999  PENDING" in cli.render(result)
    assert cli.parser().parse_args(["optimize", "O9999", "--json"]).json


def test_real_two_generations_shrink_seal_analyze_finish_and_quiesce(
    tmp_path, monkeypatch
):
    c, p, _ = fixture(tmp_path / "authority", chipzen=True)
    root = tmp_path / "runs"
    before = supervisor.plan(p, root)
    assert before["items"][0]["status"] == "BLOCKED"
    first = supervisor.execute(p, root)
    assert first["campaigns"][0]["generation"] == 1, first
    assert first["campaigns"][0]["steps"] == {"steepness": 1.0}
    second = supervisor.execute(p, root)
    row = second["campaigns"][0]
    assert (
        row["status"] == "COMPLETE"
        and row["classification"] == "NO_IMPROVEMENT_DETECTED"
    ), second
    assert row["incumbent"] == c.config["initial"]
    assert row["generation"] == 2 and row["candidate_budget_used"] == 4
    records = Index(root / "index.sqlite").runs()
    assert len(records) == 8 and all(
        journal.verify(r["run_dir"])["ok"] for r in records
    )
    state, events = optimizer.reconstruct(
        c, p, root, runner.source_state(), verify=True
    )
    assert state["classification"] == row["classification"]
    assert sum(e["kind"] == "result" for e in events) == 2
    journal_bytes = {
        str(f): f.read_bytes() for f in optimizer.directory(root, c).rglob("*.json")
    }
    reports = {str(f): f.read_bytes() for f in (root / "analysis").glob("*")}
    monkeypatch.setattr(
        scheduler, "run", Mock(side_effect=AssertionError("duplicate execution"))
    )
    monkeypatch.setattr(
        optimizer,
        "append",
        Mock(side_effect=AssertionError("duplicate generation event")),
    )
    third = supervisor.execute(p, root)
    assert third["actions"] == [] and third["nothing_to_do"]
    assert Index(root / "index.sqlite").runs() == records
    assert {
        str(f): f.read_bytes() for f in optimizer.directory(root, c).rglob("*.json")
    } == journal_bytes
    assert {str(f): f.read_bytes() for f in (root / "analysis").glob("*")} == reports


@pytest.mark.parametrize("boundary", ["batch", "shard", "analysis"])
def test_crash_resume_keeps_specs_and_completed_shards(tmp_path, monkeypatch, boundary):
    c, p, _ = fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    if boundary == "batch":
        original = optimizer.publish

        def interrupt(path, target, data):
            if target.parent.name == "specs":
                raise KeyboardInterrupt
            return original(path, target, data)

        monkeypatch.setattr(optimizer, "publish", interrupt)
    elif boundary == "shard":

        def interrupt(specs, root, *args, **kwargs):
            runner.run_shard(
                specs[0].to_dict(), specs[0].stopping.shards()[0], "fixture", str(root)
            )
            raise KeyboardInterrupt

        monkeypatch.setattr(scheduler, "run", interrupt)
    else:
        original = supervisor._analyze

        def interrupt(*args, **kwargs):
            original(*args, **kwargs)
            raise KeyboardInterrupt

        monkeypatch.setattr(supervisor, "_analyze", interrupt)
    with pytest.raises(KeyboardInterrupt):
        supervisor.execute(p, root)
    state, events = optimizer.reconstruct(c, p, root, runner.source_state())
    hashes = [s["spec_hash"] for s in state["pending"]["specs"]]
    before = optimizer._completed_records(root, state["pending"])
    monkeypatch.undo()
    outcome = supervisor.execute(p, root)
    assert outcome["campaigns"][0]["generation"] == 1, outcome
    state, events = optimizer.reconstruct(c, p, root, runner.source_state())
    batches = [e["payload"] for e in events if e["kind"] == "batch"]
    assert len(batches) == 1 and [s["spec_hash"] for s in batches[0]["specs"]] == hashes
    after = optimizer._completed_records(root, batches[0])
    assert len(after) == 4 and all(r in after for r in before)
    if boundary == "analysis":
        assert not any(
            a["action"] in {"OPTIMIZER_RUN", "OPTIMIZER_ANALYZE"}
            for a in outcome["actions"]
        )


@pytest.mark.parametrize(
    "defect",
    [
        "bytes",
        "vector",
        "extra",
        "missing",
        "journal",
        "budget",
        "report",
        "evidence",
        "unauthorized_index",
    ],
)
def test_integrity_fails_closed(tmp_path, defect):
    c, p, _ = fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    directory = optimizer.directory(root, c)
    file = next((directory / "specs").glob("*.json"))
    if defect in {"bytes", "vector"}:
        file.chmod(0o600)
        v = json.loads(file.read_text())
        if defect == "vector":
            v["spec"]["policy_config"]["params"]["steepness"] = 100.0
        file.write_bytes(canonical(v) + b" \n")
        file.chmod(0o400)
    elif defect == "extra":
        (directory / "specs/extra.json").write_bytes(file.read_bytes())
    elif defect == "missing":
        file.unlink()
    elif defect in {"journal", "budget"}:
        event = sorted((directory / "events").glob("*.json"))[1]
        event.chmod(0o600)
        value = json.loads(event.read_text())
        value["payload"]["budget_after"] += 1
        if defect == "budget":
            value["sha256"] = sha256(
                canonical({k: v for k, v in value.items() if k != "sha256"})
            )
        event.write_bytes(canonical(value))
    elif defect == "report":
        next((root / "analysis").glob("*.json")).write_text("{}")
    elif defect == "evidence":
        run = Index(root / "index.sqlite").runs()[0]
        (Path(run["run_dir"]) / "normalized/hands.jsonl").write_text("{}\n")
    else:
        value = json.loads(file.read_text())["spec"]
        value["arm"] = "unauthorized"
        Index(root / "index.sqlite").register_spec(ExperimentSpec.from_dict(value))
    result = supervisor.execute(p, root)
    assert result["campaigns"][0]["status"] == "FAILED", result
    assert not result["actions"]
    assert supervisor.plan(p, root)["campaigns"][0]["status"] == "FAILED"
    assert supervisor.execute(p, root)["actions"] == []


@pytest.mark.parametrize("identity", ["campaign", "programme", "policy"])
def test_identity_change_requires_review(tmp_path, identity):
    c, p, _ = fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    source = runner.source_state()
    if identity == "campaign":
        value = c.config
        value["parameters"][0]["max"] = 11.0
        c = replace(c, config_json=canonical(value).decode())
    elif identity == "programme":
        p = replace(p, digest="changed")
    else:
        source["policy_revision"] = "sha256:" + "0" * 64
    with pytest.raises(optimizer.ReviewRequired):
        optimizer.reconstruct(c, p, root, source)


def test_journal_rejects_secrets(tmp_path):
    with pytest.raises(ValueError, match="credential"):
        optimizer.append(tmp_path, [], "test", {"secret": "must-not-persist"})
    assert not tmp_path.joinpath("events").exists()


def test_installed_optimizer_cli(tmp_path):
    _, _, path = fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for command in (["plan"], ["optimize", "O9999"], ["auto"], ["auto"]):
        process = subprocess.run(
            [
                "uv",
                "run",
                "--locked",
                "--python",
                sys.executable,
                "research",
                *command,
                "--programme",
                str(path),
                "--root",
                str(root),
                "--json",
            ],
            cwd=runner.ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert process.returncode == 0, process.stderr
        result = json.loads(process.stdout)
    assert result["campaigns"][0]["status"] == "COMPLETE"
    assert not result["actions"]


def test_noncanonical_search_requires_separate_real_held_out_confirmation(
    tmp_path, monkeypatch
):
    c, p, _ = fixture(
        tmp_path / "authority", change=lambda v: v.update(max_generations=1)
    )
    root = tmp_path / "runs"
    paired = analysis.paired

    def controlled_search(a, b):
        if ".search." in a["spec"].experiment_id:
            # Only this branch-selection edge case uses synthetic statistics.
            # The confirmation below uses real held-out outcome statistics.
            return {
                "matches": 2,
                "mean_bb_per_100": 10.0,
                "se": 1.0,
                "ci95": [8.0, 12.0],
                "identical_match_results": 0,
            }
        return paired(a, b)

    monkeypatch.setattr(analysis, "paired", controlled_search)
    first = supervisor.execute(p, root)
    row = first["campaigns"][0]
    assert row["status"] == "CONFIRMING" and row["classification"] is None
    assert row["intended_action"] == "CONFIRM"
    assert row["incumbent"] != c.config["initial"]
    assert not any(
        "confirmation" in s.experiment_id for s in Index(root / "index.sqlite").specs()
    )
    second = supervisor.execute(p, root)
    row = second["campaigns"][0]
    assert row["status"] == "COMPLETE" and row["classification"] == "INCONCLUSIVE"
    state, events = optimizer.reconstruct(
        c, p, root, runner.source_state(), verify=True
    )
    kinds = [e["kind"] for e in events]
    assert kinds.index("stop") < [i for i, k in enumerate(kinds) if k == "batch"][1]
    assert state["used"] == 4
    batches = [e["payload"] for e in events if e["kind"] == "batch"]
    assert batches[0]["seed_block"] != batches[1]["seed_block"]
    assert (
        batches[1]["specs"][0]["spec"]["policy_config"]["params"] == c.config["initial"]
    )
    assert supervisor.execute(p, root)["actions"] == []


def test_scheduler_resources_are_not_bypassed(tmp_path, monkeypatch):
    _, p, _ = fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    monkeypatch.setattr(scheduler, "available_mib", lambda: None)
    monkeypatch.setattr(
        scheduler, "run", Mock(side_effect=AssertionError("unsafe admission"))
    )
    result = supervisor.execute(p, root)
    assert result["campaigns"][0]["status"] == "BLOCKED"
    assert not Index(root / "index.sqlite").runs()
    assert supervisor.execute(p, root)["actions"] == []


def test_strongest_qualifying_improvement_wins(tmp_path):
    c, _, _ = fixture(tmp_path, tiny=False)
    s = prepared(c)
    names = sorted(
        e["spec"]["arm"]
        for e in s["pending"]["specs"]
        if e["spec"]["arm"] != "baseline"
    )
    assert (
        optimizer.selection(c, s, report_for(s, {names[0]: 10.0, names[1]: 20.0}))[
            "selected_arm"
        ]
        == names[1]
    )


def test_programme_pins_changed_search_space(tmp_path, capsys):
    _, _, p = fixture(tmp_path)
    file = tmp_path / "campaign.json"
    value = json.loads(file.read_text())
    value["parameters"][0]["max"] = 11.0
    file.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash"):
        programme.load(p)
    assert (
        cli.main(
            ["plan", "--programme", str(p), "--root", str(tmp_path / "runs"), "--json"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "REQUIRES_REVIEW"
    assert not (tmp_path / "runs").exists()
