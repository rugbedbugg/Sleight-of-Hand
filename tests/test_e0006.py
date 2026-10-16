"""E0006: frozen SafeExploit v1 ablation authority, pairing and analysis.

Evidence tests use temporary roots and tiny derived horizons only; the real
E0006 horizon is never played here.
"""

import dataclasses
import json
import shutil
import sqlite3
import stat
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import pytest

from sleight_of_hand.experiments import (
    campaign,
    cli,
    e0006,
    programme,
    runner,
    scheduler,
    supervisor,
)
from sleight_of_hand.experiments.analysis import Z95
from sleight_of_hand.experiments.journal import read_raw
from sleight_of_hand.experiments.model import canonical, sha256
from sleight_of_hand.experiments.platforms.base import MatchContext
from sleight_of_hand.experiments.platforms.local import LocalPlatform
from sleight_of_hand.experiments.spec import StoppingRule, dump
from sleight_of_hand.experiments.spec import load as load_spec
from sleight_of_hand.experiments.storage import Index
from sleight_of_hand.holdem.agent import HoldemAgent

ROOT = runner.ROOT
STANDALONE = ROOT / "experiments" / "safe-exploit-v1"
DESIGN = STANDALONE / "DESIGN.md"
POLICY = "sha256:eff445aa51a2b78871c294df085353b4f00b3b38a0df73a954f13f2270a0cfbf"
CONTROL_HASH = "0baff5c5973ab55fada058e10ea2514c15f26690eafac50e9b6eadbe57f9ce3b"
TREATMENT_HASH = "37c8e1e3e2b0b9e422801c9ba4f9a0a4029917328ac39ceef7d0cd3a5832b33d"
#: The historical programme's canonical digest, unchanged by E0006.
HISTORICAL_DIGEST = (
    "sha256:c28fafbf567d30be0738e8df1849eb7a6c7a843a57c801554da41fd5ef2e7fc8"
)


def item():
    return programme.load(e0006.PROGRAMME).item("E0006")


def pair():
    return e0006.arms(item())


# --- authority and configuration ---------------------------------------------------


def test_standalone_programme_pins_exactly_two_e0006_arms():
    p = programme.load(e0006.PROGRAMME)
    assert [i.experiment_id for i in p.items] == ["E0006"] and not p.campaigns
    i = p.items[0]
    # Prepared, not approved. A later reviewed change to LOCAL_APPROVED (which
    # changes only the programme hash) must update this assertion with it.
    assert i.governance == "REQUIRES_REVIEW" and i.depends_on == ()
    assert {s.arm: s.spec_hash for s in i.specs} == {
        "canonical": CONTROL_HASH,
        "safe-exploit-v1": TREATMENT_HASH,
    }
    for spec in i.specs:
        assert load_spec(STANDALONE / "specs" / f"E0006-{spec.arm}.json") == spec
        assert (STANDALONE / "specs" / f"E0006-{spec.arm}.json").read_bytes() == dump(
            spec
        )


def test_arms_differ_only_in_safe_exploit():
    control, treatment = pair()
    a, b = control.to_dict(), treatment.to_dict()
    assert {k for k in a if a[k] != b[k]} == {"arm", "policy_config"}
    assert a["policy_config"] == {"opponent_memory": True, "safe_exploit": False}
    assert b["policy_config"] == {"opponent_memory": True, "safe_exploit": True}
    for spec in (control, treatment):
        assert "params" not in spec.policy_config  # the canonical PolicyParams
        assert "historical_prior" not in spec.policy_config


def test_arms_use_the_current_policy_and_integrated_source():
    for spec in pair():
        assert spec.policy_revision == POLICY == runner.policy_revision()
        assert spec.source_sha == "b4145d34bd5529b9e7a3c47881d550f4e7eb6f7a"
        assert spec.platform == "local" and spec.provenance.value == "LOCAL_SELFPLAY"


def test_frozen_horizon_cohort_and_assignment():
    control, _ = pair()
    s = control.stopping
    assert (s.min_matches, s.max_matches, s.hands_per_match) == (300, 300, 400)
    assert (s.min_hands, s.max_hands, s.early_stopping) == (120000, 120000, "none")
    assert len(s.shards()) == 30 and s.matches_per_shard == 10
    assert control.platform_config["stack_mode"] == "reset"
    cohort = [o.qualified for o in control.opponent_cohort]
    assert tuple(cohort) == e0006.EXPLOIT_SENSITIVE + e0006.REFERENCE
    assigned = [control.opponent_for(m).qualified for m in range(300)]
    assert {o: assigned.count(o) for o in cohort} == dict.fromkeys(cohort, 60)
    assert all(assigned[m] == cohort[m % 5] for m in range(300))
    LocalPlatform().prepare(control)  # every opponent is constructible


def test_seed_streams_pair_arms_and_are_fresh():
    control, treatment = pair()
    assert control.seed_policy.base_seed == 20261006
    assert control.seed_policy.common_random_numbers
    seeds = [runner.match_seeds(control, m) for m in range(300)]
    assert seeds == [runner.match_seeds(treatment, m) for m in range(300)]
    mine = {v for s in seeds for v in s.values()}
    assert len(mine) == 900  # distinct per role and match
    history = programme.load(ROOT / "experiments" / "programme.json")
    claimed = {
        v
        for i in history.items
        for spec in i.specs
        for m in range(spec.stopping.max_matches)
        for v in runner.match_seeds(spec, m).values()
    }
    for c in history.campaigns:
        claimed |= campaign.streams(c)
    assert not mine & claimed


@pytest.mark.parametrize(
    "change",
    [
        {"policy_config": {"opponent_memory": True, "safe_exploit": True}},
        {"policy_config": {"opponent_memory": False, "safe_exploit": False}},
        {
            "policy_config": {
                "opponent_memory": True,
                "safe_exploit": False,
                "params": {"aggression": 0.6},
            }
        },
        {"description": "changed"},
    ],
)
def test_check_arms_rejects_any_other_difference(change):
    control, treatment = pair()
    with pytest.raises(e0006.EvidenceError):
        e0006.check_arms(
            control.derive_arm("other", control.created_at, **change), treatment
        )


def test_check_arms_rejects_a_reordered_cohort():
    control, treatment = pair()
    cohort = tuple(reversed(control.opponent_cohort))
    with pytest.raises(e0006.EvidenceError, match="cohort"):
        e0006.check_arms(
            dataclasses.replace(control, opponent_cohort=cohort),
            dataclasses.replace(treatment, opponent_cohort=cohort),
        )


def test_historical_authority_is_unchanged_and_isolated():
    history = programme.load(ROOT / "experiments" / "programme.json")
    assert history.digest == HISTORICAL_DIGEST
    assert "E0006" not in {i.experiment_id for i in history.items}
    assert [c.identifier for c in history.campaigns] == ["O0001", "O0002"]
    standalone = programme.load(e0006.PROGRAMME)
    assert {s.spec_hash for s in standalone.items[0].specs}.isdisjoint(
        s.spec_hash for i in history.items for s in i.specs
    )


def test_standalone_plan_is_read_only_and_independent_of_e0001(tmp_path):
    root = tmp_path / "runs-e0006"
    result = supervisor.plan(programme.load(e0006.PROGRAMME), root)
    assert [r["experiment"] for r in result["items"]] == ["E0006"]
    assert result["campaigns"] == []
    row = result["items"][0]
    assert row["status"] == "REQUIRES_REVIEW" and row["intended_action"] == "REVIEW"
    assert row["policy_state"] == "CURRENT" and row["pending_shards"] == 60
    assert not root.exists()


@pytest.mark.parametrize("command", ["run", "analyze", "auto"])
def test_cli_reserves_the_historical_root_for_its_programme(
    command, tmp_path, monkeypatch, capsys
):
    reserved = tmp_path / "runs"
    monkeypatch.setattr(cli, "RUNS", reserved)
    args = [command] + (["E0006"] if command != "auto" else [])
    args += ["--programme", str(e0006.PROGRAMME), "--root", str(reserved), "--json"]
    assert cli.main(args) == 1
    assert "historical runs root is reserved" in capsys.readouterr().err
    assert not reserved.exists()


def test_cli_still_inspects_a_standalone_programme_read_only(tmp_path, capsys):
    root = tmp_path / "runs-e0006"
    args = ["plan", "--programme", str(e0006.PROGRAMME), "--root", str(root)]
    assert cli.main([*args, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["items"][0]["experiment"] == "E0006"
    assert not root.exists()


def test_analysis_refuses_the_historical_root(capsys):
    assert e0006.main(["--root", str(e0006.HISTORICAL_ROOT)]) == 1
    assert "reserved" in capsys.readouterr().err


def test_design_document_names_the_frozen_hashes():
    text = DESIGN.read_text()
    for value in (CONTROL_HASH, TREATMENT_HASH, e0006.PLAN_HASH, POLICY):
        assert value in text


def test_analysis_module_never_writes_the_index_or_plays():
    source = Path(e0006.__file__).read_text()
    for forbidden in (
        "Index(",
        "register_spec",
        "run_shard",
        "scheduler",
        "import random",
    ):
        assert forbidden not in source


# --- statistics ----------------------------------------------------------------------


def test_plan_weights_and_family_are_predeclared():
    weights = e0006.PLAN["primary"]["weights"]
    assert set(weights) == set(e0006.EXPLOIT_SENSITIVE)
    assert sum(weights.values()) == pytest.approx(1)
    assert len(set(weights.values())) == 1
    assert len(e0006.SECONDARY) == 6
    assert e0006.PLAN_HASH == "sha256:" + sha256(canonical(e0006.PLAN))


def test_stratified_contrast_weights_opponents_not_matches():
    diffs = {
        "a": [10.0, 20.0],
        "b": [0.0] * 3 + [4.0] * 3 + [8.0] * 4,
        "c": [-6.0, -2.0],
    }
    weights = dict.fromkeys(diffs, 1 / 3)
    result = e0006.stratified(diffs, weights)
    assert result["mean"] == pytest.approx((15 + 4.4 - 4) / 3)
    pooled = sum(sum(v) for v in diffs.values()) / 14
    assert result["mean"] != pytest.approx(pooled)

    def var(v):
        m = sum(v) / len(v)
        return sum((x - m) ** 2 for x in v) / (len(v) - 1)

    expected = sqrt(sum(var(v) / len(v) for v in diffs.values()) / 9)
    assert result["se"] == pytest.approx(expected)
    with pytest.raises(e0006.EvidenceError):
        e0006.stratified({"a": [1.0]}, {"a": 1.0})
    with pytest.raises(e0006.EvidenceError):
        e0006.stratified({"a": [1.0, 2.0]}, {"a": 0.5, "b": 0.5})


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (0.1, 50.0, "IMPROVEMENT_DETECTED"),
        (-50.0, -0.1, "REGRESSION_DETECTED"),
        (-30.0, 30.0, "NO_DIFFERENCE_DETECTED"),
        (-5.0, 12.0, "NO_DIFFERENCE_DETECTED"),
        (-30.1, 10.0, "INCONCLUSIVE"),
        (-1.0, 31.0, "INCONCLUSIVE"),
        (0.0, 5.0, "NO_DIFFERENCE_DETECTED"),
    ],
)
def test_classification_distinguishes_non_detection_from_inconclusive(
    low, high, expected
):
    assert e0006.classify(low, high) == expected


def test_bonferroni_critical_values():
    assert e0006.bonferroni_z(1) == pytest.approx(Z95)
    assert e0006.bonferroni_z(6) == pytest.approx(NormalDist().inv_cdf(1 - 0.05 / 12))
    assert e0006.bonferroni_z(6) > e0006.bonferroni_z(5) > Z95
    with pytest.raises(ValueError):
        e0006.bonferroni_z(0)


def synthetic(effects: dict[str, float], per: int = 4):
    """Paired per-match values: treatment = control + effect + noise."""
    cohort = e0006.EXPLOIT_SENSITIVE + e0006.REFERENCE
    left, right = {}, {}
    for m in range(per * len(cohort)):
        opponent = cohort[m % len(cohort)]
        noise = (m % 3 - 1) * 2.0
        left[m] = {"opponent": opponent, "bb_per_100": 10.0}
        right[m] = {
            "opponent": opponent,
            "bb_per_100": 10.0 + effects[opponent] + noise,
        }
    return left, right


def test_contrasts_apply_the_declared_family():
    effects = dict.fromkeys(e0006.EXPLOIT_SENSITIVE + e0006.REFERENCE, 0.0)
    effects["local/script/wide-opener"] = 60.0
    left, right = synthetic(effects)
    result = e0006.contrasts(left, right, range(20))
    primary = result["primary"]
    assert primary["mean"] == pytest.approx(
        sum(
            sum(
                right[m]["bb_per_100"] - 10.0
                for m in range(20)
                if left[m]["opponent"] == o
            )
            / 4
            for o in e0006.EXPLOIT_SENSITIVE
        )
        / 3
    )
    assert primary["ci95"] == pytest.approx(
        [primary["mean"] - Z95 * primary["se"], primary["mean"] + Z95 * primary["se"]]
    )
    family = result["secondary"]
    assert family["family_size"] == 6 and family["method"] == "bonferroni"
    z = NormalDist().inv_cdf(1 - 0.05 / 12)
    for member in family["members"].values():
        low, high = member["corrected"]
        assert high - low == pytest.approx(2 * z * member["se"])
        assert high - low > member["ci95"][1] - member["ci95"][0]
    wide = family["members"]["opponent:local/script/wide-opener"]
    assert wide["classification"] == "IMPROVEMENT_DETECTED"
    reference = family["members"]["reference_equal_weight"]
    assert set(reference["strata"]) == set(e0006.REFERENCE)


def paired(values: dict[str, list[float]]):
    """Round-robin per-match values whose differences are exactly ``values``."""
    left, right, m = {}, {}, 0
    for i in range(max(len(v) for v in values.values())):
        for opponent, diffs in values.items():
            if i < len(diffs):
                left[m] = {"opponent": opponent, "bb_per_100": 50.0}
                right[m] = {"opponent": opponent, "bb_per_100": 50.0 + diffs[i]}
                m += 1
    return left, right, range(m)


COHORT = e0006.EXPLOIT_SENSITIVE + e0006.REFERENCE
VARIED = [-40.0, 10.0, 35.0, -5.0]


def assert_degenerate(result, strata):
    assert result["classification"] == "INCONCLUSIVE"
    assert result["degenerate_strata"] == sorted(strata)
    assert result["ci95"] is None
    assert result["explanation"] == e0006.PLAN["degenerate_variance"]
    assert result["se"] >= 0 and result["mean"] is not None


@pytest.mark.parametrize("value", [0.0, 25.0, -25.0])
def test_identical_differences_are_inconclusive_not_a_claim(value):
    left, right, matches = paired(dict.fromkeys(COHORT, [value] * 4))
    result = e0006.contrasts(left, right, matches)
    primary = result["primary"]
    assert primary["mean"] == pytest.approx(value) and primary["se"] == 0.0
    assert_degenerate(primary, e0006.EXPLOIT_SENSITIVE)
    for member in result["secondary"]["members"].values():
        assert member["corrected"] is None and member["se"] == 0.0
        assert_degenerate(member, member["strata"])
    halves = e0006.stability(left, right, len(matches))
    assert halves["first_half"]["ci95"] is None
    assert halves["first_half"]["degenerate_strata"]


def test_one_zero_variance_stratum_makes_its_contrasts_inconclusive():
    values = dict.fromkeys(COHORT, VARIED)
    values["local/script/tight-opener"] = [12.0] * 4
    result = e0006.contrasts(*paired(values))
    primary = result["primary"]
    assert primary["se"] > 0  # the other strata still carry variance
    assert_degenerate(primary, ["local/script/tight-opener"])
    members = result["secondary"]["members"]
    assert_degenerate(
        members["opponent:local/script/tight-opener"], ["local/script/tight-opener"]
    )
    for label in ("opponent:local/script/wide-opener", "reference_equal_weight"):
        assert members[label]["degenerate_strata"] == []
        assert members[label]["corrected"] is not None
        assert "explanation" not in members[label]


def test_nondegenerate_results_are_unchanged():
    values = {o: [v + i for v in VARIED] for i, o in enumerate(COHORT)}
    result = e0006.contrasts(*paired(values))
    primary = result["primary"]
    expected = e0006.stratified(values, e0006.PLAN["primary"]["weights"])
    assert primary["mean"] == expected["mean"] and primary["se"] == expected["se"]
    low, high = primary["ci95"]
    assert [low, high] == [
        expected["mean"] - Z95 * expected["se"],
        expected["mean"] + Z95 * expected["se"],
    ]
    assert primary["classification"] == e0006.classify(low, high)
    assert primary["degenerate_strata"] == [] and "explanation" not in primary
    z = e0006.bonferroni_z(6)
    for member in result["secondary"]["members"].values():
        low, high = member["corrected"]
        assert high - low == pytest.approx(2 * z * member["se"])
        assert member["classification"] == e0006.classify(low, high)


def test_degenerate_rule_is_part_of_the_frozen_plan():
    assert "zero sample variance" in e0006.PLAN["degenerate_variance"]
    assert any("zero sample variance" in c for c in e0006.PLAN["classification"])
    assert e0006.PLAN_HASH in DESIGN.read_text()


def test_stability_halves_are_descriptive():
    effects = dict.fromkeys(e0006.EXPLOIT_SENSITIVE + e0006.REFERENCE, 5.0)
    left, right = synthetic(effects)
    result = e0006.stability(left, right, 20)
    assert result["binding"] is False and result["same_sign"] is True


# --- tiny real evidence in a temporary root -----------------------------------------------


TINY = StoppingRule(
    min_matches=20,
    max_matches=20,
    min_hands=800,
    max_hands=800,
    hands_per_match=40,
    matches_per_shard=5,
    evaluation_interval_matches=5,
)


def tiny_programme(directory: Path) -> Path:
    (directory / "specs").mkdir(parents=True)
    refs = []
    for spec in pair():
        small = dataclasses.replace(spec, stopping=TINY)
        path = directory / "specs" / f"E0006-{small.arm}.json"
        path.write_bytes(dump(small))
        refs.append({"path": f"specs/{path.name}", "spec_hash": small.spec_hash})
    value = {
        "schema_version": 1,
        "resources": {"workers": 2, "min_free_mib": 0},
        "experiments": [
            {
                "id": "E0006",
                "governance": "LOCAL_APPROVED",
                "reason": "tiny correctness fixture",
                "specs": refs,
                "depends_on": [],
            }
        ],
    }
    path = directory / "programme.json"
    path.write_text(json.dumps(value))
    return path


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    base = tmp_path_factory.mktemp("e0006")
    path = tiny_programme(base / "programme")
    root = base / "runs"
    specs = list(programme.load(path).item("E0006").specs)
    outcome = scheduler.run(
        specs, root, max_workers=2, min_free_mib=0, log=lambda _: None
    )
    assert outcome["completed"] == 8 and not outcome["failed"]
    return path, root


def clone(evidence, tmp_path):
    path, root = evidence
    copy = tmp_path / "runs"
    shutil.copytree(root, copy)
    db = sqlite3.connect(copy / "index.sqlite")
    db.execute(
        "UPDATE runs SET run_dir = replace(run_dir, ?, ?)", (str(root), str(copy))
    )
    db.commit()
    db.close()
    return path, copy


def test_tiny_evidence_analyzes_deterministically_and_read_only(evidence, tmp_path):
    path, root = clone(evidence, tmp_path)
    index = (root / "index.sqlite").read_bytes()
    first = e0006.analyze(root, path)
    second = e0006.analyze(root, path)
    assert canonical(first) == canonical(second)
    assert first["plan_hash"] == e0006.PLAN_HASH
    assert first["evidence"]["paired_matches"] == 20
    assert first["primary"]["classification"] in {
        "IMPROVEMENT_DETECTED",
        "REGRESSION_DETECTED",
        "NO_DIFFERENCE_DETECTED",
        "INCONCLUSIVE",
    }
    assert set(first["diagnostics"]) == set(e0006.EXPLOIT_SENSITIVE + e0006.REFERENCE)
    wide = first["diagnostics"]["local/script/wide-opener"]["arms"]
    assert wide["safe-exploit-v1"]["safe_exploit"]["activated"] > 0
    assert wide["safe-exploit-v1"]["tendencies"]["btn_open"]["opportunities"] == 80
    assert (root / "index.sqlite").read_bytes() == index
    out = e0006.write(root, first)
    data = out.read_bytes()
    e0006.write(root, second)
    assert out.read_bytes() == data == canonical(first) + b"\n"


def outputs(root):
    out = root / "analysis"
    return out / "E0006.stratified.json", out / "E0006.stratified.md"


def snapshot(paths):
    return {p: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ino) for p in paths}


def test_report_is_written_once_and_identical_rewrites_change_nothing(
    evidence, tmp_path
):
    path, root = clone(evidence, tmp_path)
    index = (root / "index.sqlite").read_bytes()
    report = e0006.analyze(root, path)
    assert e0006.write(root, report) == outputs(root)[0]
    for file in outputs(root):
        assert not file.stat().st_mode & 0o222  # published read-only
    before = snapshot(outputs(root))
    assert e0006.write(root, e0006.analyze(root, path)) == outputs(root)[0]
    assert e0006.main(["--root", str(root), "--programme", str(path)]) == 0
    assert snapshot(outputs(root)) == before
    assert (root / "index.sqlite").read_bytes() == index
    assert sorted(p.name for p in (root / "analysis").iterdir()) == [
        "E0006.stratified.json",
        "E0006.stratified.md",
    ]


def test_a_differing_report_is_never_replaced(evidence, tmp_path, capsys):
    path, root = clone(evidence, tmp_path)
    report = e0006.analyze(root, path)
    e0006.write(root, report)
    before = snapshot(outputs(root))
    changed = {**report, "note": "a different result"}
    with pytest.raises(e0006.ReportConflict, match="differs"):
        e0006.write(root, changed)
    assert snapshot(outputs(root)) == before
    # A tampered published file is a conflict too, through the CLI as well.
    json_path = outputs(root)[0]
    json_path.chmod(0o644)
    json_path.write_bytes(json_path.read_bytes().replace(b"Research", b"research"))
    tampered = snapshot(outputs(root))
    assert e0006.main(["--root", str(root), "--programme", str(path)]) == 1
    assert "ReportConflict" in capsys.readouterr().err
    assert snapshot(outputs(root)) == tampered


@pytest.mark.parametrize("missing", [0, 1])
def test_a_partial_report_pair_fails_closed(evidence, tmp_path, missing):
    path, root = clone(evidence, tmp_path)
    report = e0006.analyze(root, path)
    e0006.write(root, report)
    outputs(root)[missing].unlink()
    survivor = outputs(root)[1 - missing]
    before = snapshot([survivor])
    with pytest.raises(e0006.ReportConflict, match="partial"):
        e0006.write(root, report)
    assert snapshot([survivor]) == before
    assert not outputs(root)[missing].exists()


def test_missing_match_block_fails(evidence, tmp_path):
    path, root = clone(evidence, tmp_path)
    db = sqlite3.connect(root / "index.sqlite")
    db.execute(
        "UPDATE runs SET status = 'FAILED' WHERE rowid = (SELECT MAX(rowid) FROM runs)"
    )
    db.commit()
    db.close()
    with pytest.raises(e0006.EvidenceError, match="shards differ"):
        e0006.analyze(root, path)


def test_duplicate_completed_shard_fails(evidence, tmp_path):
    path, root = clone(evidence, tmp_path)
    db = sqlite3.connect(root / "index.sqlite")
    db.execute(
        "INSERT INTO runs SELECT run_id || '-dup', spec_hash, shard_start, shard_end, "
        "worker_id, status, started_at, finished_at, matches, hands, raw_sha256, "
        "run_dir, error FROM runs WHERE rowid = 1"
    )
    db.commit()
    db.close()
    with pytest.raises(e0006.EvidenceError, match="shards differ"):
        e0006.analyze(root, path)


def test_corrupted_raw_evidence_fails(evidence, tmp_path):
    path, root = clone(evidence, tmp_path)
    raw = min((root / "runs").glob("*/raw/events.jsonl.gz"))
    raw.chmod(stat.S_IRUSR | stat.S_IWUSR)
    raw.write_bytes(raw.read_bytes()[:-8] + b"corrupt!")
    raw.chmod(stat.S_IRUSR)
    with pytest.raises(RuntimeError):
        e0006.analyze(root, path)


def test_corrupted_derived_evidence_fails(evidence, tmp_path):
    path, root = clone(evidence, tmp_path)
    hands = min((root / "runs").glob("*/normalized/hands.jsonl"))
    hands.chmod(stat.S_IRUSR | stat.S_IWUSR)
    hands.write_text(hands.read_text().replace('"soh_net":', '"soh_net":1', 1))
    with pytest.raises(RuntimeError, match="derived"):
        e0006.analyze(root, path)


def test_foreign_evidence_in_the_root_fails(evidence, tmp_path):
    path, root = clone(evidence, tmp_path)
    other = load_spec(ROOT / "experiments" / "specs" / "E0002-canonical.json")
    Index(root / "index.sqlite").register_spec(other)
    with pytest.raises(e0006.EvidenceError, match="outside E0006"):
        e0006.analyze(root, path)


def _first_match(root):
    run = min((root / "runs").glob("E0006.canonical.*s000000*"))
    raw = [e for e in read_raw(run) if e.get("match") == 0]
    rows = {
        name: [
            json.loads(line)
            for line in (run / "normalized" / f"{name}.jsonl").read_text().splitlines()
            if json.loads(line)["match"] == 0
        ]
        for name in ("hands", "decisions")
    }
    return raw, rows


def test_mismatched_opponent_assignment_fails(evidence):
    path, root = evidence
    spec = next(
        s for s in programme.load(path).item("E0006").specs if s.arm == "canonical"
    )
    raw, rows = _first_match(root)
    e0006._match(spec, 0, raw, rows["hands"], rows["decisions"])
    for event in raw:
        if event.get("kind") == "match_meta":
            event["opponent"] = "local/script/tag_simple"
    with pytest.raises(e0006.EvidenceError, match="opponent"):
        e0006._match(spec, 0, raw, rows["hands"], rows["decisions"])


@pytest.mark.parametrize(
    ("field", "message"),
    [("opponent", "not paired"), ("seeds", "not paired"), ("deals", "deals differ")],
)
def test_unpaired_arms_fail(evidence, monkeypatch, field, message):
    path, root = evidence
    real = e0006.load_arm

    def tampered(root_, snapshot, spec):
        matches, runs = real(root_, snapshot, spec)
        if spec.arm == "safe-exploit-v1":
            matches[3] = {**matches[3], field: "tampered"}
        return matches, runs

    monkeypatch.setattr(e0006, "load_arm", tampered)
    with pytest.raises(e0006.EvidenceError, match=message):
        e0006.analyze(root, path)


def test_pre_intervention_randomness_is_identical_across_arms(evidence):
    _, root = evidence
    arms = {}
    for run in sorted((root / "runs").iterdir()):
        arm = run.name.split(".")[1]
        for event in read_raw(run):
            if event.get("kind") in {"deal", "match_meta"}:
                event.pop("seq")  # journal envelope, diverges with play
                arms.setdefault(arm, []).append(event)
    assert arms["canonical"] and arms["canonical"] == arms["safe-exploit-v1"]


# --- replay fidelity ---------------------------------------------------------------------


@pytest.mark.parametrize("index", [0, 2])  # wide-opener, threebet-heavy
def test_replay_reproduces_the_live_controller_exactly(index, monkeypatch):
    _, treatment = pair()
    spec = dataclasses.replace(
        treatment,
        stopping=StoppingRule(
            min_matches=5,
            max_matches=5,
            min_hands=600,
            max_hands=600,
            hands_per_match=120,
            matches_per_shard=5,
            evaluation_interval_matches=5,
        ),
    )
    live = []
    original = HoldemAgent.exploit_adjustment

    def record(self, context):
        result = original(self, context)
        if result is not None:
            live.append(result.to_dict())
        return result

    monkeypatch.setattr(HoldemAgent, "exploit_adjustment", record)
    events = []
    context = MatchContext(
        spec, index, spec.opponent_for(index), runner.match_seeds(spec, index)
    )
    LocalPlatform().play_match(context, lambda e: events.append({"match": index, **e}))
    replayed, _ = e0006.replay(events, 0)
    assert [{k: v for k, v in r.items() if k != "hand"} for r in replayed] == live
    assert any(r["reason"] == "adjusted" for r in live)
