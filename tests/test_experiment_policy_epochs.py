"""Policy epochs: historical evidence stays valid; execution needs current policy.

History is built under the real runtime fingerprint (OLD). A future policy is
simulated by making the supervisor report a different valid fingerprint (NEW)
while every replay, planning and execution path runs unmodified. Spawned
workers still compute the real fingerprint, so the worker firewall is real.
"""

import gc
import json
import socket
from dataclasses import replace
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
    screening,
    supervisor,
)
from sleight_of_hand.experiments.campaign import ReviewRequired, StalePolicy
from sleight_of_hand.experiments.platforms.chipzen import ChipzenExternalPlatform
from sleight_of_hand.experiments.spec import dump
from sleight_of_hand.experiments.storage import Index
from tests.test_experiment_chipzen import ENV, chipzen_spec
from tests.test_experiment_optimizer import fixture as pattern_fixture
from tests.test_experiment_screening import (
    PAIRED,
    force_qualifier,
    legacy_history,
    screening_config,
    tree,
)
from tests.test_experiment_screening import fixture as screening_fixture
from tests.test_experiment_supervisor import small, write_programme

OLD = runner.policy_revision()
NEW = "sha256:" + "1" * 64
STALE = "sha256:" + "0" * 64  # a revision the real runtime never had
RAW_DEFECTS = ("raw_flipped", "raw_truncated")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key in ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(scheduler, "available_mib", lambda: 4096)
    monkeypatch.setattr(
        ChipzenExternalPlatform,
        "play_match",
        Mock(side_effect=AssertionError("no live matches in epoch tests")),
    )

    def no_network(*args, **kwargs):
        raise AssertionError("no network use in epoch tests")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)


def durable(root):
    """Every durable file; SQLite's reader coordination files are excluded.

    A read-only WAL reader may create an empty ``-wal`` and a ``-shm`` index;
    they never hold index data, so the ``-wal`` must stay empty.
    """
    files = tree(root)
    for name in [n for n in files if n.endswith(("-wal", "-shm"))]:
        content, _ = files.pop(name)
        assert not name.endswith("-wal") or content == b""
    return files


def index_rows(root):
    index = Index(root / "index.sqlite")
    rows = (index.runs(), [s.spec_hash for s in index.specs()])
    gc.collect()
    return rows


def under_new_policy(monkeypatch):
    """The repository now reports NEW; nothing pinned to OLD may run."""
    real = runner.source_state

    def advanced():
        state = real()
        state["policy_revision"] = NEW
        return state

    monkeypatch.setattr(supervisor, "source_state", advanced)
    monkeypatch.setattr(
        scheduler, "run", Mock(side_effect=AssertionError("stale work executed"))
    )
    monkeypatch.setattr(
        supervisor, "_analyze", Mock(side_effect=AssertionError("re-analysis"))
    )
    return advanced()


def fixed_programme(directory, groups):
    return programme.load(write_programme(directory, groups))


def with_campaigns(path, *campaign_paths):
    value = json.loads(path.read_text())
    value.update(
        schema_version=2,
        campaigns=[
            {"path": c.name, "campaign_hash": campaign.load(c).digest}
            for c in campaign_paths
        ],
    )
    path.write_text(json.dumps(value))
    return programme.load(path)


def campaign_row(result, identifier):
    return next(r for r in result["campaigns"] if r["campaign"] == identifier)


# --- fixed experiments -------------------------------------------------------


def test_completed_current_fixed_experiment_is_current_complete(tmp_path):
    p = fixed_programme(tmp_path / "authority", [[small()]])
    result = supervisor.execute(p, tmp_path / "runs")
    row = result["items"][0]
    assert row["status"] == "COMPLETE" and row["policy_state"] == "CURRENT"
    assert row["recorded_policy_revision"] == row["current_policy_revision"] == OLD
    assert row["policy_note"] is None
    assert "policy:" not in cli.render(result)


def test_completed_old_fixed_experiment_is_historical_and_never_reruns(
    tmp_path, monkeypatch
):
    p = fixed_programme(tmp_path / "authority", [[small()]])
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    evidence = tree(root / "runs")
    reports = tree(root / "analysis")
    runs = Index(root / "index.sqlite").runs()
    under_new_policy(monkeypatch)
    row = supervisor.plan(p, root)["items"][0]
    assert row["status"] == "COMPLETE" and row["intended_action"] == "NONE"
    assert row["policy_state"] == "HISTORICAL_COMPLETE"
    assert row["recorded_policy_revision"] == OLD
    assert row["current_policy_revision"] == NEW
    assert "remains valid for recorded policy" in row["reason"]
    assert "no further execution is authorized" in row["reason"]
    result = supervisor.execute(p, root)
    assert result["actions"] == [] and result["disposition"] == "COMPLETE"
    text = cli.render(result)
    assert f"policy: HISTORICAL_COMPLETE — recorded {OLD}; current {NEW}" in text
    assert tree(root / "runs") == evidence and tree(root / "analysis") == reports
    assert Index(root / "index.sqlite").runs() == runs


def test_old_fixed_experiment_with_pending_shards_requires_review(
    tmp_path, monkeypatch
):
    spec = small()
    p = fixed_programme(tmp_path / "authority", [[spec]])
    root = tmp_path / "runs"
    runner.run_shard(spec.to_dict(), spec.stopping.shards()[0], "fixture", str(root))
    under_new_policy(monkeypatch)
    row = supervisor.plan(p, root)["items"][0]
    assert row["status"] == "REQUIRES_REVIEW"
    assert row["policy_state"] == "STALE_REQUIRES_REVIEW"
    assert row["pending_shards"] == 1 and "execution remains" in row["reason"]
    result = supervisor.execute(p, root)
    assert result["actions"] == [] and result["disposition"] == "REQUIRES_REVIEW"
    assert len(Index(root / "index.sqlite").runs()) == 1


def test_old_complete_evidence_without_analysis_requires_review(tmp_path, monkeypatch):
    spec = small()
    p = fixed_programme(tmp_path / "authority", [[spec]])
    root = tmp_path / "runs"
    scheduler.run([spec], root, min_free_mib=0, log=lambda _: None)
    under_new_policy(monkeypatch)
    row = supervisor.plan(p, root)["items"][0]
    assert row["pending_shards"] == 0
    assert row["status"] == "REQUIRES_REVIEW" and "analysis remains" in row["reason"]
    assert row["policy_state"] == "STALE_REQUIRES_REVIEW"
    result = supervisor.execute(p, root)
    assert result["actions"] == []  # no analysis is generated under NEW
    assert not (root / "analysis").exists()


def corrupt_fixed(root, defect):
    run = Index(root / "index.sqlite").runs()[0]
    directory = Path(run["run_dir"])
    if defect == "derived":
        (directory / "normalized/hands.jsonl").write_text("{}\n")
    elif defect in RAW_DEFECTS:
        raw = directory / journal.RAW_GZ
        data = bytearray(raw.read_bytes())
        if defect == "raw_truncated":
            data = data[:-5]  # EOFError while decompressing
        else:
            data[len(data) // 2] ^= 0xFF  # corrupt the compressed stream
        raw.chmod(0o600)
        raw.write_bytes(bytes(data))
        raw.chmod(0o400)
    elif defect == "manifest":
        manifest = directory / "manifest.json"
        value = json.loads(manifest.read_text())
        value["checksums"]["raw_sha256"] = "0" * 64
        manifest.write_text(json.dumps(value))
    elif defect == "report":
        report = root / "analysis/E9999.json"
        report.write_bytes(report.read_bytes() + b" ")
    else:
        receipt = root / "analysis/E9999.supervisor.json"
        value = json.loads(receipt.read_text())
        value["evidence_key"] = "0" * 64
        receipt.write_text(json.dumps(value))


@pytest.mark.parametrize(
    "defect", ["derived", *RAW_DEFECTS, "manifest", "report", "receipt"]
)
def test_plan_never_claims_historical_for_corrupt_fixed_evidence(
    tmp_path, monkeypatch, defect
):
    p = fixed_programme(tmp_path / "authority", [[small()]])
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    corrupt_fixed(root, defect)
    before, rows = durable(root), index_rows(root)
    under_new_policy(monkeypatch)
    row = supervisor.plan(p, root)["items"][0]  # plan/status, not execute
    assert row["status"] == "FAILED" and row["policy_state"] == "UNVERIFIED", row
    assert row["intended_action"] == "INVESTIGATE"
    assert durable(root) == before and index_rows(root) == rows


def test_old_fixed_experiment_with_corrupted_evidence_still_fails(
    tmp_path, monkeypatch
):
    p = fixed_programme(tmp_path / "authority", [[small()]])
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    run = Index(root / "index.sqlite").runs()[0]
    (Path(run["run_dir"]) / "normalized/hands.jsonl").write_text("{}\n")
    under_new_policy(monkeypatch)
    result = supervisor.execute(p, root)
    assert result["disposition"] == "FAILED" and result["failures"]
    assert result["items"][0]["status"] == "FAILED"


def test_explicit_analyze_of_historical_item_requires_review(tmp_path, monkeypatch):
    p = fixed_programme(tmp_path / "authority", [[small()]])
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    reports = tree(root / "analysis")
    under_new_policy(monkeypatch)
    result = supervisor.execute(p, root, "analyze", "E9999")
    assert result["disposition"] == "REQUIRES_REVIEW" and result["governance_stops"]
    assert "reviewed protocol" in result["governance_stops"][0]["reason"]
    assert tree(root / "analysis") == reports


def test_direct_execution_of_stale_spec_still_refuses(tmp_path):
    stale = replace(small(), policy_revision=STALE)
    with pytest.raises(RuntimeError, match="refusing to run"):
        runner.run_shard(stale.to_dict(), (0, 1), "fixture", str(tmp_path))
    outcome = scheduler.run(
        [stale], tmp_path, max_workers=1, min_free_mib=0, log=lambda _: None
    )
    assert outcome["completed"] == 0
    assert len(outcome["failed"]) == len(stale.stopping.shards())
    assert all("refusing to run" in f["error"] for f in outcome["failed"])
    assert not [
        r for r in Index(tmp_path / "index.sqlite").runs() if r["status"] == "COMPLETED"
    ]


# --- campaigns ---------------------------------------------------------------


def assert_historical(row, *, classification, incumbent, used, reason):
    assert row["status"] == "COMPLETE" and row["intended_action"] == "NONE"
    assert row["policy_state"] == "HISTORICAL_COMPLETE"
    assert row["recorded_policy_revision"] == OLD
    assert row["current_policy_revision"] == NEW
    assert row["classification"] == classification
    assert row["incumbent"] == incumbent
    assert row["candidate_budget_used"] == used
    assert row["reason"] == reason


def test_legacy_pattern_campaign_replays_as_historical_complete(tmp_path, monkeypatch):
    c, p, _, root = legacy_history(tmp_path)
    before = supervisor.plan(p, root)["campaigns"][0]
    files = tree(optimizer.directory(root, c))
    under_new_policy(monkeypatch)
    row = supervisor.plan(p, root)["campaigns"][0]
    assert_historical(
        row,
        classification=before["classification"],
        incumbent=before["incumbent"],
        used=before["candidate_budget_used"],
        reason=before["reason"],
    )
    assert row["generation"] == before["generation"]
    assert row["search_state"] == before["search_state"]
    result = supervisor.execute(p, root)
    assert result["actions"] == [] and result["disposition"] == "COMPLETE"
    assert tree(optimizer.directory(root, c)) == files


def test_authority_screening_campaign_replays_as_historical_complete(
    tmp_path, monkeypatch
):
    c, p, _ = screening_fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    before = supervisor.plan(p, root)["campaigns"][0]
    _, events = optimizer.reconstruct(c, p, root, runner.source_state(), verify=True)
    files = tree(optimizer.directory(root, c))
    reports = tree(root / "analysis")
    new = under_new_policy(monkeypatch)
    # A future default change must not make historical history unloadable.
    monkeypatch.setattr(
        screening, "_canonical_defaults", lambda: {**c.initial, "steepness": 9.0}
    )
    row = supervisor.plan(programme.load(tmp_path / "authority/programme.json"), root)[
        "campaigns"
    ][0]
    assert_historical(
        row,
        classification="NO_IMPROVEMENT_DETECTED",
        incumbent=c.initial,
        used=17,
        reason=screening.NO_QUALIFIER,
    )
    assert row["reason"] == before["reason"]
    state, replayed = optimizer.reconstruct(c, p, root, new, verify=True)
    assert replayed == events and state["classification"] == "NO_IMPROVEMENT_DETECTED"
    result = supervisor.execute(p, root)
    assert result["actions"] == []
    assert tree(optimizer.directory(root, c)) == files
    assert tree(root / "analysis") == reports


def test_incomplete_old_pattern_campaign_requires_review(tmp_path, monkeypatch):
    c, p, _ = pattern_fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    supervisor.execute(p, root)  # one of two generations
    files = tree(optimizer.directory(root, c))
    under_new_policy(monkeypatch)
    row = supervisor.plan(p, root)["campaigns"][0]
    assert row["status"] == "REQUIRES_REVIEW"
    assert row["policy_state"] == "STALE_REQUIRES_REVIEW"
    assert row["generation"] == 1 and row["candidate_budget_used"] == 2
    assert "no further execution is authorized" in row["reason"]
    result = supervisor.execute(p, root)
    assert result["actions"] == [] and result["disposition"] == "REQUIRES_REVIEW"
    assert tree(optimizer.directory(root, c)) == files


def test_incomplete_old_screening_batch_requires_review(tmp_path, monkeypatch):
    c, p, _ = screening_fixture(tmp_path / "authority")
    root = tmp_path / "runs"

    def interrupt(specs, root, *args, **kwargs):
        runner.run_shard(
            specs[1].to_dict(), specs[1].stopping.shards()[0], "fixture", str(root)
        )
        raise KeyboardInterrupt

    original = scheduler.run
    monkeypatch.setattr(scheduler, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        supervisor.execute(p, root)
    monkeypatch.setattr(scheduler, "run", original)
    files = tree(optimizer.directory(root, c))
    runs = Index(root / "index.sqlite").runs()
    under_new_policy(monkeypatch)
    row = supervisor.plan(p, root)["campaigns"][0]
    assert row["status"] == "REQUIRES_REVIEW"
    assert row["policy_state"] == "STALE_REQUIRES_REVIEW"
    assert row["candidate_budget_used"] == 17
    assert supervisor.execute(p, root)["actions"] == []
    assert tree(optimizer.directory(root, c)) == files
    assert Index(root / "index.sqlite").runs() == runs


def awaiting_confirmation(tmp_path, monkeypatch):
    c, p, _ = screening_fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    force_qualifier(monkeypatch, screening.design(c)[5]["arm"])
    assert supervisor.execute(p, root)["campaigns"][0]["status"] == "CONFIRMING"
    monkeypatch.setattr(analysis, "paired", PAIRED)
    return c, p, root


def test_old_campaign_awaiting_confirmation_requires_review(tmp_path, monkeypatch):
    c, p, root = awaiting_confirmation(tmp_path, monkeypatch)
    files = tree(optimizer.directory(root, c))
    under_new_policy(monkeypatch)
    row = supervisor.plan(p, root)["campaigns"][0]
    assert row["status"] == "REQUIRES_REVIEW"
    assert row["policy_state"] == "STALE_REQUIRES_REVIEW"
    assert row["incumbent"] == screening.design(c)[5]["vector"]
    assert supervisor.execute(p, root)["actions"] == []
    assert tree(optimizer.directory(root, c)) == files
    assert not any(
        "confirmation" in s.experiment_id for s in Index(root / "index.sqlite").specs()
    )


def test_advance_refuses_old_campaign_even_when_called_directly(tmp_path, monkeypatch):
    c, p, root = awaiting_confirmation(tmp_path, monkeypatch)
    files = tree(optimizer.directory(root, c))
    new = under_new_policy(monkeypatch)
    with pytest.raises(StalePolicy):
        optimizer.advance(c, p, root, new)
    # Even if replay were (wrongly) treated as permission, advance refuses.
    real = optimizer.reconstruct
    monkeypatch.setattr(
        optimizer,
        "reconstruct",
        lambda *a, **k: real(a[0], a[1], a[2], runner.source_state(), **k),
    )
    with pytest.raises(ReviewRequired, match="pinned to policy"):
        optimizer.advance(c, p, root, new)
    assert tree(optimizer.directory(root, c)) == files


def test_explicit_optimize_of_stale_campaign_requires_review(tmp_path, monkeypatch):
    c, p, root = awaiting_confirmation(tmp_path, monkeypatch)
    under_new_policy(monkeypatch)
    result = supervisor.execute(p, root, "optimize", c.identifier)
    assert result["disposition"] == "REQUIRES_REVIEW" and result["governance_stops"]
    assert result["actions"] == []


# --- dependencies across epochs ----------------------------------------------


def new_policy_screen(directory, depends_on):
    """A reviewed campaign pinned to NEW, as a future campaign would be."""
    template = campaign.load(runner.ROOT / "experiments/O0002.json").template
    future = replace(template, policy_revision=NEW)
    (directory / "template-new.json").write_bytes(dump(future))

    def change(v):
        v.update(policy_revision=NEW, depends_on=depends_on)
        v["template"] = {"path": "template-new.json", "spec_hash": future.spec_hash}

    return screening_config(directory, change=change, identifier="O9997")


def test_historical_campaign_satisfies_later_dependency(tmp_path, monkeypatch):
    first, _, path = pattern_fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    later = new_policy_screen(
        path.parent,
        [
            {
                "id": "O9999",
                "status": "COMPLETE",
                "classification": "NO_IMPROVEMENT_DETECTED",
            }
        ],
    )
    value = json.loads(path.read_text())
    value["campaigns"].append(
        {"path": later.name, "campaign_hash": campaign.load(later).digest}
    )
    path.write_text(json.dumps(value))
    p = programme.load(path)
    supervisor.execute(p, root)
    supervisor.execute(p, root)  # O9999 completes under OLD; O9997 is stale
    under_new_policy(monkeypatch)
    plan = supervisor.plan(p, root)
    old, new = campaign_row(plan, "O9999"), campaign_row(plan, "O9997")
    assert old["status"] == "COMPLETE" and old["policy_state"] == "HISTORICAL_COMPLETE"
    assert new["policy_state"] == "CURRENT" and new["dependencies"][0]["met"]
    assert new["status"] == "PENDING" and new["intended_action"] == "GENERATE_AND_RUN"
    rows = plan["items"] + plan["campaigns"]
    assert supervisor.disposition([], [], rows) == "PENDING"
    # The dependency still pins O9999's authority: altering it invalidates O9997.
    changed = replace(
        p,
        campaigns=(
            replace(
                first,
                config_json=first.config_json.replace('"purpose":"', '"purpose":"x'),
            ),
            p.campaigns[1],
        ),
    )
    assert changed.authority("O9997") != p.authority("O9997")


def corrupt_campaign(c, p, path, root, defect):
    directory = optimizer.directory(root, c)
    if defect == "authority":
        config = json.loads((path.parent / "O9998.json").read_text())
        config["purpose"] = "changed after completion"
        (path.parent / "O9998.json").write_text(json.dumps(config))
        value = json.loads(path.read_text())
        value["campaigns"][0]["campaign_hash"] = campaign.load(
            path.parent / "O9998.json"
        ).digest
        path.write_text(json.dumps(value))
        return programme.load(path)
    if defect in {"derived", *RAW_DEFECTS}:
        corrupt_fixed(root, defect)
    elif defect == "spec":
        file = sorted((directory / "specs").glob("*.json"))[2]
        file.chmod(0o600)
        file.write_bytes(file.read_bytes() + b" ")
        file.chmod(0o400)
    elif defect == "report":
        report = root / "analysis/O9998.screening.00.json"
        report.write_bytes(
            report.read_bytes().replace(b'"label":"AB"', b'"label":"BA"')
        )
    else:
        receipt = root / "analysis/O9998.screening.00.supervisor.json"
        receipt.write_bytes(
            receipt.read_bytes().replace(b'"binding":false', b'"binding":true')
        )
    return p


@pytest.mark.parametrize(
    "defect,expected",
    [
        ("derived", "FAILED"),
        ("raw_flipped", "FAILED"),
        ("raw_truncated", "FAILED"),
        ("spec", "FAILED"),
        ("report", "FAILED"),
        ("receipt", "FAILED"),
        ("authority", "REQUIRES_REVIEW"),
    ],
)
def test_plan_never_claims_historical_for_corrupt_campaign(
    tmp_path, monkeypatch, defect, expected
):
    c, p, path = screening_fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    p = corrupt_campaign(c, p, path, root, defect)
    before, rows = durable(root), index_rows(root)
    under_new_policy(monkeypatch)
    row = supervisor.plan(p, root)["campaigns"][0]  # plan/status, not execute
    assert row["status"] == expected and row["policy_state"] == "UNVERIFIED", row
    assert durable(root) == before and index_rows(root) == rows


def test_current_plan_does_not_deep_verify_campaign_evidence(tmp_path, monkeypatch):
    # Deep verification is the price of a historical claim only.
    _, p, _ = screening_fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    verify = Mock(side_effect=AssertionError("deep verification in current plan"))
    monkeypatch.setattr(supervisor, "_verify_evidence", verify)
    row = supervisor.plan(p, root)["campaigns"][0]
    assert row["status"] == "COMPLETE" and row["policy_state"] == "CURRENT"
    assert not verify.called


def test_valid_history_is_historical_through_plan_and_status(
    tmp_path, monkeypatch, capsys
):
    fixed_path = write_programme(tmp_path / "authority", [[small()]])
    screen = screening_config(tmp_path / "authority")
    p = with_campaigns(fixed_path, screen)
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    before, rows = durable(root), index_rows(root)
    under_new_policy(monkeypatch)
    calls = []
    real = supervisor._verify_evidence

    def counted(root, item):
        calls.append(item.experiment_id)
        return real(root, item)

    monkeypatch.setattr(supervisor, "_verify_evidence", counted)
    plan = supervisor.plan(p, root)
    assert [r["policy_state"] for r in plan["items"] + plan["campaigns"]] == [
        "HISTORICAL_COMPLETE",
        "HISTORICAL_COMPLETE",
    ]
    assert {"E9999", "O9998.screening.00"} <= set(calls)  # deep-verified claims
    for command in ("status", "plan"):
        arguments = [command, "--programme", str(fixed_path), "--root", str(root)]
        assert cli.main([*arguments, "--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        printed = result["items"] + result["campaigns"]
        assert {r["policy_state"] for r in printed} == {"HISTORICAL_COMPLETE"}
    assert durable(root) == before and index_rows(root) == rows


@pytest.mark.parametrize(
    "defect", ["authority", "evidence", "spec", "report", "receipt"]
)
def test_historical_corruption_still_fails_closed(tmp_path, monkeypatch, defect):
    c, p, path = screening_fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    directory = optimizer.directory(root, c)
    if defect == "authority":
        config = json.loads((path.parent / "O9998.json").read_text())
        config["purpose"] = "changed after completion"
        (path.parent / "O9998.json").write_text(json.dumps(config))
        value = json.loads(path.read_text())
        value["campaigns"][0]["campaign_hash"] = campaign.load(
            path.parent / "O9998.json"
        ).digest
        path.write_text(json.dumps(value))
        p = programme.load(path)
    elif defect == "evidence":
        run = Index(root / "index.sqlite").runs()[0]
        (Path(run["run_dir"]) / "normalized/hands.jsonl").write_text("{}\n")
    elif defect == "spec":
        file = sorted((directory / "specs").glob("*.json"))[2]
        file.chmod(0o600)
        file.write_bytes(file.read_bytes() + b" ")
        file.chmod(0o400)
    elif defect == "report":
        report = root / "analysis/O9998.screening.00.json"
        report.write_bytes(
            report.read_bytes().replace(b'"label":"AB"', b'"label":"BA"')
        )
    else:
        receipt = root / "analysis/O9998.screening.00.supervisor.json"
        receipt.write_bytes(
            receipt.read_bytes().replace(b'"binding":false', b'"binding":true')
        )
    under_new_policy(monkeypatch)
    result = supervisor.execute(p, root)
    row = result["campaigns"][0]
    assert row["policy_state"] == "UNVERIFIED", row
    assert row["status"] in {"FAILED", "REQUIRES_REVIEW"}
    expected = "REQUIRES_REVIEW" if defect == "authority" else "FAILED"
    assert row["status"] == expected and result["disposition"] == expected
    assert result["failures"] or result["governance_stops"]  # cycle-level, not skipped
    assert result["actions"] == []


# --- independent stale work and dispositions ----------------------------------


def stale_chipzen_and_current_screen(tmp_path, depends=False):
    online = replace(chipzen_spec(), experiment_id="E0001", policy_revision=STALE)
    path = write_programme(tmp_path / "authority", [[online]])
    deps = (
        [{"id": "E0001", "status": "COMPLETE", "classification": None}]
        if depends
        else []
    )
    screen = screening_config(
        tmp_path / "authority", change=lambda v: v.update(depends_on=deps)
    )
    return with_campaigns(path, screen)


def test_stale_e0001_style_item_does_not_mask_current_campaign(tmp_path, monkeypatch):
    p = stale_chipzen_and_current_screen(tmp_path)
    root = tmp_path / "runs"
    plan = supervisor.plan(p, root)
    item, row = plan["items"][0], plan["campaigns"][0]
    assert item["status"] == "REQUIRES_REVIEW"
    assert item["policy_state"] == "STALE_REQUIRES_REVIEW"
    assert row["status"] == "PENDING" and row["policy_state"] == "CURRENT"
    assert supervisor.disposition([], [], [item, row]) == "PENDING"
    c = p.campaigns[0]
    force_qualifier(monkeypatch, screening.design(c)[9]["arm"])
    first = supervisor.execute(p, root)
    assert first["items"][0]["status"] == "REQUIRES_REVIEW"
    assert campaign_row(first, c.identifier)["status"] == "CONFIRMING"
    assert first["disposition"] == "PENDING" and not first["governance_stops"]
    assert not any(a.get("experiment") == "E0001" for a in first["actions"])
    assert "Cycle: PENDING" in cli.render(first)
    monkeypatch.setattr(analysis, "paired", PAIRED)
    second = supervisor.execute(p, root)
    assert campaign_row(second, c.identifier)["status"] == "COMPLETE"
    # Only the stale item is left: overall review, never "blocked" or complete.
    assert second["disposition"] == "REQUIRES_REVIEW"
    assert "stale items require review" in cli.render(second)
    journal = Path(second["decision_journal"]).read_text()
    assert '"kind":"skipped"' in journal


def test_explicit_run_of_stale_item_requires_review(tmp_path, monkeypatch):
    p = stale_chipzen_and_current_screen(tmp_path)
    run = Mock(side_effect=AssertionError("stale item executed"))
    monkeypatch.setattr(scheduler, "run", run)
    result = supervisor.execute(p, tmp_path / "runs", "run", "E0001")
    assert result["disposition"] == "REQUIRES_REVIEW" and result["governance_stops"]
    assert result["actions"] == [] and not run.called


def test_current_campaign_depending_on_stale_item_stays_blocked(tmp_path, monkeypatch):
    p = stale_chipzen_and_current_screen(tmp_path, depends=True)
    monkeypatch.setattr(
        scheduler, "run", Mock(side_effect=AssertionError("blocked work executed"))
    )
    result = supervisor.execute(p, tmp_path / "runs")
    row = result["campaigns"][0]
    assert row["status"] == "BLOCKED" and not row["dependencies"][0]["met"]
    assert "found REQUIRES_REVIEW" in row["reason"]
    assert result["actions"] == []
    assert result["disposition"] == "REQUIRES_REVIEW"


def test_resource_stop_still_blocks_with_stale_item(tmp_path, monkeypatch):
    stale = replace(small("E9998"), policy_revision=STALE)
    path = write_programme(tmp_path / "authority", [[stale], [small()]])
    value = json.loads(path.read_text())
    value["resources"]["min_free_mib"] = 1024
    path.write_text(json.dumps(value))
    monkeypatch.setattr(scheduler, "available_mib", lambda: 0)
    run = Mock(side_effect=AssertionError("unsafe admission"))
    monkeypatch.setattr(scheduler, "run", run)
    result = supervisor.execute(programme.load(path), tmp_path / "runs")
    assert result["disposition"] == "BLOCKED" and result["governance_stops"]
    assert result["governance_stops"][0]["experiment"] == "E9999"
    assert not run.called


@pytest.mark.parametrize(
    "rows,stops,expected",
    [
        # historical complete only + one stale review + nothing eligible
        (
            ["COMPLETE:HISTORICAL_COMPLETE", "REQUIRES_REVIEW:STALE_REQUIRES_REVIEW"],
            [],
            "REQUIRES_REVIEW",
        ),
        # stale review + independent current pending work
        (["REQUIRES_REVIEW:STALE_REQUIRES_REVIEW", "PENDING:CURRENT"], [], "PENDING"),
        (
            ["REQUIRES_REVIEW:STALE_REQUIRES_REVIEW", "CONFIRMING:CURRENT"],
            [],
            "PENDING",
        ),
        # dependent of the stale item is blocked; no eligible work
        (
            ["REQUIRES_REVIEW:STALE_REQUIRES_REVIEW", "BLOCKED:CURRENT"],
            [],
            "REQUIRES_REVIEW",
        ),
        # a non-stale review is still global even beside pending work
        (["REQUIRES_REVIEW:CURRENT", "PENDING:CURRENT"], [], "REQUIRES_REVIEW"),
        (["REQUIRES_REVIEW:UNVERIFIED", "PENDING:CURRENT"], [], "REQUIRES_REVIEW"),
        # explicit selection of a stale item becomes a cycle-level stop
        (
            ["REQUIRES_REVIEW:STALE_REQUIRES_REVIEW"],
            ["REQUIRES_REVIEW"],
            "REQUIRES_REVIEW",
        ),
        # resource stop and failures keep their precedence
        (
            ["REQUIRES_REVIEW:STALE_REQUIRES_REVIEW", "PENDING:CURRENT"],
            ["BLOCKED"],
            "BLOCKED",
        ),
        (["REQUIRES_REVIEW:STALE_REQUIRES_REVIEW", "FAILED:UNVERIFIED"], [], "FAILED"),
        (["COMPLETE:HISTORICAL_COMPLETE", "COMPLETE:CURRENT"], [], "COMPLETE"),
    ],
)
def test_disposition_truth_table(rows, stops, expected):
    selected = [
        {"status": s, "policy_state": e} for s, e in (r.split(":") for r in rows)
    ]
    assert supervisor.disposition([], [{"status": s} for s in stops], selected) == (
        expected
    )


# --- current behaviour and read-only planning --------------------------------


def test_real_programme_is_current_under_unchanged_fingerprint(tmp_path):
    p = programme.load(runner.ROOT / "experiments/programme.json")
    result = supervisor.plan(p, tmp_path / "runs")
    assert {r["policy_state"] for r in result["items"] + result["campaigns"]} == {
        "CURRENT"
    }
    assert all(
        r["recorded_policy_revision"] == OLD
        for r in result["items"] + result["campaigns"]
    )
    assert "policy:" not in cli.render(result)
    assert not (tmp_path / "runs").exists()


def test_planning_under_new_policy_is_read_only(tmp_path, monkeypatch):
    _, p, _ = screening_fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    before = tree(root)
    under_new_policy(monkeypatch)
    for _ in range(2):
        supervisor.plan(p, root)
    assert tree(root) == before


@pytest.mark.parametrize("damage", ["flip", "truncate"])
def test_unreadable_raw_evidence_is_a_failed_check_not_a_crash(tmp_path, damage):
    p = fixed_programme(tmp_path / "authority", [[small()]])
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    corrupt_fixed(root, "raw_flipped" if damage == "flip" else "raw_truncated")
    run = Index(root / "index.sqlite").runs()[0]
    before = (Path(run["run_dir"]) / journal.RAW_GZ).read_bytes()
    assert journal.verify(Path(run["run_dir"]))["ok"] is False
    assert (Path(run["run_dir"]) / journal.RAW_GZ).read_bytes() == before
