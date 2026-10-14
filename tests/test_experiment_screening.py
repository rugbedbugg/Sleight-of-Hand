"""O0002 Resolution-V interaction screening, campaign dependencies and authority."""

import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from unittest.mock import Mock

import pytest

from sleight_of_hand.experiments import (
    analysis,
    campaign,
    cli,
    optimizer,
    programme,
    runner,
    scheduler,
    screening,
    supervisor,
)
from sleight_of_hand.experiments.campaign import ReviewRequired
from sleight_of_hand.experiments.model import canonical, sha256
from sleight_of_hand.experiments.platforms.chipzen import ChipzenExternalPlatform
from sleight_of_hand.experiments.spec import ExperimentSpec, SeedPolicy, dump
from sleight_of_hand.experiments.storage import Index
from sleight_of_hand.policy.heuristic import DEFAULT_PARAMS
from tests.test_experiment_chipzen import ENV, chipzen_spec
from tests.test_experiment_optimizer import fixture as pattern_fixture
from tests.test_experiment_supervisor import small, write_programme

CONFIG = runner.ROOT / "experiments/O0002.json"
PROGRAMME = runner.ROOT / "experiments/programme.json"
#: Programme digest recorded by the legacy O0001 v1 start event (916c0cb).
O0001_LEGACY_PROGRAMME = (
    "sha256:f7a0f305665bc930092bc5ead8ddcce3a3f35b4713310133b770e1f885d1178e"
)
TINY = {
    "min_matches": 2,
    "max_matches": 2,
    "min_hands": 2,
    "max_hands": 2,
    "hands_per_match": 1,
    "matches_per_shard": 2,
    "evaluation_interval_matches": 1,
    "early_stopping": "none",
}
TWO_FACTOR_ALIASES = {
    "AB": "CDE",
    "AC": "BDE",
    "AD": "BCE",
    "AE": "BCD",
    "BC": "ADE",
    "BD": "ACE",
    "BE": "ACD",
    "CD": "ABE",
    "CE": "ABD",
    "DE": "ABC",
}


PAIRED = analysis.paired


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key in ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(scheduler, "available_mib", lambda: 4096)
    monkeypatch.setattr(
        ChipzenExternalPlatform,
        "play_match",
        Mock(side_effect=AssertionError("no live matches in screening tests")),
    )

    def no_network(*args, **kwargs):
        raise AssertionError("no network use in screening tests")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)


def screening_config(tmp_path, *, tiny=True, change=None, identifier="O9998"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    if not (tmp_path / "template.json").exists():
        (tmp_path / "template.json").write_bytes(dump(campaign.load(CONFIG).template))
    config.update(id=identifier, depends_on=[])
    config["template"]["path"] = "template.json"
    config["resources"]["min_free_mib"] = 0
    if tiny:
        config["screening_stopping"] = dict(TINY)
        config["confirmation_stopping"] = dict(TINY)
    if change:
        change(config)
    path = tmp_path / f"{identifier}.json"
    path.write_text(json.dumps(config))
    return path


def fixture(tmp_path, *, tiny=True, change=None):
    path = screening_config(tmp_path, tiny=tiny, change=change)
    c = campaign.load(path)
    programme_path = tmp_path / "programme.json"
    programme_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "resources": {"workers": 2, "min_free_mib": 0},
                "experiments": [],
                "campaigns": [{"path": path.name, "campaign_hash": c.digest}],
            }
        )
    )
    return c, programme.load(programme_path), programme_path


def prepared(c, **state):
    s = optimizer.initial(c)
    s.update(state)
    s["pending"] = optimizer.batch(c, s)
    return s


def report_for(c, state, improvements=None, effects=None, pairs=None):
    """Synthetic already-computed statistics for selection-unit tests only."""
    improvements, effects = improvements or {}, effects or {}
    item = optimizer.item_for(state["pending"])
    max_matches = item.specs[0].stopping.max_matches
    arms = {
        s.arm: {
            "spec_hash": s.spec_hash,
            "problems": [],
            "hands": s.stopping.max_hands,
            "matches": max_matches,
            "sufficient_sample": True,
        }
        for s in item.specs
    }
    if pairs is None:
        pairs = [("baseline", a) for a in sorted(arms) if a != "baseline"]
    comparisons = []
    for a, b in pairs:
        mean = improvements.get(b, 0) - improvements.get(a, 0)
        comparison = {
            "arms": [a, b],
            "paired_difference": {
                "matches": max_matches,
                "mean_bb_per_100": mean,
                "se": 1.0,
                "ci95": [mean - analysis.Z95, mean + analysis.Z95],
            },
        }
        comparison["recommendation"] = analysis.recommend(
            comparison, True, family_size=len(pairs)
        )
        comparisons.append(comparison)
    stratum = {
        "platform": "local",
        "provenance": item.specs[0].provenance.value,
        "arms": arms,
        "comparisons": comparisons,
    }
    if state["pending"]["phase"] == "screening":
        z = NormalDist().inv_cdf(1 - 0.05 / 30)
        stratum["contrasts"] = [
            {
                **declared,
                "blocks": max_matches,
                "mean": effects.get(declared["label"], 0.0),
                "se": 1.0,
                "ci95": [-analysis.Z95, analysis.Z95],
                "multiplicity": {
                    "method": "bonferroni",
                    "family_size": 15,
                    "family_alpha": 0.05,
                    "interval": [
                        effects.get(declared["label"], 0.0) - z,
                        effects.get(declared["label"], 0.0) + z,
                    ],
                },
                "binding": False,
            }
            for declared in screening.contrasts(c)
        ]
    return {"experiment_id": item.experiment_id, "strata": [stratum]}


def treatments(state):
    return sorted(
        e["spec"]["arm"]
        for e in state["pending"]["specs"]
        if e["spec"]["arm"] != "baseline"
    )


def rewrite_events(directory, change):
    """Rebuild a temporary fixture journal's hash chain after ``change``."""
    files = sorted((directory / "events").glob("*.json"))
    events = [json.loads(f.read_text()) for f in files]
    change(events)
    previous = None
    for file, event in zip(files, events):
        event["previous"] = previous
        event["sha256"] = sha256(
            canonical({k: v for k, v in event.items() if k != "sha256"})
        )
        previous = event["sha256"]
        file.chmod(0o600)
        file.write_bytes(canonical(event) + b"\n")
        file.chmod(0o400)


def tree(directory):
    return {
        str(f): (f.read_bytes(), f.stat().st_mode)
        for f in sorted(Path(directory).rglob("*"))
        if f.is_file()
    }


def force_qualifier(monkeypatch, target, improvement=10.0, phase=".screening."):
    """Synthetic paired statistics for one phase; contrasts stay real."""
    paired = analysis.paired

    def controlled(a, b):
        if phase in a["spec"].experiment_id:
            mean = improvement if b["spec"].arm == target else 0.0
            return {
                "matches": a["spec"].stopping.max_matches,
                "mean_bb_per_100": mean,
                "se": 1.0,
                "ci95": [mean - analysis.Z95, mean + analysis.Z95],
                "identical_match_results": 0,
            }
        return paired(a, b)

    monkeypatch.setattr(analysis, "paired", controlled)


# --- reviewed authority -------------------------------------------------------


def test_exact_canonical_o0002_authority():
    c = campaign.load(CONFIG)
    v = c.config
    assert v["id"] == "O0002" and v["governance"] == "LOCAL_APPROVED"
    assert v["campaign_seed"] == 202610080002
    assert v["algorithm"] == {"kind": "resolution-v-screening", "version": 1}
    assert v["template"] == {
        "path": "specs/E0003-canonical.json",
        "spec_hash": "45d99c7cb7ad35bf1915be66ce767d56fe2ec18e5427907cbe41c60a4018bcc6",
    }
    assert c.template.spec_hash == v["template"]["spec_hash"]
    assert v["policy_revision"] == runner.policy_revision()
    assert v["fixed_policy"] == {
        "samples": 128,
        "opponent_memory": True,
        "historical_prior": None,
    }
    assert v["depends_on"] == [
        {
            "id": "O0001",
            "status": "COMPLETE",
            "classification": "NO_IMPROVEMENT_DETECTED",
        }
    ]
    assert [
        (f["symbol"], f["parameter"], f["low"], f["center"], f["high"])
        for f in v["factors"]
    ] == [
        ("A", "value_bet_threshold", 0.5, 0.55, 0.6),
        ("B", "call_threshold", 0.3, 0.35, 0.4),
        ("C", "bluff_freq", 0.1, 0.15, 0.2),
        ("D", "aggression", 0.75, 0.8, 0.85),
        ("E", "steepness", 6.0, 8.0, 10.0),
    ]
    assert v["design"]["generators"] == [
        {"factor": "E", "product": ["A", "B", "C", "D"]}
    ]
    assert v["design"]["defining_relation"] == ["ABCDE"]
    assert v["design"]["resolution"] == 5 and v["design"]["runs"] == 16
    s, h = v["screening_stopping"], v["confirmation_stopping"]
    assert (s["max_matches"], s["hands_per_match"], s["max_hands"]) == (30, 100, 3000)
    assert (s["matches_per_shard"], s["evaluation_interval_matches"]) == (5, 15)
    assert (h["max_matches"], h["hands_per_match"], h["max_hands"]) == (60, 100, 6000)
    assert (h["matches_per_shard"], h["evaluation_interval_matches"]) == (10, 20)
    assert s["early_stopping"] == h["early_stopping"] == "none"
    assert v["candidate_family"]["size"] == 16
    assert v["diagnostic_family"]["size"] == 15
    assert v["max_generated_vectors"] == 19
    assert v["resources"] == {"workers": 2, "min_free_mib": 1024}
    p = programme.load(PROGRAMME)
    assert [x.identifier for x in p.campaigns] == ["O0001", "O0002"]
    assert p.campaign("O0002").digest == c.digest


def test_campaign_hash_covers_every_authority_field(tmp_path):
    base = campaign.load(screening_config(tmp_path / "a")).digest
    for change in (
        lambda v: v.update(campaign_seed=v["campaign_seed"] + 1),
        lambda v: v["factors"][0].update(low=0.45, high=0.65),
        lambda v: v["screening_stopping"].update(matches_per_shard=1),
        lambda v: v.update(purpose="other"),
    ):
        path = screening_config(tmp_path / str(id(change)), change=change)
        assert campaign.load(path).digest != base


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(unknown=True),
        lambda v: v.update(schema_version=2),
        lambda v: v.update(governance="UNRATED_APPROVED"),
        lambda v: v.update(objective="paired-bb-per-100-bonferroni-all-pairs"),
        lambda v: v.update(seed_derivation="sha256-domain-separated-v1"),
        lambda v: v.update(algorithm={"kind": "resolution-v-screening", "version": 2}),
        lambda v: v["factors"][0].update(unknown=1),
        lambda v: v["factors"][1].update(symbol="A"),
        lambda v: v["factors"][1].update(parameter="value_bet_threshold"),
        lambda v: v["factors"][1].update(parameter="samples"),
        lambda v: v["factors"][0].update(low=0.6, high=0.5),
        lambda v: v["factors"][0].update(low=0.55),
        lambda v: v["factors"][0].update(low=0.52),
        lambda v: v["factors"][4].update(low=-14.0, high=30.0),
        lambda v: v["factors"].pop(),
        lambda v: v["factors"].reverse(),
        lambda v: v["design"].update(runs=32),
        lambda v: v["design"].update(resolution=4),
        lambda v: v["design"].update(defining_relation=["ABCD"]),
        lambda v: v["design"].update(base_factors=["A", "B", "C", "E"]),
        lambda v: v["design"]["generators"][0].update(product=["A", "B", "C"]),
        lambda v: v["design"]["generators"][0].update(product=["A", "B", "C", "F"]),
        lambda v: v["design"]["generators"].append({"factor": "E", "product": ["A"]}),
        lambda v: v["design"].update(type="plackett-burman"),
        lambda v: v["candidate_family"].update(size=136),
        lambda v: v["candidate_family"].update(comparisons="all-pairs"),
        lambda v: v["diagnostic_family"].update(size=14),
        lambda v: v["diagnostic_family"]["effects"].pop(),
        lambda v: v["diagnostic_family"].update(binding=True),
        lambda v: v.update(max_generated_vectors=20),
        lambda v: v["screening_stopping"].update(early_stopping="adaptive"),
        lambda v: v["screening_stopping"].update(min_matches=1),
        lambda v: v["fixed_policy"].update(samples=64),
        lambda v: v["resources"].update(workers=3),
        lambda v: v.update(depends_on=[{"id": "O0001"}]),
        lambda v: v.update(
            depends_on=[{"id": "O0001", "status": "FAILED", "classification": None}]
        ),
        lambda v: v.update(
            depends_on=[
                {"id": "O0001", "status": "COMPLETE", "classification": "BLOCKED"}
            ]
        ),
        lambda v: v.update(
            depends_on=[
                {"id": "O0001", "status": "COMPLETE", "classification": None},
                {"id": "O0001", "status": "COMPLETE", "classification": None},
            ]
        ),
        lambda v: v["template"].update(spec_hash="0" * 64),
        lambda v: v.update(created_at="today"),
    ],
)
def test_reject_invalid_screening_campaign(tmp_path, change):
    with pytest.raises((ValueError, TypeError)):
        campaign.load(screening_config(tmp_path, change=change))


def test_duplicate_json_fields_rejected(tmp_path):
    path = screening_config(tmp_path)
    text = path.read_text().replace('"symbol": "B"', '"symbol": "B", "symbol": "B"')
    path.write_text(text)
    with pytest.raises(ValueError, match="duplicate"):
        campaign.load(path)


def test_centre_is_exactly_canonical(tmp_path):
    c = campaign.load(CONFIG)
    assert c.initial == DEFAULT_PARAMS.__dict__
    shifted = screening_config(
        tmp_path,
        change=lambda v: v["factors"][3].update(low=0.7, center=0.75, high=0.8),
    )
    with pytest.raises(ReviewRequired, match="canonical"):
        campaign.load(shifted)


# --- design ------------------------------------------------------------------


def test_resolution_v_half_fraction_properties():
    c = campaign.load(CONFIG)
    rows = screening.design(c)
    symbols = "ABCDE"
    assert [r["row"] for r in rows] == list(range(16))
    # Lexicographic enumeration of A-D with -1 before +1.
    assert [tuple(r["signs"][s] for s in "ABCD") for r in rows] == sorted(
        {tuple(r["signs"][s] for s in "ABCD") for r in rows}
    )
    assert [r["signs"]["A"] for r in rows] == [-1] * 8 + [1] * 8
    assert [r["signs"]["D"] for r in rows] == [-1, 1] * 8
    for r in rows:
        signs = r["signs"]
        assert signs["E"] == signs["A"] * signs["B"] * signs["C"] * signs["D"]
        assert math.prod(signs.values()) == 1  # I = ABCDE
        assert all(
            r["vector"][f["parameter"]] != f["center"] for f in c.config["factors"]
        )
    assert len({canonical(r["vector"]) for r in rows}) == 16
    assert canonical(c.initial) not in {canonical(r["vector"]) for r in rows}
    columns = {
        w: screening.column(rows, w)
        for w in ["".join(x) for k in (1, 2, 3, 4) for x in combinations(symbols, k)]
    }
    for s in symbols:
        assert columns[s].count(-1) == columns[s].count(1) == 8
    for a, b in combinations(symbols, 2):
        assert sum(x * y for x, y in zip(columns[a], columns[b])) == 0
    mains = {tuple(columns[s]) for s in symbols}
    twos = {tuple(columns["".join(p)]) for p in combinations(symbols, 2)}
    negated = {tuple(-x for x in col) for col in twos}
    assert len(mains) == 5 and len(twos) == 10 and not mains & (twos | negated)
    for word, aliased in TWO_FACTOR_ALIASES.items():
        assert columns[word] == columns[aliased]  # aliased only with 3-factor terms
    for s in symbols:
        assert columns[s] == columns["".join(x for x in symbols if x != s)]
    for r in rows:
        for f in c.config["factors"]:
            assert r["vector"][f["parameter"]] == (
                f["low"] if r["signs"][f["symbol"]] < 0 else f["high"]
            )


def test_alias_labels_under_defining_relation():
    c = campaign.load(CONFIG)
    contrasts = screening.contrasts(c)
    assert [x["label"] for x in contrasts] == list("ABCDE") + list(TWO_FACTOR_ALIASES)
    for x in contrasts:
        if len(x["label"]) == 1:
            assert "main-effect screening contrast" in x["note"]
            assert x["note"].endswith(
                "aliased with four-factor term " + "ABCDE".replace(x["label"], "")
            )
        else:
            assert x["note"].startswith("two-factor screening contrast")
            assert x["note"].endswith(
                "aliased with three-factor term " + TWO_FACTOR_ALIASES[x["label"]]
            )
        assert set(x["weights"].values()) == {0.125, -0.125}
        assert len(x["weights"]) == 16 and "baseline" not in x["weights"]


def test_vectors_ids_hashes_and_seeds_are_deterministic(tmp_path, monkeypatch):
    c = campaign.load(CONFIG)
    first = optimizer.batch(c, optimizer.initial(c))
    monkeypatch.setattr(time, "time", lambda: 0.0)
    random.seed(12345)
    second = optimizer.batch(c, optimizer.initial(c))
    assert first == second
    assert first["experiment_id"] == "O0002.screening.00"
    assert [e["spec"]["arm"] for e in first["specs"]] == ["baseline"] + [
        r["arm"] for r in screening.design(c)
    ]
    assert all(
        ExperimentSpec.from_dict(e["spec"]).spec_hash == e["spec_hash"]
        and e["spec"]["created_at"] == c.config["created_at"]
        for e in first["specs"]
    )
    blocks = campaign.seed_blocks(c)
    for phase in ("screening", "confirmation"):
        material = canonical(["O0002", 202610080002, phase])
        assert blocks[f"{phase}-0"] == int(sha256(material)[:16], 16) & ((1 << 63) - 1)
    assert blocks == {
        "screening-0": 3897605199472245292,
        "confirmation-0": 2873277742887071931,
    }
    assert first["seed_block"] == blocks["screening-0"]


def test_streams_disjoint_from_o0001_and_e0003():
    p = programme.load(PROGRAMME)
    o2, o1 = p.campaign("O0002"), p.campaign("O0001")
    screening_streams = {
        SeedPolicy(campaign.seed_blocks(o2)["screening-0"]).derive(r, m)
        for r in campaign.ROLES
        for m in range(30)
    }
    confirmation_streams = {
        SeedPolicy(campaign.seed_blocks(o2)["confirmation-0"]).derive(r, m)
        for r in campaign.ROLES
        for m in range(60)
    }
    assert screening_streams.isdisjoint(confirmation_streams)
    assert campaign.streams(o2) == screening_streams | confirmation_streams
    o0001 = campaign.streams(o1)
    assert len(o0001) == 3 * (4 * 20 + 60)  # every reserved search/confirmation block
    e0003 = {
        seed
        for spec in p.item("E0003").specs
        for m in range(spec.stopping.max_matches)
        for seed in runner.match_seeds(spec, m).values()
    }
    assert campaign.streams(o2).isdisjoint(o0001 | e0003)
    assert not set(campaign.seed_blocks(o2).values()) & set(
        campaign.seed_blocks(o1).values()
    )


def test_programme_rejects_overlapping_campaign_streams(tmp_path, monkeypatch):
    _, _, path = fixture(tmp_path)
    monkeypatch.setattr(programme, "streams", lambda _: {1, 2, 3})
    value = json.loads(path.read_text())
    other = screening_config(tmp_path, identifier="O9997")
    value["campaigns"].append(
        {"path": other.name, "campaign_hash": campaign.load(other).digest}
    )
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="overlap"):
        programme.load(path)


def test_screening_batch_is_canonical_plus_sixteen_crn_treatments():
    c = campaign.load(CONFIG)
    b = optimizer.batch(c, optimizer.initial(c))
    specs = [ExperimentSpec.from_dict(e["spec"]) for e in b["specs"]]
    assert len(specs) == 17 and b["budget_after"] == 17
    assert specs[0].arm == "baseline" and specs[0].policy_config["params"] == c.initial
    assert {canonical(s.policy_config["params"]) for s in specs[1:]} == {
        canonical(r["vector"]) for r in screening.design(c)
    }
    for s in specs:
        assert s.stopping.max_matches == 30 and s.stopping.max_hands == 3000
        assert s.stopping.matches_per_shard == 5 and s.stopping.early_stopping == "none"
        assert (
            s.seed_policy == specs[0].seed_policy
            and s.seed_policy.common_random_numbers
        )
        assert s.opponent_cohort == c.template.opponent_cohort
        assert s.platform_config_json == c.template.platform_config_json
        assert s.policy_config["samples"] == 128 and s.policy_config["opponent_memory"]
        assert s.policy_config["historical_prior"] is None
    assert sum(s.stopping.max_hands for s in specs) == 51_000


# --- selection ---------------------------------------------------------------


def test_selection_family_is_exactly_sixteen_baseline_comparisons(tmp_path):
    c, _, _ = fixture(tmp_path)
    s = prepared(c)
    decision = optimizer.selection(c, s, report_for(c, s))
    assert len(decision["comparisons"]) == 16
    assert {tuple(x["arms"]) for x in decision["comparisons"]} == {
        ("baseline", t) for t in treatments(s)
    }
    assert all(x["multiplicity"]["family_size"] == 16 for x in decision["comparisons"])
    assert decision["family"] == {
        "comparator": "baseline",
        "size": 16,
        "method": "bonferroni",
        "family_alpha": 0.05,
    }
    all_pairs = list(combinations(["baseline"] + treatments(s), 2))
    assert len(all_pairs) == 136
    with pytest.raises(ValueError, match="family"):
        optimizer.selection(c, s, report_for(c, s, pairs=all_pairs))


def test_raw_mean_alone_cannot_qualify(tmp_path):
    c, _, _ = fixture(tmp_path)
    s = prepared(c)
    # 2.9 clears the unadjusted 1.96 interval but not Bonferroni m=16 (z~2.95).
    target = treatments(s)[3]
    report = report_for(c, s, {target: 2.9})
    comparison = next(
        x for x in report["strata"][0]["comparisons"] if x["arms"][1] == target
    )
    assert comparison["paired_difference"]["ci95"][0] > 0
    decision = optimizer.selection(c, s, report)
    assert decision["selected_arm"] == "baseline" and not decision["qualified"]
    assert decision["reason"] == screening.NO_QUALIFIER


def test_corrected_advantage_qualifies_and_strongest_then_name_wins(tmp_path):
    c, _, _ = fixture(tmp_path)
    s = prepared(c)
    names = treatments(s)
    decision = optimizer.selection(c, s, report_for(c, s, {names[5]: 3.5}))
    assert decision["selected_arm"] == names[5]
    assert decision["incumbent"] != c.initial
    decision = optimizer.selection(
        c, s, report_for(c, s, {names[1]: 10.0, names[2]: 20.0})
    )
    assert decision["selected_arm"] == names[2]
    tie = optimizer.selection(c, s, report_for(c, s, {names[9]: 10.0, names[4]: 10.0}))
    assert tie["selected_arm"] == names[4] and len(tie["qualified"]) == 2
    worse = optimizer.selection(c, s, report_for(c, s, {names[0]: -10.0}))
    assert worse["selected_arm"] == "baseline"


def test_factorial_diagnostics_cannot_select(tmp_path):
    c, _, _ = fixture(tmp_path)
    s = prepared(c)
    decision = optimizer.selection(
        c, s, report_for(c, s, effects={"A": 500.0, "BC": -400.0})
    )
    assert decision["selected_arm"] == "baseline"
    assert len(decision["diagnostics"]) == 15
    assert all(d["binding"] is False for d in decision["diagnostics"])
    assert {d["label"]: d["mean"] for d in decision["diagnostics"]}["A"] == 500.0


@pytest.mark.parametrize("defect", ["missing", "family", "weights", "blocks"])
def test_selection_requires_declared_diagnostics(tmp_path, defect):
    c, _, _ = fixture(tmp_path)
    s = prepared(c)
    report = report_for(c, s)
    contrasts = report["strata"][0]["contrasts"]
    if defect == "missing":
        del report["strata"][0]["contrasts"]
    elif defect == "family":
        contrasts[0]["multiplicity"]["family_size"] = 16
    elif defect == "weights":
        contrasts[3]["weights"] = {k: -v for k, v in contrasts[3]["weights"].items()}
    else:
        contrasts[0]["blocks"] = 1
    with pytest.raises(ValueError):
        optimizer.selection(c, s, report)


@pytest.mark.parametrize(
    "improvement,expected",
    [(0.0, "INCONCLUSIVE"), (10.0, "PROMISING"), (-10.0, "REJECTED")],
)
def test_confirmation_classification(tmp_path, improvement, expected):
    c, _, _ = fixture(tmp_path)
    selected = screening.design(c)[6]
    s = optimizer.initial(c)
    s.update(generation=1, incumbent=selected["vector"])
    s["stop_reason"] = optimizer.stop_reason(c, s)
    s["pending"] = optimizer.batch(c, s)
    assert s["stop_reason"] == screening.SELECTED
    assert s["pending"]["experiment_id"] == "O9998.confirmation.00"
    assert [e["spec"]["arm"] for e in s["pending"]["specs"]] == [
        "baseline",
        selected["arm"],
    ]
    assert [e["spec"]["policy_config"]["params"] for e in s["pending"]["specs"]] == [
        c.initial,
        selected["vector"],
    ]
    report = report_for(c, s, {selected["arm"]: improvement})
    before = DEFAULT_PARAMS.__dict__.copy()
    decision = optimizer.selection(c, s, report)
    assert decision["classification"] == expected
    assert decision["incumbent"] == selected["vector"]
    assert DEFAULT_PARAMS.__dict__ == before


def test_confirmation_horizon_and_budget():
    c = campaign.load(CONFIG)
    selected = screening.design(c)[0]
    s = optimizer.initial(c)
    s.update(generation=1, used=17, incumbent=selected["vector"])
    s["stop_reason"] = optimizer.stop_reason(c, s)
    b = optimizer.batch(c, s)
    assert b["budget_after"] == 19 == c.config["max_generated_vectors"]
    for e in b["specs"]:
        stopping = e["spec"]["stopping"]
        assert (stopping["max_matches"], stopping["max_hands"]) == (60, 6000)
        assert stopping["matches_per_shard"] == 10
    assert b["seed_block"] == campaign.seed_blocks(c)["confirmation-0"]
    # No adaptive second screening round exists.
    s.update(stop_reason=None)
    with pytest.raises(ValueError):
        optimizer.batch(c, s)


# --- factorial estimator -----------------------------------------------------


def synthetic_arms(c, response):
    item = optimizer.item_for(optimizer.batch(c, optimizer.initial(c)))
    arms = {}
    for spec in item.specs:
        row = next((r for r in screening.design(c) if r["arm"] == spec.arm), None)
        hands = [
            {"match": m, "soh_net": response(row, m), "big_blind": 100}
            for m in range(30)
        ]
        arms[spec.arm] = {"spec": spec, "hands": hands}
    return arms


def test_effect_scaling_recovers_injected_effects_exactly():
    c = campaign.load(CONFIG)
    beta_a, beta_bc = 3.0, -1.5

    def response(row, m):
        noise = ((m * 37) % 11 - 5) * 2.0  # shared by every arm in the block
        if row is None:
            return 100.0 + noise
        s = row["signs"]
        return 100.0 + beta_a * s["A"] + beta_bc * s["B"] * s["C"] + noise

    arms = synthetic_arms(c, response)
    results = {}
    for x in screening.contrasts(c):
        results[x["label"]] = analysis.contrast(arms, x["weights"], 15)
    # Standard two-level effect: mean(high) - mean(low) = 2 * coefficient.
    assert results["A"]["mean"] == pytest.approx(2 * beta_a, abs=1e-9)
    assert results["BC"]["mean"] == pytest.approx(2 * beta_bc, abs=1e-9)
    for label, r in results.items():
        assert r["blocks"] == 30
        if label not in {"A", "BC"}:
            assert r["mean"] == pytest.approx(0.0, abs=1e-9)
        assert r["se"] == pytest.approx(0.0, abs=1e-9)  # block noise cancels


def test_effect_estimator_matches_independent_block_contrasts():
    c = campaign.load(CONFIG)

    def response(row, m):
        if row is None:
            return 0.0
        s = row["signs"]
        noise = ((m * 13 + row["row"] * 7) % 9 - 4) * 1.5  # arm-specific noise
        return 2.0 * s["D"] + 0.75 * s["A"] * s["E"] + noise

    arms = synthetic_arms(c, response)
    rows = screening.design(c)
    for x in screening.contrasts(c):
        r = analysis.contrast(arms, x["weights"], 15)
        signs = screening.column(rows, x["label"])
        blocks = []
        for m in range(30):
            y = [response(row, m) for row in rows]
            high = [v for v, s in zip(y, signs) if s > 0]
            low = [v for v, s in zip(y, signs) if s < 0]
            blocks.append(sum(high) / 8 - sum(low) / 8)
        mean = sum(blocks) / 30
        se = math.sqrt(sum((b - mean) ** 2 for b in blocks) / 29) / math.sqrt(30)
        z = NormalDist().inv_cdf(1 - 0.05 / (2 * 15))
        assert r["mean"] == pytest.approx(mean, abs=1e-9)
        assert r["se"] == pytest.approx(se, abs=1e-9)
        assert r["ci95"] == pytest.approx(
            [mean - analysis.Z95 * se, mean + analysis.Z95 * se]
        )
        assert r["multiplicity"]["family_size"] == 15
        assert r["multiplicity"]["interval"] == pytest.approx(
            [mean - z * se, mean + z * se]
        )


def test_contrast_rejects_unknown_or_non_crn_arms():
    c = campaign.load(CONFIG)
    arms = synthetic_arms(c, lambda row, m: 0.0)
    weights = screening.contrasts(c)[0]["weights"]
    with pytest.raises(ValueError):
        analysis.contrast(arms, {**weights, "missing": 1.0}, 15)
    first = next(iter(weights))
    arms[first]["spec"] = replace(arms[first]["spec"], seed_policy=SeedPolicy(1, True))
    with pytest.raises(ValueError, match="common random numbers"):
        analysis.contrast(arms, weights, 15)


# --- dependencies ------------------------------------------------------------


def two_campaign_programme(tmp_path, classification="NO_IMPROVEMENT_DETECTED"):
    """Tiny real O9999 pattern campaign followed by a dependent O9998 screen."""
    first, _, path = pattern_fixture(tmp_path)
    second = screening_config(
        tmp_path,
        change=lambda v: v.update(
            depends_on=[
                {"id": "O9999", "status": "COMPLETE", "classification": classification}
            ]
        ),
    )
    value = json.loads(path.read_text())
    value["campaigns"].append(
        {"path": second.name, "campaign_hash": campaign.load(second).digest}
    )
    path.write_text(json.dumps(value))
    return first, programme.load(path), path


@pytest.mark.parametrize(
    "actual,met",
    [
        ({"status": "COMPLETE", "classification": "NO_IMPROVEMENT_DETECTED"}, True),
        ({"status": "COMPLETE", "classification": "PROMISING"}, False),
        ({"status": "COMPLETE", "classification": "INCONCLUSIVE"}, False),
        ({"status": "COMPLETE", "classification": "REJECTED"}, False),
        ({"status": "SEARCHING", "classification": None}, False),
        ({"status": "CONFIRMING", "classification": None}, False),
        ({"status": "FAILED", "classification": "FAILED"}, False),
        ({"status": "BLOCKED", "classification": "BLOCKED"}, False),
        ({"status": "REQUIRES_REVIEW", "classification": "REQUIRES_REVIEW"}, False),
        (None, False),
    ],
)
def test_dependency_requires_complete_and_classification(tmp_path, actual, met):
    _, p, _ = two_campaign_programme(tmp_path / "authority")
    c = p.campaign("O9998")
    preceding = {} if actual is None else {"O9999": actual}
    row = optimizer.inspect(c, p, tmp_path / "runs", runner.source_state(), preceding)
    assert row["dependencies"][0]["met"] is met
    if met:
        assert (
            row["status"] == "PENDING" and row["intended_action"] == "GENERATE_AND_RUN"
        )
    else:
        assert row["status"] == "BLOCKED"
        assert row["intended_action"] == "WAIT_FOR_DEPENDENCY"
        assert "O9999 requires COMPLETE / NO_IMPROVEMENT_DETECTED" in row["reason"]


@pytest.mark.parametrize(
    "dependency,message",
    [
        ({"id": "O9998", "status": "COMPLETE", "classification": None}, "cycle"),
        ({"id": "O9990", "status": "COMPLETE", "classification": None}, "unknown"),
        (
            {"id": "E9999", "status": "COMPLETE", "classification": "PROMISING"},
            "impossible",
        ),
    ],
)
def test_programme_rejects_bad_campaign_dependencies(tmp_path, dependency, message):
    path = write_programme(tmp_path, [[small()]])
    second = screening_config(
        tmp_path, change=lambda v: v.update(depends_on=[dependency])
    )
    value = json.loads(path.read_text())
    value.update(
        schema_version=2,
        campaigns=[
            {"path": second.name, "campaign_hash": campaign.load(second).digest}
        ],
    )
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=message):
        programme.load(path)


def test_forward_reference_and_mutual_cycle_rejected(tmp_path):
    _, _, path = two_campaign_programme(tmp_path)
    value = json.loads(path.read_text())
    value["campaigns"].reverse()  # O9998 now precedes the campaign it requires
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="cycle or forward reference"):
        programme.load(path)


def test_dependent_screen_waits_then_runs_after_real_dependency(tmp_path):
    _, p, _ = two_campaign_programme(tmp_path / "authority")
    root = tmp_path / "runs"
    plan = supervisor.plan(p, root)
    assert [r["status"] for r in plan["campaigns"]] == ["PENDING", "BLOCKED"]
    first = supervisor.execute(p, root)
    assert [r["status"] for r in first["campaigns"]] == ["SEARCHING", "BLOCKED"]
    assert not any(a.get("campaign") == "O9998" for a in first["actions"])
    second = supervisor.execute(p, root)
    rows = {r["campaign"]: r for r in second["campaigns"]}
    assert rows["O9999"]["classification"] == "NO_IMPROVEMENT_DETECTED"
    assert rows["O9998"]["status"] == "COMPLETE", rows["O9998"]
    assert rows["O9998"]["dependencies"][0]["met"]


def test_wrong_dependency_classification_never_executes(tmp_path, monkeypatch):
    _, p, _ = two_campaign_programme(tmp_path / "authority", classification="PROMISING")
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    result = supervisor.execute(p, root)
    rows = {r["campaign"]: r for r in result["campaigns"]}
    assert rows["O9999"]["classification"] == "NO_IMPROVEMENT_DETECTED"
    assert rows["O9998"]["status"] == "BLOCKED"
    assert "found COMPLETE / NO_IMPROVEMENT_DETECTED" in rows["O9998"]["reason"]
    assert not (root / "optimizer" / "O9998").exists()
    assert result["disposition"] == "BLOCKED"


# --- end to end --------------------------------------------------------------


def test_real_screening_without_qualifier_finishes_in_one_cycle(tmp_path, monkeypatch):
    c, p, _ = fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    first = supervisor.execute(p, root)
    row = first["campaigns"][0]
    assert row["status"] == "COMPLETE", row
    assert row["classification"] == "NO_IMPROVEMENT_DETECTED"
    assert row["reason"] == screening.NO_QUALIFIER
    assert row["confirmation_state"] == "NOT_STARTED"
    assert first["disposition"] == "COMPLETE"
    specs = Index(root / "index.sqlite").specs()
    assert len(specs) == 17 and {s.experiment_id for s in specs} == {
        "O9998.screening.00"
    }
    report = json.loads((root / "analysis/O9998.screening.00.json").read_text())
    stratum = report["strata"][0]
    assert len(stratum["comparisons"]) == 16
    assert all(x["arms"][0] == "baseline" for x in stratum["comparisons"])
    assert {x["multiplicity"]["family_size"] for x in stratum["comparisons"]} == {16}
    assert [x["label"] for x in stratum["contrasts"]] == list("ABCDE") + list(
        TWO_FACTOR_ALIASES
    )
    assert {x["multiplicity"]["family_size"] for x in stratum["contrasts"]} == {15}
    state, events = optimizer.reconstruct(
        c, p, root, runner.source_state(), verify=True
    )
    assert [e["kind"] for e in events] == [
        "start",
        "batch",
        "ready",
        "result",
        "stop",
        "final",
    ]
    result = events[3]["payload"]["decision"]
    assert len(result["comparisons"]) == 16 and len(result["diagnostics"]) == 15
    assert state["used"] == 17
    assert all(
        not (f.stat().st_mode & 0o222)
        for f in (optimizer.directory(root, c) / "specs").glob("*")
    )
    before = tree(optimizer.directory(root, c))
    reports = tree(root / "analysis")
    monkeypatch.setattr(
        scheduler, "run", Mock(side_effect=AssertionError("duplicate run"))
    )
    monkeypatch.setattr(optimizer, "append", Mock(side_effect=AssertionError("event")))
    again = supervisor.execute(p, root)
    assert again["actions"] == [] and again["nothing_to_do"]
    assert (
        tree(optimizer.directory(root, c)) == before
        and tree(root / "analysis") == reports
    )


def test_real_screening_selection_then_separate_held_out_confirmation(
    tmp_path, monkeypatch
):
    c, p, _ = fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    target = screening.design(c)[11]["arm"]
    force_qualifier(monkeypatch, target)
    first = supervisor.execute(p, root)
    row = first["campaigns"][0]
    assert row["status"] == "CONFIRMING" and row["classification"] is None, row
    assert row["intended_action"] == "CONFIRM"
    assert row["incumbent"] == screening.design(c)[11]["vector"]
    assert first["disposition"] == "PENDING"
    assert not any(
        "confirmation" in s.experiment_id for s in Index(root / "index.sqlite").specs()
    )
    monkeypatch.setattr(analysis, "paired", PAIRED)  # real held-out statistics
    second = supervisor.execute(p, root)
    row = second["campaigns"][0]
    assert row["status"] == "COMPLETE", row
    assert row["classification"] in {"PROMISING", "INCONCLUSIVE", "REJECTED"}
    state, events = optimizer.reconstruct(
        c, p, root, runner.source_state(), verify=True
    )
    kinds = [e["kind"] for e in events]
    assert kinds == [
        "start",
        "batch",
        "ready",
        "result",
        "stop",
        "batch",
        "ready",
        "result",
        "final",
    ]
    confirmation = events[5]["payload"]
    assert confirmation["experiment_id"] == "O9998.confirmation.00"
    assert [s["spec"]["arm"] for s in confirmation["specs"]] == ["baseline", target]
    assert confirmation["seed_block"] != events[1]["payload"]["seed_block"]
    assert state["used"] == 19
    report = json.loads((root / "analysis/O9998.confirmation.00.json").read_text())
    assert len(report["strata"][0]["comparisons"]) == 1
    assert "contrasts" not in report["strata"][0]
    assert supervisor.execute(p, root)["actions"] == []


@pytest.mark.parametrize(
    "improvement,expected",
    [(10.0, "PROMISING"), (0.0, "INCONCLUSIVE"), (-10.0, "REJECTED")],
)
def test_real_confirmation_routes_each_classification(
    tmp_path, monkeypatch, improvement, expected
):
    c, p, _ = fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    target = screening.design(c)[2]["arm"]
    force_qualifier(monkeypatch, target)
    supervisor.execute(p, root)
    force_qualifier(monkeypatch, target, improvement, phase=".confirmation.")
    row = supervisor.execute(p, root)["campaigns"][0]
    assert row["classification"] == expected and row["status"] == "COMPLETE"
    assert DEFAULT_PARAMS.__dict__ == campaign.load(CONFIG).initial


@pytest.mark.parametrize("boundary", ["shard", "analysis", "decision", "confirmation"])
def test_crash_resume_without_duplicate_work(tmp_path, monkeypatch, boundary):
    c, p, _ = fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    target = screening.design(c)[7]["arm"]
    force_qualifier(monkeypatch, target)
    if boundary == "confirmation":
        supervisor.execute(p, root)
    originals = (scheduler.run, supervisor._analyze, optimizer.append)
    if boundary in {"shard", "confirmation"}:

        def interrupt(specs, root, *args, **kwargs):
            runner.run_shard(
                specs[1].to_dict(), specs[1].stopping.shards()[0], "fixture", str(root)
            )
            raise KeyboardInterrupt

        monkeypatch.setattr(scheduler, "run", interrupt)
    elif boundary == "analysis":
        original = supervisor._analyze

        def interrupt(*args, **kwargs):
            original(*args, **kwargs)
            raise KeyboardInterrupt

        monkeypatch.setattr(supervisor, "_analyze", interrupt)
    else:
        original = optimizer.append

        def interrupt(path, events, kind, payload):
            if kind == "result":
                raise KeyboardInterrupt
            return original(path, events, kind, payload)

        monkeypatch.setattr(optimizer, "append", interrupt)
    with pytest.raises(KeyboardInterrupt):
        supervisor.execute(p, root)
    state, events = optimizer.reconstruct(c, p, root, runner.source_state())
    pending = state["pending"]
    before = optimizer._completed_records(root, pending)
    assert before
    monkeypatch.setattr(scheduler, "run", originals[0])
    monkeypatch.setattr(supervisor, "_analyze", originals[1])
    monkeypatch.setattr(optimizer, "append", originals[2])
    outcome = supervisor.execute(p, root)
    row = outcome["campaigns"][0]
    expected = "COMPLETE" if boundary == "confirmation" else "CONFIRMING"
    assert row["status"] == expected, row
    state, events = optimizer.reconstruct(
        c, p, root, runner.source_state(), verify=True
    )
    batches = [e["payload"] for e in events if e["kind"] == "batch"]
    assert len(batches) == (2 if boundary == "confirmation" else 1)
    assert [s["spec_hash"] for s in batches[-1]["specs"]] == [
        s["spec_hash"] for s in pending["specs"]
    ]
    after = optimizer._completed_records(root, batches[-1])
    assert all(r in after for r in before)
    shards = [(r["spec_hash"], r["shard_start"]) for r in after]
    assert len(shards) == len(set(shards)) == len(batches[-1]["specs"])
    if boundary in {"analysis", "decision"}:
        assert not any(
            a["action"] in {"OPTIMIZER_RUN", "OPTIMIZER_ANALYZE"}
            for a in outcome["actions"]
        )


def test_crash_before_confirmation_specs_resumes_same_batch(tmp_path, monkeypatch):
    c, p, _ = fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    target = screening.design(c)[4]["arm"]
    force_qualifier(monkeypatch, target)
    supervisor.execute(p, root)
    original = optimizer.publish

    def interrupt(path, file, data):
        if ".confirmation." in file.name:
            raise KeyboardInterrupt
        return original(path, file, data)

    monkeypatch.setattr(optimizer, "publish", interrupt)
    with pytest.raises(KeyboardInterrupt):
        supervisor.execute(p, root)
    state, _ = optimizer.reconstruct(c, p, root, runner.source_state())
    hashes = [s["spec_hash"] for s in state["pending"]["specs"]]
    monkeypatch.setattr(optimizer, "publish", original)
    row = supervisor.execute(p, root)["campaigns"][0]
    assert row["status"] == "COMPLETE"
    _, events = optimizer.reconstruct(c, p, root, runner.source_state(), verify=True)
    batches = [e["payload"] for e in events if e["kind"] == "batch"]
    assert len(batches) == 2 and [s["spec_hash"] for s in batches[1]["specs"]] == hashes


@pytest.mark.parametrize(
    "defect", ["spec", "design_row", "receipt", "report", "evidence", "extra_spec"]
)
def test_tampering_fails_closed(tmp_path, defect):
    c, p, _ = fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    directory = optimizer.directory(root, c)
    file = sorted((directory / "specs").glob("*.json"))[3]
    if defect == "spec":
        file.chmod(0o600)
        file.write_bytes(file.read_bytes() + b" ")
        file.chmod(0o400)
    elif defect == "design_row":

        def change(events):
            batch = events[1]["payload"]
            entry = batch["specs"][4]
            entry["spec"]["policy_config"]["params"]["steepness"] = 12.0
            spec = ExperimentSpec.from_dict(entry["spec"])
            entry.update(spec_hash=spec.spec_hash, file_sha256=sha256(dump(spec)))

        rewrite_events(directory, change)
    elif defect == "receipt":
        receipt = root / "analysis/O9998.screening.00.supervisor.json"
        receipt.write_bytes(
            receipt.read_bytes().replace(b'"binding":false', b'"binding":true')
        )
    elif defect == "report":
        report = root / "analysis/O9998.screening.00.json"
        report.write_bytes(
            report.read_bytes().replace(b'"label":"AB"', b'"label":"BA"')
        )
    elif defect == "evidence":
        run = Index(root / "index.sqlite").runs()[0]
        (Path(run["run_dir"]) / "normalized/hands.jsonl").write_text("{}\n")
    else:
        (directory / "specs/extra.json").write_bytes(file.read_bytes())
    result = supervisor.execute(p, root)
    assert result["campaigns"][0]["status"] == "FAILED", result["campaigns"][0]
    assert not result["actions"] and result["disposition"] == "FAILED"


@pytest.mark.parametrize("identity", ["campaign", "dependency", "policy"])
def test_authority_changes_require_review(tmp_path, identity):
    path = write_programme(tmp_path / "authority", [[small()]])
    second = screening_config(
        tmp_path / "authority",
        change=lambda v: v.update(
            depends_on=[{"id": "E9999", "status": "COMPLETE", "classification": None}]
        ),
    )
    c = campaign.load(second)
    value = json.loads(path.read_text())
    value.update(
        schema_version=2, campaigns=[{"path": second.name, "campaign_hash": c.digest}]
    )
    path.write_text(json.dumps(value))
    p = programme.load(path)
    root = tmp_path / "runs"
    assert supervisor.execute(p, root)["campaigns"][0]["status"] == "COMPLETE"
    source = runner.source_state()
    if identity == "campaign":
        config = c.config
        config["purpose"] = "changed"
        c = replace(c, config_json=canonical(config).decode())
    elif identity == "dependency":
        p = replace(
            p, items=tuple(replace(i, governance="REQUIRES_REVIEW") for i in p.items)
        )
    else:
        source["policy_revision"] = "sha256:" + "0" * 64
    with pytest.raises(ReviewRequired):
        optimizer.reconstruct(c, p, root, source)


# --- legacy O0001 history across programme evolution ------------------------


def legacy_history(tmp_path):
    """A completed v1 journal whose start event binds the whole programme digest."""
    c, p, path = pattern_fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    supervisor.execute(p, root)
    assert supervisor.execute(p, root)["campaigns"][0]["status"] == "COMPLETE"

    def legacy(events):
        start = events[0]["payload"]
        del start["authority"]
        assert set(start) == optimizer.LEGACY_START
        assert start["programme_hash"] == p.digest

    rewrite_events(optimizer.directory(root, c), legacy)
    return c, p, path, root


def append_campaign(path, tmp_path):
    second = screening_config(path.parent)
    value = json.loads(path.read_text())
    value["campaigns"].append(
        {"path": second.name, "campaign_hash": campaign.load(second).digest}
    )
    path.write_text(json.dumps(value))
    return programme.load(path)


def test_legacy_history_survives_appended_campaign_byte_for_byte(tmp_path, monkeypatch):
    c, p, path, root = legacy_history(tmp_path)
    before = tree(optimizer.directory(root, c))
    extended = append_campaign(path, tmp_path)
    assert extended.digest != p.digest
    assert extended.prefix_digest(c.identifier) == p.digest
    monkeypatch.setattr(scheduler, "run", Mock(side_effect=AssertionError("rerun")))
    monkeypatch.setattr(optimizer, "append", Mock(side_effect=AssertionError("event")))
    row = supervisor.plan(extended, root)["campaigns"][0]
    assert (
        row["status"] == "COMPLETE"
        and row["classification"] == "NO_IMPROVEMENT_DETECTED"
    )
    result = supervisor.execute(extended, root, "optimize", c.identifier)
    assert result["actions"] == [] and result["campaigns"][0]["status"] == "COMPLETE"
    assert tree(optimizer.directory(root, c)) == before


@pytest.mark.parametrize(
    "change", ["experiments", "campaign", "policy", "spec", "evidence"]
)
def test_legacy_history_still_fails_closed(tmp_path, change):
    c, _, path, root = legacy_history(tmp_path)
    extended = append_campaign(path, tmp_path)
    source = runner.source_state()
    if change == "experiments":
        # Legacy events bound the whole prefix: new fixed work needs review.
        spec = small()
        (path.parent / "E9999-canonical.json").write_bytes(dump(spec))
        value = json.loads(path.read_text())
        value["experiments"].append(
            {
                "id": "E9999",
                "governance": "LOCAL_APPROVED",
                "reason": "unrelated reviewed fixed work",
                "specs": [
                    {"path": "E9999-canonical.json", "spec_hash": spec.spec_hash}
                ],
                "depends_on": [],
            }
        )
        path.write_text(json.dumps(value))
        extended = programme.load(path)
    elif change == "campaign":
        config = c.config
        config["purpose"] = "changed"
        c = replace(c, config_json=canonical(config).decode())
    elif change == "policy":
        source["policy_revision"] = "sha256:" + "0" * 64
    if change in {"experiments", "campaign", "policy"}:
        with pytest.raises(ReviewRequired):
            optimizer.reconstruct(c, extended, root, source)
        return
    if change == "spec":
        file = next((optimizer.directory(root, c) / "specs").glob("*.json"))
        file.chmod(0o600)
        file.write_bytes(file.read_bytes() + b" ")
        file.chmod(0o400)
        with pytest.raises(ValueError):
            optimizer.reconstruct(c, extended, root, source)
    else:
        run = Index(root / "index.sqlite").runs()[0]
        (Path(run["run_dir"]) / "normalized/hands.jsonl").write_text("{}\n")
        with pytest.raises((ValueError, RuntimeError)):
            optimizer.reconstruct(c, extended, root, source, verify=True)


def test_legacy_rule_is_narrow(tmp_path):
    c, _, path, root = legacy_history(tmp_path)
    # A legacy event cannot be satisfied by the full digest of a later programme
    # or by a non-pattern campaign kind.
    extended = append_campaign(path, tmp_path)
    start = optimizer._events(optimizer.directory(root, c))[0]["payload"]
    assert optimizer._authorized(c, extended, start)
    assert not optimizer._authorized(
        c, replace(extended, prefix_digests=((c.identifier, extended.digest),)), start
    )
    s = campaign.load(path.parent / "O9998.json")
    assert not optimizer._authorized(
        s, extended, {**start, "programme_hash": extended.prefix_digest("O9998")}
    )


def test_committed_programme_still_reproduces_o0001_legacy_digest():
    # The real O0001 v1 start event recorded this digest; appending O0002 must
    # leave the programme prefix through O0001 byte-identical.
    p = programme.load(PROGRAMME)
    assert p.prefix_digest("O0001") == O0001_LEGACY_PROGRAMME
    assert p.digest != O0001_LEGACY_PROGRAMME
    assert p.campaigns[0].digest == (
        "sha256:11cabc1fbe990217896ee4be436ccebb3142d8820018b265937e11e9a0f576ff"
    )


# --- supervisor disposition --------------------------------------------------


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["BLOCKED", "PENDING"], "PENDING"),
        (["BLOCKED", "SCREENING"], "PENDING"),
        (["BLOCKED", "CONFIRMING"], "PENDING"),
        (["BLOCKED", "COMPLETE", "COMPLETE"], "BLOCKED"),
        (["BLOCKED", "BLOCKED"], "BLOCKED"),
        (["COMPLETE", "COMPLETE"], "COMPLETE"),
        (["BLOCKED", "PENDING", "FAILED"], "FAILED"),
        (["BLOCKED", "PENDING", "REQUIRES_REVIEW"], "REQUIRES_REVIEW"),
    ],
)
def test_blocked_item_does_not_mask_eligible_work(statuses, expected):
    rows = [{"status": s} for s in statuses]
    assert supervisor.disposition([], [], rows) == expected


def test_resource_stop_still_blocks_everything():
    stop = {"status": "BLOCKED", "reason": "insufficient memory"}
    assert supervisor.disposition([], [stop], [{"status": "PENDING"}]) == "BLOCKED"


def test_blocked_chipzen_with_pending_screen_reports_pending(tmp_path, monkeypatch):
    online = replace(
        chipzen_spec(), experiment_id="E0001", policy_revision=runner.policy_revision()
    )
    path = write_programme(tmp_path / "authority", [[online]])
    second = screening_config(tmp_path / "authority")
    value = json.loads(path.read_text())
    value.update(
        schema_version=2,
        campaigns=[
            {"path": second.name, "campaign_hash": campaign.load(second).digest}
        ],
    )
    path.write_text(json.dumps(value))
    p = programme.load(path)
    root = tmp_path / "runs"
    c = p.campaigns[0]
    force_qualifier(monkeypatch, screening.design(c)[1]["arm"])
    first = supervisor.execute(p, root)
    assert [r["status"] for r in first["items"]] == ["BLOCKED"]
    assert first["campaigns"][0]["status"] == "CONFIRMING"
    assert first["disposition"] == "PENDING"
    text = cli.render(first)
    assert "Cycle: PENDING" in text and "No eligible work remains" not in text
    second = supervisor.execute(p, root)
    assert second["campaigns"][0]["status"] == "COMPLETE"
    assert second["disposition"] == "BLOCKED"
    assert "No eligible work remains" in cli.render(second)


# --- governance, CLI and firewall ---------------------------------------------


def test_plan_is_read_only_and_renders_screening(tmp_path, monkeypatch):
    _, p, _ = fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    monkeypatch.setattr(
        scheduler, "run", Mock(side_effect=AssertionError("planning ran"))
    )
    result = supervisor.plan(p, root)
    assert not root.exists()
    row = result["campaigns"][0]
    assert row["status"] == "PENDING" and row["intended_action"] == "GENERATE_AND_RUN"
    assert row["confirmation_state"] == "NOT_STARTED"
    assert row["production"] == "LOCKED / untouched"
    assert row["screening"]["arms"] == 17 and row["candidate_family"]["size"] == 16
    text = cli.render(result)
    for expected in (
        "O9998  PENDING",
        "algorithm: resolution-v-screening v1",
        "canonical + 16 treatments",
        "candidate comparison family: 16 canonical-vs-treatment",
        "5 main + 10 two-factor contrasts (non-binding)",
        "confirmation: NOT_STARTED",
        "next: GENERATE_AND_RUN",
        "production: LOCKED / untouched",
    ):
        assert expected in text


def test_real_programme_plan_shows_o0002_pending_without_execution(
    tmp_path, monkeypatch
):
    p = programme.load(PROGRAMME)
    monkeypatch.setattr(
        scheduler, "run", Mock(side_effect=AssertionError("planning ran"))
    )
    preceding = {
        "O0001": {"status": "COMPLETE", "classification": "NO_IMPROVEMENT_DETECTED"}
    }
    row = optimizer.inspect(
        p.campaign("O0002"), p, tmp_path / "runs", runner.source_state(), preceding
    )
    assert row["status"] == "PENDING" and row["intended_action"] == "GENERATE_AND_RUN"
    assert row["screening"]["matches_per_arm"] == 30
    assert row["screening"]["hands_per_arm"] == 3000
    assert not (tmp_path / "runs").exists()
    fresh = supervisor.plan(p, tmp_path / "runs")
    rows = {r["campaign"]: r for r in fresh["campaigns"]}
    # Without evidence E0003 is incomplete, so both campaigns wait in order.
    assert rows["O0001"]["status"] == rows["O0002"]["status"] == "BLOCKED"
    assert (
        "O0001 requires COMPLETE / NO_IMPROVEMENT_DETECTED" in rows["O0002"]["reason"]
    )


def test_production_and_policy_remain_locked(tmp_path, monkeypatch):
    refs = ["refs/heads/deploy/chipzen", "refs/remotes/origin/deploy/chipzen"]
    pointer = subprocess.check_output(["git", "for-each-ref", *refs], cwd=runner.ROOT)
    before = (runner.policy_revision(), DEFAULT_PARAMS.__dict__.copy())
    c, p, _ = fixture(tmp_path / "authority")
    force_qualifier(monkeypatch, screening.design(c)[0]["arm"])
    supervisor.execute(p, tmp_path / "runs")
    monkeypatch.setattr(analysis, "paired", PAIRED)
    row = supervisor.execute(p, tmp_path / "runs")["campaigns"][0]
    assert row["status"] == "COMPLETE" and row["production"] == "LOCKED / untouched"
    assert (runner.policy_revision(), DEFAULT_PARAMS.__dict__) == before
    assert (
        subprocess.check_output(["git", "for-each-ref", *refs], cwd=runner.ROOT)
        == pointer
    )


def test_installed_cli_routes_screening(tmp_path):
    _, _, path = fixture(tmp_path / "authority")
    root = tmp_path / "runs"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    results = []
    for command in (["status"], ["plan"], ["optimize", "O9998"], ["auto"]):
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
            timeout=120,
            check=False,
        )
        assert process.returncode == 0, process.stderr
        results.append(json.loads(process.stdout))
    assert results[1]["campaigns"][0]["status"] == "PENDING"
    assert results[2]["campaigns"][0]["status"] == "COMPLETE"
    assert results[3]["actions"] == [] and results[3]["nothing_to_do"]
