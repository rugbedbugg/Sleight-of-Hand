"""Finite configuration proposals; all poker execution belongs to the control plane.

The canonical state is a chain of immutable decision events, not a cache.
Each event is atomically published and fsynced before its dependent action.
One invocation evaluates at most one batch (search or held-out confirmation).
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

from . import campaign as authority
from . import journal, scheduler, supervisor
from .campaign import ReviewRequired
from .model import Redactor, assert_no_secrets, canonical, exact, sha256
from .programme import Item, Programme
from .spec import ExperimentSpec, dump

VERSION = 1


def directory(root, campaign):
    return Path(root) / "optimizer" / campaign.identifier


def initial(campaign):
    c = campaign.config
    return {
        "generation": 0,
        "used": 0,
        "incumbent": c["initial"],
        "steps": {p["parameter"]: p["initial_step"] for p in c["parameters"]},
        "pending": None,
        "ready": False,
        "stop_reason": None,
        "classification": None,
        "confirmation": None,
        "converged": False,
    }


def stop_reason(campaign, state):
    c = campaign.config
    if state["converged"]:
        return "minimum steps evaluated without improvement"
    if state["generation"] >= c["max_generations"]:
        return "maximum generations reached"
    candidates = authority.neighbors(campaign, state["incumbent"], state["steps"])
    if not candidates:
        return "no bounded neighbors"
    # Include baselines and reserve both held-out arms before admitting search.
    if state["used"] + len(candidates) + 1 + 2 > c["max_generated_vectors"]:
        return "candidate budget exhausted (confirmation reserved)"
    return None


def batch(campaign, state):
    c = campaign.config
    confirmation = state["stop_reason"] is not None
    if not confirmation and stop_reason(campaign, state):
        raise ValueError("search batch exceeds generation/vector budget or convergence")
    phase = "confirmation" if confirmation else "search"
    generation = 0 if confirmation else state["generation"]
    vectors = (
        [c["initial"], state["incumbent"]]
        if confirmation
        else [
            state["incumbent"],
            *authority.neighbors(campaign, state["incumbent"], state["steps"]),
        ]
    )
    if state["used"] + len(vectors) > c["max_generated_vectors"]:
        raise ValueError("candidate budget exceeded")
    identifier = f"{c['id']}.{phase}.{generation:02d}"
    seed = authority.seed_blocks(campaign)[f"{phase}-{generation}"]
    specs = []
    for index, vector in enumerate(vectors):
        authority.validate_vector(campaign, vector)
        arm = (
            "baseline" if index == 0 else "candidate-" + sha256(canonical(vector))[:20]
        )
        value = campaign.template.to_dict()
        value.update(
            experiment_id=identifier,
            arm=arm,
            description=f"{c['id']} {phase}; authority {campaign.digest}; non-binding research",
            created_at=c["created_at"],
            policy_revision=c["policy_revision"],
            policy_config={**c["fixed_policy"], "params": vector},
            seed_policy={"base_seed": seed, "common_random_numbers": True},
            stopping=c["confirmation_stopping" if confirmation else "search_stopping"],
        )
        spec = ExperimentSpec.from_dict(value)
        specs.append(
            {
                "spec": spec.to_dict(),
                "spec_hash": spec.spec_hash,
                "file_sha256": sha256(dump(spec)),
            }
        )
    if len({s["spec"]["arm"] for s in specs}) != len(specs):
        raise ValueError("duplicate generated vectors or arm hash collision")
    return {
        "phase": phase,
        "incumbent": state["incumbent"],
        "steps": state["steps"],
        "generation": generation,
        "experiment_id": identifier,
        "seed_block": seed,
        "specs": specs,
        "budget_after": state["used"] + len(specs),
    }


def item_for(prepared):
    return Item(
        prepared["experiment_id"],
        "LOCAL_APPROVED",
        "reviewed bounded campaign",
        tuple(ExperimentSpec.from_dict(s["spec"]) for s in prepared["specs"]),
        (),
    )


def selection(campaign, state, report):
    """Consume existing corrected verdicts; a larger raw mean never suffices."""
    pending = state["pending"]
    item = item_for(pending)
    if report["experiment_id"] != item.experiment_id or len(report["strata"]) != 1:
        raise ValueError("unexpected optimizer analysis stratum")
    stratum = report["strata"][0]
    if (
        stratum["platform"] != "local"
        or stratum["provenance"] != item.specs[0].provenance.value
    ):
        raise ValueError("unexpected optimizer platform/provenance")
    if set(stratum["arms"]) != {s.arm for s in item.specs}:
        raise ValueError("analysis arms differ from generated specs")
    for spec in item.specs:
        arm = stratum["arms"][spec.arm]
        if (
            arm["problems"]
            or arm["spec_hash"] != spec.spec_hash
            or arm["hands"] != spec.stopping.max_hands
            or arm["matches"] != spec.stopping.max_matches
            or not arm["sufficient_sample"]
        ):
            raise ValueError("complete valid generation evidence required")
    expected_pairs = {tuple(sorted(p)) for p in combinations(stratum["arms"], 2)}
    comparisons = stratum["comparisons"]
    if (
        len(comparisons) != len(expected_pairs)
        or {tuple(c["arms"]) for c in comparisons} != expected_pairs
    ):
        raise ValueError("incomplete comparison family")
    qualified = []
    for comparison in comparisons:
        correction = comparison.get("multiplicity", {})
        if (
            correction.get("method") != "bonferroni"
            or correction.get("family_size") != len(expected_pairs)
            or correction.get("family_alpha") != 0.05
        ):
            raise ValueError("unapproved statistical comparison rule")
        if (
            comparison["paired_difference"]["matches"]
            != item.specs[0].stopping.max_matches
        ):
            raise ValueError("incomplete paired evidence")
        recommendation = comparison["recommendation"]
        if recommendation["verdict"] not in {
            "DIFFERENCE_DETECTED",
            "NO_DIFFERENCE_DETECTED",
        }:
            raise ValueError("generation has no valid statistical verdict")
        if "baseline" not in comparison["arms"]:
            continue
        higher = recommendation.get("higher_arm")
        if recommendation["verdict"] == "DIFFERENCE_DETECTED" and higher != "baseline":
            if higher not in comparison["arms"]:
                raise ValueError("invalid higher arm")
            improvement = comparison["paired_difference"]["mean_bb_per_100"]
            if comparison["arms"][0] != "baseline":
                improvement = -improvement
            if improvement <= 0:
                raise ValueError("verdict and improvement disagree")
            qualified.append((improvement, higher))
    qualified.sort(key=lambda pair: (-pair[0], pair[1]))
    winner = qualified[0][1] if qualified else "baseline"
    vector = next(s.policy_config["params"] for s in item.specs if s.arm == winner)
    result = {
        "selected_arm": winner,
        "incumbent": vector,
        "comparisons": comparisons,
        "qualified": qualified,
        "reason": (
            "largest positive paired bb/100 improvement among corrected detectable higher arms; ties by arm name"
            if qualified
            else "no corrected detectable advantage over incumbent"
        ),
        "step_shrink": not bool(qualified),
        "steps": state["steps"]
        if qualified
        else authority.shrink(campaign, state["steps"]),
    }
    if pending["phase"] == "confirmation":
        verdict = comparisons[0]["recommendation"]
        result["classification"] = (
            "PROMISING"
            if qualified
            else "REJECTED"
            if verdict["verdict"] == "DIFFERENCE_DETECTED"
            else "INCONCLUSIVE"
            if verdict["verdict"] == "NO_DIFFERENCE_DETECTED"
            else "FAILED"
        )
        # Confirmation classifies the selected candidate; it does not search.
        result.update(
            incumbent=state["incumbent"], steps=state["steps"], step_shrink=False
        )
    return result


def _events(path):
    events, previous = [], None
    for seq, file in enumerate(sorted((path / "events").glob("*"))):
        if file.name != f"{seq:06d}.json":
            raise ValueError("noncontiguous or unauthorized optimizer event")
        if file.stat().st_mode & 0o222:
            raise ValueError("optimizer event is no longer immutable")
        event = json.loads(file.read_text())
        exact(event, {"seq", "previous", "kind", "payload", "sha256"})
        material = {k: v for k, v in event.items() if k != "sha256"}
        if (
            event["seq"] != seq
            or event["previous"] != previous
            or event["sha256"] != sha256(canonical(material))
        ):
            raise ValueError("optimizer journal checksum/sequence failure")
        assert_no_secrets(event, "optimizer event")
        events.append(event)
        previous = event["sha256"]
    return events


def append(path, events, kind, payload):
    assert_no_secrets(payload, "optimizer event")
    value = {
        "seq": len(events),
        "previous": events[-1]["sha256"] if events else None,
        "kind": kind,
        "payload": payload,
    }
    value["sha256"] = sha256(canonical(value))
    target = path / "events" / f"{len(events):06d}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("optimizer event already exists")
    publish(path, target, canonical(value) + b"\n")
    events.append(value)


def publish(path, target, data):
    """Publish a fully durable, already read-only file without overwriting one.

    The scratch file is not history. A process can die at any instruction;
    readers see either the complete immutable target or no target at all.
    The supervisor root lock serializes writers.
    """
    scratch = path / ".pending"
    journal.atomic_write(scratch, data)
    scratch.chmod(0o400)
    with scratch.open("rb") as stream:
        os.fsync(stream.fileno())
    os.link(scratch, target)
    fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    scratch.unlink()


def _completed_records(root, prepared):
    hashes = {s["spec_hash"] for s in prepared["specs"]}
    return sorted(
        [
            {
                k: r[k]
                for k in (
                    "run_id",
                    "spec_hash",
                    "shard_start",
                    "shard_end",
                    "raw_sha256",
                )
            }
            for r in supervisor.Snapshot(root).records
            if r["spec_hash"] in hashes and r["status"] == "COMPLETED"
        ],
        key=lambda r: r["run_id"],
    )


def reconstruct(campaign, programme, root, source, *, verify=False):
    """Read-only replay. Every transition is checked against its authority/evidence."""
    path = directory(root, campaign)
    events = _events(path)
    c, state, batches = campaign.config, initial(campaign), []
    if (
        c["policy_revision"] != source["policy_revision"]
        or c["governance"] != "LOCAL_APPROVED"
    ):
        raise ReviewRequired("campaign policy fingerprint/governance mismatch")
    for index, event in enumerate(events):
        kind, payload = event["kind"], event["payload"]
        if state["classification"]:
            raise ValueError("events after campaign completion")
        if index == 0:
            if kind != "start":
                raise ValueError("optimizer journal missing start")
            if (
                payload["campaign_hash"] != campaign.digest
                or payload["programme_hash"] != programme.digest
                or payload["policy_revision"] != c["policy_revision"]
                or payload["optimizer_version"] != VERSION
            ):
                raise ReviewRequired(
                    "campaign/programme/version identity changed mid-run"
                )
            if (
                payload["seed_blocks"] != authority.seed_blocks(campaign)
                or payload["campaign_seed"] != c["campaign_seed"]
            ):
                raise ValueError("campaign seed authority changed")
            continue
        if kind == "batch":
            if (
                state["pending"]
                or state["confirmation"]
                or (not state["stop_reason"] and stop_reason(campaign, state))
            ):
                raise ValueError("batch outside campaign budget/state")
            if payload != batch(campaign, state):
                raise ValueError(
                    "generated batch differs from authorized vectors/specs/budget"
                )
            state.update(pending=payload, ready=False, used=payload["budget_after"])
            batches.append([payload, False])
        elif kind == "ready":
            if (
                not state["pending"]
                or state["ready"]
                or payload != {"experiment_id": state["pending"]["experiment_id"]}
            ):
                raise ValueError("invalid spec materialization event")
            state["ready"] = True
            batches[-1][1] = True
        elif kind == "result":
            if not state["pending"] or not state["ready"]:
                raise ValueError("result without durable materialized batch")
            prepared = state["pending"]
            report_path = Path(root) / "analysis" / f"{prepared['experiment_id']}.json"
            data = report_path.read_bytes()
            if sha256(data) != payload["report_sha256"] or payload[
                "runs"
            ] != _completed_records(root, prepared):
                raise ValueError("analysis or completed evidence changed")
            receipt = supervisor._analysis_path(
                Path(root), item_for(prepared)
            ).read_bytes()
            if sha256(receipt) != payload["receipt_sha256"]:
                raise ValueError("analysis receipt changed")
            decision = selection(campaign, state, json.loads(data))
            # canonical JSON normalizes tuple/list representations.
            if canonical(decision) != canonical(payload["decision"]):
                raise ValueError("selection differs from verified analysis")
            if prepared["phase"] == "search":
                state["converged"] = decision["step_shrink"] and state[
                    "steps"
                ] == authority.shrink(campaign, state["steps"])
                state.update(
                    generation=state["generation"] + 1,
                    incumbent=decision["incumbent"],
                    steps=decision["steps"],
                )
            else:
                state["confirmation"] = decision["classification"]
            state.update(pending=None, ready=False)
        elif kind == "stop":
            reason = stop_reason(campaign, state)
            if (
                state["pending"]
                or state["stop_reason"]
                or not reason
                or payload != {"reason": reason}
            ):
                raise ValueError("invalid search termination")
            state["stop_reason"] = reason
        elif kind == "final":
            classification = (
                "NO_IMPROVEMENT_DETECTED"
                if state["incumbent"] == c["initial"]
                else state["confirmation"]
            )
            if (
                not state["stop_reason"]
                or state["pending"]
                or not classification
                or payload != {"classification": classification, "binding": False}
            ):
                raise ValueError("invalid campaign classification")
            state["classification"] = classification
        elif kind == "halt":
            if payload["classification"] not in {
                "FAILED",
                "BLOCKED",
                "REQUIRES_REVIEW",
            }:
                raise ValueError("invalid halt classification")
            state.update(
                classification=payload["classification"], stop_reason=payload["reason"]
            )
        else:
            raise ValueError("unknown optimizer event")
    expected, hashes = {}, set()
    for prepared, ready in batches:
        for entry in prepared["specs"]:
            spec = ExperimentSpec.from_dict(entry["spec"])
            file = path / "specs" / f"{spec.experiment_id}.{spec.arm}.json"
            expected[file] = (entry["file_sha256"], ready)
            hashes.add(spec.spec_hash)
        if verify:
            supervisor._verify_evidence(Path(root), item_for(prepared))
    actual = set((path / "specs").glob("*"))
    if actual - set(expected):
        raise ValueError("unauthorized generated spec file")
    for file, (digest, required) in expected.items():
        if file.exists():
            if sha256(file.read_bytes()) != digest or file.stat().st_mode & 0o222:
                raise ValueError("generated spec bytes or immutability changed")
        elif required:
            raise ValueError("materialized generated spec missing")
    for spec in supervisor.Snapshot(root).registered:
        if (
            spec.experiment_id.startswith(campaign.identifier + ".")
            and spec.spec_hash not in hashes
        ):
            raise ValueError("unauthorized ExperimentSpec in campaign")
    return state, events


def inspect(campaign, programme, root, source, fixed):
    state = initial(campaign)
    status, reason, action = "PENDING", "approved local campaign", "GENERATE_AND_RUN"
    try:
        state, _ = reconstruct(campaign, programme, root, source)
        if campaign.identifier in supervisor._past_failures(
            Path(root), programme.digest
        ):
            status, reason, action = (
                "FAILED",
                "previous campaign failure requires investigation",
                "INVESTIGATE",
            )
        elif state["classification"]:
            status = (
                state["classification"]
                if state["classification"] in {"FAILED", "BLOCKED", "REQUIRES_REVIEW"}
                else "COMPLETE"
            )
            reason, action = state["stop_reason"], "NONE"
        elif any(
            fixed[d]["status"] != "COMPLETE" for d in campaign.config["depends_on"]
        ):
            status, reason, action = (
                "BLOCKED",
                "campaign dependencies incomplete",
                "WAIT_FOR_DEPENDENCY",
            )
        else:
            status = (
                "CONFIRMING"
                if state["stop_reason"]
                else "SEARCHING"
                if state["generation"] or state["pending"]
                else "PENDING"
            )
            if state["pending"]:
                p = Programme(
                    (item_for(state["pending"]),),
                    programme.workers,
                    programme.min_free_mib,
                    programme.digest,
                )
                row = supervisor.plan(p, root)["items"][0]
                action = row["intended_action"]
                if row["status"] in {"BLOCKED", "FAILED", "REQUIRES_REVIEW"}:
                    status, reason = row["status"], row["reason"]
            elif state["stop_reason"]:
                action = (
                    "FINALIZE"
                    if state["confirmation"]
                    or state["incumbent"] == campaign.config["initial"]
                    else "CONFIRM"
                )
            elif stop_reason(campaign, state):
                action = "STOP_SEARCH"
    except ReviewRequired as exc:
        status, reason, action = "REQUIRES_REVIEW", str(exc), "REVIEW"
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        status, reason, action = "FAILED", str(exc), "INVESTIGATE"
    return {
        "campaign": campaign.identifier,
        "campaign_hash": campaign.digest,
        "governance": campaign.config["governance"],
        "algorithm": campaign.config["algorithm"],
        "status": status,
        "reason": reason,
        "intended_action": action,
        "incumbent": state["incumbent"],
        "generation": state["generation"],
        "steps": state["steps"],
        "candidate_budget_used": state["used"],
        "candidate_budget_max": campaign.config["max_generated_vectors"],
        "candidate_budget_remaining": campaign.config["max_generated_vectors"]
        - state["used"],
        "search_state": state["stop_reason"] or "ACTIVE"
        if state["generation"] or state["pending"] or state["stop_reason"]
        else "PENDING",
        "confirmation_state": state["confirmation"]
        or (
            "PENDING"
            if state["stop_reason"] and state["incumbent"] != campaign.config["initial"]
            else "NOT_STARTED"
        ),
        "classification": status
        if status in {"FAILED", "REQUIRES_REVIEW"}
        else state["classification"],
        "production": "LOCKED / untouched",
    }


def advance(campaign, programme, root, source):
    """Called only under the supervisor's existing cycle lock."""
    path = directory(root, campaign)
    state, events = reconstruct(campaign, programme, root, source, verify=True)
    if state["classification"]:
        return []
    actions = []

    def record(kind, payload):
        append(path, events, kind, payload)
        actions.append(
            {"campaign": campaign.identifier, "action": "OPTIMIZER_" + kind.upper()}
        )

    if not events:
        record(
            "start",
            {
                "optimizer_version": VERSION,
                "campaign_hash": campaign.digest,
                "programme_hash": programme.digest,
                "source": source,
                "policy_revision": source["policy_revision"],
                "campaign_seed": campaign.config["campaign_seed"],
                "seed_blocks": authority.seed_blocks(campaign),
            },
        )

    def finish_search():
        nonlocal state
        if not state["pending"] and not state["stop_reason"]:
            reason = stop_reason(campaign, state)
            if reason:
                record("stop", {"reason": reason})
                state["stop_reason"] = reason
        if state["stop_reason"] and not state["pending"]:
            classification = (
                "NO_IMPROVEMENT_DETECTED"
                if state["incumbent"] == campaign.config["initial"]
                else state["confirmation"]
            )
            if classification:
                record("final", {"classification": classification, "binding": False})
                return True
        return False

    if finish_search():
        return actions
    if not state["pending"]:
        prepared = batch(campaign, state)
        record("batch", prepared)
        state.update(pending=prepared, used=prepared["budget_after"], ready=False)
    prepared = state["pending"]
    item = item_for(prepared)
    if not state["ready"]:
        target = path / "specs"
        target.mkdir(parents=True, exist_ok=True)
        for spec in item.specs:
            file = target / f"{spec.experiment_id}.{spec.arm}.json"
            if file.exists():
                if file.read_bytes() != dump(spec):
                    raise ValueError("generated spec changed before materialization")
            else:
                publish(path, file, dump(spec))
        record("ready", {"experiment_id": item.experiment_id})
        state["ready"] = True
    resources = campaign.config["resources"]
    work = Programme(
        (item,), resources["workers"], resources["min_free_mib"], programme.digest
    )
    row = supervisor.plan(work, root)["items"][0]
    if row["status"] in {"FAILED", "REQUIRES_REVIEW", "BLOCKED"}:
        record("halt", {"classification": row["status"], "reason": row["reason"]})
        return actions
    try:
        if row["pending_shards"]:
            outcome = scheduler.run(
                list(item.specs),
                root,
                work.workers,
                work.min_free_mib,
                log=lambda _: None,
                stop_on_failure=True,
                require_resources=True,
            )
            actions.append(
                {
                    "campaign": campaign.identifier,
                    "action": "OPTIMIZER_RUN",
                    "outcome": outcome,
                }
            )
            if outcome["failed"] or outcome["resource_blocked"]:
                record(
                    "halt",
                    {
                        "classification": "FAILED" if outcome["failed"] else "BLOCKED",
                        "reason": "scheduler failure"
                        if outcome["failed"]
                        else outcome["resource_blocked"],
                    },
                )
                return actions
        row = supervisor.plan(work, root)["items"][0]
        if row["pending_shards"] or row["status"] not in {"COMPLETE", "PENDING"}:
            raise ValueError("generation horizon incomplete or unauthorized")
        supervisor._verify_evidence(root, item)
        if row["analysis_status"] != "CURRENT":
            supervisor._analyze(root, item, familywise=True)
            actions.append(
                {"campaign": campaign.identifier, "action": "OPTIMIZER_ANALYZE"}
            )
        report = (root / "analysis" / f"{item.experiment_id}.json").read_bytes()
        decision = selection(campaign, state, json.loads(report))
        record(
            "result",
            {
                "decision": decision,
                "report_sha256": sha256(report),
                "receipt_sha256": sha256(
                    supervisor._analysis_path(root, item).read_bytes()
                ),
                "runs": _completed_records(root, prepared),
            },
        )
        state, _ = reconstruct(campaign, programme, root, source)
        finish_search()
    except Exception as exc:  # noqa: BLE001 - preserve failed research state
        record(
            "halt",
            {
                "classification": "FAILED",
                "reason": Redactor().text(f"{type(exc).__name__}: {exc}"),
            },
        )
    return actions
