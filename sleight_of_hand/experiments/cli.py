"""Public ``uv run research`` interface to the existing control plane."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from . import programme, supervisor
from .campaign import ReviewRequired
from .model import Redactor
from .runner import ROOT


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="research",
        description="Sleight-of-Hand Research Supervisor — approved research only; production LOCKED",
    )
    commands = result.add_subparsers(dest="command", required=True)
    for command in ("status", "plan", "run", "analyze", "auto", "optimize"):
        sub = commands.add_parser(command)
        if command in {"run", "analyze", "optimize"}:
            sub.add_argument("experiment", help="approved experiment or campaign ID")
        sub.add_argument(
            "--programme", type=Path, default=ROOT / "experiments" / "programme.json"
        )
        sub.add_argument("--root", type=Path, default=ROOT / "runs")
        sub.add_argument("--json", action="store_true", help="machine-readable result")
    return result


def render(result: dict) -> str:
    source = result["source"]
    lines = [
        "Sleight-of-Hand Research Supervisor",
        f"source        {source['git_head']}",
        f"policy        {source['policy_revision']}",
        f"workers       {result['workers']}",
        "production    LOCKED / untouched",
        "",
    ]
    for item in result["items"]:
        lines += [
            f"{item['experiment']}  {item['status']}  [{item['governance']}]",
            f"  {item['platform']} / {item['provenance']}; arms: "
            + ", ".join(s["arm"] for s in item["arms"]),
            f"  shards: {item['completed_shards']} complete, {item['pending_shards']} pending, {item['incomplete_runs']} incomplete runs, {item['running_runs']} running",
            f"  analysis: {item['analysis_status']}; next: {item['intended_action']}",
            f"  {item['reason']}",
        ]
        for state in item["platform_statuses"]:
            lines.append(
                f"  platform {state['arm']}: {state['status']} — {state['reason']}"
            )
    for item in result.get("campaigns", []):
        lines += [
            f"{item['campaign']}  {item['status']}  [{item['governance']}]",
            f"  campaign hash: {item['campaign_hash']}",
            f"  algorithm: {item['algorithm']['kind']} v{item['algorithm']['version']}",
            "  incumbent: " + json.dumps(item["incumbent"], sort_keys=True),
            f"  generation: {item['generation']}",
            "  steps: " + json.dumps(item["steps"], sort_keys=True),
            f"  candidate budget: {item['candidate_budget_used']} / {item['candidate_budget_max']} ({item['candidate_budget_remaining']} remaining)",
            f"  search: {item['search_state']}; confirmation: {item['confirmation_state']}",
            f"  classification: {item['classification'] or 'PENDING'}; next: {item['intended_action']}",
            f"  {item['reason']}",
            "  production: LOCKED / untouched",
        ]
    for action in result.get("actions", []):
        if action["action"] == "ANALYZE":
            for conclusion in action["analysis"]["conclusions"]:
                lines.append(
                    f"{action['experiment']} non-binding conclusion: "
                    + json.dumps(conclusion, sort_keys=True)
                )
    for failure in result.get("failures", []) + result.get("governance_stops", []):
        lines.append(f"{failure['status']}: {failure['reason']}")
    if "disposition" in result:
        lines += [
            "",
            f"Cycle: {result['disposition']}",
            {
                "COMPLETE": "Nothing to do."
                if result["nothing_to_do"]
                else "All selected approved work is complete.",
                "BLOCKED": "No eligible work remains; blocked items require action.",
                "REQUIRES_REVIEW": "Stopped at a governance boundary.",
                "FAILED": "Stopped for investigation.",
                "PENDING": "Pending work remains.",
            }[result["disposition"]],
            f"Decision journal: {result['decision_journal']}",
        ]
    lines += [
        "",
        "Governance: production unchanged; recommendations are non-binding.",
        "Additional experiments or policy changes require review.",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(arguments)
    try:
        approved = programme.load(args.programme)
        if args.command in {"status", "plan"}:
            result = supervisor.plan(approved, args.root)
        else:
            result = supervisor.execute(
                approved,
                args.root,
                args.command,
                getattr(args, "experiment", None),
                arguments,
            )
        print(json.dumps(result, indent=2) if args.json else render(result))
        return 1 if result.get("disposition") == "FAILED" else 0
    except ReviewRequired as exc:
        reason = Redactor().text(str(exc))
        print(
            json.dumps(
                {
                    "status": "REQUIRES_REVIEW",
                    "reason": reason,
                    "production": "LOCKED / untouched",
                }
            )
            if args.json
            else f"research: REQUIRES_REVIEW — {reason}; production LOCKED / untouched"
        )
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        error = Redactor().text(f"{type(exc).__name__}: {exc}")
        print(
            json.dumps({"status": "FAILED", "error": error})
            if args.json
            else f"research: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
