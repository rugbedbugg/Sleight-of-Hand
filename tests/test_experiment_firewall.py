"""Production firewall: the control plane cannot reach the shipped runtime."""

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "sleight_of_hand" / "experiments"
SCRIPTS = [
    ROOT / "scripts" / name
    for name in (
        "run_experiment.py",
        "run_experiment_batch.py",
        "analyze_experiment.py",
        "build_local_prior.py",
    )
]


def sources():
    return sorted(EXPERIMENTS.rglob("*.py")) + SCRIPTS


def test_experiment_package_is_absent_from_chipzen_staging(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "build_chipzen", ROOT / "scripts" / "build_chipzen.py"
    )
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    monkeypatch.setattr(build, "OUTPUT", tmp_path / "chipzen")
    build.main()
    staged = {
        p.relative_to(tmp_path / "chipzen").as_posix()
        for p in (tmp_path / "chipzen").rglob("*")
        if p.is_file()
    }
    assert staged and not any("experiments" in p for p in staged)
    assert not any(
        p.startswith(("tests/", "scripts/", "docs/", "runs/")) for p in staged
    )


def test_runtime_never_imports_the_control_plane():
    runtime = [
        ROOT / "bots" / "chipzen" / "bot.py",
        ROOT / "bots" / "chipzen" / "accounting_observer.py",
    ]
    for package in ("engine", "policy", "holdem"):
        runtime += sorted((ROOT / "sleight_of_hand" / package).glob("*.py"))
    for path in runtime:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any("experiments" in n for n in names), path


def test_git_access_is_read_only_and_nothing_deploys():
    for path in sources():
        text = path.read_text()
        for forbidden in (
            "deploy/chipzen",
            "docker",
            "git push",
            '"push"',
            '"tag"',
            '"commit"',
        ):
            assert forbidden not in text, (path, forbidden)
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_git":
                args = [a.value for a in node.args if isinstance(a, ast.Constant)]
                assert args[0] in {"rev-parse", "status"}, (path, args)


def test_production_credentials_are_never_used_for_research():
    text = (EXPERIMENTS / "platforms" / "chipzen.py").read_text()
    tree = ast.parse(text)
    reads = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and getattr(node.value, "attr", None) == "environ"
    }
    assert reads and all(
        name.startswith(("CHIPZEN_RESEARCH_", "CHIPZEN_PROBE_")) for name in reads
    )
