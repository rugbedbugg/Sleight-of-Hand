"""One dependency source of truth: pyproject.toml + uv.lock.

The uploaded ChipZen runtime installs ``bots/chipzen/requirements.txt`` and
nothing else; it must stay an exact projection of the ``chipzen`` group.
"""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; tomli is locked for pytest there
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
LOCK = tomllib.loads((ROOT / "uv.lock").read_text())
LOCKED = {}
for package in LOCK["package"]:
    LOCKED.setdefault(package["name"], set()).add(package["version"])


def production_requirements():
    lines = (ROOT / "bots" / "chipzen" / "requirements.txt").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def pins(requirements):
    pairs = [requirement.split("==") for requirement in requirements]
    assert all(len(pair) == 2 for pair in pairs), requirements
    return {name.strip(): version.strip() for name, version in pairs}


def test_production_requirements_equal_the_chipzen_group():
    group = PYPROJECT["dependency-groups"]["chipzen"]
    assert production_requirements() == group
    assert pins(group) == {"chipzen-bot": "0.4.0", "websockets": "15.0.1"}


def test_dev_group_contains_the_chipzen_group_and_pinned_ruff():
    dev = PYPROJECT["dependency-groups"]["dev"]
    assert {"include-group": "chipzen"} in dev
    assert "ruff==0.16.3" in dev
    assert PYPROJECT["tool"]["uv"]["default-groups"] == ["dev"]


def test_lock_resolves_the_exact_pins_for_every_supported_python():
    assert LOCK["requires-python"] == PYPROJECT["project"]["requires-python"]
    exact = pins(PYPROJECT["dependency-groups"]["chipzen"]) | {"ruff": "0.16.3"}
    for name, version in exact.items():
        assert LOCKED[name] == {version}, name


def test_no_second_requirements_manifest():
    skip = {"build", "dist", "evals", "runs"}  # generated or local-only
    manifests = sorted(
        str(relative)
        for relative in (p.relative_to(ROOT) for p in ROOT.rglob("requirements*.txt"))
        if not any(part in skip or part.startswith(".") for part in relative.parts)
    )
    assert manifests == ["bots/chipzen/requirements.txt"]


def test_project_installs_the_research_console_script():
    assert PYPROJECT["tool"]["uv"]["package"] is True
    assert PYPROJECT["project"]["scripts"] == {
        "research": "sleight_of_hand.experiments.cli:main"
    }
    project = next(p for p in LOCK["package"] if p["name"] == "sleight-of-hand")
    assert project["source"] == {"editable": "."}
