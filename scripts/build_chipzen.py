"""Stage a small, self-contained ChipZen Docker build context."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "build" / "chipzen"


def main() -> None:
    # This dedicated generated directory is disposable; refresh it so old
    # modules/dependencies cannot silently enter a later upload image.
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    for name in ("bot.py", "requirements.txt", "Dockerfile", ".dockerignore"):
        shutil.copy2(ROOT / "bots" / "chipzen" / name, OUTPUT / name)
    shutil.copy2(ROOT / "LICENSE", OUTPUT / "LICENSE")
    package = OUTPUT / "sleight_of_hand"
    package.mkdir()
    shutil.copy2(ROOT / "sleight_of_hand" / "__init__.py", package / "__init__.py")
    for name in ("engine", "policy", "holdem"):
        shutil.copytree(
            ROOT / "sleight_of_hand" / name,
            package / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    print(OUTPUT)


if __name__ == "__main__":
    main()
