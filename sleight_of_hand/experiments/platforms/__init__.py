"""Platform registry. Adding a platform requires a qualification report."""

from __future__ import annotations

from .base import Platform


def get_platform(name: str) -> Platform:
    if name == "local":
        from .local import LocalPlatform

        return LocalPlatform()
    if name == "chipzen":
        from .chipzen import ChipzenExternalPlatform

        return ChipzenExternalPlatform()
    raise ValueError(f"unknown or unqualified platform {name!r}")
