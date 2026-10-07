"""Thin platform contract.

A platform owns reality: authentication, remote IDs, wire protocol, rate
limits, lifecycle and its raw event format. The common layer only sees
:class:`MatchContext` in and raw events plus a :class:`MatchSummary` out.
Nothing here knows about any particular platform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from ..model import Availability, OpponentRef
from ..spec import ExperimentSpec

#: Receives one raw platform event (a JSON-compatible dict).
Emit = Callable[[dict], None]


@dataclass(frozen=True)
class MatchContext:
    spec: ExperimentSpec
    match_index: int
    opponent: OpponentRef
    #: Seeds the platform may control; uncontrolled ones are reported back.
    seeds: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MatchSummary:
    match_index: int
    hands: int
    #: Platform-assigned identifiers (match IDs etc.), never credentials.
    platform_ids: dict = field(default_factory=dict)
    #: Factors the platform could not hold fixed (live opponents, timing...).
    uncontrolled: tuple[str, ...] = ()
    ended: str = "complete"


@dataclass(frozen=True)
class PlatformStatus:
    availability: Availability
    reason: str


class Platform(ABC):
    name: str
    adapter_version: str

    @abstractmethod
    def status(self, spec: ExperimentSpec | None = None) -> PlatformStatus:
        """Whether this platform can run (this spec) now, and why not."""

    @abstractmethod
    def metadata(self) -> dict:
        """Versions and conventions recorded with every run. No secrets."""

    @abstractmethod
    def prepare(self, spec: ExperimentSpec) -> None:
        """Validate the spec's platform configuration; raise if unusable."""

    @abstractmethod
    def play_match(self, context: MatchContext, emit: Emit) -> MatchSummary:
        """Launch one match, stream its raw events to ``emit``, then close it."""

    def require_available(self, spec: ExperimentSpec) -> None:
        state = self.status(spec)
        if state.availability is not Availability.AVAILABLE:
            raise PlatformUnavailable(f"{self.name}: {state.availability.value}")


class PlatformUnavailable(RuntimeError):
    """Raised before any match starts; never after partial play."""
