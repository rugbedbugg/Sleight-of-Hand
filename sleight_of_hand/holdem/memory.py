"""Policy-inert opponent evidence. No clocks, RNG, logging or implicit storage."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .observations import preflop_observations
from .profile_store import NullProfileStore, OpponentProfileStore
from .profiles import (
    TENDENCIES,
    BeliefConfig,
    Counts,
    HistoricalOpponentProfile,
    IdentityResolution,
    OpponentIdentity,
    identifier,
    resolve_match_identity,
)

MAX_PENDING_STARTS = 16


@dataclass(frozen=True)
class OpponentBelief:
    historical: Counts
    current: Counts
    config: BeliefConfig

    @property
    def historical_ess(self) -> float:
        return min(self.historical.opportunities, self.config.historical_ess_cap)

    @property
    def alpha(self) -> float:
        h = self.historical
        weight = self.historical_ess / h.opportunities if h.opportunities else 0.0
        return self.config.alpha_base + weight * h.successes + self.current.successes

    @property
    def beta(self) -> float:
        h = self.historical
        weight = self.historical_ess / h.opportunities if h.opportunities else 0.0
        return (
            self.config.beta_base
            + weight * (h.opportunities - h.successes)
            + self.current.opportunities
            - self.current.successes
        )

    @property
    def effective_sample_size(self) -> float:
        """Evidence ESS; excludes the separately reported base pseudocounts."""
        return self.historical_ess + self.current.opportunities

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return a * b / ((a + b) ** 2 * (a + b + 1))

    def interval(self, coverage: float = 0.95) -> tuple[float, float]:
        """Conservative Chebyshev posterior bound, NOT equal-tailed quantiles.

        Has at least requested posterior mass under the weighted Beta model;
        no claim of empirical frequentist coverage or shift probability.
        """
        if type(coverage) not in (int, float) or not 0 < coverage < 1:
            raise ValueError("coverage must lie in (0, 1)")
        radius = math.sqrt(self.variance / (1 - coverage))
        return max(0.0, self.mean - radius), min(1.0, self.mean + radius)

    def to_dict(self) -> dict:
        return {
            "historical": asdict(self.historical),
            "current": asdict(self.current),
            "config": asdict(self.config),
            "historical_ess": self.historical_ess,
            "evidence_ess": self.effective_sample_size,
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "variance": self.variance,
            "conservative_95_interval": self.interval(),
        }


class OpponentMemory:
    """Independent match state; explicit lifecycle loading, never autosaving.

    The default resolver conservatively treats transport participant IDs as
    match-scoped. A trusted external resolver must provide explicit persistent
    key provenance before any store lookup is allowed.
    """

    def __init__(
        self,
        store: OpponentProfileStore | None = None,
        config: BeliefConfig | None = None,
        identity_resolver: Callable[
            [str, str, object], IdentityResolution
        ] = resolve_match_identity,
    ):
        self.store = store if store is not None else NullProfileStore()
        self.config = config if config is not None else BeliefConfig()
        self.identity_resolver = identity_resolver
        self.match_key: str | None = None
        self.identity = OpponentIdentity("unknown")
        self.resolution_reason = "not_started"
        self.prior: HistoricalOpponentProfile | None = None
        self.current: dict[str, Counts] = {}
        self.processed: set[str] = set()
        self.rejected: set[str] = set()
        self.starts: dict[str, dict] = {}
        self.ended = False
        self.identity_conflict = False
        self._anonymous_matches = 0

    def begin_match(self, key: str, resolution: IdentityResolution) -> None:
        identifier(key)
        identity = resolution.identity
        if self.match_key == key:
            # Repeated starts/reconnects cannot reload or count history twice.
            if identity.key is None:
                return
            if self.identity.to_dict() == identity.to_dict():
                self.identity = identity  # rename is metadata only
                return
            # Never attach already-collected evidence to a different identity.
            self.identity_conflict = True
            self.prior = None
            self.current = {}
            self.processed = set()
            self.rejected = set()
            self.starts = {}
            self.resolution_reason = "identity_changed_within_match"
            return
        prior = None
        reason = resolution.reason
        if identity.persistent_key is not None:
            try:
                loaded = self.store.load(identity)
                if loaded is not None:
                    checked = HistoricalOpponentProfile.from_dict(loaded.to_dict())
                    if checked.identity.persistent_key == identity.persistent_key:
                        prior = checked
                    else:
                        reason = "profile_identity_mismatch"
            except Exception:  # noqa: BLE001 - optional storage is fail-closed
                reason = "profile_unavailable"
        self.match_key, self.identity, self.prior = key, identity, prior
        self.resolution_reason = reason
        self.current, self.processed, self.starts = {}, set(), {}
        self.rejected = set()
        self.ended = self.identity_conflict = False

    def record_start(self, key: str | None, state: object) -> None:
        try:
            identifier(key)
        except ValueError:
            return
        if (
            self.match_key is None
            or self.ended
            or self.identity_conflict
            or key in self.processed
            or key in self.rejected
        ):
            return
        if type(state) is not dict:
            return
        try:
            identifier(key)
        except ValueError:
            return
        stacks = state.get("stacks")
        if (
            type(stacks) is not list
            or len(stacks) != 2
            or any(type(x) is not int or x <= 0 for x in stacks)
        ):
            return
        clean = {"stacks": list(stacks)}
        if key in self.starts:
            if self.starts[key] != clean:
                # A conflicting duplicate invalidates this hand, not the match.
                self.starts.pop(key)
                self.rejected.add(key)
            return
        self.starts[key] = clean
        while len(self.starts) > MAX_PENDING_STARTS:
            self.starts.pop(next(iter(self.starts)))

    def observe_result(self, key: str | None, result: object, hero: int | None) -> bool:
        try:
            identifier(key)
        except ValueError:
            return False
        if self.ended or self.identity_conflict or key is None or key in self.processed:
            return False
        start = self.starts.get(key)
        if start is None or type(result) is not dict:
            return False
        observations = preflop_observations(start, result, key, hero)
        if observations is None:
            return False  # no partial delta, permits later complete redelivery
        updated = dict(self.current)
        for tendency, count in observations:
            updated[tendency] = updated.get(tendency, Counts()) + count
        self.current = updated
        self.processed.add(key)
        self.starts.pop(key)
        return True

    def belief(self, tendency: str) -> OpponentBelief:
        if tendency not in TENDENCIES:
            raise ValueError("unknown tendency")
        historical = (
            dict(self.prior.stats).get(tendency, Counts()) if self.prior else Counts()
        )
        return OpponentBelief(
            historical, self.current.get(tendency, Counts()), self.config
        )

    def snapshot(self) -> dict:
        """Explicit research diagnostics only; no names, cards or raw messages."""
        names = sorted(
            set(self.current) | (set(dict(self.prior.stats)) if self.prior else set())
        )
        return {
            "schema_version": 1,
            "identity": self.identity.to_dict(),
            "resolution": self.resolution_reason,
            "match_key": self.match_key,
            "prior_hands": self.prior.total_hands if self.prior else 0,
            "prior_provenance": asdict(self.prior.provenance) if self.prior else None,
            "current_match_hands": len(self.processed),
            "config": asdict(self.config),
            "tendencies": {k: self.belief(k).to_dict() for k in names},
            "ended": self.ended,
            "identity_conflict": self.identity_conflict,
        }

    def notify(self, event: str, message: object, hero: int | None) -> None:
        """Mapping adapter. Only completed rounds update tendencies.

        turn_result deliberately adds no evidence (no double counting and no
        inference from incomplete action prefixes). No decide hook exists.
        """
        if type(message) is not dict:
            return
        if event in {"match_start", "reconnected"}:
            key = message.get("match_id")
            if key is None:
                if self.match_key is not None and not self.ended:
                    key = self.match_key
                else:
                    self._anonymous_matches += 1
                    key = f"anonymous-match-{self._anonymous_matches}"
            try:
                identifier(key)
            except ValueError:
                return
            resolved = self.identity_resolver("chipzen", key, message.get("seats"))
            self.begin_match(key, resolved)
        elif event == "match_end":
            if message.get("match_id", self.match_key) == self.match_key:
                self.ended = True
                self.starts = {}
        elif event in {"round_start", "round_result"}:
            if message.get("match_id", self.match_key) != self.match_key:
                return
            key = message.get("round_id")
            value = message.get("state" if event == "round_start" else "result")
            if key is None and type(value) is dict:
                hand = value.get("hand_number")
                if type(hand) is int and hand > 0:
                    key = f"hand:{hand}"
            if event == "round_start":
                self.record_start(key, value)
            else:
                self.observe_result(key, value, hero)
