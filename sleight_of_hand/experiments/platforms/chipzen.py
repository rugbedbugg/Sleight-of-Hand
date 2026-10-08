"""Chipzen research worker on the official external-API remote-play path.

Uses the SDK's documented ``run_external_bot`` (lobby -> ``matched`` ->
per-match gateway). The research identity is separate from production: it
reads only ``CHIPZEN_RESEARCH_*`` (and, for the accounting probe,
``CHIPZEN_PROBE_*``) and refuses to run if a research token equals the
production ``CHIPZEN_TOKEN``. It never uploads images, never joins rated
queues or tournaments, and never creates accounts, bots or tokens.

Matches come only from an *unrated* challenge created by the account owner
in the Chipzen dashboard (the documented route for same-owner matches, which
are never rated). The owner signals that the challenge exists by setting
``CHIPZEN_RESEARCH_CHALLENGE_READY=1``. The ``matched.rated`` flag is
recorded from the SDK's structured log record, attributed to the worker that
received it. The check fails closed: the run fails unless every connected
worker logged exactly one well-formed ``matched`` record with ``rated`` exactly
``False``, all for the one match the SDK session returned.

Raw evidence is every message the SDK delivers to SOH's hooks. The SDK does
not expose raw ``turn_request`` frames, so decision requests are rebuilt
from the SDK-parsed ``GameState`` and flagged as such.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import logging
import os

from ..model import Availability, Provenance
from ..spec import ExperimentSpec
from .base import Emit, MatchContext, MatchSummary, Platform, PlatformStatus
from .local import AccountingProbe, build_soh

ADAPTER_VERSION = "chipzen-external/1"
RESEARCH_ENV = ("CHIPZEN_RESEARCH_TOKEN", "CHIPZEN_RESEARCH_BOT_ID")
PROBE_ENV = ("CHIPZEN_PROBE_TOKEN", "CHIPZEN_PROBE_BOT_ID")
READY_FLAG = "CHIPZEN_RESEARCH_CHALLENGE_READY"
CONFIG_KEYS = {"environment", "match_source", "accounting_observer", "timeout_seconds"}
ALLOWED_PROVENANCE = {Provenance.LIVE_UNRATED, Provenance.SCRIPTED_PROBE}
MATCHED_PREFIX = "lobby: matched"
MATCHED_FORMAT = "lobby: matched -> match %s (rated=%s); playing"
_WORKER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "chipzen_research_worker", default=None
)


def _missing(names) -> list[str]:
    return [n for n in names if not os.environ.get(n)]


def state_dict(state) -> dict:
    """A turn_request-shaped record rebuilt from an SDK ``GameState``."""
    return {
        "type": "turn_request",
        "reconstructed_from": "sdk_gamestate",
        "round_id": state.round_id,
        "request_id": state.request_id,
        "valid_actions": list(state.valid_actions),
        "state": {
            "hand_number": state.hand_number,
            "phase": state.phase,
            "board": [str(c) for c in state.board],
            "your_hole_cards": [str(c) for c in state.hole_cards],
            "pot": state.pot,
            "your_stack": state.your_stack,
            "opponent_stacks": list(state.opponent_stacks),
            "your_seat": state.your_seat,
            "dealer_seat": state.dealer_seat,
            "to_call": state.to_call,
            "min_raise": state.min_raise,
            "max_raise": state.max_raise,
            "action_history": copy.deepcopy(list(state.action_history)),
        },
    }


class Recorder:
    """Wraps a Chipzen bot; journals every delivered message, then delegates."""

    HOOKS = (
        "on_match_start",
        "on_round_start",
        "on_turn_result",
        "on_phase_change",
        "on_round_result",
        "on_match_end",
        "on_reconnected",
    )

    def __init__(self, bot, emit: Emit):
        self.bot, self.emit = bot, emit
        self.seat = None
        self.rounds = 0

    def wrap(self):
        recorder, bot = self, self.bot
        for hook in self.HOOKS:
            original = getattr(bot, hook)

            def hooked(message, _original=original, _hook=hook):
                if _hook in ("on_match_start", "on_reconnected"):
                    for seat in (message or {}).get("seats", []) or []:
                        if isinstance(seat, dict) and seat.get("is_self"):
                            recorder.seat = seat.get("seat")
                            recorder.emit(
                                {"kind": "match_meta", "soh_seat": recorder.seat}
                            )
                recorder.emit(
                    {"kind": "message", "to": [recorder.seat], "message": message}
                )
                if _hook == "on_round_result":
                    recorder.rounds += 1
                return _original(message)

            setattr(bot, hook, hooked)
        decide = bot.decide

        def recorded_decide(state):
            recorder.emit(
                {"kind": "message", "to": [recorder.seat], "message": state_dict(state)}
            )
            action = decide(state)
            recorder.emit(
                {
                    "kind": "action",
                    "seat": recorder.seat,
                    "action": action.action,
                    "amount": action.amount if action.action == "raise" else 0,
                }
            )
            return action

        bot.decide = recorded_decide
        return bot


class _MatchedLog(logging.Handler):
    """Reads the SDK's structured ``matched`` log record (match_id, rated).

    Each record is attributed to the worker whose task logged it. A record
    that mentions ``matched`` but does not have the expected shape is kept
    as malformed, so a changed SDK format fails the run instead of vanishing.
    """

    def __init__(self):
        super().__init__(logging.INFO)
        self.matched: list[dict] = []

    def emit(self, record):
        msg, args = record.msg, record.args
        if not isinstance(msg, str) or not msg.startswith(MATCHED_PREFIX):
            return
        entry: dict = {"worker": _WORKER.get()}
        if msg != MATCHED_FORMAT or not isinstance(args, tuple) or len(args) != 2:
            entry["malformed"] = True
        else:
            entry["match_id"], entry["rated"] = args
        self.matched.append(entry)


def verify_unrated(matched: list[dict], results: dict[str, list]) -> str:
    """Return the one match ID only on positive, consistent unrated evidence.

    ``results`` maps each connected worker to its ``run_external_bot`` result.
    Absent, malformed, unattributed, duplicate, ``None``/``True``-rated or
    mismatched evidence raises; nothing is ever inferred from missing data.
    """
    if not matched:
        raise RuntimeError("no matched notification was observed; rating unknown")
    unknown = [m for m in matched if m.get("worker") not in results]
    if unknown:
        raise RuntimeError("a matched notification came from an unknown worker")
    match_ids = set()
    for worker, worker_results in results.items():
        records = [m for m in matched if m["worker"] == worker]
        if len(records) != 1:
            raise RuntimeError(
                f"{worker}: expected one matched notification, saw {len(records)}"
            )
        record = records[0]
        if record.get("malformed"):
            raise RuntimeError(f"{worker}: matched notification was malformed")
        if record["rated"] is not False:
            raise RuntimeError(
                f"{worker}: matched notification was not explicitly unrated"
            )
        match_id = record["match_id"]
        if not isinstance(match_id, str) or not match_id:
            raise RuntimeError(f"{worker}: matched notification had no match ID")
        returned = [
            r.get("match_id") if isinstance(r, dict) else None
            for r in worker_results or []
        ]
        if returned != [match_id]:
            raise RuntimeError(
                f"{worker}: session result does not match the matched notification"
            )
        match_ids.add(match_id)
    if len(match_ids) != 1:
        raise RuntimeError("workers were matched into different matches")
    return match_ids.pop()


def make_probe_bot(seed: int, margin: int):
    """The scripted accounting probe as a Chipzen bot (its own identity)."""
    import random

    from chipzen import Bot

    class ProbeBot(AccountingProbe, Bot):
        pass

    return ProbeBot(random.Random(seed), margin)


class ChipzenExternalPlatform(Platform):
    name = "chipzen"
    adapter_version = ADAPTER_VERSION

    def status(self, spec: ExperimentSpec | None = None) -> PlatformStatus:
        if spec is not None and spec.provenance not in ALLOWED_PROVENANCE:
            return PlatformStatus(
                Availability.BLOCKED_PENDING_RULE_CONFIRMATION,
                f"{spec.provenance.value} play needs separate explicit approval",
            )
        missing = _missing(RESEARCH_ENV)
        if spec is not None and any(o.kind == "probe" for o in spec.opponent_cohort):
            missing += _missing(PROBE_ENV)
        if missing:
            return PlatformStatus(
                Availability.READY_FOR_CREDENTIALS,
                "dashboard-created external-API research bot(s) and tokens needed: "
                + ", ".join(missing),
            )
        production = os.environ.get("CHIPZEN_TOKEN")
        if production and production in (
            os.environ.get("CHIPZEN_RESEARCH_TOKEN"),
            os.environ.get("CHIPZEN_PROBE_TOKEN"),
        ):
            return PlatformStatus(
                Availability.BLOCKED_PENDING_RULE_CONFIRMATION,
                "research token equals the production token; refusing",
            )
        if os.environ.get(READY_FLAG) != "1":
            return PlatformStatus(
                Availability.BLOCKED_PENDING_USER_ACTION,
                "create an unrated same-owner challenge in the Chipzen dashboard, "
                f"then set {READY_FLAG}=1",
            )
        return PlatformStatus(Availability.AVAILABLE, "credentials and challenge ready")

    def metadata(self) -> dict:
        import chipzen

        return {
            "adapter_version": ADAPTER_VERSION,
            "chipzen_sdk": chipzen.__version__,
            "path": "official external-API remote play (run_external_bot)",
            "raw_events": "SDK-delivered hook messages; decisions rebuilt from GameState",
            "match_source": "owner-created unrated dashboard challenge",
            "uncontrolled": [
                "server deck and seating",
                "opponent behavior (unless probe)",
                "network timing",
                "round_start stack timing (pre/post blinds) is platform-defined",
            ],
        }

    def prepare(self, spec: ExperimentSpec) -> None:
        config = spec.platform_config
        if set(config) != CONFIG_KEYS:
            raise ValueError(f"chipzen config needs exactly {sorted(CONFIG_KEYS)}")
        if config["environment"] not in {"prod", "staging"}:
            raise ValueError("environment must be prod or staging")
        if config["match_source"] != "dashboard_unrated_challenge":
            raise ValueError("only owner-created unrated challenges are supported")
        if type(config["accounting_observer"]) is not bool:
            raise ValueError("accounting_observer must be a boolean")
        if (
            type(config["timeout_seconds"]) is not int
            or not 60 <= config["timeout_seconds"] <= 7200
        ):
            raise ValueError("timeout_seconds must be 60..7200")
        if spec.provenance not in ALLOWED_PROVENANCE:
            raise ValueError("chipzen research runs are unrated only")
        if not all(o.kind in {"house_bot", "probe"} for o in spec.opponent_cohort):
            raise ValueError("cohort must be house bots or the accounting probe")

    def play_match(self, context: MatchContext, emit: Emit) -> MatchSummary:
        from chipzen import run_external_bot

        self.require_available(context.spec)
        config = context.spec.platform_config
        recorder = Recorder(
            build_soh(
                context.spec.policy_config,
                context.seeds["soh"],
                observer=config["accounting_observer"],
            ),
            emit,
        )
        soh = recorder.wrap()
        if config["accounting_observer"]:
            _capture_observer_output(soh, emit)
        handler = _MatchedLog()
        sdk_log = logging.getLogger("chipzen.external")
        sdk_log.addHandler(handler)
        previous = sdk_log.level
        sdk_log.setLevel(logging.INFO)

        workers = {
            "soh": (
                soh,
                os.environ["CHIPZEN_RESEARCH_BOT_ID"],
                os.environ["CHIPZEN_RESEARCH_TOKEN"],
            )
        }
        if context.opponent.kind == "probe":
            workers["probe"] = (
                make_probe_bot(context.seeds["opponent"], margin=200),
                os.environ["CHIPZEN_PROBE_BOT_ID"],
                os.environ["CHIPZEN_PROBE_TOKEN"],
            )

        async def worker(name, bot, bot_id, token):
            _WORKER.set(name)  # this task's context; SDK subtasks inherit it
            return await run_external_bot(
                bot,
                bot_id=bot_id,
                token=token,
                env=config["environment"],
                max_matches=1,
            )

        async def session():
            jobs = [worker(name, *args) for name, args in workers.items()]
            return await asyncio.wait_for(
                asyncio.gather(*jobs), timeout=config["timeout_seconds"]
            )

        try:
            results = asyncio.run(session())
        finally:
            sdk_log.removeHandler(handler)
            sdk_log.setLevel(previous)
        emit({"kind": "lobby_matched", "matched": handler.matched})
        match_id = verify_unrated(handler.matched, dict(zip(workers, results)))
        return MatchSummary(
            match_index=context.match_index,
            hands=recorder.rounds,
            platform_ids={"matches": [match_id]},
            uncontrolled=tuple(self.metadata()["uncontrolled"]),
        )


def _capture_observer_output(bot, emit: Emit) -> None:
    """Journal the observer's stdout frames exactly as a log reader would."""
    import io
    import sys

    original = bot.on_round_result

    def captured(message):
        buffer = io.StringIO()
        real = sys.stdout
        sys.stdout = buffer
        try:
            return original(message)
        finally:
            sys.stdout = real
            if buffer.getvalue():
                emit(
                    {"kind": "seat_output", "seat": "soh", "stdout": buffer.getvalue()}
                )

    bot.on_round_result = captured
