"""Chipzen research adapter: gating, recording and parsing, all offline."""

import logging

import pytest
from chipzen import GameState, conformance

from sleight_of_hand.experiments.journal import RunJournal, read_raw
from sleight_of_hand.experiments.model import Availability, OpponentRef, Provenance
from sleight_of_hand.experiments.normalization import normalize
from sleight_of_hand.experiments.platforms import chipzen as adapter
from sleight_of_hand.experiments.platforms.base import MatchContext, PlatformUnavailable
from sleight_of_hand.experiments.platforms.local import build_soh
from tests.test_experiment_spec import make

CONFIG = {
    "environment": "prod",
    "match_source": "dashboard_unrated_challenge",
    "accounting_observer": True,
    "timeout_seconds": 3600,
}
ENV = (
    *adapter.RESEARCH_ENV,
    *adapter.PROBE_ENV,
    adapter.READY_FLAG,
    "CHIPZEN_TOKEN",
)


def chipzen_spec(
    provenance=Provenance.SCRIPTED_PROBE, kind="probe", key="accounting_cover"
):
    return make(
        platform="chipzen",
        platform_config=CONFIG,
        opponent_cohort=(OpponentRef("chipzen", kind, key),),
        provenance=provenance,
        metrics=("accounting",),
    )


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def credentials(env, probe=True):
    env.setenv("CHIPZEN_RESEARCH_TOKEN", "cz_extbot_researchAAAA")
    env.setenv("CHIPZEN_RESEARCH_BOT_ID", "11111111-1111-1111-1111-111111111111")
    if probe:
        env.setenv("CHIPZEN_PROBE_TOKEN", "cz_extbot_probeBBBB")
        env.setenv("CHIPZEN_PROBE_BOT_ID", "22222222-2222-2222-2222-222222222222")


def test_status_gates_every_external_boundary(clean_env):
    platform = adapter.ChipzenExternalPlatform()
    spec = chipzen_spec()
    state = platform.status(spec)
    assert state.availability is Availability.READY_FOR_CREDENTIALS
    assert "CHIPZEN_PROBE_TOKEN" in state.reason
    rated = chipzen_spec(Provenance.LIVE_RATED, "house_bot", "h1")
    assert (
        platform.status(rated).availability
        is Availability.BLOCKED_PENDING_RULE_CONFIRMATION
    )
    credentials(clean_env, probe=False)
    assert platform.status(spec).availability is Availability.READY_FOR_CREDENTIALS
    credentials(clean_env)
    assert (
        platform.status(spec).availability is Availability.BLOCKED_PENDING_USER_ACTION
    )
    clean_env.setenv(adapter.READY_FLAG, "1")
    assert platform.status(spec).availability is Availability.AVAILABLE
    clean_env.setenv("CHIPZEN_TOKEN", "cz_extbot_researchAAAA")
    assert (
        platform.status(spec).availability
        is Availability.BLOCKED_PENDING_RULE_CONFIRMATION
    )


def test_prepare_rejects_anything_but_unrated_owner_challenges():
    platform = adapter.ChipzenExternalPlatform()
    platform.prepare(chipzen_spec())
    for change in (
        {"match_source": "matchmaking_queue"},
        {"environment": "local"},
        {"timeout_seconds": 5},
    ):
        with pytest.raises(ValueError):
            platform.prepare(
                make(
                    platform="chipzen",
                    platform_config={**CONFIG, **change},
                    opponent_cohort=(
                        OpponentRef("chipzen", "probe", "accounting_cover"),
                    ),
                    provenance=Provenance.SCRIPTED_PROBE,
                )
            )
    with pytest.raises(ValueError, match="unrated"):
        platform.prepare(chipzen_spec(Provenance.LIVE_RATED, "house_bot", "h1"))


def test_no_connection_is_attempted_without_credentials(clean_env):
    called = []

    async def forbidden(*args, **kwargs):
        called.append(1)

    clean_env.setattr("chipzen.run_external_bot", forbidden)
    spec = chipzen_spec()
    with pytest.raises(PlatformUnavailable):
        adapter.ChipzenExternalPlatform().play_match(
            MatchContext(spec, 0, spec.opponent_cohort[0], {"soh": 1, "opponent": 2}),
            lambda e: None,
        )
    assert called == []


def drive(bot, log=True):
    """Feed the SDK's own conformance messages through the hook sequence."""
    if log:
        logging.getLogger("chipzen.external").info(adapter.MATCHED_FORMAT, "m-1", False)
    bot.on_match_start(conformance._match_start())
    bot.on_round_start(conformance._round_start())
    action = bot.decide(GameState.from_turn_request(conformance._turn_request()))
    bot.on_turn_result(conformance._turn_result())
    bot.on_round_result(conformance._round_result())
    bot.on_match_end(conformance._match_end())
    return action


def test_recorder_preserves_decisions_and_parses_into_the_common_schema(tmp_path):
    plain = drive(build_soh({}, 5))
    events = []
    recorder = adapter.Recorder(
        build_soh({}, 5), lambda e: events.append({"match": 0, **e})
    )
    recorded = drive(recorder.wrap())
    assert (recorded.action, recorded.amount) == (plain.action, plain.amount)
    assert recorder.rounds == 1
    kinds = [e["kind"] for e in events]
    assert kinds.count("match_meta") == 1 and kinds.count("action") == 1
    request = next(
        e["message"]
        for e in events
        if e["kind"] == "message" and e["message"].get("type") == "turn_request"
    )
    assert request["reconstructed_from"] == "sdk_gamestate"
    normalized = normalize(events)
    assert len(normalized["decisions"]) == 1 and len(normalized["hands"]) == 1


def test_recorded_messages_are_redacted_in_the_journal(tmp_path, clean_env):
    credentials(clean_env)
    j = RunJournal.create(tmp_path / "r", {"run_id": "r"})
    message = conformance._match_start()
    message["session_token"] = "anything"
    message["echo"] = "cz_extbot_researchAAAA"
    j.append({"match": 0, "kind": "message", "to": [0], "message": message})
    j.seal_raw()
    stored = read_raw(tmp_path / "r")[0]["message"]
    assert stored["session_token"] == "[REDACTED]" and stored["echo"] == "[REDACTED]"


MATCH = conformance._match_end()["match_id"]
SOH_TOKEN, PROBE_TOKEN = "cz_extbot_researchAAAA", "cz_extbot_probeBBBB"


def log_matched(match_id=MATCH, rated=False):
    logging.getLogger("chipzen.external").info(adapter.MATCHED_FORMAT, match_id, rated)


def play(clean_env, behaviors, opponent="probe"):
    """Run play_match against a fake SDK; ``behaviors`` maps token -> callable.

    Each callable receives the bot, may log ``matched`` evidence, and returns
    the session result list (default: one clean result for ``MATCH``).
    """
    credentials(clean_env)
    clean_env.setenv(adapter.READY_FLAG, "1")
    seen = []

    async def fake_run_external_bot(bot, **kwargs):
        assert kwargs["max_matches"] == 1 and kwargs["token"].startswith("cz_extbot_")
        seen.append(kwargs["token"])
        returned = behaviors[kwargs["token"]](bot)
        if kwargs["token"] == SOH_TOKEN:
            drive(bot, log=False)
        return [{"match_id": MATCH, "end": {}}] if returned is None else returned

    clean_env.setattr("chipzen.run_external_bot", fake_run_external_bot)
    if opponent == "probe":
        spec = chipzen_spec()
    else:
        spec = chipzen_spec(Provenance.LIVE_UNRATED, "house_bot", "h1")
    events = []
    context = MatchContext(spec, 0, spec.opponent_cohort[0], {"soh": 1, "opponent": 2})
    summary = adapter.ChipzenExternalPlatform().play_match(context, events.append)
    return summary, events, seen


def unrated(bot):
    log_matched()


def test_probe_mode_with_both_workers_explicitly_unrated_is_allowed(clean_env):
    summary, events, seen = play(clean_env, {SOH_TOKEN: unrated, PROBE_TOKEN: unrated})
    assert sorted(seen) == sorted([SOH_TOKEN, PROBE_TOKEN])
    assert summary.hands == 1 and summary.uncontrolled
    assert summary.platform_ids == {"matches": [MATCH]}
    lobby = [e for e in events if e["kind"] == "lobby_matched"]
    assert sorted(lobby[0]["matched"], key=lambda m: m["worker"]) == [
        {"worker": "probe", "match_id": MATCH, "rated": False},
        {"worker": "soh", "match_id": MATCH, "rated": False},
    ]


def test_house_bot_mode_needs_only_the_soh_worker(clean_env):
    summary, _, seen = play(clean_env, {SOH_TOKEN: unrated}, opponent="house_bot")
    assert seen == [SOH_TOKEN]
    assert summary.platform_ids == {"matches": [MATCH]}


def silent(bot):
    return None


def logs(*calls):
    def behavior(bot):
        for args in calls:
            log_matched(*args)

    return behavior


def raw_log(msg, *args):
    def behavior(bot):
        logging.getLogger("chipzen.external").info(msg, *args)

    return behavior


REJECTED = {
    "rated_true": ({SOH_TOKEN: logs((MATCH, True)), PROBE_TOKEN: unrated}, "unrated"),
    "rated_none": ({SOH_TOKEN: logs((MATCH, None)), PROBE_TOKEN: unrated}, "unrated"),
    "rated_string_false": (
        {SOH_TOKEN: logs((MATCH, "False")), PROBE_TOKEN: unrated},
        "unrated",
    ),
    "rated_zero": ({SOH_TOKEN: logs((MATCH, 0)), PROBE_TOKEN: unrated}, "unrated"),
    "zero_records": ({SOH_TOKEN: silent, PROBE_TOKEN: silent}, "no matched"),
    "probe_evidence_missing": ({SOH_TOKEN: unrated, PROBE_TOKEN: silent}, "probe"),
    "soh_evidence_missing": ({SOH_TOKEN: silent, PROBE_TOKEN: unrated}, "soh"),
    "probe_rated_true": (
        {SOH_TOKEN: unrated, PROBE_TOKEN: logs((MATCH, True))},
        "probe",
    ),
    "duplicate_records": (
        {SOH_TOKEN: logs((MATCH, False), (MATCH, False)), PROBE_TOKEN: unrated},
        "expected one",
    ),
    "contradictory_records": (
        {SOH_TOKEN: logs((MATCH, False), (MATCH, True)), PROBE_TOKEN: unrated},
        "expected one",
    ),
    "changed_format": (
        {
            SOH_TOKEN: raw_log("lobby: matched => %s rated=%s", MATCH, False),
            PROBE_TOKEN: unrated,
        },
        "malformed",
    ),
    "missing_args": (
        {
            SOH_TOKEN: raw_log("lobby: matched -> match (rated unknown)"),
            PROBE_TOKEN: unrated,
        },
        "malformed",
    ),
    "empty_match_id": (
        {SOH_TOKEN: logs(("", False)), PROBE_TOKEN: logs(("", False))},
        "match ID",
    ),
    "different_matches": (
        {
            SOH_TOKEN: unrated,
            PROBE_TOKEN: lambda bot: (
                log_matched("m-other"),
                [{"match_id": "m-other"}],
            )[1],
        },
        "different matches",
    ),
    "session_result_mismatch": (
        {
            SOH_TOKEN: lambda bot: (log_matched(), [{"match_id": "m-other"}])[1],
            PROBE_TOKEN: unrated,
        },
        "session result",
    ),
    "session_result_failed": (
        {
            SOH_TOKEN: lambda bot: (
                log_matched(),
                [{"match_id": None, "reason": "exception"}],
            )[1],
            PROBE_TOKEN: unrated,
        },
        "session result",
    ),
    "session_result_empty": (
        {SOH_TOKEN: lambda bot: (log_matched(), [])[1], PROBE_TOKEN: unrated},
        "session result",
    ),
}


@pytest.mark.parametrize("case", sorted(REJECTED))
def test_missing_or_inconsistent_rating_evidence_fails_closed(clean_env, case):
    behaviors, reason = REJECTED[case]
    with pytest.raises(RuntimeError, match=reason):
        play(clean_env, behaviors)


def test_unattributed_matched_record_is_rejected():
    handler = adapter._MatchedLog()
    record = logging.LogRecord(
        "chipzen.external",
        logging.INFO,
        "x",
        1,
        adapter.MATCHED_FORMAT,
        (MATCH, False),
        None,
    )
    handler.emit(record)  # logged outside any research worker's context
    assert handler.matched == [{"worker": None, "match_id": MATCH, "rated": False}]
    with pytest.raises(RuntimeError, match="unknown worker"):
        adapter.verify_unrated(handler.matched, {"soh": [{"match_id": MATCH}]})


def test_matched_handler_ignores_non_string_and_unrelated_messages():
    handler = adapter._MatchedLog()
    for msg, args in (
        (RuntimeError("boom"), ()),
        ({"type": "matched"}, ()),
        ("match %s: connecting gateway", (MATCH,)),
    ):
        handler.emit(
            logging.LogRecord("chipzen.external", logging.INFO, "x", 1, msg, args, None)
        )
    assert handler.matched == []
    with pytest.raises(RuntimeError, match="no matched"):
        adapter.verify_unrated(handler.matched, {"soh": [{"match_id": MATCH}]})


def test_probe_bot_is_a_chipzen_bot_with_its_own_identity():
    from chipzen import Bot

    probe = adapter.make_probe_bot(3, 200)
    assert isinstance(probe, Bot)
    action = drive(probe)
    assert action.action in {"fold", "check", "call", "raise"}
