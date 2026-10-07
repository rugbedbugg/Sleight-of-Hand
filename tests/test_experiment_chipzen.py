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


def drive(bot, rated=False):
    """Feed the SDK's own conformance messages through the hook sequence."""
    logging.getLogger("chipzen.external").info(
        "lobby: matched -> match %s (rated=%s); playing", "m-1", rated
    )
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


@pytest.mark.parametrize("rated", [False, True, None])
def test_play_match_records_lobby_rating_and_refuses_rated(clean_env, rated):
    credentials(clean_env)
    clean_env.setenv(adapter.READY_FLAG, "1")

    async def fake_run_external_bot(bot, **kwargs):
        assert kwargs["max_matches"] == 1 and kwargs["token"].startswith("cz_extbot_")
        drive(bot, rated=rated)
        return [{"match_id": "m-1", "end": {}}]

    clean_env.setattr("chipzen.run_external_bot", fake_run_external_bot)
    spec = chipzen_spec()
    events = []
    platform = adapter.ChipzenExternalPlatform()
    context = MatchContext(spec, 0, spec.opponent_cohort[0], {"soh": 1, "opponent": 2})
    if rated is False:
        summary = platform.play_match(context, events.append)
        assert summary.hands == 1 and summary.uncontrolled
        lobby = [e for e in events if e["kind"] == "lobby_matched"]
        assert lobby[0]["matched"] == [{"match_id": "m-1", "rated": False}] * 2
    else:
        with pytest.raises(RuntimeError, match="unrated"):
            platform.play_match(context, events.append)


def test_probe_bot_is_a_chipzen_bot_with_its_own_identity():
    from chipzen import Bot

    probe = adapter.make_probe_bot(3, 200)
    assert isinstance(probe, Bot)
    action = drive(probe)
    assert action.action in {"fold", "check", "call", "raise"}
