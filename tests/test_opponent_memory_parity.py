"""Canonical 3f8291d oracle: never normalize away a policy/lifecycle mismatch."""

import ast
import copy
import gzip
import io
import json
import os
import subprocess
import sys
import tarfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from itertools import zip_longest
from pathlib import Path
from unittest.mock import patch

BASE = "3f8291d9b1ba7c6ac7b9cde99f046d0478fb54fe"


def canonical_records(stream):
    # Imports are intentionally local: this driver runs under archived main too.
    from chipzen import Action, GameState

    from bots.chipzen.bot import SleightOfHandBot
    from sleight_of_hand.holdem.hands import CLASSES
    from tests.test_chipzen import state
    from tests.test_preflop import sdk_state, shove
    from tests.test_preflop_gate import MATRIX, fallback_states
    from tests.test_shove_model import MALFORMED, report

    class Sink:
        def append(self, record):
            stream.write(json.dumps({"canonical": record}) + "\n")

    records = Sink()
    memory_hands = 0

    def invoke(bot, method, argument):
        nonlocal memory_hands
        err, out = io.StringIO(), io.StringIO()
        with redirect_stderr(err), redirect_stdout(out):
            try:
                value = getattr(bot, method)(copy.deepcopy(argument))
                if isinstance(value, Action):
                    result = {
                        "type": type(value).__module__ + "." + type(value).__qualname__,
                        "fields": asdict(value),
                    }
                else:
                    result = value
                exception = None
            except Exception as exc:  # noqa: BLE001 - exceptions are oracle outputs
                result = None
                exception = (
                    type(exc).__module__ + "." + type(exc).__qualname__,
                    str(exc),
                )
        model = bot.shove_model
        observer = bot._accounting_observer
        records.append(
            copy.deepcopy(
                {
                    "method": method,
                    "result": result,
                    "exception": exception,
                    "stderr": err.getvalue(),
                    "stdout": out.getvalue(),
                    "rng": bot.rng.getstate(),
                    "seat": bot._seat,
                    "warned": bot._warned_preflop,
                    "model": {
                        "stats": [(k, asdict(v)) for k, v in model.stats.items()],
                        "processed": sorted(model.processed),
                        "starts": list(model._starts.items()),
                        "config": asdict(model.config),
                        "prior_strength": model.prior_strength,
                    },
                    "accounting": vars(observer) if observer else None,
                }
            )
        )
        memory = getattr(bot, "opponent_memory", None)
        if memory is not None:
            memory_hands = max(memory_hands, memory.snapshot()["current_match_hands"])

    for trace in (False, True):
        bot = SleightOfHandBot(seed=734, samples=8, trace_preflop=trace)
        for _, spot in MATRIX:
            for label in CLASSES:
                hole = [
                    label[0] + "s",
                    label[1] + ("s" if label.endswith("s") else "h"),
                ]
                invoke(bot, "decide", sdk_state(spot, hole))
        for observer_enabled in (False, True):
            if observer_enabled:
                os.environ["SLEIGHT_CHIPZEN_ACCOUNTING_OBSERVER"] = "1"
                os.environ["SLEIGHT_CHIPZEN_ACCOUNTING_STDOUT"] = "1"
            else:
                os.environ.pop("SLEIGHT_CHIPZEN_ACCOUNTING_OBSERVER", None)
                os.environ.pop("SLEIGHT_CHIPZEN_ACCOUNTING_STDOUT", None)
            os.environ.pop("SLEIGHT_CHIPZEN_ACCOUNTING_LOG", None)
            bot = SleightOfHandBot(seed=17, samples=8, trace_preflop=trace)
            match = {
                "match_id": "oracle",
                "seats": [
                    {"seat": 0, "is_self": True},
                    {"seat": 1, "participant_id": "p-1", "display_name": "metadata"},
                ],
            }
            invoke(bot, "on_match_start", match)
            invoke(bot, "on_match_start", match)
            for spot in [
                sdk_state(shove(14), ["Qs", "Jh"]),
                *fallback_states().values(),
            ]:
                invoke(bot, "decide", spot)
            for number in range(1, 25):
                button = number % 2
                voluntary = (
                    [(1, "raise", 5000), (0, "fold", 0)]
                    if button
                    else [(0, "raise", 250), (1, "raise", 5000), (0, "call", 0)]
                )
                start, result = report.hand_messages(
                    number, button, (5000, 5000), voluntary
                )
                invoke(bot, "on_round_start", start)
                invoke(bot, "decide", sdk_state(shove(50), ["Kh", "Td"]))
                invoke(
                    bot,
                    "on_turn_result",
                    {"details": {"seat": 1, "action": "raise", "amount": 5000}},
                )
                invoke(bot, "on_round_result", result)
                invoke(bot, "on_round_result", result)
            invoke(bot, "on_reconnected", match)
            # Pending-start overflow, out-of-order results, malformed lifecycles.
            for number in range(30, 50):
                start, _ = report.hand_messages(
                    number, 1, (5000, 5000), [(1, "raise", 5000), (0, "fold", 0)]
                )
                invoke(bot, "on_round_start", start)
            for number in (49, 32, 40, 49):
                _, result = report.hand_messages(
                    number, 1, (5000, 5000), [(1, "raise", 5000), (0, "fold", 0)]
                )
                invoke(bot, "on_round_result", result)
            for fault in MALFORMED.values():
                start, result = report.hand_messages(
                    80, 1, (5000, 5000), [(1, "raise", 5000), (0, "fold", 0)]
                )
                start, result = fault(start, result)
                if start is not None:
                    invoke(bot, "on_round_start", start)
                invoke(bot, "on_round_result", result)
            for value in (
                None,
                {},
                "bad",
                {"state": None},
                {"state": {"your_hole_cards": ["bad"]}},
            ):
                invoke(bot, "on_round_start", value)
                invoke(bot, "on_round_result", value)
            for history in (None, [None], ["bad"], [{"seat": [], "action": "fold"}]):
                for phase in ("preflop", "river"):
                    invoke(bot, "decide", state(phase=phase, action_history=history))
            for valid in (
                [],
                ["draw"],
                ["check"],
                ["call"],
                ["raise"],
                ["all_in"],
                ["fold", "call", "raise"],
            ):
                for cards in ([], ["As", "As"], ["As", "Kh"]):
                    invoke(bot, "decide", state(valid_actions=valid, hole_cards=cards))
            invoke(bot, "on_match_end", {"match_id": "oracle"})
            invoke(bot, "on_match_start", {**match, "match_id": "next"})
            # Preserve disagreement between lifecycle seat and decision seat.
            invoke(
                bot,
                "on_reconnected",
                {"match_id": "next", "seats": [{"seat": "1", "is_self": True}]},
            )
            invoke(bot, "decide", sdk_state(shove(50), ["Kh", "Td"]))
    fixtures = json.loads(
        (Path(__file__).parent / "fixtures/postflop_baseline.json").read_text()
    )["fixtures"]
    for case in fixtures:
        bot = SleightOfHandBot(seed=17, samples=128)
        invoke(bot, "decide", GameState(**case["state"]))
    stream.write(json.dumps({"memory_hands": memory_hands}) + "\n")


def test_canonical_full_state_oracle(tmp_path):
    root = Path(__file__).resolve().parents[1]
    archive = tmp_path / "main.tar"
    source = tmp_path / "main"
    source.mkdir()
    with archive.open("wb") as stream:
        subprocess.run(["git", "archive", BASE], stdout=stream, check=True, cwd=root)
    with tarfile.open(archive) as stream:
        stream.extractall(source, filter="data")
    driver = "import sys,runpy,gzip; sys.path.insert(0,sys.argv[1]); n=runpy.run_path(sys.argv[2]); stream=gzip.open(sys.argv[3],'wt'); n['canonical_records'](stream); stream.close()"
    results = []
    for label, path in [("main", source), ("current", root)]:
        out = tmp_path / (label + ".jsonl.gz")
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("SLEIGHT_CHIPZEN_ACCOUNTING_")
        }
        env["PYTHONPATH"] = str(path)
        subprocess.run(
            [
                sys.executable,
                "-c",
                driver,
                str(path),
                str(Path(__file__).resolve()),
                str(out),
            ],
            cwd=path,
            env=env,
            check=True,
            capture_output=True,
        )
        results.append(out)
    count = 0
    with gzip.open(results[0], "rt") as left, gzip.open(results[1], "rt") as right:
        for index, (a, b) in enumerate(zip_longest(left, right)):
            assert a is not None and b is not None, "Oracle length mismatch"
            old, new = json.loads(a), json.loads(b)
            if "canonical" in old:
                assert old == new, f"Canonical observable mismatch at step {index}"
                count += 1
            else:
                assert old["memory_hands"] == 0 and new["memory_hands"] > 0
    (tmp_path / "oracle-summary.json").write_text(
        json.dumps({"steps": count, "result": "PASS"})
    )


def test_agent_diff_is_composition_only():
    root = Path(__file__).resolve().parents[1]
    old = ast.parse(
        subprocess.check_output(
            ["git", "show", BASE + ":sleight_of_hand/holdem/agent.py"],
            cwd=root,
            text=True,
        )
    )
    new = ast.parse((root / "sleight_of_hand/holdem/agent.py").read_text())

    def policy(tree):
        nodes = []
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name != "__init__"
            ):
                nodes.append(ast.dump(node))
        return nodes

    assert policy(old) == policy(new)


def test_adapter_existing_lifecycle_order_and_inputs():
    from chipzen import Bot

    from bots.chipzen.bot import SleightOfHandBot
    from sleight_of_hand.holdem.memory import OpponentMemory

    events = []
    bot = SleightOfHandBot(seed=17)
    for event in (
        "match_start",
        "reconnected",
        "round_start",
        "round_result",
        "turn_result",
    ):
        events.clear()
        message = {
            "round_id": "h",
            "state": {"stacks": [5000, 5000]},
            "result": {},
            "seats": [],
        }
        with (
            patch.object(
                Bot, "on_" + event, side_effect=lambda msg: events.append(("sdk", msg))
            ),
            patch.object(
                bot, "_observe", side_effect=lambda *a: events.append(("accounting", a))
            ),
            patch.object(
                bot, "_learn_seat", side_effect=lambda msg: events.append(("seat", msg))
            ),
            patch.object(
                bot.shove_model,
                "record_start",
                side_effect=lambda *a: events.append(("shove_start", a)),
            ),
            patch.object(
                bot.shove_model,
                "observe_result",
                side_effect=lambda *a: events.append(("shove_result", a)),
            ),
            patch.object(
                OpponentMemory,
                "notify",
                side_effect=lambda *a: events.append(("memory", a)),
            ),
        ):
            getattr(bot, "on_" + event)(message)
        expected = {
            "match_start": ["seat", "sdk", "accounting", "memory"],
            "reconnected": ["seat", "sdk", "accounting", "memory"],
            "round_start": ["shove_start", "sdk", "accounting", "memory"],
            "round_result": ["shove_result", "sdk", "accounting", "memory"],
            "turn_result": ["sdk", "accounting", "memory"],
        }[event]
        assert [e[0] for e in events] == expected
        assert next(e[1] for e in events if e[0] == "sdk") is message
        assert events[-1][1][1] is message


def test_memory_failure_cannot_change_policy_or_existing_hooks(capsys):
    from bots.chipzen.bot import SleightOfHandBot
    from sleight_of_hand.holdem.memory import OpponentMemory
    from tests.test_chipzen import state

    class Broken(OpponentMemory):
        def notify(self, *args):
            raise RuntimeError("SECRET_TOKEN_SENTINEL")

    a, b = (
        SleightOfHandBot(seed=17),
        SleightOfHandBot(seed=17, opponent_memory=Broken()),
    )
    streams = []
    for bot in (a, b):
        bot.on_match_start({"seats": [{"seat": 0, "is_self": True}]})
        decision = bot.decide(state())
        streams.append((decision, capsys.readouterr()))
    assert b.opponent_memory is None
    assert streams[0][0] == streams[1][0]
    assert a.rng.getstate() == b.rng.getstate()
    assert vars(a.shove_model) == vars(b.shove_model) and a._seat == b._seat
    # Canonical stderr (e.g. the Season 6 fallback notice) is unchanged and
    # the swallowed memory failure leaks nothing.
    assert streams[0][1] == streams[1][1]
    assert "SECRET_TOKEN_SENTINEL" not in streams[1][1].err


def test_injected_historical_priors_have_no_policy_consumer():
    from bots.chipzen.bot import SleightOfHandBot
    from sleight_of_hand.holdem.memory import OpponentMemory
    from sleight_of_hand.holdem.profile_store import InMemoryProfileStore
    from sleight_of_hand.holdem.profiles import IdentityResolution
    from tests.test_opponent_profiles import identity, profile
    from tests.test_preflop import sdk_state, shove

    store = InMemoryProfileStore()
    store.save(profile(success=10))
    m = OpponentMemory(
        store,
        identity_resolver=lambda *a: IdentityResolution(
            identity(), "fixture-mapping", 1
        ),
    )
    a, b = SleightOfHandBot(seed=17), SleightOfHandBot(seed=17, opponent_memory=m)
    for bot in (a, b):
        bot.on_match_start({"match_id": "m", "seats": [{"seat": 0, "is_self": True}]})
    assert m.prior is not None
    for _ in range(10):
        assert a.decide(sdk_state(shove(50), ["Kh", "Td"])) == b.decide(
            sdk_state(shove(50), ["Kh", "Td"])
        )
        assert a.rng.getstate() == b.rng.getstate() and vars(a.shove_model) == vars(
            b.shove_model
        )
