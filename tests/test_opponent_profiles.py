"""Evidence serialization, identity trust boundary, stores and local compiler."""

import copy
import errno
import hashlib
import os
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.build_opponent_priors import compile_priors
from sleight_of_hand.holdem.observations import NormalizedObservation
from sleight_of_hand.holdem.profile_store import (
    BundledProfileStore,
    FileProfileStore,
    InMemoryProfileStore,
    NullProfileStore,
    bundle_bytes,
)
from sleight_of_hand.holdem.profiles import (
    BeliefConfig,
    Counts,
    HistoricalOpponentProfile,
    OpponentIdentity,
    Provenance,
    decode,
    encode,
    resolve_match_identity,
)


def identity(key="bot-a", name=None):
    return OpponentIdentity(
        "research", key, "external_bot_id", "persistent", "fixture-mapping-v1", name
    )


def profile(key="bot-a", success=7, total=10):
    return HistoricalOpponentProfile(
        identity(key),
        (("btn_open", Counts(success, total)),),
        total,
        1,
        Provenance("synthetic-v1", ("a" * 64,), ("fixture",), total),
    )


def dataset():
    observations = [
        NormalizedObservation(
            identity(),
            "match-1",
            f"hand-{i}",
            "fixture",
            "a" * 64,
            "synthetic",
            (("btn_open", Counts(i % 2, 1)),),
        ).to_dict()
        for i in range(8)
    ]
    return {
        "schema_version": 1,
        "data_version": "synthetic-v1",
        "observations": observations,
    }


def test_rename_and_same_name_collisions():
    assert identity(name="old").persistent_key == identity(name="new").persistent_key
    assert identity("a", "same").persistent_key != identity("b", "same").persistent_key
    assert "old" not in encode(identity(name="old").to_dict()).decode()


@pytest.mark.parametrize(
    "kind,scope,key,proof,allowed",
    [
        (
            "bot_uuid",
            "persistent",
            "00000000-0000-0000-0000-000000000001",
            "mapping",
            True,
        ),
        ("external_bot_id", "persistent", "external-1", "mapping", True),
        ("participant_id", "match", "p-1", None, False),
        ("participant_id", "unknown", "p-1", None, False),
        ("bot_uuid", "unknown", "00000000-0000-0000-0000-000000000001", None, False),
        ("unknown", "unknown", None, None, False),
    ],
)
def test_identity_scope_matrix(kind, scope, key, proof, allowed):
    value = OpponentIdentity("chipzen", key, kind, scope, proof)
    assert (value.persistent_key is not None) == allowed


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "key_kind": "participant_id",
            "scope": "persistent",
            "key": "p1",
            "persistence_evidence": "claim",
        },
        {"key_kind": "external_bot_id", "scope": "persistent", "key": "b1"},
        {"key_kind": "bot_uuid", "key": "invalid"},
        {"key": "../../private"},
        {"key": True},
        {"scope": "global"},
    ],
)
def test_invalid_identity_rejected(kwargs):
    with pytest.raises(ValueError):
        OpponentIdentity("chipzen", **kwargs)


@pytest.mark.parametrize("participant", [None, "", 4, True, [], "../x", "p-1"])
def test_resolver_does_not_promote_participant_or_unverified_bot_id(participant):
    seats = [
        {"seat": 0, "is_self": True},
        {
            "seat": 1,
            "participant_id": participant,
            "bot_id": "00000000-0000-0000-0000-000000000001",
            "display_name": "same",
        },
    ]
    value = resolve_match_identity("chipzen", "m", seats)
    assert value.identity.persistent_key is None
    assert value.identity.key_kind != "bot_uuid"
    assert value.opponent_seat == 1


@pytest.mark.parametrize(
    "seats",
    [
        None,
        [],
        [{}],
        [{"seat": 0}, {"seat": 1}],
        [{"seat": 0, "is_self": True}, {"seat": 0}],
        [{"seat": False}, {"seat": 1}],
    ],
)
def test_unusable_seats_are_anonymous(seats):
    assert resolve_match_identity("chipzen", "m", seats).identity.key is None


@pytest.mark.parametrize("store_kind", ["null", "memory", "file", "bundle"])
def test_store_roundtrip_and_uncertain_identity(tmp_path, store_kind):
    p = profile()
    stores = {
        "null": NullProfileStore(),
        "memory": InMemoryProfileStore(),
        "file": FileProfileStore(tmp_path / "explicit"),
        "bundle": BundledProfileStore(bundle_bytes([p], "synthetic-v1")),
    }
    store = stores[store_kind]
    if store_kind != "bundle":
        store.save(p)
    assert store.load(identity()) == (None if store_kind == "null" else p)
    assert store.load(identity("missing")) is None
    uncertain = OpponentIdentity("research", "bot-a", "external_bot_id", "unknown")
    assert store.load(uncertain) is None
    if store_kind == "bundle":
        with pytest.raises(TypeError):
            store.save(p)


def test_file_atomicity(tmp_path):
    store = FileProfileStore(tmp_path)
    original = profile()
    store.save(original)
    old_bytes = next(tmp_path.iterdir()).read_bytes()
    for operation in ("os.replace", "os.fsync"):
        with (
            patch(
                "sleight_of_hand.holdem.profile_store." + operation,
                side_effect=OSError("injected"),
            ),
            pytest.raises(OSError),
        ):
            store.save(profile(success=2))
        assert store.load(identity()) == original
        assert [f.read_bytes() for f in tmp_path.iterdir()] == [old_bytes]


def directory_fsync_spy(monkeypatch, error=None):
    """Record (kind, fd-or-path) calls; optionally fail only directory fsync."""
    calls, real_fsync, real_open = [], os.fsync, os.open

    def fsync(fd):
        is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
        calls.append(("fsync_dir" if is_dir else "fsync_file", fd))
        if is_dir and error is not None:
            raise OSError(error, os.strerror(error))
        real_fsync(fd)

    def open_(path, flags, *args):
        fd = real_open(path, flags, *args)
        if stat.S_ISDIR(os.fstat(fd).st_mode):  # tempfile also uses os.open
            calls.append(("open", Path(path)))
        return fd

    monkeypatch.setattr("sleight_of_hand.holdem.profile_store.os.fsync", fsync)
    monkeypatch.setattr("sleight_of_hand.holdem.profile_store.os.open", open_)
    return calls


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="POSIX directory sync")
def test_file_save_syncs_parent_directory_after_replace(tmp_path, monkeypatch):
    calls = directory_fsync_spy(monkeypatch)
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(
        "sleight_of_hand.holdem.profile_store.os.replace",
        lambda a, b: (replaced.append(len(calls)), real_replace(a, b)),
    )
    FileProfileStore(tmp_path / "profiles").save(profile())
    kinds = [kind for kind, _ in calls]
    assert kinds == ["fsync_file", "open", "fsync_dir"]
    assert replaced == [1]  # file fsync, then replace, then directory sync
    assert calls[1][1] == tmp_path / "profiles"


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="POSIX directory sync")
@pytest.mark.parametrize("code", [errno.EINVAL, errno.EBADF])
def test_unsupported_directory_fsync_is_skipped(tmp_path, monkeypatch, code):
    directory_fsync_spy(monkeypatch, error=code)
    store = FileProfileStore(tmp_path)
    store.save(profile(success=3))
    assert store.load(identity()) == profile(success=3)
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="POSIX directory sync")
def test_directory_fsync_io_error_propagates_without_temp_or_fd_leak(
    tmp_path, monkeypatch
):
    store = FileProfileStore(tmp_path)
    store.save(profile())
    calls = directory_fsync_spy(monkeypatch, error=errno.EIO)
    with pytest.raises(OSError) as raised:
        store.save(profile(success=2))
    assert raised.value.errno == errno.EIO
    # The rename completed (visible, possibly not durable); nothing is left over.
    assert store.load(identity()) == profile(success=2)
    assert len(list(tmp_path.iterdir())) == 1
    directory_fd = next(fd for kind, fd in calls if kind == "fsync_dir")
    with pytest.raises(OSError):
        os.fstat(directory_fd)  # closed despite the error


def test_directory_sync_noop_without_o_directory(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "O_DIRECTORY", raising=False)
    calls = directory_fsync_spy(monkeypatch)
    store = FileProfileStore(tmp_path)
    store.save(profile())
    assert [kind for kind, _ in calls] == ["fsync_file"]
    assert store.load(identity()) == profile()


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="POSIX directory sync")
def test_unopenable_directory_is_skipped(tmp_path, monkeypatch):
    real_open = os.open

    def refuse_directories(path, flags, *args):
        if flags & os.O_DIRECTORY:
            raise PermissionError(errno.EACCES, "denied")
        return real_open(path, flags, *args)

    monkeypatch.setattr(
        "sleight_of_hand.holdem.profile_store.os.open", refuse_directories
    )
    store = FileProfileStore(tmp_path)
    store.save(profile())
    assert store.load(identity()) == profile()


def test_file_store_construction_performs_no_io(tmp_path):
    forbidden = AssertionError("constructor I/O")
    with (
        patch("sleight_of_hand.holdem.profile_store.os.open", side_effect=forbidden),
        patch("pathlib.Path.mkdir", side_effect=forbidden),
        patch("pathlib.Path.open", side_effect=forbidden),
    ):
        FileProfileStore(tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize(
    "bad",
    [
        b"",
        b"{",
        b"[]",
        b'{"schema_version":99}',
        b'{"x":1,"x":2}',
        b"null",
        b"x" * 1_048_577,
    ],
)
def test_corrupt_file_and_bundle_fail_closed(tmp_path, bad):
    store = FileProfileStore(tmp_path)
    store.save(profile())
    next(tmp_path.iterdir()).write_bytes(bad)
    assert store.load(identity()) is None
    bundle = BundledProfileStore(bad)
    assert not bundle.valid and bundle.load(identity()) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema_version=True),
        lambda p: p.update(schema_version=2),
        lambda p: p["stats"]["btn_open"].update(successes=11),
        lambda p: p["config"].update(alpha_base=float("nan")),
        lambda p: p["historical_ess"].update(btn_open=99),
        lambda p: p.update(token="SECRET_TOKEN_SENTINEL"),
        lambda p: p["identity"].update(display_name="PLAYER_NAME_SENTINEL"),
        lambda p: p["provenance"].update(observation_count=11),
    ],
)
def test_profile_validation_rejects_malformed_and_private_fields(mutate):
    data = profile().to_dict()
    mutate(data)
    with pytest.raises((ValueError, TypeError)):
        HistoricalOpponentProfile.from_dict(data)


def test_wrong_identity_file_fails_closed(tmp_path):
    store = FileProfileStore(tmp_path)
    store.save(profile())
    next(tmp_path.iterdir()).write_bytes(encode(profile("other").to_dict()))
    assert store.load(identity()) is None


def test_compiler_counts_dedup_provenance_and_determinism():
    value = dataset()
    value["observations"].append(copy.deepcopy(value["observations"][0]))
    raw = encode(value)
    first = compile_priors(raw)
    assert first == compile_priors(raw)
    store = BundledProfileStore(first)
    p = store.load(identity())
    assert p.total_hands == 8 and p.total_matches == 1
    assert dict(p.stats)["btn_open"] == Counts(4, 8)
    assert p.provenance.observation_count == 8
    assert hashlib.sha256(raw).hexdigest() in p.provenance.input_hashes
    assert p.config == BeliefConfig()
    assert p.to_dict()["historical_ess"] == {"btn_open": 8}
    assert first == bundle_bytes([p], "synthetic-v1")


@pytest.mark.parametrize(
    "fault", ["schema", "private", "conflict", "scope", "kind", "count"]
)
def test_compiler_rejects_bad_records(fault):
    data = dataset()
    row = data["observations"][0]
    if fault == "schema":
        data["schema_version"] = 2
    if fault == "private":
        row["hole_cards"] = ["PRIVATE_CARD_SENTINEL"]
    if fault == "scope":
        row["identity"]["scope"] = "match"
    if fault == "kind":
        row["source_kind"] = "private_log"
    if fault == "count":
        row["stats"]["btn_open"]["opportunities"] = 2
    if fault == "conflict":
        duplicate = copy.deepcopy(row)
        duplicate["stats"]["btn_open"]["successes"] = 1
        data["observations"].append(duplicate)
    with pytest.raises(ValueError):
        compile_priors(encode(data))


def test_compiler_cli_is_local_repeatable_and_preserves_output_on_error(tmp_path):
    source, out = tmp_path / "input.json", tmp_path / "out.json"
    source.write_bytes(encode(dataset()))
    command = [
        sys.executable,
        "scripts/build_opponent_priors.py",
        "--input",
        str(source),
        "--output",
        str(out),
    ]
    a = subprocess.run(command, check=True, capture_output=True)
    first = out.read_bytes()
    b = subprocess.run(command, check=True, capture_output=True)
    assert first == out.read_bytes() and a.stdout == b.stdout
    source.write_text('{"token":"SECRET_TOKEN_SENTINEL"}')
    failed = subprocess.run(command, capture_output=True, check=False)
    assert failed.returncode == 2 and b"SECRET" not in failed.stderr
    assert out.read_bytes() == first


def test_bundle_order_and_serialization_without_names():
    a, b = profile("a"), profile("b")
    assert bundle_bytes([a, b], "synthetic-v1") == bundle_bytes([b, a], "synthetic-v1")
    renamed = replace(
        a, identity=replace(a.identity, display_name="PLAYER_NAME_SENTINEL")
    )
    data = bundle_bytes([renamed], "synthetic-v1")
    assert b"PLAYER_NAME" not in data
    assert decode(data)["schema_version"] == 1
