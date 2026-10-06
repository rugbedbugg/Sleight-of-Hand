"""Explicit evidence stores; no store performs I/O during construction."""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Protocol

from .profiles import (
    HistoricalOpponentProfile,
    OpponentIdentity,
    decode,
    encode,
    fields,
    identifier,
)

MAX_PROFILE_BYTES = 1_048_576
MAX_BUNDLE_BYTES = 33_554_432
MAX_BUNDLE_PROFILES = 10_000


class OpponentProfileStore(Protocol):
    def load(self, identity: OpponentIdentity) -> HistoricalOpponentProfile | None: ...
    def save(self, profile: HistoricalOpponentProfile) -> None: ...


class NullProfileStore:
    def load(self, identity: OpponentIdentity) -> None:
        return None

    def save(self, profile: HistoricalOpponentProfile) -> None:
        pass


class InMemoryProfileStore:
    def __init__(self):
        self._profiles: dict[str, HistoricalOpponentProfile] = {}

    def load(self, identity: OpponentIdentity) -> HistoricalOpponentProfile | None:
        return (
            self._profiles.get(identity.persistent_key)
            if identity.persistent_key
            else None
        )

    def save(self, profile: HistoricalOpponentProfile) -> None:
        # Validate at the storage boundary as well as at compilation.
        checked = HistoricalOpponentProfile.from_dict(profile.to_dict())
        self._profiles[checked.identity.persistent_key] = checked


#: fsync errors meaning "directories cannot be synced here", not lost data.
_DIRECTORY_SYNC_UNSUPPORTED = {
    errno.EINVAL,
    errno.EBADF,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}


def _fsync_directory(directory: Path) -> None:
    """Persist a completed rename where the platform can sync directories.

    Without ``O_DIRECTORY`` (Windows) a directory cannot be opened for fsync,
    so this is a no-op there; so is a directory that cannot be opened or a
    filesystem that rejects directory fsync. Other fsync failures propagate:
    the new file is visible but its rename may not be durable.
    """
    flag = getattr(os, "O_DIRECTORY", None)
    if flag is None:
        return
    try:
        fd = os.open(directory, os.O_RDONLY | flag)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in _DIRECTORY_SYNC_UNSUPPORTED:
            raise
    finally:
        os.close(fd)


def atomic_write(path: Path, data: bytes) -> None:
    """Same-directory replacement; old destination survives failed writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".profile-", delete=False
        ) as stream:
            temp = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


class FileProfileStore:
    """Explicit research/remote directory. No runtime autosave or home defaults."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _path(self, identity: OpponentIdentity) -> Path | None:
        key = identity.persistent_key
        return (
            self.directory / (hashlib.sha256(key.encode("ascii")).hexdigest() + ".json")
            if key
            else None
        )

    def load(self, identity: OpponentIdentity) -> HistoricalOpponentProfile | None:
        path = self._path(identity)
        if path is None:
            return None
        try:
            with path.open("rb") as stream:
                data = stream.read(MAX_PROFILE_BYTES + 1)
            if len(data) > MAX_PROFILE_BYTES:
                return None
            profile = HistoricalOpponentProfile.from_dict(decode(data))
            return (
                profile
                if profile.identity.persistent_key == identity.persistent_key
                else None
            )
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            OverflowError,
            RecursionError,
        ):
            return None

    def save(self, profile: HistoricalOpponentProfile) -> None:
        checked = HistoricalOpponentProfile.from_dict(profile.to_dict())
        data = encode(checked.to_dict())
        if len(data) > MAX_PROFILE_BYTES:
            raise ValueError("profile exceeds byte limit")
        atomic_write(self._path(checked.identity), data)


def bundle_bytes(profiles: list[HistoricalOpponentProfile], data_version: str) -> bytes:
    identifier(data_version)
    keyed = {}
    for profile in profiles:
        checked = HistoricalOpponentProfile.from_dict(profile.to_dict())
        key = checked.identity.persistent_key
        if key in keyed:
            raise ValueError("duplicate bundled identity")
        if checked.provenance.data_version != data_version:
            raise ValueError("inconsistent bundle data version")
        keyed[key] = checked.to_dict()
    if len(keyed) > MAX_BUNDLE_PROFILES:
        raise ValueError("too many bundled profiles")
    data = encode(
        {"schema_version": 1, "data_version": data_version, "profiles": keyed}
    )
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("bundle exceeds byte limit")
    return data


class BundledProfileStore:
    """Read-only immutable snapshot; malformed entire bundle becomes empty.

    Parsing is explicit before injection, never in HoldemAgent construction.
    No populated bundle is installed in Chipzen staging.
    """

    def __init__(self, data: bytes):
        self._profiles: dict[str, HistoricalOpponentProfile] = {}
        self.valid = False
        try:
            if len(data) > MAX_BUNDLE_BYTES:
                return
            value = fields(decode(data), {"schema_version", "data_version", "profiles"})
            if type(value["schema_version"]) is not int or value["schema_version"] != 1:
                return
            identifier(value["data_version"])
            if (
                type(value["profiles"]) is not dict
                or len(value["profiles"]) > MAX_BUNDLE_PROFILES
            ):
                return
            parsed = {}
            for key, entry in value["profiles"].items():
                profile = HistoricalOpponentProfile.from_dict(entry)
                if (
                    profile.identity.persistent_key != key
                    or profile.provenance.data_version != value["data_version"]
                ):
                    return
                parsed[key] = profile
            self._profiles = parsed
            self.valid = True
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            OverflowError,
            RecursionError,
        ):
            pass

    @classmethod
    def from_path(cls, path: Path) -> BundledProfileStore:
        try:
            with Path(path).open("rb") as stream:
                return cls(stream.read(MAX_BUNDLE_BYTES + 1))
        except OSError:
            return cls(b"")

    def load(self, identity: OpponentIdentity) -> HistoricalOpponentProfile | None:
        return (
            self._profiles.get(identity.persistent_key)
            if identity.persistent_key
            else None
        )

    def save(self, profile: HistoricalOpponentProfile) -> None:
        raise TypeError("bundled profiles are read-only")
