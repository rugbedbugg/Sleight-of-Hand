"""Compile LOCAL normalized observations into a deterministic evidence bundle.

No HTTP client, credentials, system clock, randomness or policy actions.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleight_of_hand.holdem.observations import NormalizedObservation
from sleight_of_hand.holdem.profile_store import atomic_write, bundle_bytes
from sleight_of_hand.holdem.profiles import (
    BeliefConfig,
    Counts,
    HistoricalOpponentProfile,
    Provenance,
    decode,
    fields,
    identifier,
)

MAX_INPUT_BYTES = 33_554_432
MAX_OBSERVATIONS = 100_000


def compile_priors(data: bytes, config: BeliefConfig | None = None) -> bytes:
    config = config if config is not None else BeliefConfig()
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("input exceeds byte limit")
    dataset = fields(decode(data), {"schema_version", "data_version", "observations"})
    if type(dataset["schema_version"]) is not int or dataset["schema_version"] != 1:
        raise ValueError("unsupported dataset schema")
    version = identifier(dataset["data_version"])
    rows = dataset["observations"]
    if type(rows) is not list or len(rows) > MAX_OBSERVATIONS:
        raise ValueError("invalid observation list")
    unique = {}
    for row in rows:
        observation = NormalizedObservation.from_dict(row)
        if observation.identity.persistent_key is None:
            raise ValueError("cannot compile history for unproven identity")
        key = observation.dedup_key
        if key in unique and unique[key] != observation:
            raise ValueError("conflicting duplicate observation")
        unique[key] = observation
    groups = {}
    for observation in unique.values():
        groups.setdefault(observation.identity.persistent_key, []).append(observation)
    profiles = []
    for key in sorted(groups):
        group = groups[key]
        identity = group[0].identity
        if any(o.identity != identity for o in group):
            raise ValueError("conflicting identity provenance")
        stats = {}
        for observation in group:
            for name, count in observation.stats:
                stats[name] = stats.get(name, Counts()) + count
        provenance = Provenance(
            version,
            tuple(
                sorted(
                    {
                        hashlib.sha256(data).hexdigest(),
                        *(o.source_sha256 for o in group),
                    }
                )
            ),
            tuple(sorted({o.source_id for o in group})),
            len(group),
        )
        profiles.append(
            HistoricalOpponentProfile(
                identity,
                tuple(sorted(stats.items())),
                len(group),
                len({o.match_id for o in group}),
                provenance,
                config,
            )
        )
    return bundle_bytes(profiles, version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha-base", type=float, default=1.0)
    parser.add_argument("--beta-base", type=float, default=1.0)
    parser.add_argument("--historical-ess-cap", type=float, default=10.0)
    args = parser.parse_args()
    try:
        config = BeliefConfig(args.alpha_base, args.beta_base, args.historical_ess_cap)
        with args.input.open("rb") as stream:
            data = stream.read(MAX_INPUT_BYTES + 1)
        output = compile_priors(data, config)
        atomic_write(args.output, output)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        parser.exit(
            2,
            "Prior compilation failed: invalid local input, configuration or output path.\n",
        )
    print(hashlib.sha256(output).hexdigest())


if __name__ == "__main__":
    main()
