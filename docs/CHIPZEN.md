# ChipZen Hold'em port

The port implements `chipzen.Bot.decide(GameState) -> Action` using the
existing five-parameter betting formula. The SDK owns the WebSocket
connection, authentication, request IDs, and rejected-action retries.
Only No-Limit Hold'em is supported. Leduc's local engine and entry points
are unchanged; this does not register a local `--gamemode holdem` engine.

## Installation and quick start

From the repository root, use the existing Python 3.13 pin:

```sh
uv venv
uv pip install -r requirements-dev.txt
uv run python -m pytest tests/ -q
uv run python scripts/build_chipzen.py
uv run chipzen-sdk validate build/chipzen --check-connectivity
```

The SDK and its WebSocket dependency are pinned exactly in
`bots/chipzen/requirements.txt`. Existing project dependency constraints
are unchanged. The bot runtime needs neither NumPy nor Matplotlib; the
staging script copies only the required Python modules and license.
`build/chipzen/` is disposable and replaced whenever staging runs.

## Usage and upload

To run against a supplied match WebSocket, set `CHIPZEN_WS_URL` and either
`CHIPZEN_TOKEN` or `CHIPZEN_TICKET` in the environment, then run:

```sh
uv run python -m bots.chipzen.bot
```

The hosted platform injects those values into the container. Never bake
credentials into the image. The SDK's external-API lobby is a separate
connection flow; a match URL is not a lobby URL.

With Docker installed and running, build the staged context and export
the image that ChipZen accepts:

```sh
docker build --platform linux/amd64 --provenance=false -t sleight-of-hand:chipzen-v2 build/chipzen
docker image inspect sleight-of-hand:chipzen-v2 --format '{{.Size}}'
docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --memory 256m --cpus 1 --entrypoint python sleight-of-hand:chipzen-v2 -u -m chipzen validate /bot --check-connectivity
mkdir -p dist
set -o pipefail
docker save sleight-of-hand:chipzen-v2 | gzip -n > dist/sleight-of-hand-chipzen-v2.tar.gz
gzip -t dist/sleight-of-hand-chipzen-v2.tar.gz
sha256sum dist/sleight-of-hand-chipzen-v2.tar.gz > dist/sleight-of-hand-chipzen-v2.tar.gz.sha256
ls -lh dist/sleight-of-hand-chipzen-v2.tar.gz
```

Upload the **Docker image archive**, not the source directory, through
ChipZen's developer UI, then run practice matches. The SDK documentation
lists a 200 MB built-image limit and a 250 MB compressed upload limit;
verify both before uploading. This simple container follows the reference
bot's runtime contract: digest-pinned Python 3.13.15, unbuffered output, non-root user,
and no required filesystem writes. It includes readable MIT-licensed
strategy source; it does not use the optional Cython protection starter.

## Configuration

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `SLEIGHT_PARAMS` | Existing `PolicyParams` defaults | JSON object with any of the five parameter names below |
| `SLEIGHT_EQUITY_SAMPLES` | `128` | Monte Carlo trials per decision, integer from 1 to 512 |
| `SLEIGHT_SEED` | Unseeded | Integer for reproducible local decisions |

The five parameters remain `value_bet_threshold`, `call_threshold`,
`bluff_freq`, `aggression`, and `steepness`. For example:

```sh
export SLEIGHT_PARAMS='{"aggression":0.75,"bluff_freq":0.10}'
```

Keep the seed unset for normal play. Parameters are clipped to the
existing policy bounds. Invalid configuration fails at startup.

## Strategy and limitations

Version 2 adjusts the call threshold to the current pot odds. Calling a
pot-sized bet needs 1/3 equity; the existing `call_threshold` keeps its
margin relative to that reference price. Cheaper calls reduce the required
equity, while expensive calls increase it. Value-raise thresholds also rise
above the reference price, without making cheap calls trigger looser raises.
The other three parameters and bet-size formula are unchanged. An all-in
call uses only the chips we can pay, though side-pot eligibility is still
an approximation. This is a local policy adjustment, not learned strategy.


- Strength is estimated showdown pot share against uniformly sampled
  live opponents and unseen runouts, including split-pot ties. Folded
  seats are removed; all-in seats remain opponents. Supports 2–6 seats.
- The same sigmoid policy converts strength into fold, check/call, and
  raise probabilities. The Leduc card-strength function still feeds the
  same formula for existing agents.
- Raise amounts are **total bets**: start at the legal minimum and add
  `aggression * (pot + affordable_call)`, capped at the server maximum.
  Short-stack raises use the server's all-in ceiling. Legacy `all_in`
  is emitted only if explicitly offered; ordinary shoves use `raise`.
- Malformed/missing Hold'em cards use check, then fold, when available.
  The bot never constructs draw or OFC placement actions.
- This first port does not include Bayesian range inference, exhaustive
  search, side-pot equity, or Hold'em
  GA training. Existing Leduc parameter values have not been tuned for
  Hold'em. Legal actions and protocol conformance do not establish strength.

## Development and validation

Version 2 adds tests for call prices, short stacks, risk margins, and the
actual decision path. In two controlled terminal call-or-fold examples
with 40% equity (no future betting or side pots), call probability changes
from 59.9% to 91.2% when the current pot is 100 and calling costs 10, and
from 59.9% to 32.2% when the current pot is 110 and calling costs 100.
Expected incremental chips improve from 20.36 to 31.01 and from -9.58 to
-5.16 respectively. These illustrate the response to price; they do not
measure full-game win rate against opponents.

The original image is retained as `sleight-of-hand:chipzen-v1` and the
original `dist/sleight-of-hand-chipzen.tar.gz` is preserved for comparison.
The updated image is `sleight-of-hand:chipzen-v2` and its upload archive is
`dist/sleight-of-hand-chipzen-v2.tar.gz`. Both use the same exact dependency
and base-image pins.

Version 2 passes 87 tests on each of Python 3.10 and 3.13, plus SDK
conformance inside the same restricted container configuration described
below. Its archive passed gzip validation and a Docker reload, and the
version 1 archive's checksum remained unchanged. Measured archive sizes
and SHA-256 are recorded in `dist/chipzen-v2-validation.json`; a separate
`.sha256` file accompanies the upload archive. No arena comparison has
been run yet.


`tests/test_chipzen.py` checks hand categories and kickers, best-five
selection, equity ties, multiway seats, malformed-card fallback, raise
bounds, and SDK wire parsing. The existing Leduc tests cover regression
of the shared policy. Install the port requirements to enable the SDK
tests; CI installs them explicitly. Run the SDK check on the **staged**
directory to catch missing runtime modules before building a container.

Version 1 local validation: 81 tests passed on both Python 3.10 and 3.13; the SDK's full
conformance scenarios passed on Python 3.10 and 3.13. A 40-decision,
six-seat sample on Python 3.13 measured 15 ms median and 27 ms maximum
with 128 equity trials. These timings are local observations, not a
platform latency guarantee. The validator emits its generic `os` import
warning because the entry point reads the platform-provided environment,
as the official starter does.

The version 1 Linux/amd64 image was built and its SDK conformance checks passed
inside a non-root, read-only container with no network, all capabilities
dropped, no-new-privileges, a 256 MB memory limit, and one CPU. The upload
archive is `dist/sleight-of-hand-chipzen.tar.gz`: 46,210,068 bytes compressed
(46.2 MB), with 132,310,016 bytes of uncompressed layer tar data (132.3 MB),
both below the documented limits. Gzip integrity and the Docker archive's
manifest, platform, and entry point were checked. The archive was also
loaded back into Docker successfully. Its SHA-256 is recorded alongside it;
`dist/chipzen-validation.json` records the measured sizes and check result.
Live arena matches and platform review have not been run.

Protocol sources: [SDK quickstart](https://github.com/chipzen-ai/chipzen-sdk/blob/main/docs/QUICKSTART.md),
[NLHE state protocol](https://github.com/chipzen-ai/chipzen-sdk/blob/main/docs/protocol/POKER-GAME-STATE-PROTOCOL.md),
and [developer manual](https://github.com/chipzen-ai/chipzen-sdk/blob/main/docs/DEV-MANUAL.md).
Implementation reference inspected at SDK commit
`e6acc70f5a3247b6d345813d09e05b9c19aedf1e`; runtime pin: `chipzen-bot==0.4.0`.

## License

Sleight-of-Hand is [MIT licensed](../LICENSE). ChipZen's separately
installed SDK is Apache-2.0 licensed.
