# ChipZen Hold'em port

The port implements `chipzen.Bot.decide(GameState) -> Action`. Version 3
sends heads-up preflop decisions through an explicit context-aware policy
([below](#version-3-context-aware-heads-up-preflop)); every other decision
uses the existing five-parameter betting formula unchanged. The SDK owns the WebSocket
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

## Version 3: context-aware heads-up preflop

Version 2 is the build tagged `season-6` (signed tag on commit `ceecfcc`)
and evaluated on ChipZen in Season 6. The tag is the reproducibility
baseline and is not moved. Version 3 changes **only heads-up preflop
decisions**. Postflop, multiway and unrecognised states still take the
Season 6 path, decision for decision under the same seed; a regression
test replays decisions recorded from the tag.

### Why: PLUMBER findings

ChipZen's PLUMBER report (23 September 2026, 27,394 hands: 825 real hands
over 17 matches, plus 2,000 simulated matches against Fableous and
Chatterbox) found no invalid actions or timeouts, and three preflop leaks:

| Metric (40bb+ unless noted) | Real | Sim (Fab / Chat) | Reference |
| --- | --- | --- | --- |
| Button VPIP | 86.8% | 88.0 / 87.0% | 84–96% |
| Button raise first in | 26.5% | 33.0 / 33.4% | 64–85% |
| Button limp | 60.3% | 55.0 / 53.7% | 0–32% |
| Big-blind PFR | 32.2% | 37.4 / 38.1% | 15–25% |
| Big-blind 3-bet vs button open | 28.0% | 33.6 / 33.6% | 15–22% |
| Big-blind fold vs all-in, under 20bb | 23.8% (N=21) | 17.8 / 13.1% | 38–78% |

The cause is structural. Season 6 scored every preflop hand by equity
against a uniformly random hand, adjusted by price, and fed that into the
same five global parameters, so it could not tell apart the button acting
first, the big blind facing a limp, an open or a shove, or different stack
depths. Its equity estimate also treats a shover as holding a random hand.

The local calibration tool reproduces PLUMBER's measurements of the
Season 6 policy (button raise first in 33.1%, limp 54.7%, VPIP 87.8%;
big-blind 3-bet 33.1%). It also finds a Season 6 pot-odds defect: the
price counts chips a covering shover bets beyond our stack, which cannot
be won. Against a 60bb shove, a 7bb big blind calls 94.6% of hands and a
14bb big blind 90.0%. This likely explains why PLUMBER measured far fewer
folds than equal-stack shoves predict. `betting_params` is unchanged in
version 3 because it also drives postflop play; the preflop policy
removes the excess itself.

### What changed

- `sleight_of_hand/holdem/preflop.py` derives a typed `PreflopContext`
  (position, facing situation, current big blind, bets, stacks, effective
  stack in big blinds, voluntary action sequence) from public state only:
  - The current big blind comes from the synthetic `post_big_blind` entry,
    because blinds rise every 20 hands.
  - Position comes from the seat that posted the small blind; in heads-up
    the button posts it.
  - Raise amounts are read as raise-to totals, per the protocol.
  - Call amounts are never read: the protocol's own examples disagree on
    whether they are totals or increments.
  - The context stores the full `amount_owed` and the real `call_cost`
    (owed, capped at our stack) separately. The protocol does not say
    whether `to_call` is the full amount or the capped one, so either is
    accepted and both give an identical context and decision.
  - Rebuilt bets must reconcile with `pot` and `to_call`. If they don't,
    or blind posts are missing, the table is multiway, or the order is
    impossible, the context is `None`. The bot then logs one warning and
    uses Season 6.
- Facing situations: `first_in`, `limp`, `open`, `limp_raised`,
  `three_bet`, `four_bet_plus`, and `shove`. A shove means the villain is
  all-in, or calling would commit our whole stack.
- `sleight_of_hand/holdem/hands.py` defines the 169 starting-hand classes
  and their combination weights (6 per pair, 4 suited, 12 offsuit).
- `sleight_of_hand/holdem/preflop_tables.py` is generated by
  `scripts/generate_preflop_tables.py`. It holds heads-up all-in equity for
  each class against a random hand, which defines the strength order, and
  against the top 5%, 10%, …, 100% of that order, all measured with the
  repo's evaluator. Every cell is string-seeded; regenerating produces a
  byte-identical file.
- `bots/chipzen/bot.py`: `decide()` tries `preflop_action()` first.
  Preflop raises use explicit sizes, not `aggression`:
  - Open to 2.5bb, or 2bb below 25bb effective.
  - Iso-raise to 4bb.
  - 3-bet to 4× the open, re-raise a limp-raise to 3×, and 4-bet to 2.25×.
  - Jam against a 4-bet or more.
  - Any raise that commits 40% of the effective stack becomes all-in.

  Folding is never chosen when checking is free, and illegal choices
  degrade to the passive legal action.

### Policy and assumptions

Raise regions are combo-weighted tops of the strength order. Only the
single class on each boundary mixes, so each region is exactly its stated
size. Calls compare equity against an **assumed** villain range with the
exact price of the decision:

- **Button first in:** raise the strongest 70% of combos, limp the next
  17%, and fold the weakest 13%. Participation stays at Season 6's
  healthy level; the raise/limp split changes.
- **Big blind vs limp:** iso-raise the strongest 30%, check the rest.
- **Big blind vs open:** 3-bet the strongest 16%. Call when equity against
  an assumed 80% opening range, times a 0.75 out-of-position realization
  factor, meets the price; otherwise fold. Defence therefore tightens as
  the open size grows.
- **Button vs 3-bet:** 4-bet the strongest 5%. Call on price against an
  assumed 30% range. A linear 20% assumption folded ~62% of opens to a
  3-bet.
- **Facing a shove:** call when all-in equity against an assumed shove
  range meets the price. The price excludes chips we cannot match. The
  assumed range depends on effective stack (≤5, ≤8, ≤12, ≤16, ≤20 and
  over 20bb) and on whether we had already acted:

  | Shove type | ≤5bb | ≤8bb | ≤12bb | ≤16bb | ≤20bb | >20bb |
  | --- | --- | --- | --- | --- | --- | --- |
  | Open shove | 80% | 65% | 50% | 45% | 38% | 25% |
  | Reshove over our limp or raise | 60% | 45% | 35% | 28% | 22% | 15% |

**Everything above is an empirical, provisional baseline, not GTO or a
Nash chart.** Only the equities are measured. Region sizes, assumed
ranges, realization factors and bet sizes are documented judgement,
collected in `PreflopConfig` so they can be replaced by sourced ranges
(for example HoldemResources' heads-up push/fold tables, which PLUMBER
cites but which are not checked in) or by measured opponent data.

### Predicted frequencies

Output of `scripts/preflop_calibration.py`: combo-weighted shares of all
1,326 hands at 50/100 blinds. Season 6 values are expectations of its
decision branch over repeated 128-sample equity estimates.

| Scenario | Version 3 | Season 6 |
| --- | --- | --- |
| Button first in, 100bb (also 40/15/8bb) | raise 70.0%, limp 17.0%, fold 13.0%, VPIP 87.0% | raise 33.1%, limp 54.7%, VPIP 87.8% |
| Big blind vs limp, 100bb | iso-raise 30.0%, check 70.0% | raise 41.9% |
| Big blind vs 2.0bb open | 3-bet 16.0%, call 77.0%, fold 7.0% | 3-bet 33.1%, fold 12.2% |
| Big blind vs 2.5bb open | 3-bet 16.0%, call 51.6%, fold 32.4% | 3-bet 33.1%, fold 16.3% |
| Big blind vs 3.0bb open | 3-bet 16.0%, call 38.5%, fold 45.5% | 3-bet 33.1%, fold 19.5% |
| Button open vs 3-bet to 10bb, among opened hands | 4-bet 7.1%, call 44.1%, fold 48.8% | — |
| Big blind vs shove, 4 / 7 / 10 / 14 / 18bb (fold) | 25.0 / 53.2 / 67.8 / 72.2 / 78.6% | 32.1 / 40.8 / 44.4 / 46.8 / 48.2% |
| Big blind 7 / 14bb vs covering 60bb shove (fold), with `to_call` as full owed or capped at stack | 53.2 / 72.2% | 5.4 / 10.0% |

Big-blind PFR depends on how often the opponent opens rather than limps.
With 50–90% opens it is 17–23% (the test covers this range), inside the
15–25% reference.

Fold to shove at 4bb (25%) is below PLUMBER's 38–78% band, and 18bb
(78.6%) sits at its top. That band spans 5–20bb, and very short stacks
correctly call wide when the price is good.

### Limitations

- The preflop policy covers heads-up only; multiway uses Season 6.
- Button opening frequency does not yet change with stack depth. At short
  stacks the commit rule turns large raises into shoves, but there is no
  dedicated push/fold chart for the button.
- Ranges are linear in equity against a random hand, which undervalues
  some suited and connected hands. There are no opponent-specific
  adjustments.
- ChipZen's conformance harness sends an empty `action_history`, so it
  exercises only the Season 6 fallback. The version 3 path is covered by
  unit tests, including the protocol's section 4 example message. On the
  first live matches, check the logs for the one-time warning `Preflop
  context unavailable`.
- Predicted frequencies describe the policy, not results. Only matches
  against opponents can show whether win rate improves.

### Reproduce and compare

```sh
uv run --no-project python -m pytest tests/test_preflop.py -q
uv run --no-project python scripts/preflop_calibration.py   # v3 and Season 6 side by side
uv run --no-project python scripts/generate_preflop_tables.py  # ~4 min on 8 cores; byte-identical
git worktree add ../soh-season6 season-6   # the untouched Season 6 source
```

Stage and validate as above. Build version 3 under a new tag, for
example `sleight-of-hand:chipzen-v3`, with archive
`dist/sleight-of-hand-chipzen-v3.tar.gz`, so the Season 6 image remains
available for comparison. The next experiment should play version 3
against the `season-6` image and rerun PLUMBER or equivalent controlled
matches. Check the metrics in the table above before starting further
strategy work.

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
