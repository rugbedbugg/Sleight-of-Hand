# Opponent memory foundation

Sleight-of-Hand's Hold'em identity is **play sound poker when little is known;
play the opponent increasingly as evidence accumulates**. This foundation
implements evidence and memory, not the second step's strategy deviations.
It is not reinforcement learning, solver imitation, self-play or policy training.

## Implemented now

- Explicit identity kind, scope and persistence provenance.
- Match-local, card-free preflop opportunity observations.
- Immutable historical sufficient statistics and independently inspectable
  current evidence and posterior belief.
- Null, in-memory, explicit-directory file, and read-only bundled stores.
- Bounded historical effective sample size (ESS).
- Deterministic local normalized-data compiler.
- Policy-inert runtime composition and lifecycle forwarding.
- Explicit snapshot diagnostics; no automatic logging or environment switches.

The existing `ShoveModel` is unchanged and remains strategically authoritative.
The general memory does not supply range widths, actions, sizes or parameters.
`decide`, preflop decisions, equity estimation and existing traces do not read it.

## Identity is not strategy

`OpponentIdentity` holds `provider`, `key`, `key_kind`, `scope`, optional
`persistence_evidence` and optional `display_name`. Provider, kind and key form a
collision-free tuple-encoded persistent lookup key. Name is metadata only,
excluded from profile, bundle and diagnostic serialization.

| Identity | Cross-match profile lookup |
| --- | --- |
| Canonical bot UUID, persistent scope, explicit mapping provenance | Allowed |
| External bot ID, persistent scope, explicit mapping provenance | Allowed |
| Bot UUID without proven persistent scope | Denied |
| Transport `participant_id` | Denied; default resolver uses match scope |
| Name only, missing ID, malformed ID | Denied; anonymous match-local learning |
| Unknown identity scope | Denied |

The installed SDK passes the `seats` mapping to lifecycle hooks. The inspected
[transport protocol](https://github.com/chipzen-ai/chipzen-sdk/blob/main/docs/protocol/TRANSPORT-PROTOCOL.md)
describes `participant_id` as stable and opaque, but does not establish that it
is the persistent uploaded-bot UUID. The default resolver explicitly ignores an
unverified `bot_id` field. It never upgrades `participant_id` to `bot_uuid`.

An external application with a verified mapping may explicitly inject an identity
resolver returning `IdentityResolution`. The application is responsible for the
truth of its evidence reference; this code validates structure, not that external
assertion. No such mapping or populated bundle ships in the Chipzen runtime.

Two identities sharing a name never share profiles. Duplicate starts/reconnects
with the same match and identity do not reload priors. Rename changes only
in-memory metadata. A conflicting identity within the same match suspends memory
and discards attribution-dependent evidence, without affecting policy. A missing
identity on reconnect does not erase a previously resolved identity.

## Data model and trust boundaries

`Counts(successes, opportunities)` uses nonnegative integers with successes no
greater than opportunities. Historical profiles are frozen dataclasses with
sorted, immutable context/count pairs, total observed hands and matches,
configuration, schema version and provenance. They contain evidence, never
policy instructions. Runtime current counts live separately in `OpponentMemory`.
`OpponentBelief` is a derived immutable view; computing it does not mutate either.

Only persistent identities can appear in historical profiles. Stores revalidate
profiles. Profile absence, corruption, schema mismatch or failed lookup produces
no prior. Entire malformed bundles become empty rather than partially trusted.
Strict schemas reject unknown fields and duplicate JSON keys. No raw SDK object,
private cards, deck reveal, credentials, cookies or arbitrary metadata is copied.

Identifiers are bounded opaque strings, not filesystem paths. File names are
SHA-256 of persistent identity keys. This validation cannot identify a secret
that a caller deliberately mislabels as an allowed identifier; trusted local
normalized-input producers must honor the public-data contract.

## Reliable opportunity definitions

All current metrics require a complete, validated HU preflop prefix in a delivered
round result, a corresponding start containing two positive integer stacks, a
known hero seat, distinct SB/BB posts and non-forced choices after antes/blinds.
The opponent is the other seat; button is determined from the SB post. Call
amounts are not used because the documented legacy call representation is
ambiguous. Raise amounts are raise-to totals.

| Metric | Opportunity | Success |
| --- | --- | --- |
| `btn_open` | Opponent is button and has first voluntary choice | First action is raise/all_in |
| `btn_limp` | Same | First action is call |
| `btn_fold` | Same | First action is fold |
| `bb_iso` | Opponent BB responds to an observed BTN limp | Raise/all_in |
| `bb_check_limp` | Same | Check |
| `bb_3bet` | Opponent BB responds to a non-covering opening raise | Raise/all_in |
| `bb_call_open` | Same | Call |
| `bb_fold_open` | Same | Fold |
| `open_shove/bucket` | Unchanged pure `parse_hand` open opportunity, on validated subset | Its `open_shove` boolean |
| `reshove/bucket` | Unchanged pure `parse_hand` reshove opportunity, on validated subset | Its `reshove` boolean |

Shove metrics retain the existing parser's definition: a raise committing the
shorter effective stack can count even when the covering player retains chips.
Effective stack is the minimum starting stack; existing bucket boundaries are
5, 8, 12, 16, 20, infinity BB. The parser's ante treatment remains unchanged.
These are empirical frequencies, not inferred top-X% hand ranges.

Excluded: missing/invalid blinds or starting stacks; equal/short blind posts;
blind/ante forced all-ins; wrong first actor; repeated actors; invalid or
non-increasing raise totals; unknown actions/streets; backward street order;
explicit timeout actions; incomplete terminal prefixes; oversized histories.
Ordinary BB response metrics exclude covering opens, where a reraise opportunity
is absent. Missing timeout flags are not proof an action was voluntary: results
without such information are measured as delivered actions, with that limitation.

The new parser is intentionally stricter than legacy `ShoveModel`. Comparable
accepted observations have identical shove labels; rejecting more uncertain
hands never changes legacy counters. No postflop pot/stack reconstruction or
second authoritative game engine is introduced.

`limp_reraise`, c-bet, fold-to-c-bet, barrels, check-raises and showdown tendencies
are deferred pending explicit opportunity and line-context semantics. Raw
aggression counts are not substitutes. Archetypes and bluff-rate inference are
speculative and unimplemented.

## Statistical model

For one tendency, let historical counts be `(s_h, n_h)`, current counts be
`(s_m, n_m)`, base pseudocounts be `a_0, b_0`, and historical ESS cap be `H`:

```text
w = min(1, H / n_h)   if n_h > 0, otherwise 0
alpha = a_0 + w*s_h + s_m
beta  = b_0 + w*(n_h-s_h) + (n_m-s_m)
posterior mean = alpha / (alpha + beta)
historical ESS = min(n_h, H)
evidence ESS   = historical ESS + n_m
posterior variance = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
```

Defaults are explicit: `a_0=1`, `b_0=1`, `H=10`. The symmetric base is a uniform
Beta prior, not an assertion that a poker tendency is 50%. Ten historical
pseudo-opportunities is an initial inspectable engineering budget, not a
calibrated optimum or a poker tuning recommendation. It bounds historical
influence; callers can configure it, including zero. Configuration is recorded
in profiles and diagnostics. Runtime configuration controls weighting, and raw
historical counts remain available even if compiler configuration differs.

This is a **Beta-Binomial-inspired weighted prior**. Capping an arbitrarily large
history is a deliberate discount, not a claim that all original hands were
independent draws from one stationary generative model. Current evidence is
uncapped and therefore eventually dominates. There is no time decay, implicit
wall clock, calibrated strategy-change probability or automatic reset-on-shift.
A strategy switch affects counts progressively; fast change detection is future
work.

`interval(coverage)` returns a conservative Chebyshev interval intersected with
[0,1], using radius `sqrt(variance/(1-coverage))`. It is an at-least-mass bound
under this Beta model, **not** an equal-tailed credible interval, frequentist
coverage guarantee or exploitation confidence score. Sparse evidence often
returns [0,1]. Do not feed this interval directly into strategy without validation.

## Stores and uploaded versus remote execution

- `NullProfileStore`: no I/O, no historical persistence; current default.
- `InMemoryProfileStore`: explicit research fixtures/experiments; immutable values.
- `FileProfileStore(directory)`: explicit path, deterministic JSON, exclusive
  temporary file, flush/fsync then same-directory atomic replacement. A failed
  pre-replacement write preserves the old file. No hidden home-directory state.
  After the replacement, the parent directory is fsynced where the platform
  supports opening directories (`os.O_DIRECTORY`, i.e. POSIX), so the rename
  itself survives a crash. Without `O_DIRECTORY` (Windows), when the directory
  cannot be opened, or when the filesystem rejects directory fsync
  (`EINVAL`/`EBADF`/`ENOTSUP`), that step is skipped. Any other directory-fsync
  error is raised: the new file is already visible but may not be durable.
  Concurrent writers are last-writer-wins, not transactional merging;
  multi-process aggregation is not promised.
- `BundledProfileStore`: explicitly parses bytes or an explicitly opened path;
  read-only lookup. No runtime scan or constructor-time implicit file discovery.

Profile cap: 1 MiB. Bundle cap: 32 MiB and 10,000 profiles. Pending starts: 16;
input history: 256 entries. Match dedup/rejected sets grow with delivered hands
and reset with the match. This growth is deliberate to avoid recounting an old
round after eviction. No cross-match profiles are saved automatically.

Remote and uploaded bots use these same classes. A remote application may
explicitly persist compiled evidence. An uploaded application can eventually
load an offline read-only snapshot, but the current image contains no snapshot
and performs no profile reads by default. No packaging files are changed.
The existing recursive core staging naturally includes the new Python modules;
its recipe is unchanged, not its code-bearing output bytes.

## Local normalized dataset and compiler

Input is one JSON object with exactly `schema_version: 1`, `data_version`, and
`observations`. Each observation contains exactly:

```text
schema_version, identity, match_id, hand_id,
source_id, source_sha256, source_kind, stats
```

Identity uses the five serialized fields (`display_name` is excluded). Stats is
`{tendency: {successes: 0|1, opportunities: 1}}`. Allowed source kinds are
`synthetic`, `public_history`, `delivered_match`; this is a producer attestation,
not proof that a remote source was public. Do not ingest private history.

The compiler deduplicates by persistent identity + match + hand, accepts exact
repeats, and rejects conflicting repeats. It never averages conflicts. Counts
are integers; ordering is stable. Input bytes and source-content SHA-256 values,
source IDs, data version, input schema, observation count and match/hand coverage
are retained in provenance. Exact input-byte changes change provenance even if
aggregate statistics remain equal. No timestamps are synthesized.

```bash
uv run python scripts/build_opponent_priors.py \
  --input /explicit/normalized.json --output /explicit/priors.json \
  --alpha-base 1 --beta-base 1 --historical-ess-cap 10
```

Input is bounded to 32 MiB / 100,000 observations. Output replacement is atomic.
Errors are generic and do not echo private input. The script prints the output
SHA-256. Two builds of identical bytes/configuration produce identical bytes.
The bundle has schema version, data version and a stable-key-ordered profiles
mapping. Each profile contains raw counts, coverage, provenance, configuration
and per-context capped historical ESS. It never emits an action.

No network collector or raw-response storage is implemented. Future raw public
source retention must remain separate from normalized observations. Only an
allowlisted public accounting/action subset may be normalized; private cards,
user data and raw SDK response objects must not be blindly serialized.

## Public source discovery: incomplete, no collection authorized

The official SDK documentation is a documented public protocol reference, not a
public match-history download API. The [developer manual](https://github.com/chipzen-ai/chipzen-sdk/blob/main/docs/DEV-MANUAL.md)
describes replay/log UI access and a parsed bot-log route. Its public-history
access authorization, pagination and rate-limit contract have not been
established; it is not an approved collection source. The website alone does
not establish an API contract. No history endpoint has been queried by this
implementation, and no cookies, credentials or browser sessions are reused.

A future collector requires a separate checkpoint covering URL/interface,
authentication, public/private boundary, pagination, rate limits and stable bot
identity semantics. Then design a finite, single-worker CLI with explicit base
URL/output, bounded retry/backoff, Retry-After handling, resumable cursor and
content deduplication. Neither guessed endpoints nor an automatic daemon are
acceptable. No concrete request rate is presented as a platform guarantee.

## Runtime lifecycle and failure isolation

`HoldemAgent.__init__` retains an injected memory or constructs a fresh inert
memory. It performs no new I/O, lookup, environment discovery, clock read, RNG
operation or logging. `self.shove_model = ShoveModel(preflop_config)` remains in
its original position before memory composition.

**Default memory.** `HoldemAgent(...)` without `opponent_memory` gets its own
fresh `OpponentMemory()` (never a shared singleton) with a `NullProfileStore`,
the default `BeliefConfig` and the match-scoped default resolver. Consequences:

- construction reads no files, environment, clock or network and consumes no
  RNG state;
- policy output is unchanged: no decision path reads memory, and the canonical
  oracle compares the default bot against `3f8291d` (which has no memory);
- nothing is persisted: `NullProfileStore.save` is a no-op and memory never
  calls `save`;
- a store lookup happens only when an identity resolves to persistent scope,
  which the default resolver never produces; with the default store even that
  lookup returns no prior.

The default memory therefore still learns match-local counts in the background
from completed rounds; that state is visible only through `snapshot()`.

The adapter forwards memory only **after** existing operations:

```text
match_start / reconnect: seat learning → SDK → accounting → memory
round_start:            ShoveModel.record_start → SDK → accounting → memory
round_result:           ShoveModel.observe_result → SDK → accounting → memory
turn_result:            SDK → accounting → memory (no new evidence)
match_end:              SDK → memory closes pending state; no autosave
```

No legacy hand hooks are overridden, so SDK round-to-hand delegation cannot
double-count memory. Results without starts are ignored. Matching starts are
copied using an allowlist containing only stacks. Identical duplicates do not
reset evidence; conflicting duplicate starts reject that hand. Malformed
results do not partially update and can be replaced by a later complete result.
Explicit mismatching match IDs are ignored. Reconnect does not invent missing
starts or backfill evidence. Unexpected memory errors disable only the optional
component; existing hooks, observer, policy exceptions and diagnostics are
unchanged. Store errors produce generic-prior fallback instead.

`snapshot()` is the explicit diagnostic interface; callers may choose to display
it. No new log output occurs automatically, and existing preflop trace bytes
remain unchanged. Per-action work is unchanged: memory has no `decide` callback.

## Validation and future boundary

Tests cover identity scope/rename/collisions, corruption/atomic writes, exact
compiler output, synthetic limp/open/shove/tight/aggressive/switch patterns,
posterior limits, constructor side effects, lifecycle ordering and failure
isolation. The canonical 3f8291d oracle compares every full RNG state, SDK action
(type, amount, params), exception, stdout/stderr, seat, existing ShoveModel
counters/config/pending ordering/dedup and accounting state. It includes all
preflop classes/contexts, lifecycle edge cases, malformed states and all 28
postflop fixtures, with traces both enabled and disabled. Snapshots stream to
compressed files to bound test memory; comparison still uses full values.

Future flow is one core, not uploaded/remote strategy forks:

```text
Authoritative platform state/history → derived context
                                      + OpponentBelief
                                      → BasePolicy / SafeExploitController
                                      → Decision
```

The platform owns reality; SOH owns interpretation and memory. A future safe
exploit controller may consume validated behavioral uncertainty, never literal
player names or ID-specific strategy branches. Its design and controlled,
ablated policy integration is the next strategic milestone, after this substrate
is validated. All baseline/match-local/history variants currently choose the
same actions; only their separately inspected model state differs.

## Deferred

Persistent hosted bot-ID mapping; public-history collector; populated bundles;
postflop context metrics; age-based decay; change-point probabilities; archetype
posteriors; inferred hand ranges; SafeExploitController; policy deviations;
pricing repair; any deployment. Existing Leduc research and older roadmap text
are not claims that those techniques are used by this Hold'em memory component.
