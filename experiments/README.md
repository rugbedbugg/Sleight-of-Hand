# Experiment control plane

Controlled, reproducible comparisons of **already-defined** Sleight-of-Hand
configurations. The control plane selects which configuration is evaluated;
it is not a policy layer and never changes policy during a match. Results
are research evidence: nothing here is promoted to production automatically.

    experiment result -> candidate recommendation -> review
        -> regression validation -> explicit promotion

Doctrine: *platform owns reality, SOH owns interpretation and memory,
experimentation owns comparison.*

## Pipeline

```
ExperimentSpec (immutable, hashed) --> scheduler (bounded worker processes)
    --> platform adapter (local dealer | chipzen external API)
    --> raw journal (append-only, sealed, checksummed, read-only)
    --> normalization (regenerable from raw)
    --> metrics (outcome, behavior, model, accounting)
    --> analysis (paired comparisons within one provenance stratum)
    --> recommendation (research evidence only)
```

Code lives in `sleight_of_hand/experiments/`. It is never imported by the
Chipzen runtime and is not staged into its image (`tests/test_experiment_firewall.py`).

## Research supervisor v1

The canonical public interface is installed by `uv sync --locked`:

```sh
uv run research status
uv run research plan
uv run research run E0003
uv run research analyze E0003
uv run research auto
uv run research optimize O0001 --json
uv run research optimize O0002 --json
```

`auto` executes approved research inside a declared authority envelope. It is
not an RL trainer or a policy layer. It never influences an action inside a
hand. Each command accepts `--programme PATH`, `--root PATH`, and `--json`
after the subcommand. Defaults are this checkout's `experiments/programme.json`
and `runs/`. The editable install includes the existing Chipzen adapter package
needed by local workers; it does not change the production staging contract.

`status` and `plan` read metadata and validate configuration without registering
specs, recovering runs, creating an index, or starting matches. They show every
item's arms, pinned hashes, platform/provenance, availability and reason, shard
counts, incomplete/live runs, analysis status, governance and intended action.
The current source SHA and policy fingerprint are separate from a historical
spec's `source_sha`. A policy-inert infrastructure change does not stale a spec;
a fingerprint change never rewrites the historical spec (see
[policy epochs](#policy-epochs-and-historical-evidence)).
Production is always shown as `LOCKED / untouched`.

### Reviewed programme

`programme.json` is versioned JSON, not a directory scan. Its required fields:

| Field | Contract |
|---|---|
| `schema_version` | `1` for fixed-only programmes; `2` adds required `campaigns` |
| `resources.workers` | Integer 1–32; the checked-in programme uses 2 |
| `resources.min_free_mib` | Nonnegative memory floor; the checked-in programme uses 1024 MiB |
| `experiments` | Ordered list, each with `id`, `governance`, `reason`, `specs`, `depends_on` |
| `specs` | Explicit `{path, spec_hash}` references; relative paths stay inside the programme directory |
| `governance` | `LOCAL_APPROVED`, `UNRATED_APPROVED`, or `REQUIRES_REVIEW` |
| `depends_on` | Unique experiment IDs appearing earlier in the programme |
| `campaigns` (v2) | Ordered explicit `{path, campaign_hash}` references to reviewed local campaigns |

Unknown fields, duplicate IDs/paths/arms, mismatched hashes, mixed strata,
cyclic/forward dependencies, unapproved platforms, and unsupported provenance
are rejected. Local provenance is limited to LOCAL_SELFPLAY, SYNTHETIC,
SCRIPTED_PROBE and BENCHMARK. Chipzen permits only LIVE_UNRATED/SCRIPTED_PROBE
and still delegates availability and validation to its existing adapter.
Changing this file changes authority and requires normal repository review.
The [bounded optimizer](OPTIMIZER.md) generates candidates only inside a
separately pinned campaign. It cannot expand its search space or budget.
[O0002](SCREENING.md) is a fixed Resolution-V screening campaign run by the same
campaign machinery.

A campaign may depend on fixed experiments or on **earlier** campaigns, with
an optional required classification (for example O0002 requires O0001
`COMPLETE / NO_IMPROVEMENT_DETECTED`). Unknown IDs, forward references and
cycles (including self-dependency), duplicate dependencies, and impossible
requirements (a classification on a fixed experiment, a non-completion
classification, or a status other than COMPLETE) are rejected at load. Every
campaign's declared seed streams must be disjoint from every fixed experiment's
streams and from earlier campaigns' streams.

The initial programme explicitly lists E0001–E0005. E0004 depends on E0002 and
requires the artifact hash already pinned by its immutable spec. The supervisor
does not invent or substitute a prior: a missing bundle is BLOCKED, a mismatched
bundle fails validation. The existing `build_local_prior.py` remains the explicit
prior-compilation interface; the resulting hash must match before E0004 runs.

### Finite execution and evidence

An auto cycle makes one pass in dependency order, re-planning after each item,
then advances at most one batch for each eligible reviewed campaign, in order.
It delegates unfinished shards to the existing spawned-process scheduler and
uses the existing analysis implementation with raw verification enabled.
Completed work is verified and skipped; analysis receipts bind reports to the
approved specs and completed evidence. Raw and derived checksums are verified
before analysis or accepting an already-complete item. Altered evidence fails
closed. No observed result changes a fixed horizon or stopping rule.

Unavailable platforms are BLOCKED, so missing Chipzen credentials/challenge do
not prevent later local work. Dependencies and missing artifacts are also
explicit blocks. A governance boundary or stale fingerprint requires review.
A failed experiment stops the cycle and retains evidence. Recorded failures
with unfinished work are not automatically retried on later invocations of the
same programme; investigation and reviewed recovery are required. Dead-worker
shards resume through scheduler recovery; live workers block duplicate work.

The supervisor opts into scheduler admission checks that also protect the first
worker: insufficient or unknown free memory stops admission and returns a
resource block once any in-flight work drains. It also stops admitting shards
after a failure; already-running workers finish and retain their evidence.
Existing developer scripts keep their previous scheduler defaults. There is
no indefinite credentials/resource polling and no automatic retry loop.
A nonblocking per-root process lock prevents overlapping supervisor cycles.
Independent legacy scheduler invocations should not target the same root while
a supervisor cycle is active.

The result distinguishes COMPLETE, PENDING, BLOCKED, FAILED and REQUIRES_REVIEW.
FAILED and REQUIRES_REVIEW take precedence; a cycle-level resource stop is
BLOCKED. Otherwise the cycle is PENDING while any selected item or campaign
still has autonomous work, even if another independent item (such as E0001
without credentials) is BLOCKED. BLOCKED ("no eligible work remains") is
reported only when every unfinished selected row is itself blocked.
FAILED returns exit code 1; completed, blocked and review-boundary cycles exit
cleanly with their explicit disposition. Invalid commands/programmes fail.
Conclusions are non-binding INCONCLUSIVE or REQUIRES_REVIEW summaries of the
existing statistical verdicts/accounting evidence. More samples require a new
reviewed experiment/spec. A winning result never changes policy or production.

Each execution command writes a unique `runs/supervisor/<cycle-id>.jsonl`:
append-only, sequence-numbered, timestamped, flushed/fsynced events, read-only
when closed. Start records include schema/version, source/policy fingerprint,
programme hash, arguments and the complete initial plan. Item inspections,
action starts/results, analysis receipts, failures, governance stops and the
final disposition follow. A crash can leave an unfinished record; it is never
reported as a successful cycle. These records are supervisor decisions, not raw
poker evidence. The existing credential redactor scrubs all output and records;
environment values are never copied into decision records.

### Policy epochs and historical evidence

Two statements are kept separate:

| Term | Meaning |
|---|---|
| policy revision | Identity of the runtime policy that produced evidence: the `policy_revision` pinned in a spec or campaign and recorded in run manifests and campaign events. It is never rewritten. |
| current runtime revision | `runner.policy_revision()` of this checkout: the only policy that may execute now. |
| historical complete | Verified, immutable evidence produced under an **older** revision. |

Every experiment and campaign row reports `policy_state`,
`recorded_policy_revision`, `current_policy_revision` and `policy_note`
(non-current rows also print a `policy:` line):

| Recorded vs current | Work | Status / `policy_state` | Execution |
|---|---|---|---|
| same | complete | COMPLETE / CURRENT | none needed |
| same | unfinished | normal governance / CURRENT | as approved |
| older | complete, verified | COMPLETE / HISTORICAL_COMPLETE | never again |
| older | unfinished | REQUIRES_REVIEW / STALE_REQUIRES_REVIEW | never |
| any | integrity or authority failure | FAILED or REQUIRES_REVIEW / UNVERIFIED | never |

A fixed experiment is historical complete only when every approved shard is
complete and its existing analysis receipt is current. Unauthorized specs,
out-of-horizon or failed shards still require review or fail first, and
`auto` still verifies raw and derived evidence before accepting it. Complete
evidence whose analysis receipt is missing, stale or unverifiable requires
review: no analysis is generated under a newer policy, and an explicit
`research analyze` of a historical item stops for review. Historical
re-analysis would need a separately reviewed protocol.

A campaign replays its whole journal against the revision and authority it
recorded (campaign hash, legacy programme prefix or per-campaign authority,
dependency identities, generated specs, reports, receipts, hash chain). Only
then is its epoch decided: a durable final classification under an older
revision is historical complete and keeps its exact classification, incumbent,
budget, stop reason and hashes; anything unfinished (pending search or
screening, awaiting or partial confirmation, generated specs awaiting work)
requires review. A FAILED halt stays FAILED. Successful replay never grants
execution: `advance()` separately refuses any new event, spec or shard unless
the current revision equals the campaign's reviewed revision (and, for a
screen, its centre equals today's canonical defaults; that check runs at
execution, not load, so a future default change cannot make history
unreadable). The worker firewall in `runner.run_shard` is unchanged: a spec
whose revision differs from the current one always refuses to run.

Dependencies state facts about verified results, not executability. A
historical complete campaign or experiment satisfies a later dependency's
COMPLETE/classification requirement; its identity remains pinned in the
dependent campaign's authority, so altering it still invalidates the
dependent. A dependency on stale unfinished work is unmet (BLOCKED).

A verified stale-policy REQUIRES_REVIEW is **item-local**: a full `auto` pass
records it as skipped and continues with unrelated current work. Selecting it
explicitly (`research run E0001`, `research analyze ...`,
`research optimize O000N`) stops the cycle for review. Every other
REQUIRES_REVIEW (governance, authority, unapproved specs, horizon) is still a
cycle-level stop. Cycle disposition, in order: FAILED; cycle-level
REQUIRES_REVIEW; resource stop BLOCKED; PENDING while any selected row has
eligible work; REQUIRES_REVIEW when only item-local reviews remain; BLOCKED;
COMPLETE. So after a policy change a stale E0001 cannot mask an eligible new
campaign, and once only stale work remains the cycle says it needs review.

Historical complete does **not** mean executable today, applicable unchanged
to a newer policy, promoted, or production-approved. It only means the
historical research claim is still internally valid for the policy revision
that produced it. Comparing epochs requires an explicit new reviewed
experiment or campaign pinned to the current revision.

### Developer compatibility entry points

Existing scripts remain available, with their existing arguments:

```sh
uv run python scripts/run_experiment.py --spec experiments/specs/E0002-canonical.json
uv run python scripts/run_experiment_batch.py --match 'E0003-*.json' --workers 2
uv run python scripts/analyze_experiment.py --experiment E0002
uv run python scripts/build_local_prior.py --experiment E0002 --arm canonical \
    --opponent script.tag_simple --data-version e0002-tag-simple-v1 \
    --output runs/priors/e0002-tag-simple.json
```

All generated work lives under the git-ignored `runs/` hierarchy: `index.sqlite`,
`runs/<run-id>/`, `analysis/`, `batches/`, `priors/`, `supervisor/`, and `optimizer/`.

## ExperimentSpec

A frozen dataclass identified by the SHA-256 of its canonical JSON
(`sort_keys`, compact separators, finite numbers). Spec files store the spec
plus its hash; loading recomputes and compares it. Unknown fields,
credential-like keys and credential-like values are rejected.

| Field | Meaning |
|---|---|
| `experiment_id`, `arm` | One experiment has many arms; a config change is a new arm (`derive_arm`) |
| `source_sha` | Commit the evaluated policy was taken from |
| `policy_revision` | `sha256:` digest of the runtime policy sources (line-ending insensitive); a run refuses to start if the code differs |
| `policy_config` | Only approved surfaces: `opponent_memory`, `params` (the five `PolicyParams`), `samples`, `historical_prior` |
| `platform`, `platform_config` | Platform name and its explicit conventions |
| `game_variant` | `nlhe_hu` |
| `opponent_cohort` | Namespaced opponents on the spec's own platform; match `i` plays `cohort[i % n]` |
| `seed_policy` | `base_seed` + `common_random_numbers`; every stream derives from the seed, role and match index |
| `stopping` | `min/max_matches`, `min/max_hands`, `hands_per_match`, `matches_per_shard`, `evaluation_interval_matches`, `early_stopping` |
| `metrics` | Any of `outcome`, `behavior`, `model`, `accounting` |
| `provenance` | `LOCAL_SELFPLAY`, `SCRIPTED_PROBE`, `LIVE_UNRATED`, `LIVE_RATED`, `BENCHMARK`, `HISTORICAL_PUBLIC`, `SYNTHETIC` |
| `created_at` | Immutable authoring timestamp |

**Stopping** is fixed-horizon and declared before the first match. Interim
evaluations at each interval are recorded in the index as *non-binding* looks;
`early_stopping` must be `"none"`, so no result can stop a run early.
`min_*` are the sample sizes analysis requires before it reports a verdict.

**Identity firewall.** Opponents are `platform/kind/key`; equality includes the
platform, so identities never match across platforms, and display names are
metadata only. Local scripted opponents use a `local`-provider persistent
identity (evidence `local_policy_registry`) that cannot collide with any
Chipzen identity. The existing `OpponentIdentity` rules are unchanged:
Chipzen participant IDs remain match-scoped.

## Evidence layout

```
runs/<run-id>/
    manifest.json            spec, provenance, source/HEAD, policy revision,
                             runtime versions, platform metadata, progress,
                             status, checksums
    raw/events.jsonl.gz      every platform event (deterministic gzip, 0400)
    normalized/hands.jsonl
    normalized/decisions.jsonl
    normalized/observations.jsonl
    metrics.json
    result.json
```

Raw events are appended during the run and fsynced after every match. Sealing
compresses them deterministically, verifies the round trip, records the
SHA-256 of the uncompressed stream and makes the file read-only. Normalized
data is always derived from the sealed raw stream (`normalization.VERSION`).
A failed run keeps its raw events read-only and is marked `FAILED`. A run left
`RUNNING` by a dead worker is marked `INCOMPLETE`; it is never completed,
and its shard is re-run with a new run ID.

### SQLite index (`runs/index.sqlite`)

Migrations are an append-only, numbered list applied once each inside a
transaction (`storage.MIGRATIONS`).

| Table | Columns |
|---|---|
| `schema_migrations` | `version`, `name` |
| `specs` | `spec_hash` (PK), `experiment_id`, `arm`, `platform`, `provenance`, `policy_config_hash`, `created_at`, `spec_json` |
| `runs` | `run_id` (PK), `spec_hash`, `shard_start`, `shard_end`, `worker_id`, `status`, `started_at`, `finished_at`, `matches`, `hands`, `raw_sha256`, `run_dir`, `error` |
| `evaluations` | `spec_hash`, `completed_matches`, `recorded_at`, `binding` (always 0), `summary_json` |

## Metrics

- **Outcome:** chips/hand and bb/100 with match-clustered standard errors and
  95% intervals; hands/matches won and lost; showdown rate.
- **Behavior** (both seats, Wilson intervals): `btn_rfi`, `btn_limp`,
  `btn_fold`, `bb_3bet`, `bb_call_open`, `bb_fold_to_open`, `bb_iso`,
  `preflop_shove`, `postflop_aggression`, `cbet`, `fold_to_cbet`.
- **Model:** OpponentMemory replayed over SOH-delivered messages. Each
  prediction is the belief mean immediately before the hand that resolves
  it: Brier score, log loss, 10-bin calibration error, Brier skill against a
  uniform prior, first-half vs second-half convergence, final ESS.
- **Accounting:** observer captures reassembled (checksum before parse) and
  classified per match; the primary verdict is the single agreed conclusive
  class, else `INCONCLUSIVE`.

Analysis compares arms only within one `(platform, provenance)` stratum, and
only arms with identical seeds (under common random numbers), cohort, platform
configuration and match length. It reports per-match paired differences and
decision-by-decision identity.

## Platforms

| Platform | Status | Notes |
|---|---|---|
| `local` | AVAILABLE | In-process heads-up NLHE dealer driving the real `SleightOfHandBot` adapter through the SDK hook order and `GameState.from_turn_request`. Its pot, `to_call`, stack-timing and bet-cap conventions are explicit configuration; local results describe those conventions, not Chipzen's |
| `chipzen` | READY_FOR_CREDENTIALS | Official external-API remote play. See [PLATFORMS.md](PLATFORMS.md) |

Local opponents: `script/calling_station`, `script/random_legal`,
`script/tag_simple`, `soh/canonical` (self-play) and `probe/accounting_cover`.

## Experiment registry

| ID | Arms | Platform / provenance | Purpose |
|---|---|---|---|
| E0001 | accounting-probe | chipzen / SCRIPTED_PROBE | Live reachability of unmatched covering chips in the delivered postflop pot. Blocked on credentials and a dashboard challenge |
| E0002 | canonical, memory-off | local / LOCAL_SELFPLAY | OpponentMemory is policy-inert; the orchestration layer changes nothing |
| E0003 | canonical, aggression-0.6, call-threshold-0.45, bluff-0.05 | local / LOCAL_SELFPLAY | Offline sweep of the existing `PolicyParams` surface |
| E0004 | no-prior, e0002-prior | local / SYNTHETIC | Historical-prior machinery: predictive value of a compiled prior, with identical decisions. The bundle lives in `runs/priors/` (not committed); `build_local_prior.py` rebuilds it byte-identically from E0002's runs, and the spec pins its hash |
| E0005 | committed, contestable, capped-bets | local / SCRIPTED_PROBE | Rehearsal of the E0001 probe pipeline under each dealer convention |

### E0001 on Chipzen

The research worker never uses the production bot, token or image. It needs:

1. two **External API** bots created in the Chipzen dashboard: the SOH
   research bot and the scripted probe bot, each with its own token;
2. `CHIPZEN_RESEARCH_TOKEN`, `CHIPZEN_RESEARCH_BOT_ID`, `CHIPZEN_PROBE_TOKEN`
   and `CHIPZEN_PROBE_BOT_ID` in the environment (never in files);
3. an **unrated** same-owner challenge between the two bots created in the
   dashboard (same-owner matches are never rated), then
   `CHIPZEN_RESEARCH_CHALLENGE_READY=1`.

Then inspect `uv run research plan` and use `uv run research run E0001`.
Only already-approved research identities and explicitly unrated challenges
are permitted. Implementation tests use fixtures and never live matches.
A `matched` notification that is not explicitly unrated fails the run.
The classification is measurement only and never triggers a pricing change.
