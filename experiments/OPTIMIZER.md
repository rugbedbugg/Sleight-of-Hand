# Bounded Research Optimizer v1

The optimizer proposes immutable configurations inside a reviewed local search
space. It is black-box, derivative-free parameter optimization, not reinforcement
learning. It never runs inside a hand and has no authority to promote a result.

| Layer | Authority |
|---|---|
| Policy | Decide poker actions |
| Learning | Update opponent beliefs |
| Experiment Control Plane | Execute immutable ExperimentSpecs |
| Research Supervisor | Decide which approved work executes next |
| Bounded Optimizer | Propose configurations within campaign bounds and budget |

## Commands

```sh
uv run research status
uv run research plan
uv run research auto
uv run research optimize O0001 --json
```

`status` and `plan` inspect authority and replay campaign events without writing
files, registering specs, recovering shards, or starting experiments. `auto`
processes fixed experiments and at most **one batch per campaign**. A batch is
one search generation or the independent confirmation experiment. Run `auto`
again to progress another batch. Dependencies are evaluated after fixed work,
so a blocked E0001 does not block O0001. `optimize ID` selects one approved
campaign, inside the same supervisor lock and resource controls.

One batch per cycle gives each generation a clear durability boundary and keeps
every invocation finite. Search termination is durably recorded before a
confirmation batch can be proposed. After completion, subsequent invocations
verify/replay history and perform zero new optimizer actions or experiment work.

Example before execution:

```text
O0001  PENDING  [LOCAL_APPROVED]
  algorithm: bounded-pattern-search v1
  incumbent: {"aggression": 0.8, "bluff_freq": 0.15, "call_threshold": 0.35,
              "steepness": 8.0, "value_bet_threshold": 0.55}
  generation: 0
  steps: {"aggression": 0.05, "bluff_freq": 0.05, "call_threshold": 0.05,
          "steepness": 2.0, "value_bet_threshold": 0.05}
  candidate budget: 0 / 45 (45 remaining)
  search: PENDING; confirmation: NOT_STARTED
  classification: PENDING; next: GENERATE_AND_RUN
  production: LOCKED / untouched
```

Actual output also includes the campaign hash. Campaign states are PENDING,
SEARCHING, CONFIRMING, COMPLETE, BLOCKED, FAILED, and REQUIRES_REVIEW. The
generation counter counts completed search generations; a pending experiment
uses that number as its zero-based generation ID.

## Reviewed authority

Programme schema 2 adds a required `campaigns` list of explicit
`{path, campaign_hash}` references. Schema 1 remains supported for fixed-only
programmes. Campaign IDs cannot duplicate experiment or campaign IDs. Campaign
dependencies must name fixed experiments in the same programme. Paths remain
inside the programme/campaign directory. No directory discovery grants authority.

Campaign schema 1 requires every field below and rejects unknown fields:

| Field | Contract |
|---|---|
| `schema_version` | `1` |
| `id`, `purpose`, `governance` | Explicit identity, purpose, LOCAL_APPROVED or REQUIRES_REVIEW |
| `campaign_seed`, `seed_derivation` | Integer seed; `sha256-domain-separated-v1` |
| `template` | Relative ExperimentSpec path and pinned `spec_hash` |
| `policy_revision` | Expected runtime fingerprint, matching the template |
| `initial` | Full five-parameter starting vector |
| `parameters` | Unique supported parameter names with min/max, initial/min steps |
| `algorithm` | `{kind: bounded-pattern-search, version: 1}` |
| `search_stopping`, `confirmation_stopping` | Exact fixed-horizon StoppingRules; at least two matches |
| `max_generations`, `max_generated_vectors` | Positive integers, capped at 4 and 45 in v1 |
| `objective` | `paired-bb-per-100-bonferroni-all-pairs` |
| `resources` | At most two workers; memory floor no weaker than programme |
| `depends_on` | Unique fixed-experiment dependencies |
| `created_at` | Declared reproducible spec timestamp |
| `fixed_policy` | samples=128, opponent_memory=true, historical_prior=null |

The campaign hash is `sha256:` plus SHA-256 of canonical, sorted, compact finite
JSON. Any search-space or other authority change changes this hash. Updating
the committed campaign and its programme pin requires review. A changed
campaign/programme hash during a run requires review; existing history is
never widened or silently repaired.

## O0001

[O0001.json](O0001.json) is the reviewed authority. It depends only on E0003
COMPLETE and pins E0003's canonical spec
`45d99c7cb7ad35bf1915be66ce767d56fe2ec18e5427907cbe41c60a4018bcc6`.
The template supplies the local platform conventions, LOCAL_SELFPLAY provenance,
heads-up NLHE game, calling_station/random_legal/tag_simple cohort, and
behavior/outcome metric groups. E0003's specs remain untouched.

| Parameter | Initial | Minimum | Maximum | Initial step | Minimum step |
|---|---:|---:|---:|---:|---:|
| value_bet_threshold | 0.55 | 0.45 | 0.65 | 0.05 | 0.0125 |
| call_threshold | 0.35 | 0.25 | 0.45 | 0.05 | 0.0125 |
| bluff_freq | 0.15 | 0.05 | 0.25 | 0.05 | 0.0125 |
| aggression | 0.80 | 0.70 | 0.95 | 0.05 | 0.0125 |
| steepness | 8.0 | 4.0 | 12.0 | 2.0 | 0.5 |

Campaign seed: `202610080001`. Declared creation time: `2026-10-08T00:00:00Z`.
Budget: at most four search generations and 45 materialized arm vectors.
Resources: two workers, 1024 MiB minimum free memory.

| Horizon | Search generation | Held-out confirmation |
|---|---:|---:|
| Matches per arm | 20 | 60 |
| Hands per match | 100 | 100 |
| Planned hands per arm | 2,000 | 6,000 |
| Matches per shard | 5 | 10 |
| Evaluation interval (matches) | 10 | 20 |
| Early stopping | none | none |

E0003 motivates this bounded region, including exclusion of aggression=0.60.
Its findings do not establish an optimum. Its outcomes and hands are never
pooled into O0001 or used for confirmation.

## Deterministic pattern search

Pattern search needs no third-party dependency, surrogate model, or hidden
fitness function. Every proposal is one interpretable coordinate move.

For the current incumbent, visit parameter names in sorted order and consider
minus/plus the active step using decimal arithmetic. Omit directions outside
the approved bounds; never clip. Deduplicate full vectors by canonical JSON,
sort them by those bytes, and prepend the incumbent baseline. Revalidate every
full vector. Untunable parameters must equal their initial values.

Each batch becomes normal ExperimentSpecs with identical environment, cohort,
stopping rule and common random numbers. IDs are `O0001.search.NN` or
`O0001.confirmation.00`. Arms are `baseline` or `candidate-` plus the first 20
hex digits of SHA-256 of the full vector. Hash collisions are rejected.
Spec identity uses the campaign's declared timestamp and the template's policy
source SHA; actual execution HEAD is separately recorded in run manifests and
the campaign start event. Infrastructure commits therefore do not introduce
wall-clock identity changes. The existing policy fingerprint remains enforced.

Generated specs are written under `runs/optimizer/O0001/specs/`, read-only
before publication. They never enter committed `experiments/specs/`.

### Seeds

For phase `search` and generation N, canonicalize
`[campaign_id, campaign_seed, N, "search"]`, SHA-256 it, take the first 16 hex
digits as an integer, and mask to 63 bits. Confirmation uses generation zero
and literal `"confirmation"`. The campaign declares this derivation before
search. All possible blocks are computed before execution, independent of
outcomes, time, global RNG, scheduling, or OS entropy.

| O0001 block | Base seed |
|---|---:|
| search-0 | 9118584578214162485 |
| search-1 | 7376750585917583887 |
| search-2 | 975674925447566693 |
| search-3 | 5128848628376879690 |
| confirmation-0 | 6646139538147470181 |

The existing SeedPolicy derives deck/SOH/opponent streams from base seed, role
and match index. Validation exhaustively checks all declared stream seeds for
collisions across generations, confirmation, and the historical template.
Any collision fails closed; it does not silently choose replacement seeds.
All arms within a batch intentionally share streams.

### Selection and multiplicity

The original analysis provides paired per-match differences in bb/100,
their mean, standard error, normal 95% interval and recommendation. Audit found
no existing multiplicity correction. The central analysis now accepts
`familywise=True`: it uses the same estimator and standard error with the
Bonferroni critical value `NormalDist().inv_cdf(1 - 0.05/(2*m))`, where `m`
is the full comparable unordered arm-pair count in the stratum. With eleven
arms, `m=55`. Both the original interval and corrected interval/method/family
size remain in the report. Historical callers retain their original verdicts.

The optimizer selects only incumbent comparisons whose corrected recommendation
is DIFFERENCE_DETECTED **and** whose higher arm is the candidate. Among those,
choose the largest signed paired mean improvement over the incumbent, then
the lexicographically smallest deterministic arm name on an exact tie.
Record every comparison and the exact selection reason. Raw means alone,
behavior metrics and model metrics cannot select a candidate.

This correction controls the declared within-generation comparison family
under the existing normal-interval approximation. Adaptive search does not
establish a campaign-level advantage. Only the independent two-arm held-out
confirmation can classify the final candidate as promising.

No qualifying candidate means retain the incumbent and halve every step,
bounded below by its declared minimum. A no-improvement generation evaluated
with all steps already at minimum ends the search.

### Budgets and termination

Every materialized arm counts, including repeated incumbent baselines and the
two confirmation arms. Admission reserves two confirmation vectors before
starting search. A batch is admitted only if the **whole** batch fits; it is
never truncated based on results. This conservative accounting can stop an
interior four-generation search at three generations (33 search arms) because
44 search arms plus two confirmation arms would exceed 45. Four generations
remain possible when boundaries reduce a batch. Unused budget grants no
additional work.

Stop at maximum generations, minimum-step non-improvement, no bounded neighbor,
or insufficient remaining candidate budget. Scheduler failure, unavailable
resources, evidence failure and governance boundaries also stop execution.
Resource failures retain a BLOCKED campaign requiring reviewed recovery;
there is no automatic retry/polling daemon.

## Campaign provenance and recovery

`runs/optimizer/O0001/events/NNNNNN.json` is an append-only, sequence-numbered
hash chain. Each event is atomically published read-only after fsync; the
directory is fsynced. A replaceable `.pending` scratch file is never canonical
history. The supervisor's existing nonblocking per-root lock serializes writes.

Events are start, batch, ready, result, stop, final, or halt. They record version,
campaign/programme hashes, source HEAD/policy fingerprint, all seed blocks,
incumbent and steps, generated vectors/spec hashes/file hashes, budget consumed,
experiment IDs, completed run IDs/raw hashes, analysis/report receipt hashes,
all comparisons, selection/shrink reasons, stop reason and final classification.
Credential-like content is rejected; errors pass through the existing redactor.

Replay validates every transition, rederives authorized spec identities, and
recomputes selection from its hash-bound report. There is no trusted mutable
state cache. Runtime execution also verifies raw and derived evidence through
the existing supervisor. Unexpected specs in the generated directory or index,
altered bytes/permissions, missing materialized specs, inconsistent budgets,
changed reports, seed overlap, or an invalid journal fail closed. Governance,
programme/campaign identity or policy fingerprint changes require review.

The batch event reserves budget before files or work exist. Partial spec
materialization resumes the same batch; a ready event makes missing specs an
integrity failure. Completed shards are skipped and dead-worker shards use
the existing recovery path. A complete report/receipt left by interruption is
reused. A durable result advances the generation exactly once; interruption
before that event reuses the same specs and completed evidence. A separate
stop event must precede confirmation. History is never silently repaired.

## Non-binding classification and production firewall

| Condition | Classification |
|---|---|
| Search ends at canonical | NO_IMPROVEMENT_DETECTED |
| Confirmation detects candidate advantage | PROMISING |
| Confirmation detects no difference | INCONCLUSIVE |
| Confirmation detects canonical advantage | REJECTED |
| Evidence integrity or execution failure | FAILED |
| Resource/dependency boundary | BLOCKED |
| Governance/policy/authority mismatch | REQUIRES_REVIEW |

PROMISING is a research recommendation. Human review, separate regression and
validation, and an explicit promotion decision are still required. The optimizer
cannot change defaults, strategy source, deployment pointers/configuration,
images, releases, tags, or production PRs. It cannot generate online experiments.
The existing staging/import firewall excludes all experiment infrastructure.

## Validation

`tests/test_experiment_optimizer.py` tests schema strictness, deterministic
vectors/IDs/hashes/seeds, bounds/budgets, selection and correction, confirmation,
restart, tampering, resource limits, supervisor integration and the installed CLI.
Its temporary campaigns use tiny fixed local horizons. One real two-generation
fixture runs scheduling, evidence sealing, analysis, step shrink and completion,
then checks zero duplicate work. A separate routing test controls the search
statistics and runs actual held-out confirmation with unmocked outcome analysis.
No canonical O0001 work or online match belongs in implementation validation/CI.
