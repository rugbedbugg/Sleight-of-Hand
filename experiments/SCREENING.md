# O0002 — Resolution-V Interaction Screening v1

O0002 is a **designed experiment over fixed policy parameters**. It is not
reinforcement learning, policy-gradient training, reward learning or
value-function learning, and it is not an open-ended optimizer. It asks one
predeclared question:

> Is there evidence for a useful **multivariate** configuration of the five
> existing PolicyParams that O0001's one-coordinate-at-a-time search could not
> see?

O0001 completed `NO_IMPROVEMENT_DETECTED`: 30 single-coordinate moves around
canonical over three step resolutions (plus three canonical baselines) and no
corrected detectable improvement. That does not show canonical is globally
optimal. O0002 changes all five parameters simultaneously, at O0001's original
coarse step amplitudes, inside the existing policy domain.

Candidate **selection** and factorial **diagnostics** are different things.
Selection decides whether one complete parameter vector deserves held-out
confirmation. Diagnostics only help explain which parameter combinations may
matter; they can never select or promote anything.

```sh
uv run research plan          # O0002 PENDING once O0001 is COMPLETE / NO_IMPROVEMENT_DETECTED
uv run research auto          # first cycle: screening only
uv run research auto          # later cycle: held-out confirmation, only if a treatment qualified
uv run research optimize O0002
```

## Reviewed authority

[O0002.json](O0002.json) is `LOCAL_APPROVED`, pinned in
[programme.json](programme.json) as
`sha256:6e14745c56975321d42ec59a896cfef0cebbd20707ee2d709a575e47ea4ac3c1`.
Its per-campaign authority (see [OPTIMIZER.md](OPTIMIZER.md#per-campaign-authority))
is `sha256:774349840613d8063a4f1e703607d00b9f174ead066875d95f6de5429ad2b22a`.

| Field | Contract |
|---|---|
| `schema_version` | `1` |
| `id`, `purpose`, `governance` | `O0002`; LOCAL_APPROVED or REQUIRES_REVIEW |
| `campaign_seed`, `seed_derivation` | `202610080002`; `sha256-campaign-phase-v1` |
| `template` | `specs/E0003-canonical.json`, `45d99c7c…bcc6` |
| `policy_revision` | `sha256:8e51fc38…6f99`, matching the template |
| `fixed_policy` | samples=128, opponent_memory=true, historical_prior=null |
| `depends_on` | `[{id: O0001, status: COMPLETE, classification: NO_IMPROVEMENT_DETECTED}]` |
| `algorithm` | `{kind: resolution-v-screening, version: 1}` |
| `factors` | Five `{symbol, parameter, low, center, high}`; one per PolicyParam |
| `design` | Regular two-level fraction: runs, base factors, generator, relation, resolution, enumeration |
| `screening_stopping`, `confirmation_stopping` | Exact fixed horizons; at least two matches |
| `candidate_family` | Every treatment vs `baseline`; size 16; Bonferroni α=0.05; selection rule |
| `diagnostic_family` | The 15 declared effects; size 15; Bonferroni α=0.05; `binding: false` |
| `objective` | `paired-bb-per-100-bonferroni-vs-canonical` |
| `max_generated_vectors` | Exactly runs + 1 + 2 = 19 |
| `resources` | At most two workers; memory floor no weaker than the programme |
| `created_at` | `2026-10-14T00:00:00Z`, the declared spec timestamp |

Unknown or duplicate fields, duplicate factor symbols or parameters, a missing
PolicyParam, `low < center < high` violations, a centre that is not the exact
midpoint, levels outside the policy domain, a generator or defining relation
that does not reference exactly the declared factors, a resolution below V, a
different family size, or any other comparison family are rejected. Before any
execution the centre vector must equal the current canonical `DEFAULT_PARAMS`;
otherwise the campaign requires review and nothing runs. This check is made at
execution rather than load so completed history stays readable after a future
policy change (see [policy epochs](README.md#policy-epochs-and-historical-evidence)).
Every generated vector is revalidated.

## Factors

| Symbol | Parameter | Low (−1) | Centre (canonical) | High (+1) |
|---|---|---:|---:|---:|
| A | value_bet_threshold | 0.50 | 0.55 | 0.60 |
| B | call_threshold | 0.30 | 0.35 | 0.40 |
| C | bluff_freq | 0.10 | 0.15 | 0.20 |
| D | aggression | 0.75 | 0.80 | 0.85 |
| E | steepness | 6.0 | 8.0 | 10.0 |

## Design: 2^(5−1), E = ABCD, I = ABCDE (Resolution V)

A–D are enumerated lexicographically over `{−1, +1}^4` with −1 first and A most
significant. For every row, `E = A·B·C·D`, so the product of all five signs is
+1 (the defining relation `I = ABCDE`). −1 maps to the declared low level and
+1 to the declared high level. No wall-clock time or RNG participates.

| Row | A | B | C | D | E=ABCD | value_bet | call | bluff | aggression | steepness | Arm |
|---:|:-:|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|---|
| 0 | − | − | − | − | + | 0.5 | 0.3 | 0.1 | 0.75 | 10.0 | `treatment-00-ad3f6ff3479d` |
| 1 | − | − | − | + | − | 0.5 | 0.3 | 0.1 | 0.85 | 6.0 | `treatment-01-9a17c4cfbdbe` |
| 2 | − | − | + | − | − | 0.5 | 0.3 | 0.2 | 0.75 | 6.0 | `treatment-02-3a764b9a079d` |
| 3 | − | − | + | + | + | 0.5 | 0.3 | 0.2 | 0.85 | 10.0 | `treatment-03-207f5955ddc8` |
| 4 | − | + | − | − | − | 0.5 | 0.4 | 0.1 | 0.75 | 6.0 | `treatment-04-f24986c77d2f` |
| 5 | − | + | − | + | + | 0.5 | 0.4 | 0.1 | 0.85 | 10.0 | `treatment-05-adbe693870e2` |
| 6 | − | + | + | − | + | 0.5 | 0.4 | 0.2 | 0.75 | 10.0 | `treatment-06-16a357128a2c` |
| 7 | − | + | + | + | − | 0.5 | 0.4 | 0.2 | 0.85 | 6.0 | `treatment-07-420168bf8f62` |
| 8 | + | − | − | − | − | 0.6 | 0.3 | 0.1 | 0.75 | 6.0 | `treatment-08-9a787728d1fb` |
| 9 | + | − | − | + | + | 0.6 | 0.3 | 0.1 | 0.85 | 10.0 | `treatment-09-1291bd9f72b6` |
| 10 | + | − | + | − | + | 0.6 | 0.3 | 0.2 | 0.75 | 10.0 | `treatment-10-7d1211bc91bd` |
| 11 | + | − | + | + | − | 0.6 | 0.3 | 0.2 | 0.85 | 6.0 | `treatment-11-ee874b5ee1d5` |
| 12 | + | + | − | − | + | 0.6 | 0.4 | 0.1 | 0.75 | 10.0 | `treatment-12-26bec5cb975c` |
| 13 | + | + | − | + | − | 0.6 | 0.4 | 0.1 | 0.85 | 6.0 | `treatment-13-49d1409c71fe` |
| 14 | + | + | + | − | − | 0.6 | 0.4 | 0.2 | 0.75 | 6.0 | `treatment-14-81eb930ae6ac` |
| 15 | + | + | + | + | + | 0.6 | 0.4 | 0.2 | 0.85 | 10.0 | `treatment-15-fb28511a58c9` |

Every treatment moves all five parameters away from centre; each factor is
8 low / 8 high; the five factor columns are mutually orthogonal; the 16
treatment vectors are unique. Canonical is a separate 17th **baseline** arm.

### Alias structure (explicit limitation)

With `I = ABCDE`, every effect word is aliased with its complement:

| Effect | Aliased with | Effect | Aliased with |
|---|---|---|---|
| A | BCDE | AE | BCD |
| B | ACDE | BC | ADE |
| C | ABDE | BD | ACE |
| D | ABCE | BE | ACD |
| E | ABCD | CD | ABE |
| AB | CDE | CE | ABD |
| AC | BDE | DE | ABC |
| AD | BCE | | |

Main effects are never aliased with two-factor interactions (Resolution V).
Main-effect contrasts are aliased with four-factor terms, and **two-factor
contrasts are aliased with three-factor terms**. A two-factor estimate is
therefore a screening contrast, not proof of a pure two-factor interaction;
report it as, for example, "two-factor screening contrast, aliased with CDE".
No diagnostic is causally proven.

## Horizons and budget

| Horizon | Screening (17 arms) | Held-out confirmation (2 arms) |
|---|---:|---:|
| Matches per arm | 30 | 60 |
| Hands per match | 100 | 100 |
| Planned hands per arm | 3,000 | 6,000 |
| Matches per shard | 5 | 10 |
| Evaluation interval (matches) | 15 | 20 |
| Early stopping | none | none |
| Total planned hands | 51,000 | 12,000 |

The maximum is fixed: 17 screening vectors plus at most 2 confirmation vectors,
19 materialized vectors in total. There are no generations, no adaptive
candidate creation after screening, no second screening round, and no
"keep searching while better".

## Seeds

For phase `screening` or `confirmation`, canonicalize
`[campaign_id, campaign_seed, phase]`, SHA-256 it, take the first 16 hex digits
as an integer and mask to 63 bits:

| O0002 block | Material | Base seed |
|---|---|---:|
| screening-0 | `["O0002",202610080002,"screening"]` | 3897605199472245292 |
| confirmation-0 | `["O0002",202610080002,"confirmation"]` | 2873277742887071931 |

All 17 screening arms share the screening block with common random numbers, as
do both confirmation arms. The existing SeedPolicy derives SOH/opponent/deck
streams per match. At campaign load the screening and confirmation streams are
checked against each other and the E0003 template; at programme load every
campaign's declared streams are checked against every fixed experiment's
streams (including all E0003 arms) and every earlier campaign's reserved
streams (all four O0001 search blocks and its confirmation block). Any
collision is a configuration failure; seeds are never regenerated.

## Candidate selection

The analysis is the existing paired per-match bb/100 estimator, invoked with a
predeclared comparator. The central `analysis.analyze` accepts
`comparator="baseline"`, which restricts the comparison family to every arm
versus that arm before any outcome is read. For screening the family is
exactly the **16** canonical-vs-treatment comparisons (never the 136 unordered
pairs) with Bonferroni critical value `NormalDist().inv_cdf(1 − 0.05/(2·16))`.
Historical callers and O0001 still get the all-pairs report.

A treatment qualifies only if its corrected verdict is `DIFFERENCE_DETECTED`
**and** the higher arm is the treatment. A larger raw mean, a significant
unadjusted interval, or any factorial diagnostic is insufficient. If several
qualify, choose the largest positive paired mean improvement over canonical,
then the lexicographically smallest arm name. All 16 comparisons, the
qualifying set and the exact reason are recorded in the decision event.

## Factorial diagnostics (non-binding)

For each matched block (match index) `m`, with `y_i(m)` the per-match bb/100
of treatment row `i` and `s_i` its ±1 column for an effect word:

```text
effect_block(m) = (2/16) · Σ_i s_i · y_i(m)     # = mean(high) − mean(low)
```

Blocks are aggregated with the same mean / sample-SE / normal 95% interval as
the paired estimator, plus a Bonferroni interval over the declared family of
**15** effects (5 main + 10 two-factor). Each diagnostic records its label,
weights, alias note, block count, mean, SE, ordinary and corrected intervals,
and `binding: false`. Under a model `y = μ + β·x + …`, the reported effect is
`2β`. Diagnostics are written to the hash-bound analysis report and copied into
the screening decision, but selection never reads them.

## Classification

| Condition | Classification |
|---|---|
| No treatment clears the corrected canonical comparison | NO_IMPROVEMENT_DETECTED (no confirmation) |
| Confirmation: selected treatment corrected-detectably better | PROMISING |
| Confirmation: no detectable difference | INCONCLUSIVE |
| Confirmation: canonical corrected-detectably better | REJECTED |
| Evidence integrity or execution failure | FAILED |
| Authority / policy mismatch | REQUIRES_REVIEW |
| Resource or dependency boundary | BLOCKED |

When nothing qualifies the reason is "no multivariate treatment cleared the
corrected canonical comparison" and O0002 completes in the screening cycle.
Otherwise a durable stop event records the selection and the status becomes
CONFIRMING; a later cycle runs `O0002.confirmation.00` comparing exactly
canonical and the selected treatment on fresh streams. Screening and
confirmation never run in the same cycle.

PROMISING is a non-binding research recommendation. O0002 cannot change
`DEFAULT_PARAMS`, policy or preflop/shove/memory/equity source, production
configuration, deployment pointers, releases, tags, uploads or production PRs.

## Durability, replay and recovery

O0002 uses the shared campaign journal under `runs/optimizer/O0002/`:
immutable, fsynced, hash-chained `events/NNNNNN.json` (start, batch, ready,
result, stop, final, halt) and read-only generated specs in `specs/`. Spec IDs
are `O0002.screening.00` / `O0002.confirmation.00`; arms are `baseline` and
`treatment-<row>-<12 hex of the vector hash>`. Generated specs never enter
`experiments/specs/`. Identical authority and evidence reproduce identical
spec hashes.

Replay re-derives the design, specs, budget and selection from the hash-bound
report and receipt, and verifies completed evidence before acting. Tampered
specs, an altered design row, a changed report or receipt, altered evidence or
extra spec files fail closed. Recovery is the same as O0001: an interruption
during screening shards, after analysis, before the decision event, before
confirmation spec publication or during confirmation resumes the same logical
batch without duplicate shards and without generating a second campaign path.
Selection never happens before the full screening horizon is complete and
analysed, and confirmation is proposed only after a durable stop event.

## Validation

`tests/test_experiment_screening.py` covers schema strictness, design
properties and aliases, determinism, seed disjointness, the 16-comparison
family, raw-mean/corrected qualification, tie-breaks, confirmation routing,
the factorial estimator against injected known effects, dependency semantics,
legacy O0001 history across programme evolution, crash recovery, tampering,
disposition, the read-only plan and the installed CLI. Its campaigns are tiny
temporary fixtures run through the real design, specs, scheduler, local dealer,
evidence sealing, analysis and decisions. Canonical O0002 is never executed in
implementation validation or CI, and nothing uses the network.
