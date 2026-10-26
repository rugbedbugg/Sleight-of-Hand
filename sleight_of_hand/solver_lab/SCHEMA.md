# M1 data contract (schema 1)

`contract.py`, `tree.py` and `dataset.py` provide executable validation.
Files are ASCII canonical JSON: sorted object keys, compact separators,
finite numbers, followed by one newline. Hashes use SHA-256 over canonical
JSON **without** that newline, prefixed `sha256:`. Source-file hashes use raw
file bytes. Duplicate JSON keys, NaN/infinity and inputs over 2 MB are rejected.

## Artifact

| Field | Meaning |
| --- | --- |
| `sha256` | Hash of `payload`; output checksum, excluding itself |
| `payload.manifest` | Complete generation configuration, including all input games |
| `payload.input_sha256` | Hash of that manifest |
| `payload.source_files` | Exact external Python modules and LICENSE hashes, checked against `source.lock.json` |
| `payload.generator_files` | Hashes of all local lab Python modules and the existing Hold'em evaluator used by the oracle, checked against the current checkout |
| `payload.python` | Exact CPython generation version |
| `payload.games[]` | Solved trees in manifest order; missing/extra/reordered trees are rejected |

Manifest fields are `schema_version=1`, `source={url,revision,license}`,
`algorithm=vector_cfr_plus_alternating_linear_naive`, strictly increasing
`checkpoints`, `limits`, and `games`. The source is pinned to the approved MIT
teacher revision. Algorithm means alternating CFR+, linear averaging, the
native pairwise reference evaluator, no sampling and no random seed.
The checked-in iteration checkpoints are 100, 500 and 2,000; no adaptive
stopping or post-hoc budget increase is performed. Final exploitability must
be at most 0.001 times root pot. Earlier checkpoints are diagnostic only.

## Game

Each game contains exactly:

- `id`, `split` (`train`, `validation`, `test`), `variant=nlhe_hu_river`.
- `board`: five distinct valid card strings. Their supplied order is preserved
  in state identity; no suit isomorphism or nearest-board mapping is applied.
- `pot`: settled pre-river pot in integer chips; `stacks`: OOP/IP chips behind;
  `big_blind`: positive minimum-bet unit. Current scope requires equal stacks
  equal to the even root pot. The effective root stack is therefore `stacks[0]`.
- `ranges`: two explicit lists of `{cards:[card,card],weight:number}` in OOP/IP
  order, with positive finite weights no greater than 1. Hand cards must be
  in rank-major `23456789TJQKA` / `cdhs` order. Range order is preserved and
  indexes the teacher matrix. Range weights need not individually sum to 1.
- `betting={bet_fractions:[0.5,1.0],include_all_in:true,max_wagers:2}`.

Invalid/duplicate/board-blocked combos and combos without any compatible
opponent hand are rejected. Inter-player overlap is allowed in the *ranges*;
deals sharing a card are excluded. Independent starting weights are combined
as w0×w1 and normalized once over compatible pairs. The 1,326-combination
universe remains physical cards; current execution caps ranges at 16 combos,
and does not claim full-range coverage.

Distinct board sets and tree IDs are required across the whole manifest,
including within a split. Board order changes cannot evade this check.
The dataset has one whole board/tree per split, six hand combinations per
player and 48 information sets per tree. Splits are an input declaration,
not a random action-level split. This is insufficient for a credible student
generalization experiment or probability calibration report.

## Solved tree

`game_sha256` hashes the exact game. `teacher_profile` is the unrounded native
average-policy export: player keys `"0"`/`"1"`, public keys `root` or slash-joined
tokens, then `{actions:[token,...],strategy:[[prob,...],...]}`. Each matrix row
corresponds to that player's input range order. Every possible public node and
private hand must be present, including zero-reach continuations. Such rows
are the solver's average strategies, not observed play frequencies.

`diagnostics[]` records iterations, native exploitability in chips, and an
independent oracle's two player EVs, BR values, exploitability, compatible deal
count and joint weight. Final oracle values are recomputed on load. The
absolute diagnostic tolerance is `1e-10` for cross-Python summation differences;
native/oracle consistency checks allow `1e-8` chips. Normalization tolerance
is `1e-12`. Probabilities must be finite, nonnegative, at most 1 and sum to 1.

`samples[]` contains `{state, probabilities}` for every information set.
Probabilities must exactly equal the corresponding native teacher matrix row.
No rounding, renormalization, missing-node uniform fallback or nearest-action
mapping is permitted in ingestion or consumption.

## Query and action amounts

A lookup takes the **complete original game object** and a state containing
exactly `phase`, `board`, `hero_position`, `hole_cards`, `history`, `pot`,
`stacks`, `street_contributions`, `to_call`, and `legal_actions`.

Positions are OOP=0 and IP/button=1. History is a river-root token sequence,
not a preflop-to-river transcript. Current pot includes prior river payments
but excludes the pending action. Stacks are remaining chips; contributions
are street totals. Only private-card order is normalized at lookup.

Each legal action contains `id` (teacher token), `action` (check/call/fold/raise),
`raise_to` (integer street total for raises, otherwise null), `pay` (chips
added now), and `all_in` (whether that payment exhausts the actor's stack).
Thus the state after `b50` has pot 150, contributions [50,0], to_call 50.
Its `r50` action is raise-to 100, pay 100, all_in true. After `b50/r50`, pot
is 250 and OOP's all-in call pays 50, producing a 300-chip showdown pot.

`TeacherPolicy.lookup` returns `{status:supported,actions,probabilities}` or
`{status:abstain,reason}`. A changed range, amount, position, history, board,
stack, action set or unsupported hand cannot inherit another state's advice.
The consumer never samples a fallback action and never claims unsupported
states have equilibrium coverage.

## Evidence limitations

`execution.json` records nondeterministic timing/RSS separately from the
deterministic dataset; `open_spiel.json` records independent oracle checks.
Both link to the dataset checksum. The source hashes and reproduction tests
establish traceability and detect corruption, not cryptographic attestation.
The targets approximate an equilibrium of a restricted, weighted-range river
game. They are not an exact equilibrium of full heads-up no-limit Hold'em.
