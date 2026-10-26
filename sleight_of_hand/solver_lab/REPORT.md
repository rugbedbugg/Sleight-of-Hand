# M1 engineering report

M1 produces a real, reproducible solver-to-policy pipeline. The research
consumer is an exact lookup with explicit abstention. No supervised student,
runtime integration, strategy promotion, gameplay improvement claim, merge,
deployment or online poker action is part of this milestone.

## Teacher and provenance

Selected teacher: `noambrown/poker_solver`, MIT, source revision
`6a10442877ffc8fd28af93e16e279b9bbdd97b2a`. Its unmodified Python
`VectorCFRTrainer` runs alternating CFR+ with linear average-strategy weights
for exactly 2,000 iterations, recording checkpoints at 100 and 500.
The native pairwise reference evaluator is selected explicitly. The optimized
evaluator fails on sparse asymmetric strength sets; the pinned defect and the
working reference path are covered by tests. [Full audit](SOURCES.md).

The raw probability matrices came from actual executions. There are 144
private-hand information sets across 24 public decision nodes in three
independently specified trees. Each tree has six physical combinations per
player. Every information set has a complete, unrounded teacher distribution,
including off-path continuations. The 82 KB dataset includes exact source
hashes, generation inputs, generator hashes, iteration evidence and checksums.
No external binaries or large archives are committed.

Canonical dataset payload checksum:
`sha256:9794ee261229cd063672bbdbed3a923d641e442138ab82cd0b34123d0a8da6b4`.

## Supported games and partitions

All roots have pot 100 chips, OOP and IP stacks 100 chips each, minimum bet
unit 2 chips, no current river contribution, and OOP acting first. The full
weighted starting ranges are in [manifest.json](manifest.json).

| Partition / tree | Board | Compatible private deals | Information sets |
| --- | --- | ---: | ---: |
| Train / dry | Ks Th 7s 4d 2s | 32 | 48 |
| Validation / paired | Qc Qh 8d 5s 2c | 32 | 48 |
| Test / connected | Js Td 9c 6h 3s | 35 | 48 |

Both players may check or open for 50 or 100. Facing 50, the responder may
fold, call or raise-to 100. Facing 100, only fold/call remain. Each game is a
restricted betting abstraction of real river NLHE with weighted hand ranges,
not the equilibrium of unrestricted full heads-up NLHE. Supported positions,
histories, stacks and action sizes are exact; every other query abstains.

Whole boards/trees were assigned before solving; there is no action-level
random split. Reordering board cards cannot bypass the leakage validator.
The interface retains physical-combination fidelity and card removal; its
bounded range size must not be mistaken for complete 1,326-combination coverage.

## Convergence and independent checks

Exploitability is `(BR0 + BR1 - settled_pot) / 2`, in chips. The predeclared
final gate is 0.1 chips, or 0.1% of this pot.

| Tree | 100 iterations | 500 iterations | 2,000 iterations |
| --- | ---: | ---: | ---: |
| Dry | 0.0802370915 | 0.0068651406 | 0.0009015208 |
| Paired | 0.1549540248 | 0.0092225300 | 0.0010812723 |
| Connected | 0.0976297596 | 0.0047643387 | 0.0003423767 |

All pass. The independent SOH finite-tree oracle aggregates opponent hands
before maximizing, avoiding hidden-card strategy fusion, and agrees with the
teacher's native BR calculation to floating-point precision. The tree's
integer accounting and every legal edge also agree with SOH's local dealer.
All supported hand strengths agree between independent hand evaluators.

OpenSpiel 1.6.15 independently evaluates all three river profiles: the largest
exploitability discrepancy is `1.6653345369377348e-15` chips. This bridge shares
SOH's research tree and evaluator, so it validates the BR algorithm rather
than constituting an independent third NLHE rules implementation. Built-in
OpenSpiel Kuhn poker provides the additional independent reference game:
after 10,000 teacher CFR+ iterations, player-0 EV is `-0.05555555911117002`,
versus known `-1/18`; exploitability is `9.632756980813839e-06`, agreeing with
the teacher evaluator. [Machine-readable evidence](data/open_spiel.json).

## Consumer and training decision

The lookup reproduces **144/144** teacher distributions exactly, with maximum
absolute error **0**. This also implies identical deterministic argmax action
selection for supported states. Distribution normalization, hand removal,
state identity, exact raise sizes and absent coverage are tested. Returned
data cannot mutate the internal lookup table.

No supervised student was trained. One board per partition cannot support a
credible held-out generalization or calibration claim. Student KL divergence,
cross-entropy, calibration and held-out action agreement are therefore **not
applicable**. A copied teacher table is not described as a trained neural policy.

## Reproduction and resources

Exact commands are in [README.md](README.md). Separate repeat runs produce
identical dataset bytes on each interpreter. The canonical artifact uses
CPython 3.13.15; CPython 3.10.21 produces numerically equivalent results, with
the largest observed probability difference `1.8907098109366416e-13`.
Cross-interpreter checks use absolute `1e-10` tolerance. The difference is
reported, not hidden by rounding the teacher's probabilities.

The final run's measured solver work takes about 0.6 seconds per tree and
about 22 MiB peak worker RSS. All three run sequentially, with one Python
thread and affinity limited to two available CPUs. Each worker has hard
512 MiB address-space and 120-second CPU limits; the parent enforces a
180-second wall timeout. Inputs/output files are capped at 2 MB, ranges at
16 combinations per player, trees at 64 public nodes and manifests at six
trees. A test deliberately exceeds a child memory bound and confirms failure.

On this machine, 2,880 supported lookups measure about 54 microseconds median,
124 microseconds p99 and 308 microseconds maximum, after an approximately
17 ms validated load. Timing is descriptive and is not a live-server SLA.
OpenSpiel verification uses about 82 MiB RSS. Exact run measurements are
separate from deterministic data in [execution.json](data/execution.json).

## Verification and isolation

Focused tests: **57 pass on Python 3.10.21 and 57 on Python 3.13.15**, including
real solver reproduction and external source integrity checks. Normal runs
without the external checkout skip three integration tests. Ruff, formatter
and workflow validation are required; CI performs the same artifact checks
and fetches the exact teacher revision into runner temporary storage.

Full existing regression suites pass on both interpreters: **2,807 passed,
3 optional integration tests skipped** on Python 3.13.15 (482.42 s) and
Python 3.10.21 (637.30 s). The separately run focused suites include and pass
all three real-source tests, including after final provenance validation.
Ruff lint/format, `git diff --check`, and actionlint 1.7.12 with shellcheck
0.10.0 pass. Chipzen SDK 0.4.0 passes smoke, timing and all four protocol
conformance scenarios, with its existing restricted-`os` import warning.
The initial SDK check stalled during executor shutdown in the network
sandbox; the unrestricted rerun passes. No server match or deployment occurs.

[Reproduction evidence](data/reproduction.json) records two identical runs
per Python version and the measured cross-version numeric difference.

The branch starts at the requested `1b1536241e29a77a45a7398629456b6831c3625e`.
No files in `bots`, `deploy`, `experiments`, production `holdem`, `engine`,
`policy`, or the existing experiment framework are changed. `pyproject.toml`,
`uv.lock` and runtime requirements are unchanged. The rebuilt 29-file Chipzen
staging tree equals the baseline archive after CRLF normalization; its
normalized tree SHA-256 is
`a1fb777d81cb42c0ecec4e231632a5fb0ea63252a70bd5bed5068ab74edc921f`.
Five existing files differ from Git's archive only in LF/CRLF representation.

The canonical policy fingerprint remains
`sha256:eff445aa51a2b78871c294df085353b4f00b3b38a0df73a954f13f2270a0cfbf`.
E0006 report hashes remain unchanged from the initial inspection:

- JSON: `0c9df0b631cfa00e098daf84c8d0ee2c0d25dd7c52e6ee914cccfadd9d6b60c9`.
- Markdown: `163e2231805db96f206921c3b285294c82e733ccb58b3fb45ba4ee9dfdc770b7`.

No historical evidence was regenerated or used as training/test data.
E0006 remains +2.85 bb/100, 95% CI [-4.32,+10.02], NO_DIFFERENCE_DETECTED.

## Coverage gaps and recommended M2

The main limitation is coverage: three boards, small manually declared ranges,
one SPR, two opening sizes and one all-in raise. There are no flop/turn trees,
unequal-stack/short-all-in contexts, side pots, full-range solves, continuous
bet sizes, range inference, suit-isomorphism mapping or full-game equilibrium
claims. The fast upstream evaluator remains unsuitable for these sparse ranges.

Recommended M2 is a separately authorized expansion to many independently
partitioned river boards and plausible weighted physical-hand ranges, with
fixed resource/convergence gates and the same independent correctness checks.
First validate a faster teacher path against the pairwise reference and add
proper minimum-raise/unequal-stack handling before expanding the action space.
Then compare a small supervised student's held-out KL, cross-entropy,
calibration and action agreement against this exact lookup baseline. Keep
board families and every derived tree in the same partition. Only a later
controlled outcome comparison could support a stronger-gameplay claim.
M1 ends here; none of this recommendation authorizes execution or deployment.
