# Source and licensing audit

Verified against public GitHub repositories and actual pinned source on
2026-10-26. No commercial trainer, proprietary strategy archive, account or
online poker service was accessed. No AGPL implementation was executed,
copied or linked into SOH. Runtime dependencies and `uv.lock` are unchanged.

| Source | Inspected revision | License | Decision |
| --- | --- | --- | --- |
| [noambrown/poker_solver](https://github.com/noambrown/poker_solver/tree/6a10442877ffc8fd28af93e16e279b9bbdd97b2a) | `6a10442877ffc8fd28af93e16e279b9bbdd97b2a` | [MIT](https://github.com/noambrown/poker_solver/blob/6a10442877ffc8fd28af93e16e279b9bbdd97b2a/LICENSE) | Selected, Python reference evaluator |
| [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel/tree/fcb98881bffd25a324d5505041c837ca9b7ed0bf) | `fcb98881bffd25a324d5505041c837ca9b7ed0bf` (v1.6.15); audited current HEAD `5886e0b822bcc469cacb53a82c18ac1e8748d820` | [Apache-2.0](https://github.com/google-deepmind/open_spiel/blob/fcb98881bffd25a324d5505041c837ca9b7ed0bf/LICENSE) | Independent oracle; viable alternative |
| [bupticybee/TexasSolver](https://github.com/bupticybee/TexasSolver/tree/bb53f8cbfff4841da2d75bd060d0ab269c3d5881) | `bb53f8cbfff4841da2d75bd060d0ab269c3d5881` | [AGPL-3.0](https://github.com/bupticybee/TexasSolver/blob/bb53f8cbfff4841da2d75bd060d0ab269c3d5881/LICENSE) | Comparison only |
| [b-inary/postflop-solver](https://github.com/b-inary/postflop-solver/tree/9d1509fe5077d019825f833eed04b16d342dfda1) | `9d1509fe5077d019825f833eed04b16d342dfda1` | [AGPL-3.0-or-later](https://github.com/b-inary/postflop-solver/blob/9d1509fe5077d019825f833eed04b16d342dfda1/LICENSE) | Comparison only |

## Selected teacher

`poker_solver` supports river NLHE plus Kuhn/Leduc reference games, CFR/CFR+,
DCFR and other algorithms, with Python and C++ implementations. Its Python
`VectorCFRTrainer.average_strategy_profile()` exposes action tokens and a
probability row per physical hand at each public decision node. No inference
from charts or extraction from opaque binaries is needed. The Python river
path imports only the standard library; no compiler, GPU or extra package is
needed. The default full-range/deep-stack example was not run.

Sparse asymmetric ranges expose an upstream `vector_eval.showdown_values`
`KeyError`: the fast evaluator assumes a hero strength exists in the opposing
strength table. The checked-in test reproduces `(3, 13, 10, 7)` on the train
board. M1 selects the existing `use_naive_eval=True` option and
`exploitability_naive`, which enumerate compatible hand pairs. The external
source is unmodified. Its minimum-raise and unequal-stack limitations are
handled through a deliberately narrow validated game family, not silent rule
changes. Measured peak RSS is about 22 MiB per solve.

## OpenSpiel

OpenSpiel provides CFR, tabular policy exports, best-response evaluation,
Kuhn/Leduc, and universal poker including no-limit rules. Arbitrary river
ranges require a custom game bridge or more setup than the selected teacher.
The optional 1.6.15 wheel is 14.8 MiB; its isolated dependencies are exactly
pinned and hash locked. Oracle execution uses about 82 MiB RSS and no GPU.
The wheel's source provenance is its corresponding release tag, not a claim
that SOH rebuilt and verified all binary build inputs. The recorded binary
SHA-256 identifies the actual tested artifact.

## Comparison-only alternatives

TexasSolver supports Hold'em and short deck, C++ console/GUI workflows and
JSON strategy exports. Its published example benchmark reports about 1,600 MB
and six threads for a particular flop tree; this is neither a minimum nor a
measurement on this machine. Its AGPL license excludes incorporation from M1.

The Rust postflop-solver library supports postflop trees, per-node strategy
inspection and optional serialization/compression, with optional Rayon
parallelism. Memory depends on the configured tree and precision; no verified
fixed minimum was found. Upstream marks development suspended. It was not
built or benchmarked because M1 has no AGPL licensing authorization.

## Provenance boundary

`source.lock.json` records hashes of every Python source module in the teacher
checkout and its LICENSE, tied to the exact Git revision. Execution verifies
both revision and files and rejects extra Python modules. The dataset embeds
these hashes, the full game manifest, local generator hashes, interpreter,
raw teacher matrices and derived samples. Checksums detect corruption; they
are not signed attestations and cannot prove an adversary never forged a
dataset. Actual deterministic re-execution supplies the reproducibility
evidence. Upstream MIT copyright is Noam Brown (2025); its license remains
with the isolated checkout. SOH contains no copied solver implementation.
