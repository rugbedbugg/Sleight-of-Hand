# Solver-guided policy development: M1

This offline lab runs an external, pinned CFR+ teacher on three small heads-up
NLHE river trees. It validates the resulting probabilities and exposes an exact
lookup policy with explicit abstention. It is research code, not part of the
competition bot. No supervised student has been trained.

- [Source and licensing audit](SOURCES.md)
- [Baseline and boundary audit](BOUNDARIES.md)
- [Dataset and query contract](SCHEMA.md)
- [Engineering results and limitations](REPORT.md)
- [Predeclared game manifest](manifest.json), [source hashes](source.lock.json)
- [Small generated dataset](data/dataset.json), [execution measurements](data/execution.json),
  [OpenSpiel checks](data/open_spiel.json)

## Reproduce

Run from the SOH repository checkout on Linux. The generation interpreter is
CPython **3.13.15**, managed by uv. The teacher uses the standard library only;
neither the project dependency graph nor the production environment needs an
additional package. Source checkouts, virtual environments, and new outputs
stay outside the repository. About 25 MiB resident memory per solve suffices
for these inputs; hard limits are described below.

```sh
git init /tmp/soh-m1-teacher
git -C /tmp/soh-m1-teacher remote add origin https://github.com/noambrown/poker_solver.git
git -C /tmp/soh-m1-teacher fetch --depth 1 origin 6a10442877ffc8fd28af93e16e279b9bbdd97b2a
git -C /tmp/soh-m1-teacher checkout --detach FETCH_HEAD

uv run --no-project --python 3.13.15 python -m sleight_of_hand.solver_lab solve \
  --source /tmp/soh-m1-teacher \
  --manifest sleight_of_hand/solver_lab/manifest.json \
  --output /tmp/soh-m1-reproduction
uv run --no-project --python 3.13.15 python -m sleight_of_hand.solver_lab validate \
  /tmp/soh-m1-reproduction/dataset.json
cmp sleight_of_hand/solver_lab/data/dataset.json /tmp/soh-m1-reproduction/dataset.json
```

Use fresh output directories: the CLI refuses to overwrite evidence. Dataset
bytes are deterministic under the pinned interpreter and inputs. Timing and
RSS in `execution.json` are measurements and will differ. CPython 3.10.21 is
also tested; cross-version arithmetic is compared at absolute tolerance
`1e-10`, not by claiming the two interpreters produce identical file hashes.

The manifest fixes board/range partitions, weighted physical combinations,
root geometry, betting abstraction, iteration checkpoints and resource caps.
The interface accepts at most six trees, 16 combinations per player, even
pots of 4–10,000 chips, equal behind-stacks equal to the settled pot, and a
big blind no larger than the half-pot opening bet. M1's checked-in inputs
use six combinations per range, a 100-chip pot, 100-chip stacks and a 2-chip BB.
Changing these inputs defines a new experiment; it does not extend the
coverage of the existing lookup artifact.

## Independent OpenSpiel oracle

The optional oracle uses a separate exact, hash-locked environment. The
OpenSpiel 1.6.15 binary distribution corresponds to release tag revision
`fcb98881bffd25a324d5505041c837ca9b7ed0bf`; it is not represented as a locally
reproducible source build. Its binary hash is recorded with the evidence.

```sh
uv venv --python 3.13.15 /tmp/soh-m1-oracle
uv pip sync --python /tmp/soh-m1-oracle/bin/python --require-hashes \
  sleight_of_hand/solver_lab/oracle-requirements.txt
timeout 180s /tmp/soh-m1-oracle/bin/python -m sleight_of_hand.solver_lab.reference \
  --source /tmp/soh-m1-teacher \
  --dataset /tmp/soh-m1-reproduction/dataset.json \
  --output /tmp/soh-m1-reproduction/open_spiel.json
```

An optional `pokerkit_wrapper` import notice is emitted by OpenSpiel 1.6.15;
PokerKit is not used by these checks. The oracle computes best responses for
all three river profiles and checks the selected teacher on built-in Kuhn
poker. The river bridge shares the independently written SOH research tree
and SOH evaluator; OpenSpiel's best-response algorithm is independent. The
built-in Kuhn check additionally uses OpenSpiel's own game implementation.

## Consume

```python
from sleight_of_hand.solver_lab.contract import ROOT, read_json
from sleight_of_hand.solver_lab.dataset import TeacherPolicy

artifact = read_json(ROOT / "data/dataset.json")
teacher = TeacherPolicy(artifact)
game = artifact["payload"]["manifest"]["games"][0]
query = artifact["payload"]["games"][0]["samples"][0]["state"]
advice = teacher.lookup(game, query)
assert advice["status"] == "supported"
assert sum(advice["probabilities"]) > 0.999999999999
```

Supply the complete [query contract](SCHEMA.md), including the original
ranges, history, exact legal amounts and current chip accounting. Unsupported
queries return `status=abstain` and a reason, with no fallback distribution.
The output action list is the legal restricted abstraction, not every legal
integer bet in unrestricted NLHE. No Chipzen state converter is included.

## Validation and resource limits

```sh
uv run --locked ruff check --isolated sleight_of_hand/solver_lab tests/test_solver_lab.py
uv run --locked ruff format --isolated --check sleight_of_hand/solver_lab tests/test_solver_lab.py
SOH_SOLVER_SOURCE=/tmp/soh-m1-teacher uv run --locked pytest tests/test_solver_lab.py -q

UV_PROJECT_ENVIRONMENT=/tmp/soh-m1-tests310 uv sync --locked --python 3.10.21
SOH_SOLVER_SOURCE=/tmp/soh-m1-teacher /tmp/soh-m1-tests310/bin/python -m pytest tests/test_solver_lab.py -q
uv run --locked pytest tests/ -q
/tmp/soh-m1-tests310/bin/python -m pytest tests/ -q
```

The parent launches one worker at a time. Each worker is restricted to two
available CPUs and one Python thread, 512 MiB address space, 120 CPU seconds,
180 wall seconds, a 2 MB file limit and no core dumps. Linux enforces CPU and
address-space hard limits; the parent kills a worker that exceeds wall time.
Oversized ranges/trees/JSON and convergence failures are rejected. Normal CI
fetches the exact small teacher revision into runner temporary storage and
tests real reproduction on Python 3.10 and 3.13. Without `SOH_SOLVER_SOURCE`,
three external-checkout tests are skipped; artifact and contract tests remain.

M1 must stop here. M2 proposals in the report are not execution authority.

## License

SOH research code follows the repository's [MIT license](../../LICENSE).
The external teacher retains its own MIT notice in its checkout. OpenSpiel
retains Apache-2.0 and its distribution notices in the optional environment.
No external solver implementation or binary is vendored into SOH.
