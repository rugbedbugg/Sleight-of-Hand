# Repository standards baseline

This workflow adaptation follows the shared reference baseline in ReAgent's
`.github/STANDARDS.md` (the `ReAgent-ci-standards` checkout): pull-request
and branch validation, explicit read-only permissions, cached isolated
dependencies, cancellation of obsolete runs, lint, tests, and artifact
validation. The README follows its purpose, installation, quick-start,
usage/configuration, development/testing, and license structure, with the
ChipZen-specific sections in `docs/CHIPZEN.md`.

CI retains the Python 3.10/3.13 compatibility matrix. mise provides the
matrix interpreter and `uv sync --locked --python` builds `.venv` from the
committed `uv.lock` (failing rather than re-resolving if it is stale), so the
development `.python-version` never replaces the matrix interpreter and CI
never updates the lock; a final step checks the lock and checkout are
unchanged. The ChipZen SDK and WebSocket runtime are exactly pinned in the
`chipzen` dependency group and mirrored in `bots/chipzen/requirements.txt`;
Ruff is pinned in the `dev` group. Lint/format currently cover the
new port files; existing Leduc modules retain their formatting and tests.
The staged runtime is checked by the SDK's protocol-conformance harness.

There is no CD workflow. Container build/export remains a local operation;
the SDK source checks do not certify an image's size or production review.
No release or CI-status badges are invented: GitHub publication could not
be verified with the available connection during this port. This document
records the scoped CI changes, not a completed repository-wide audit.
