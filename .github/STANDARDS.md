# Repository standards baseline

This workflow adaptation follows the shared reference baseline in ReAgent's
`.github/STANDARDS.md` (the `ReAgent-ci-standards` checkout): pull-request
and branch validation, explicit read-only permissions, cached isolated
dependencies, cancellation of obsolete runs, lint, tests, and artifact
validation. The README follows its purpose, installation, quick-start,
usage/configuration, development/testing, and license structure, with the
ChipZen-specific sections in `docs/CHIPZEN.md`.

CI retains the Python 3.10/3.13 compatibility matrix. `uv run --no-project`
uses the matrix environment without replacing it with the development
`.python-version`. The ChipZen SDK and WebSocket runtime are exactly pinned;
Ruff is pinned in `requirements-dev.txt`. Lint/format currently cover the
new port files; existing Leduc modules retain their formatting and tests.
The staged runtime is checked by the SDK's protocol-conformance harness.

There is no CD workflow. Container build/export remains a local operation;
the SDK source checks do not certify an image's size or production review.
No release or CI-status badges are invented: GitHub publication could not
be verified with the available connection during this port. This document
records the scoped CI changes, not a completed repository-wide audit.
