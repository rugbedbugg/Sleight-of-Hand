# ChipZen strategy improvement plan

Status: deferred; resume when ready. No implementation is authorized by this
document alone.

## Starting point

The ChipZen port lives on `feature/chipzen-port`; its changes are currently
uncommitted. Version 2 reuses the five-parameter policy with Monte Carlo
Hold'em equity, pot-odds-adjusted thresholds, and legal raise sizing.

- Version 1 archive: `dist/sleight-of-hand-chipzen.tar.gz`.
- Version 2 archive: `dist/sleight-of-hand-chipzen-v2.tar.gz`.
- Version 2 passed 87 tests on Python 3.10 and 3.13, plus SDK conformance
  inside a restricted Docker container.
- No live arena comparison has been run. Compatibility and controlled
  decision tests do not establish a full-game strength improvement.
- Keep the conservative exact dependency pins unless a change is discussed.

The largest modeling limitation is that opponents' unseen hands are sampled
uniformly, even after strong betting. Equity against random hands can differ
substantially from equity against the hands an opponent would actually play.

## 1. Establish a reproducible match benchmark

Compare versions 1 and 2 against the same opponent pool before adding more
strategy changes. Begin with ChipZen practice matches. For automated local
experiments, select and validate a No-Limit Hold'em simulator; the existing
Leduc harness is not a Hold'em simulator.

- Rotate seats and use matched deals/seeds where the environment permits.
- Record bot version, parameter values, opponent, table size, stack/blind
  structure, hand count, and seed when available.
- Measure chips won in big-blind units with uncertainty, accounting for
  dependence between hands within a match. Record rejected actions,
  timeouts, decision latency, and large-loss hands separately.
- Keep evaluation opponents and seeds separate from tuning data.
- Assess cash-style chip performance separately from tournament placement;
  chip expected value is not the same objective as tournament payout.

Completion criterion: a repeatable comparison with saved results and clear
uncertainty. If results are inconclusive, collect more evidence rather than
declaring a winner.

## 2. Add opponent hand ranges

Bring the Bayesian design into Hold'em by weighting possible two-card
holdings using public betting history, position, street, and bet size.

- Respect known-card removal and avoid overlapping cards across opponents.
- Start with a simple, documented action-likelihood model. Treat its
  assumptions as hypotheses to evaluate, not observations of hidden cards.
- Retain nonzero probability for plausible bluffs and unusual actions;
  avoid becoming certain from a small sample.
- Recompute equity against the weighted ranges, including all-in opponents.
- Use only information available to the bot at that decision. Revealed
  showdown hands can support later evaluation but must not leak backward.

Completion criterion: range normalization and card-consistency checks pass,
runtime stays within budget, and benchmark results support the change.

## 3. Improve preflop decisions

Account for position, effective stacks, previous raises, and players still
to act. Raw showdown equity alone misses the cost of acting early and the
possibility of further raises.

- Distinguish opening, calling, reraising, and responding to reraises.
- Handle heads-up and multiway tables explicitly.
- Begin with an interpretable baseline and tune it through the benchmark.
- Keep the existing five parameters as strategic controls where sensible;
  document any proposed expansion before increasing tuning complexity.

Completion criterion: fewer demonstrable preflop mistakes and supported
improvement against held-out opponents.

## 4. Improve bet sizing and bluff selection

Replace the single sizing rule with a small set of legal candidate sizes.
Evaluate their tradeoffs using estimated opponent responses.

- Consider pot fractions and stack-aware all-in choices, using the SDK's
  total-bet semantics and legal bounds.
- Favor bluff candidates with useful draws or blockers when justified by
  the opponent range and board.
- Account for the number of opponents that must fold and their estimated
  calling tendencies.
- Improve side-pot eligibility and equity accounting before relying on
  large multiway all-in decisions.

Completion criterion: legality and chip-accounting checks pass; experiments
separate the effects of sizing changes from bluff-selection changes.

## 5. Tune the five parameters on Hold'em

Adapt the genetic-algorithm evaluation loop to the validated Hold'em
environment. Existing Leduc fitness results are not Hold'em evidence.

- Train against diverse opponents and retain older bot versions as baselines.
- Use fixed evaluation conditions for comparable learning curves.
- Evaluate candidates on opponents and seeds excluded from training.
- Save genomes, run configuration, results, and uncertainty together.
- Reject candidates that gain average chips through unacceptable timeout,
  legality, or stability regressions.

Completion criterion: reproducible held-out improvement over the untuned
baseline, with no claim based only on training fitness.

## Longer-term option: self-play and limited search

After the benchmark and range model are reliable, assess self-play training
and depth-limited search with a small betting abstraction. Do not transplant
Leduc's exhaustive search into Hold'em: its state space is much larger, and
decisions must remain within the arena's runtime budget.

Pluribus provides a research reference for self-play combined with search,
not a drop-in implementation or a promised outcome:
[Brown and Sandholm, *Superhuman AI for multiplayer poker*](https://noambrown.github.io/papers/19-Science-Superhuman.pdf).

## Recommended resumption order

Start with the match benchmark, then opponent ranges. Introduce one strategy
change at a time, preserve the previous image and results, and rerun unit
tests, SDK conformance, container validation, and the match benchmark before
promoting a new version. Keep upload instructions in [CHIPZEN.md](CHIPZEN.md).
