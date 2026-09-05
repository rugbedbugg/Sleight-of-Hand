# Sleight of Hand — Project Specification

Status: **draft**, revision 1 (2026-09-05)

This document defines what Sleight of Hand is becoming, the rules of the
games it plays, the interfaces its components share, and how the resulting
agent is measured. It is the reference for design decisions; `README.md`
stays the front door for users.

---

## 1. Goal

Build a strong heads-up **fixed-limit 2-7 Triple Draw** agent, developed
over a long horizon, whose strength is demonstrated by published,
reproducible measurements rather than by a leaderboard position.

The existing Leduc hold'em agent is retained — not as a legacy artifact but
as a **verification test bed**: a game small enough to solve exactly, so
that every approximation introduced for 2-7 can be checked against ground
truth in a game where ground truth exists.

### 1.1 Why 2-7 Triple Draw

Chosen over heads-up hold'em and Pineapple OFC on four grounds:

1. **Code reuse.** Assuming the arena's hold'em is no-limit, 2-7 preserves
   roughly 61% of the existing source against hold'em's ~45%, and the
   surviving fraction is the load-bearing part rather than the scaffolding.
   2-7 stays fixed-limit, so per-round bet sizes and the raise cap in
   `engine/state.py` carry over directly, and betting actions remain
   fold/call/raise.
2. **Architecture fit.** The project's thesis is Bayesian opponent
   modelling plus search — exploitative play. The hold'em field is built
   around equilibrium approximation, against which opponent modelling buys
   little. 2-7's field is hand-built agents with real, exploitable leaks.
   Additionally, draw counts are a far stronger likelihood signal than any
   Leduc bet, so the belief model gains power rather than losing it.
3. **Tractability on the available hardware.** See §2.
4. **Empty field.** No public 2-7 bot, solver, or benchmark exists. Nothing
   downloadable to lose to; a credible published result becomes the
   reference point.

### 1.2 Non-goals

* Solving 2-7. Infeasible for anyone (see §4.1); the agent plays well
  without solving.
* Deploying against humans for money on any real-money site. Prohibited by
  the terms of every licensed operator and enforced with lifetime bans and
  confiscation. Play-money and self-hosted play only.
* Competing in heads-up no-limit hold'em as a primary effort. A hold'em
  adapter may later exist purely as a calibration harness (§6.6).
* Multiway play in the first iterations — though the engine is written for
  N players from the start (§4.2).

---

## 2. Hardware envelope

All design decisions assume the primary development machine:

| Resource | Available |
|---|---|
| CPU | AMD Ryzen 3 7320U — 4 cores / 8 threads @ 2.4 GHz, 4 MiB L3 |
| RAM | 7.0 GB total; budget **≤ 3 GB** resident for any training run |
| GPU | Radeon 610M integrated — no CUDA, no ROCm. Treat as absent. |
| Disk | ~360 GB free |

Measured throughput on this machine (see `scripts/bench.py`, to be added):

| Operation | Rate |
|---|---|
| Pure-Python dict/regret updates, single core | 1.60 M/s |
| Leduc hands/sec, random vs random | 17,131 /s |
| `eval7` 7-card evaluations | 1.11 M/s |

**Consequences.** Tabular CFR is capped near ~10M information sets
(~250 MB as float32); anything finer exceeds the RAM budget. Deep
reinforcement learning is out. The agent is therefore built around
**runtime search plus belief**, not a precomputed blueprint — which is also
the better fit for an exploitative architecture.

---

## 3. Game rules

### 3.1 Leduc hold'em (`--gamemode leduc`)

Unchanged from the current implementation. 6-card deck {J,J,Q,Q,K,K}, ante
1, two betting rounds with bet sizes 2 and 4, at most 2 raises per round,
one community card. Retained as a regression test bed and, at this size,
exactly solvable.

### 3.2 Deuce-to-seven triple draw (`--gamemode deuce27`)

The product. Rules are **pinned here** so that published measurements are
reproducible and comparable:

* **Deck.** Standard 52 cards, no jokers.
* **Players.** 2 (heads-up) for all current work. The engine supports up to
  6, which is the hard maximum for the game — six players drawing three
  times can exhaust the deck.
* **Hand ranking.** Standard five-card *high*-hand ranking, **inverted**.
  Aces are always high (and therefore bad); straights and flushes count
  against the holder. The best hand is 7-5-4-3-2 not all of one suit.
  Implementation note: evaluate with a conventional high-hand evaluator and
  take the minimum — no separate lowball evaluator is required.
* **Blinds.** Small blind 1, big blind 2. (Leduc's ante structure does not
  apply.)
* **Betting rounds.** Four: pre-draw, after draw 1, after draw 2, after
  draw 3.
* **Bet sizes.** Small bet (= 2) pre-draw and after draw 1; big bet (= 4)
  after draws 2 and 3.
* **Raise cap.** 4 bets per round (one bet plus three raises).
* **Draws.** Three. On each, a player discards any subset of their five
  cards (0–5, so 32 choices) and is dealt replacements.
* **Deck exhaustion.** If the stub cannot cover a requested draw, the
  previously discarded cards are shuffled to form a new stub, excluding
  burn cards and excluding the drawing player's own current discards.
  Unreachable heads-up; implemented for forward compatibility.

### 3.3 Mini 2-7 (`--gamemode mini27`)

A scaled-down 2-7 built for one purpose: to be **small enough to solve
exactly** on the hardware in §2, so that the belief and search components
can be validated against ground truth before being trusted at full scale.
It is a test fixture and is never shipped as a product.

Provisional parameters (to be tuned to fit the RAM budget once the exact
solver exists):

* Ranks 2–6 plus an ace, two suits — a 12-card deck
* Three-card hands
* Two draws, three betting rounds
* Same inverted-ranking rule, same fixed-limit structure

---

## 4. Architecture

### 4.1 What is and isn't computable

Heads-up limit hold'em has ~3.19 × 10¹⁴ information sets and its solution
(Cepheus) required 4,800 cores for 68 days and 10.9 TB. Full 2-7 is far
larger: the private state is a five-card holding — C(52,5) = 2,598,960
against hold'em's 1,326 — and each player faces 2⁵ = 32 discard choices on
each of three draws, branching with no hold'em analogue.

The agent therefore **never enumerates the game tree**. It computes equity
by Monte Carlo rollout, searches a bounded number of betting nodes ahead,
and maintains a belief over hand *classes* rather than exact holdings.
Every one of those is an approximation, which creates a new obligation
absent from Leduc: measuring the error they introduce (§6).

### 4.2 Shared pipeline

Unchanged in shape from the Leduc implementation:

```
engine/    immutable state; legal actions, betting, chance events, payoffs
   |
   +-- bayes/    P(opponent hand class | history), Bayes' rule
   +-- search/   depth-limited expectiminimax over the betting tree
   +-- ga/       evolves the policy parameters
        `-------> policy/ :: action_probs()
```

All three consume one parametrized policy, exactly as now. The engine is
written for **N players** throughout — `private` and `contrib` are tuples
of length N, `to_act` advances modulo the active players — even though only
N=2 is exercised. This costs little now and avoids rewriting every module
when multiway work begins.

### 4.3 The `Game` protocol

Games are addressed through a single interface so that the belief, search,
evaluation and GA layers are game-agnostic. `--gamemode` selects an
implementation from a registry.

```python
class Game(Protocol):
    spec: GameSpec                      # name, num_players, bet sizes, blinds, raise cap

    def new_hand(self, rng) -> State: ...
    def legal_actions(self, state) -> list[Action]: ...
    def apply_action(self, state, action, rng=None) -> State: ...

    # chance nodes: the Leduc community card, or 2-7 draw replacements
    def awaiting_chance(self, state) -> bool: ...
    def chance_outcomes(self, state) -> list[tuple[Outcome, float]]: ...
    def apply_chance(self, state, outcome) -> State: ...

    def payoffs(self, state) -> tuple[float, ...]: ...
    def hand_strength(self, state, player) -> float: ...
```

Generalizations from the current Leduc-specific names:

| Leduc-specific | Generalized |
|---|---|
| `awaiting_community` | `awaiting_chance` |
| `possible_community_cards` | `chance_outcomes` |
| `deal_community` | `apply_chance` |
| `BET_SIZE`, `MAX_RAISES` module constants | fields on `GameSpec` |
| `BIG_BLIND` in `eval/harness.py` | `spec.big_blind` |

### 4.4 Actions, including draws

**Draws are actions, not a separate agent method.** A draw decision is
represented as an action carrying a discard mask; the search tree keeps
uniform decision nodes, `action_probs` generalizes to
`P(action | hand, context)` without a special case, and the `Agent`
interface is unchanged:

```python
Action = ActionType | DrawAction     # ActionType: FOLD | CALL | RAISE
```

Leduc only ever produces `ActionType`, so the existing engine, agents and
tests are untouched by this addition.

### 4.5 Policy

`policy/heuristic.py::action_probs` is already written against an abstract
strength score in [0,1]; its sigmoid machinery, `to_call` branching, legal
-action masking and normalization are game-neutral and are retained. Only
`hand_strength` is game-specific and moves behind the `Game` protocol.

For 2-7 the genome widens beyond the current five parameters to cover
drawing standards (how many cards to draw from which holdings), pat-hand
thresholds, breaking standards, and snowing frequency. Gene bounds live in
`ga/genome.py`, which is already variant-agnostic.

### 4.6 Belief

The belief is over **hand classes** — pat 7-low, pat 8-low, pat 9-plus,
drawing-one-to-X, drawing-two, and so on — not over exact holdings. This is
tractable, and it matches how strong players reason.

The replay-from-history design in `bayes/opponent_model.py::infer_belief`
is retained; it stays correct by construction. Leduc's
`update_on_public_card` (a hypergeometric update on the community card) is
deleted for 2-7 and replaced by `update_on_draw(n_discarded)`, which
occupies the same slot in the replay loop and carries far more information.

### 4.7 Search

Depth-limited expectiminimax. Draws become chance nodes over replacement
cards; leaves are evaluated by Monte Carlo rollout equity. The existing
three-node structure (MAX / chance / opponent) is retained; exhaustive
enumeration is not.

The per-decision budget is **5 seconds** (matching the arena's constraint);
the search must be interruptible and return its best action so far.

---

## 5. Performance plan

* Precompute the 5-card rank table once — C(52,5) = 2,598,960 entries,
  stored as a flat integer array with an index scheme rather than a Python
  dict of tuples, so it costs single-digit MB and every later evaluation is
  a lookup.
* Rollouts: budget 100–300k full rollouts/sec in pure Python after deck
  handling overhead, i.e. on the order of 10⁶ rollouts inside a 5 s
  decision. Revisit with `numba` or a C extension only if measurement shows
  it is the binding constraint.
* Parallelism across 4 real cores via `multiprocessing` (the GIL rules out
  threads for CPU-bound rollouts).

---

## 6. Evaluation

No external benchmark exists for 2-7 (§1.1). The measuring instruments are
therefore built in-project. They measure different and partly opposing
things, and are tracked together.

### 6.1 Exploitability via Local Best Response — primary metric

A depth-limited best-response adversary queries the agent's action
probabilities and plays the EV-maximizing reply, scored over Monte Carlo
hands. Yields a **lower bound** on exploitability in mbb/hand: absolute,
opponent-free, and adversarial, so it finds leaks no hand-built opponent
would. Strengthens the existing `eval/exploitability.py`, whose documented
caveats (strategy fusion, single-sided) are addressed here.

Being a lower bound, it can prove the agent weak but never prove it strong.

### 6.2 Exact solution in `mini27` — ground truth

Exact best-response and equilibrium computation in the miniature, giving
true exploitability rather than a bound. Any algorithm is validated here
before it is trusted at full scale.

### 6.3 Version ladder

`eval/harness.py::round_robin` over saved checkpoints, tracking relative
progress continuously and cheaply.

### 6.4 Off-model baselines

Hand-built opponents drawn from real 2-7 convention — a rock that never
snows, a maniac that snows constantly, a player who breaks pat hands too
readily, one who draws two too often. These are **outside** the agent's own
`action_probs` family, which is the specific flaw in the current Leduc
results: every published number there is against opponents sharing the
agent's generative model.

### 6.5 Human play

Play-money 2-7 on a licensed site, self-play through `scripts/play.py`, and
recruited players from the mixed-game community. The external reality
check, and the source of any reputation the project earns.

### 6.6 Hold'em calibration harness (optional, later)

A heads-up no-limit hold'em adapter behind the same `Game` protocol,
existing only to play free public benchmarks (Slumbot, the GTO Wizard API)
and reveal where the search and belief code break against a strong
opponent. Not a competitive entry.

### 6.7 The tension to manage

Exploitability measures **defense**; the baseline pool measures
**offense**. They trade off — exploiting requires deviating from safe play,
which increases one's own exploitability. Neither number is optimized
alone; both are tracked, and trades between them are made deliberately and
recorded.

### 6.8 Publication

For the measurements to constitute a benchmark others can meet, the
artifact must include the pinned game configuration (§3.2), the **LBR
implementation itself**, and a queryable agent — not merely a number. A
lower bound measured by a stronger adversary is not comparable to one
measured by a weaker one.

---

## 7. Roadmap

| Phase | Content | Status |
|---|---|---|
| 0 | `Game` protocol, registry, `--gamemode`, CI | **done** |
| 1 | 2-7 engine (N-player), 5-card rank table, `mini27` | |
| 2 | Off-model baselines from real 2-7 convention | |
| 3 | Belief over hand classes; `update_on_draw` | |
| 4 | Depth-limited search with rollout leaves | |
| 5 | LBR exploitability; exact solver for `mini27` | |
| 6 | GA over the widened genome | |
| 7 | Publication: numbers, method, playable agent | |

Phases 5's instruments are wanted as early as they can be built: without
them, later phases tune blind and mistake self-play drift for progress.

Throughout, keep a running strategy log. The findings — what the agent
converges to on drawing standards, snowing frequency, breaking thresholds —
are the publishable artifact, and are far easier to record as they happen
than to reconstruct.

---

## 8. Open questions

1. **Arena structure.** Whether chipzen.ai's hold'em is limit or no-limit,
   whether its 2-7 is triple or single draw, and whether it offers anything
   multiway. Its site renders nothing without JavaScript, so this needs a
   browser. The answer does not change the plan but does change whether the
   arena is a usable external check.
2. **Raise cap heads-up.** Casinos frequently uncap heads-up betting.
   Pinned at 4 here for reproducibility; revisit if the arena differs.
3. **`mini27` parameters.** Deck size, hand size and draw count need tuning
   so that exact solution fits the RAM budget in §2.
4. **Hand-class granularity.** How coarse the belief's class partition can
   be before it costs measurable strength. To be determined empirically in
   `mini27`, where the exact answer is computable.
