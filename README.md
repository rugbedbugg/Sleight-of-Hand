# Sleight-of-Hand

**Calling the Bluff: An AI That Learns to Play Poker Without the Tells**

A heads-up Leduc hold'em agent built on Bayesian opponent modeling,
expectiminimax game-tree search, and genetic-algorithm strategy tuning.

## Architecture

```
                    Leduc Engine (sleight_of_hand/engine)
                 state, legal actions, betting, showdown, payoffs
                                   |
        +--------------------------+--------------------------+
        |                          |                          |
  Bayesian opponent          Expectiminimax              GA strategy
  model (bayes/)     -----> search (search/) <-----      tuner (ga/)
  P(card | history)   belief   over betting tree   params    evolves policy/heuristic.py's
                                                              PolicyParams genome
```

The three techniques share one load-bearing object:
`policy/heuristic.py`'s `action_probs(card, public, legal, to_call, params)`
— a smooth, parametrized model of "how does a player with this hand act
in this spot?" It plays four roles at once:

- As the **rule-based baseline agent**, with hand-tuned params.
- As the **likelihood term** `P(action | card, context)` in the Bayesian
  opponent model's Bayes'-rule update.
- As the **opponent-response model at opponent nodes** inside
  expectiminimax search.
- As the **genome** the genetic algorithm evolves — its five parameters
  (`value_bet_threshold`, `call_threshold`, `bluff_freq`, `aggression`,
  `steepness`) are exactly what it tunes.

## Techniques and where they live

| Technique | Component | Where |
|---|---|---|
| Environment model | Leduc as a partially-observable, stochastic, two-agent environment | `engine/` (state/action/reward design) |
| Game-tree search | Expectiminimax over the betting tree: MAX nodes (our decisions), chance nodes (community card), opponent nodes (belief- and policy-weighted expectation), plus a worst-case `minimax` variant | `search/expectiminimax.py` |
| Evolutionary search | Genetic algorithm evolving bet/call/fold thresholds, bluff frequency, and aggression | `ga/` |
| Bayesian inference | `P(opponent card \| betting history)` maintained via Bayes' rule; updated on every action and on the community-card reveal | `bayes/opponent_model.py` |

## Repository layout

```
sleight_of_hand/
  engine/       rules engine (pure functions over immutable GameState)
  policy/       the shared parametrized heuristic policy (genome / likelihood / opponent model)
  bayes/        Bayesian opponent model
  search/       expectiminimax search
  agents/       baselines (random, always-call, rule-based) + the full BayesSearchAgent
  ga/           genetic algorithm (genome, selection, crossover, mutation, elitism)
  eval/         evaluation harness (mbb/hand), belief-accuracy, ablation, exploitability
tests/          unit tests for every component above
scripts/        experiment runners that produce results/ (plots, tables, best genome)
demo.py         live demo: agent playing hands with belief state shown per decision
results/        generated plots, CSVs, and the GA's best genome (produced by scripts/)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running things

```bash
# unit tests (engine, policy, Bayes model, search, GA, eval) -- ~10s
pytest tests/ -q

# live demo: watch the agent play, with belief state and search EVs per decision
python demo.py --hands 5 --seed 1

# run everything and populate results/ (takes a few minutes; runs the
# steps below in order)
python scripts/run_all_experiments.py
```

Or run individual experiments:

```bash
python scripts/run_ga.py --population 40 --generations 30      # -> results/ga_fitness_curve.png, ga_best_genome.json
python scripts/run_baseline_eval.py --hands 4000                # -> results/win_rate_matrix.{csv,png}
python scripts/plot_belief_convergence.py                       # -> results/belief_convergence.png
python scripts/ablation.py                                      # -> results/ablation.png
python scripts/run_exploitability.py                            # -> results/exploitability.png (optional/advanced)
```

## Evaluation

- **Win-rate (mbb/hand).** Leduc has no literal blind, so — following the
  usual poker-AI convention of pegging the unit to a fixed bet size — one
  "big blind" is defined as the round-1 bet (2 chips). `eval/harness.py`
  plays large round-robin matches with seats alternated each hand to
  cancel positional variance, and reports mean ± 95% CI in mbb/hand.
- **Belief accuracy.** `eval/belief_accuracy.py` tracks
  `P(opponent's true card)` under the Bayesian model at every observation
  within a hand, averaged over thousands of hands — the belief-convergence
  plot.
- **GA fitness curve.** Best/mean/worst population fitness per generation,
  fitness = mean mbb/hand against a *fixed* baseline pool (kept fixed
  across generations, plus a small coevolutionary term against the
  previous generation's champion, so the curve is a meaningful,
  comparable trend rather than a moving target).
- **Ablation.** The same `BayesSearchAgent`, once updating belief from
  observed opponent actions and once restricted to the deck-combinatoric
  prior only — isolating what the Bayesian layer is worth, in mbb/hand.
- **Exploitability (optional, advanced).** Because Leduc's game tree is
  small enough to search exhaustively, `eval/exploitability.py` computes
  an exact best response to any fixed strategy (by setting the search
  agent's assumed opponent parameters *and* Bayesian likelihood model to
  match that strategy precisely) and Monte-Carlo estimates its
  best-response value — a genuine, tractable measure of how exploitable a
  given strategy is. See that module's docstring for exactly what is and
  isn't claimed (single-sided best response to a stationary strategy, not
  full two-sided Nash-equilibrium exploitability).

## Results from this run

Artifacts are in `results/` (regenerate with `python scripts/run_all_experiments.py`).

- **Win rate.** Round robin, 4000 hands/pairing, mean mbb/hand vs the rest
  of the field: `bayes_search +341`, `ga_tuned +150`, `bayes_minimax +95`,
  `rule_based +55`, `random -316`, `always_call -326`. The full pipeline
  agent (expectiminimax) is the clear headline result. See
  `win_rate_matrix.png` / `.csv`.
- **Belief convergence.** Averaged over 3000 hands, the posterior on the
  opponent's *true* card rises from **0.36** (prior, close to the ~0.33-0.4
  uniform baseline) to **0.79** by the last observation in the hand. See
  `belief_convergence.png`.
- **Ablation.** Search with Bayesian updates beats the same search using
  only the deck-combinatoric prior by **+139 mbb/hand** (`+311` vs `+172`
  against a rule-based opponent) — a direct, quantified measure of what
  the Bayesian layer is worth. See `ablation.png`.
- **GA fitness.** Mean population fitness rises from `+51` to a
  `~+260–310` mbb/hand plateau over 30 generations against the fixed
  baseline pool; best-of-generation fluctuates noisily around `+400–500`
  once past the initial climb (elitism preserves the champion, so this
  noise is opponent-pool variance, not regression). See
  `ga_fitness_curve.png`.
- **GA overfits its training pool — a genuine, reportable finding.**
  `ga_tuned` does well against the exploitable field it was trained on
  (headline `+150` mbb/hand, second only to `bayes_search`, and it beats
  plain `rule_based` head-to-head by `+55`). But it is decisively beaten by
  the best-responding search agent (`bayes_search` earns `+357` mbb/hand
  off `ga_tuned`), and it is *more* exploitable by an exact best responder
  (`+379.5` mbb/hand) than either `rule_based` (`+300.2`) or the untuned
  `default_heuristic` (`+314.1`) — see `exploitability.png`. The evolved
  genome (`results/ga_best_genome.json`) has low bluff frequency (`0.07`)
  and maximal aggression (`1.0`): a strategy that presses opponents who
  fold too little, and is correspondingly transparent/exploitable against
  one that adapts. This is a textbook premature-convergence-to-training-
  opponents failure mode, caught empirically rather than theoretically.
- **Exploitability.** As a sanity check, a deliberately bad strategy
  (`always_bet_never_fold`) is estimated at **+1160 mbb/hand** exploitable
  — roughly 3-4x every reasonable strategy tested — confirming the
  best-response search is discriminating real strategy quality, not just
  noise.

## Design notes worth knowing before reading the code

- **The engine never hides information.** `GameState.private` holds both
  players' hole cards; agents are trusted not to peek at the wrong index.
  `BayesSearchAgent` and `bayes/opponent_model.infer_belief` are careful to
  only ever read `state.private[my_player]`.
- **Belief is recomputed from history, not mutated per-hand.** Rather than
  a stateful model updated via engine hooks, `infer_belief(state,
  my_player, params)` replays `state.history` from scratch on every call.
  This makes correctness a property of one small pure function instead of
  a lifecycle of mutations to keep in sync — and it composes cleanly with
  search, which explores hypothetical futures the real belief model never
  actually visits.
- **Search is exact, not depth-limited.** The full Leduc betting tree
  (given a fixed hypothesis for the opponent's card) is small enough to
  enumerate completely, so `search/expectiminimax.py` never truncates or
  heuristically evaluates a non-terminal node.
