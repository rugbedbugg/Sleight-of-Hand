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
  - engine/       rules engine (pure functions over immutable GameState)
  - policy/       the shared parametrized heuristic policy (genome / likelihood / opponent model)
  - bayes/        Bayesian opponent model
  - search/       expectiminimax search
  - agents/       baselines (random, always-call, rule-based) + the full BayesSearchAgent
  - ga/           genetic algorithm (genome, selection, crossover, mutation, elitism)
  - eval/         evaluation harness (mbb/hand), belief-accuracy, ablation, exploitability
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

## Results

Artifacts are in `results/` (regenerate with `python scripts/run_all_experiments.py`).