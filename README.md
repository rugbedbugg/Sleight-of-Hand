# Sleight-of-Hand

**Calling the Bluff: An AI That Learns to Play Poker Without the Tells**

A heads-up Leduc hold'em agent built on Bayesian opponent modeling,
expectiminimax game-tree search, and genetic-algorithm strategy tuning.

## Architecture

```
engine/     immutable GameState; legal actions, betting, showdown, payoffs
   |
   +-- bayes/    P(opp card | history), Bayes' rule
   +-- search/   expectiminimax over the betting tree
   +-- ga/       evolves PolicyParams
        `-------> policy/heuristic.py :: action_probs()
```

All three techniques consume one function:
`policy/heuristic.py::action_probs(card, public, legal, to_call, params)`,
a sigmoid model of `P(action | hand, context)` over five real-valued
parameters (`value_bet_threshold`, `call_threshold`, `bluff_freq`,
`aggression`, `steepness`).

| Role | Consumer |
|---|---|
| Rule-based baseline policy | `agents/baselines.py::RuleBasedAgent` |
| Likelihood `P(action \| card)` in the Bayes update | `bayes/opponent_model.py` |
| Opponent-response model at opponent nodes | `search/expectiminimax.py` |
| Genome (5 genes, bounded in `ga/genome.py`) | `ga/evolve.py` |

Changing the parameters changes the baseline, the belief update, the
search's opponent model, and the GA phenotype at once.

### Search

`search/expectiminimax.py`, three node types:

* **MAX**: our decision. `max` over legal actions.
* **chance**: community card. Deck-weighted expectation.
* **opponent**: expectation under `action_probs`, not `min`.

The opponent's hidden card is fixed as a hypothesis and marginalized over
the Bayesian belief at the root. The betting tree is enumerated
exhaustively (no depth limit), which Leduc's size permits.

### Belief

`bayes/opponent_model.py::infer_belief` replays `state.history` from
scratch each decision, reading only `state.private[my_player]`. Prior is
deck combinatorics after removing our own card; updates are the action
likelihood from `action_probs`, plus a hypergeometric update on the
community reveal.

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

Win rates in **mbb/hand** (milli-big-blinds per hand). Leduc antes rather
than blinds, so `BIG_BLIND` is pegged to the round-1 bet size (2 chips):
`mbb = 1000 * chips_won_per_hand / 2`.

| Experiment | Script | Output |
|---|---|---|
| Round-robin win rate | `run_baseline_eval.py` | `win_rate_matrix.{csv,png}` |
| Belief convergence | `plot_belief_convergence.py` | `belief_convergence.png` |
| GA fitness | `run_ga.py` | `ga_fitness_curve.png`, `ga_best_genome.json` |
| Bayes ablation | `ablation.py` | `ablation.png` |
| Exploitability | `run_exploitability.py` | `exploitability.png` |

**Win rate.** `eval/harness.py::play_match` alternates seat 0 every hand to
cancel positional asymmetry and returns per-hand mean and standard error.
`MatchResult.__str__` renders a normal-approximation 95% interval
(1.96 SE). `round_robin` plays each unordered pair once and mirrors.

**Belief convergence.** `eval/belief_accuracy.py` records
`P(opponent's true card)` after every observation (each opponent action,
plus the community reveal), averaged across hands. Both seats play the
same `RULE_BASED_PARAMS`, so the likelihood is exactly matched to the
generative process: this measures calibration under a correctly specified
model, not robustness to model mismatch.

**GA fitness.** Fitness is mean mbb/hand over `n_hands_per_opponent`
against a fixed pool (`random`, `always_call`, `rule_based`), blended
`0.75 / 0.25` with a match against the previous generation's champion
(`GAConfig.coevolve_frac`). The pool is held fixed across generations so
the curve is a comparable trend. Best/mean/worst reported per generation.

**Ablation.** One `BayesSearchAgent`, `use_bayes=True` vs `False`. The
disabled arm keeps the deck-combinatoric prior and the community-card
conditioning and drops only the action-likelihood updates, isolating the
value of observing betting behaviour.

**Exploitability.** `eval/exploitability.py` estimates a **lower bound**
on a fixed strategy `sigma`'s exploitability: a `BayesSearchAgent` with
both its likelihood model and its opponent model set to `sigma`, scored by
realized mbb/hand over Monte Carlo hands. Not a true best response: the
search takes `max` inside each fixed card hypothesis, so future decisions
are clairvoyant (strategy fusion). Also single-sided, against one
stationary strategy, not two-sided Nash exploitability
## Results

Artifacts are in `results/` (regenerate with `python scripts/run_all_experiments.py`).