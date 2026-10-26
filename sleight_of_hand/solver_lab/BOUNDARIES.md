# Existing baseline and semantic boundaries

Baseline source: `1b1536241e29a77a45a7398629456b6831c3625e`.
Canonical runtime fingerprint:
`sha256:eff445aa51a2b78871c294df085353b4f00b3b38a0df73a954f13f2270a0cfbf`.

Inspected `holdem/agent.py`, `preflop.py`, `preflop_tables.py`, `equity.py`,
`policy/heuristic.py`, `experiments/platforms/local.py`, `spec.py`, `runner.py`,
`bots/chipzen/bot.py`, SDK 0.4.0 models/conformance, platform qualification,
postflop characterization/known-defect tests, and E0006 design/result.

## Production behavior

The preflop policy derives position and betting context from blind posts and
history, uses combo-weighted 169-class rankings and Monte Carlo equity tables,
and applies heuristic range widths and sizes. The tables explicitly describe
measured showdown equity, not Nash strategy. Class weighting is 6 pair,
4 suited and 12 offsuit combinations, totaling 1,326 physical starting hands.

Postflop, the five parameters are value threshold, call threshold, bluff
frequency, aggression and sigmoid steepness. Default equity uses 128 random
runouts/opponent hands; supported sample count is 1–512. This is sampled
showdown equity against uniform remaining opponent cards, not inferred river
ranges. The sampled equity feeds heuristic action probabilities and a single
pot-based raise size. ShoveModel learns match-local preflop shove behavior.
SafeExploit is optional and default off. No external solver trained this policy.

E0006 remains 240,000 hands, +2.85 bb/100, 95% CI [-4.32, +10.02],
NO_DIFFERENCE_DETECTED. Mechanical SafeExploit activation is not established
performance improvement. M1 uses none of those hands or opponents as labels.

## Accounting and translation

| Concept | SOH / local dealer / SDK | Teacher and M1 boundary |
| --- | --- | --- |
| Heads-up position | Button posts SB, acts first preflop and last postflop; local button alternates | Player 0 is OOP/BB; player 1 is IP/button. These are roles, not raw Chipzen seat numbers. Root is OOP's first river decision |
| Blinds and stacks | Local round-start stacks are before blind posts; decision stacks are chips behind. Histories record posted chips | River root has no new blinds or antes. Settled pot includes earlier contributions; both behind-stacks are 100. BB=2 validates minimum bets |
| Pot | Local `committed` sums chips paid; `contestable` subtracts amounts that cannot be matched | Root pot 100 plus both river contributions. Equal stacks prevent unmatched chips in every supported node |
| Call | Local `owed` is street gap; `capped` is min(gap, stack). SDK supplies `to_call` | Every supported call is affordable and equals the gap; an all-in call closes the hand |
| Raise | SDK/SOH amount is total contributed this street, including previous chips | Teacher `bX` adds X chips; `rX` adds call cost plus X. Convert rX to opponent's current total + X. M1 b50/r50 means raise-to 100, not 50 or 150 |
| Actions | SDK fold/check/call/raise, with legacy all_in support; local offers check or fold/call plus raise | Teacher c means check or call by context; f is fold. All-in is a raise-to or call with `all_in=true`, never a separate invented size |
| Raising rules | Local tracks minimum increments and short-all-in reopening | Upstream does not fully enforce these rules or short calls. M1 allows only half-pot opening bet, full-stack bet, or full-stack raise over half-pot. Every edge is checked against the local dealer |
| Private ranges | Production samples remaining cards, with preflop classes used for heuristic decisions | Explicit weighted two-card combinations. Joint deal weight is w0×w1, conditioned on board and mutual card removal. No equally weighted class substitution |
| Cards | Canonical strings rank `23456789TJQKA`, suit `cdhs`; SOH integer is rank_index×4+suit_index | Teacher integer is suit_index×13+rank_index. Only validated strings cross the interface; all 52 cards and supported hand strengths are independently checked |
| Utility | Local payouts return uncalled chips and conserve stacks; historical outcomes are net match chips | Teacher utilities are river payout minus river contribution, summing to the settled pot. Exploitability=(BR0+BR1−pot)/2. OpenSpiel bridge subtracts pot/2 from each utility |

The external solver's `max_raises` counts the opening bet too; M1 calls this
`max_wagers=2`. Root and checked-to bets allow 50 and 100; facing 50 permits
call/fold or raise-to 100; facing 100 permits call/fold. Every terminal branch
has matched contributions or a fold. There are no side pots, rake, later
streets, multiple players, unequal root stacks, or arbitrary bet sizes.

## Unresolved hosted semantics and budget

The SDK documents raise-to amounts but does not establish every hosted pot
convention. Local platform configuration is explicit and must not be treated
as observed Chipzen behavior. Existing synthetic postflop fixtures freeze a
known untrimmed covering-shove price defect: supplied pot 6,100 and affordable
call 600 produce 6/67 instead of the modeled contestable 3/7. Live reachability
remains inconclusive. M1 neither fixes nor relies on that scenario.

SDK conformance examples advertise 5,000 ms turn timeouts, consistent with the
repository's five-second design budget. `DecisionState` does not expose an
enforced wall-clock budget and HoldemAgent bounds samples, not elapsed time.
The local dealer likewise does not enforce a per-decision timeout. These are
not verified live-server guarantees. M1 solves offline; a measured lookup
is about 54 microseconds median, after one validated artifact load. No live
competition integration or runtime performance claim follows from this test.
