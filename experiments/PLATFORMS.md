# Platform qualification

A platform is eligible only if automated play is explicitly permitted, an
official or stable interface exists, game semantics are observable,
experiments can be run ethically and reproducibly, and initial tests need no
rated or paid participation. Adapters exist only for qualified platforms.
Permission is never inferred from the mere existence of an endpoint.

## Chipzen — qualified; adapter implemented; READY_FOR_CREDENTIALS

| | |
|---|---|
| Permission | Bot platform. The official SDK (`chipzen-bot` 0.4.0) and the SDK repository's `docs/EXTERNAL-API-BOT-PROTOCOL.md` and `docs/external-api/FIRST-30-MINUTES.md` document remote bots ("External API" bot kind) run from the developer's own machine |
| Interface | `chipzen.run_external_bot`: lobby WebSocket `/ws/external/bot/{bot_id}` -> `matched` -> per-match gateway; token in `Sec-WebSocket-Protocol` |
| Game | NLHE, same match data plane as uploaded bots |
| Identity | `matched` carries `match_id`, `participant_id`, `rated`; `match_start.seats` carries `is_self`. No opponent bot ID is documented, so opponent identity stays match-scoped |
| Match semantics | Casual house-bot challenges and same-owner matches are unrated; same-owner unrated challenges are created from the dashboard; tournaments and the matchmaking queue are rated (never used) |
| Rate limits | Connection rate limit (close 1008); 10 messages/s and 5 invalid actions per round per participant; per-token concurrent-match cap; per-account free-tier match and lobby-hour metering |
| Cost | Free tier, metered. Runs are bounded by `max_matches` |
| Observability | SOH's own delivered messages; opponent hole cards only at showdown; the opt-in accounting observer for E0001 |
| Adapter complexity | Low: reuses the official SDK; raw evidence is SDK-delivered hook messages (decision requests rebuilt from `GameState`) |
| Boundaries | Dashboard-only bot creation and token issuance; dashboard-only same-owner unrated challenges; no credentials present in this environment |
| Recommendation | Use for E0001 once the owner creates the two research bots, tokens and an unrated challenge |

## GTO Wizard Benchmark — candidate; BLOCKED_PENDING_RULE_CONFIRMATION; no adapter

| | |
|---|---|
| Permission | Described as "a public API and standardized evaluation framework for benchmarking algorithms in Heads-Up No-Limit Texas Hold'em" (arXiv 2603.23660), so automated agents are its purpose |
| Interface | A RESTful API managing game state (per the paper); official endpoint documentation, authentication and terms were not available to verify in this session |
| Game | HUNL against GTO Wizard AI |
| Identity | Single fixed benchmark opponent |
| Match semantics | Benchmark hands with AIVAT variance reduction |
| Rate limits / cost | Not verified |
| Observability | Expected to be good (AIVAT needs full information), not verified |
| Recommendation | Strongest next platform (`BENCHMARK` provenance). Qualify after reading the official API documentation and terms: access, cost, rate limits and permitted volume |

## Slumbot — not qualified

| | |
|---|---|
| Permission | No official bot-play documentation could be verified. Published work (OpenHoldem, arXiv 2012.06168) describes the site as intended for human players and notes researchers resorting to browser simulation, which this program does not do |
| Recommendation | Rejected unless the operator publishes an official API with permission for automated play |

## Unvetted "AI arena" sites — not qualified

Hobby sites surfaced by search (an "AI vs AI" poker site on a dynamic-DNS
host whose TLS endpoint failed; an "AI × Human" arena) have no verifiable
operator, terms or permission, and mixed human/AI rooms are out of scope.
Rejected.

## Local — internal platform

The in-process heads-up NLHE dealer (`platforms/local.py`). Explicit
conventions: `pot_convention` (`committed` | `contestable`),
`to_call_convention` (`owed` | `capped`), `cap_bets_to_effective`,
`stack_mode` (`reset` | `carry`), round-start stacks before blinds,
action-history raise amounts as raise-to totals. These are modelling choices;
they are not claims about any hosted platform.
