"""ChipZen NLHE adapter: heads-up preflop policy, then the five-parameter policy."""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys

from chipzen import Action, Bot, GameState
from chipzen.client import run_bot

if __package__:
    from . import accounting_observer
else:  # Flat staged/container runtime.
    import accounting_observer

from sleight_of_hand.holdem import agent, preflop
from sleight_of_hand.holdem.agent import HoldemAgent
from sleight_of_hand.holdem.decision import Decision
from sleight_of_hand.holdem.equity import estimate_equity
from sleight_of_hand.holdem.opponent import _round_key
from sleight_of_hand.policy.heuristic import DEFAULT_PARAMS, PolicyParams


def live_opponents(state: GameState) -> int:
    """Compatibility wrapper for the Hold'em live-opponent count."""
    return agent.live_opponents(state)


def raise_amount(state: GameState, aggression: float) -> int:
    """Compatibility wrapper; the result is a raise-to total."""
    return agent.raise_amount(state, aggression)


def betting_params(state: GameState, params: PolicyParams) -> PolicyParams:
    """Compatibility wrapper for the existing pot-odds adjustment."""
    return agent.betting_params(state, params)


def _sdk_action(decision: Decision | None) -> Action | None:
    if decision is None:
        return None
    return Action(action=decision.action, amount=decision.amount)


def _decision(action: Action | None) -> Decision | None:
    if action is None:
        return None
    return Decision(action.action, action.amount)


class SleightOfHandBot(HoldemAgent, Bot):
    """Heads-up preflop policy first; otherwise the Season 6 equity policy.

    No exhaustive search, opponent ranges or learned Hold'em strategy.
    """

    def __init__(
        self,
        params: PolicyParams = DEFAULT_PARAMS,
        seed: int | None = None,
        samples: int = 128,
        preflop_config: preflop.PreflopConfig = preflop.DEFAULT_PREFLOP,
        trace_preflop: bool = False,
    ) -> None:
        if not 1 <= samples <= 512:
            raise ValueError("samples must be between 1 and 512")
        # The SDK owns one bot per match and invokes hooks serially.
        # Construct the RNG here and inject that same object into the policy.
        super().__init__(
            params=params.clipped(),
            rng=random.Random(seed),
            samples=samples,
            preflop_config=preflop_config,
            trace_preflop=trace_preflop,
        )
        self._seat: int | None = None
        self._accounting_observer = accounting_observer.from_environment()

    def _observe(self, event, *args) -> None:
        if self._accounting_observer is not None:
            try:
                self._accounting_observer.notify(event, *args)
            except Exception:  # noqa: BLE001 - isolate diagnostics, not policy
                self._accounting_observer = None
                accounting_observer.warning()

    def _learn_seat(self, message: dict) -> None:
        for seat in message.get("seats", []) or []:
            if isinstance(seat, dict) and seat.get("is_self"):
                self._seat = int(seat.get("seat", 0))
                return

    def on_match_start(self, match_info: dict) -> None:
        self._learn_seat(match_info)
        super().on_match_start(match_info)
        self._observe("match_start")

    def on_reconnected(self, message: dict) -> None:
        self._learn_seat(message)
        super().on_reconnected(message)
        self._observe("reconnected")

    def on_round_start(self, message: dict) -> None:
        if isinstance(message, dict):
            state = message.get("state")
            if isinstance(state, dict):
                self.shove_model.record_start(_round_key(message), state)
        super().on_round_start(message)
        self._observe("round_start", message, self._seat)

    def on_round_result(self, message: dict) -> None:
        # Observation never raises and applies a hand entirely or not at all.
        try:
            key = _round_key(message)
            result = message.get("result")
        except (AttributeError, TypeError, ValueError):
            pass
        else:
            self.shove_model.observe_result(key, result, self._seat)
        super().on_round_result(message)
        self._observe("round_result", message)

    def on_turn_result(self, message: dict) -> None:
        super().on_turn_result(message)
        self._observe("turn_result", message)

    def shove_estimate(self, context: preflop.PreflopContext) -> dict | None:
        return super().shove_estimate(context)

    def preflop_action(self, state: GameState) -> Action | None:
        if state.phase == "preflop" and self._seat is None:
            self._seat = state.your_seat
        return _sdk_action(super().preflop_decision(state))

    def preflop_decision(self, state: GameState) -> Decision | None:
        # Keep existing subclasses overriding preflop_action effective.
        return _decision(self.preflop_action(state))

    def _legal_preflop(
        self, state: GameState, context: preflop.PreflopContext, choice: str
    ) -> Action | None:
        return _sdk_action(super().legal_preflop_decision(state, context, choice))

    def legal_preflop_decision(
        self, state: GameState, context: preflop.PreflopContext, choice: str
    ) -> Decision | None:
        return _decision(self._legal_preflop(state, context, choice))

    def _trace(
        self,
        state: GameState,
        context: preflop.PreflopContext | None,
        reason: str,
        label: str | None = None,
        probs: dict[str, float] | None = None,
        action: Action | None = None,
        shove: dict | None = None,
    ) -> None:
        super()._trace(state, context, reason, label, probs, action, shove)

    def decide(self, state: GameState) -> Action:
        # Resolve the estimator here to retain the adapter's patchable hook.
        self._observe("decision_state", state)
        action = _sdk_action(super().decide(state, equity_estimator=estimate_equity))
        self._observe("selected_action", action)
        return action


def main() -> None:
    url = os.environ.get("CHIPZEN_WS_URL") or (
        sys.argv[1] if len(sys.argv) > 1 else None
    )
    if not url:
        print(
            "Set CHIPZEN_WS_URL or pass the WebSocket URL as the first argument.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    params = PolicyParams(**json.loads(os.environ.get("SLEIGHT_PARAMS", "{}")))
    seed_text = os.environ.get("SLEIGHT_SEED")
    bot = SleightOfHandBot(
        params=params,
        seed=int(seed_text) if seed_text is not None else None,
        samples=int(os.environ.get("SLEIGHT_EQUITY_SAMPLES", "128")),
        trace_preflop=os.environ.get("SLEIGHT_TRACE_PREFLOP") == "1",
    )
    asyncio.run(
        run_bot(
            url,
            bot,
            token=os.environ.get("CHIPZEN_TOKEN"),
            ticket=os.environ.get("CHIPZEN_TICKET"),
        )
    )


if __name__ == "__main__":
    main()
