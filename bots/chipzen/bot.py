"""ChipZen NLHE adapter: heads-up preflop policy, then the five-parameter policy."""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from dataclasses import replace

from chipzen import Action, Bot, GameState
from chipzen.client import run_bot

from sleight_of_hand.engine.actions import ActionType
from sleight_of_hand.holdem import preflop
from sleight_of_hand.holdem.equity import estimate_equity
from sleight_of_hand.holdem.hands import hand_class
from sleight_of_hand.policy.heuristic import (
    DEFAULT_PARAMS,
    PolicyParams,
    action_probs_from_strength,
    sample_action,
)


def live_opponents(state: GameState) -> int:
    """Seat-indexed stacks include folded players; zero stacks can be all-in."""
    seats = set(range(len(state.opponent_stacks) + 1)) - {state.your_seat}
    folded = {
        entry.get("seat")
        for entry in state.action_history
        if entry.get("action") == "fold"
    }
    return len(seats - folded)


def raise_amount(state: GameState, aggression: float) -> int:
    """Use a pot-based raise-to target, bounded by the server's legal totals.

    The minimum already includes the current bet and minimum increment.
    Add up to a pot-sized amount above that minimum as aggression increases;
    never interpret the resulting amount as chips to add to a previous bet.
    """
    upper = state.max_raise
    lower = min(state.min_raise, upper)  # short all-in below the normal minimum
    extra = round((state.pot + min(state.to_call, state.your_stack)) * aggression)
    return max(lower, min(upper, lower + extra))


def betting_params(state: GameState, params: PolicyParams) -> PolicyParams:
    """Adjust equity thresholds for the price of continuing.

    Calling a pot-sized bet requires 1/3 equity to break even. Preserve the
    original call threshold at that reference price; its offset from 1/3
    remains the strategy's risk margin at other prices. Cap the payment
    at our stack for short all-in calls. Pot is the server's current pot,
    including the opponent's bet, but not our pending call.

    This is a one-pot approximation: future betting, equity realization
    and side-pot eligibility still require a richer model.
    """
    if "check" in state.valid_actions or state.to_call <= 0:
        return params
    cost = min(state.to_call, state.your_stack)
    if cost <= 0 or state.pot < 0:
        return params
    price = cost / (state.pot + cost)
    price_adjustment = price - 1.0 / 3.0
    threshold = max(0.0, min(1.0, params.call_threshold + price_adjustment))
    return replace(
        params,
        call_threshold=threshold,
        value_bet_threshold=min(
            1.0,
            max(threshold, params.value_bet_threshold + max(0.0, price_adjustment)),
        ),
    )


class SleightOfHandBot(Bot):
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
        self.params = params.clipped()
        self.rng = random.Random(seed)
        self.samples = samples
        self.preflop_config = preflop_config
        self.trace_preflop = trace_preflop
        self._warned_preflop = False

    def preflop_action(self, state: GameState) -> Action | None:
        """Decide a heads-up preflop state, or ``None`` to use Season 6.

        Declines multiway tables, states whose history cannot be reconciled
        with the pot, and unreadable cards (Season 6's safe fallback owns
        those). Unsupported choices degrade to the passive legal action;
        folding is never chosen when checking is free.
        """
        if state.phase != "preflop":
            return None
        context, reason = preflop.diagnose_context(state)
        if context is None:
            if len(state.opponent_stacks) == 1 and not self._warned_preflop:
                self._warned_preflop = True
                print(
                    "Preflop context unavailable; using the Season 6 policy.",
                    file=sys.stderr,
                )
            self._trace(state, None, reason)
            return None
        try:
            label = hand_class([str(c) for c in state.hole_cards])
        except ValueError:
            self._trace(state, context, "unreadable_hole_cards")
            return None
        probs = preflop.action_distribution(context, label, self.preflop_config)
        choices = [a for a, p in probs.items() if p > 0]
        choice = self.rng.choices(choices, weights=[probs[a] for a in choices])[0]
        action = self._legal_preflop(state, context, choice)
        self._trace(
            state, context, "ok" if action else "no_legal_action", label, probs, action
        )
        return action

    def _legal_preflop(
        self, state: GameState, context: preflop.PreflopContext, choice: str
    ) -> Action | None:
        valid = set(state.valid_actions)
        if choice == preflop.RAISE:
            if "raise" in valid and state.max_raise > 0:
                return Action.raise_to(
                    preflop.raise_to(
                        context, state.min_raise, state.max_raise, self.preflop_config
                    )
                )
            if "all_in" in valid:
                return Action.all_in()
        if choice == preflop.FOLD and "fold" in valid and "check" not in valid:
            return Action.fold()
        if "check" in valid:
            return Action.check()
        if "call" in valid:
            return Action.call()
        return Action.fold() if "fold" in valid else None

    def _trace(
        self,
        state: GameState,
        context: preflop.PreflopContext | None,
        reason: str,
        label: str | None = None,
        probs: dict[str, float] | None = None,
        action: Action | None = None,
    ) -> None:
        """One JSON line per preflop decision when tracing is enabled.

        Only public state and our own hand class; no hole cards, credentials
        or connection details. Tracing never touches the RNG or the choice.
        """
        if not self.trace_preflop:
            return
        record = {
            "event": "preflop",
            "hand": state.hand_number,
            "round_id": state.round_id,
            "seat": state.your_seat,
            "context_ok": context is not None,
            "reason": reason,
            "valid": list(state.valid_actions),
            "to_call": state.to_call,
            "pot": state.pot,
            "your_stack": state.your_stack,
            "opponent_stacks": list(state.opponent_stacks),
            "min_raise": state.min_raise,
            "max_raise": state.max_raise,
            "history": [
                [e.get("seat"), e.get("action"), e.get("amount")]
                if isinstance(e, dict)
                else repr(e)
                for e in state.action_history
            ],
        }
        if context is not None:
            record.update(
                position=context.position.value,
                facing=context.facing.value,
                big_blind=context.big_blind,
                effective_bb=round(context.effective_stack_bb, 2),
                hero_bet=context.hero_bet,
                villain_bet=context.villain_bet,
                amount_owed=context.amount_owed,
                call_cost=context.call_cost,
                sequence=list(context.action_sequence),
            )
        if label is not None:
            record["hand_class"] = label
        if probs is not None:
            record["probs"] = {a: round(p, 4) for a, p in probs.items()}
        if action is not None:
            record["action"] = action.action
            if action.action == "raise":
                record["raise_to"] = action.amount
        print(json.dumps(record, separators=(",", ":")), file=sys.stderr)

    def decide(self, state: GameState) -> Action:
        valid = set(state.valid_actions)
        if not valid:
            raise ValueError("ChipZen requested a decision without legal actions")
        passive = "check" if "check" in valid else "call" if "call" in valid else None
        mapped = {}
        if passive:
            mapped[ActionType.CALL] = (
                Action.check() if passive == "check" else Action.call()
            )
        if "fold" in valid:
            mapped[ActionType.FOLD] = Action.fold()
        if "raise" in valid and state.max_raise > 0:
            mapped[ActionType.RAISE] = Action.raise_to(
                raise_amount(state, self.params.aggression)
            )
        elif "all_in" in valid:
            # Older SDK vocabulary: only emit this if the server offers it.
            mapped[ActionType.RAISE] = Action.all_in()
        if not mapped:
            raise ValueError(f"no supported Hold'em actions: {sorted(valid)}")
        if len(mapped) == 1:
            return next(iter(mapped.values()))
        action = self.preflop_action(state)
        if action is not None:
            return action

        try:
            strength = estimate_equity(
                [str(c) for c in state.hole_cards],
                [str(c) for c in state.board],
                live_opponents(state),
                self.rng,
                self.samples,
            )
        except ValueError as exc:
            # Missing/invalid cards should lose one decision, not the session.
            print(f"Hold'em state unavailable: {exc}", file=sys.stderr)
            for action in (ActionType.CALL, ActionType.FOLD):
                if action in mapped and (
                    action == ActionType.FOLD or passive == "check"
                ):
                    return mapped[action]
            return next(iter(mapped.values()))

        probs = action_probs_from_strength(
            strength,
            list(mapped),
            0 if passive == "check" else state.to_call,
            betting_params(state, self.params),
        )
        return mapped[sample_action(self.rng, probs)]


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
