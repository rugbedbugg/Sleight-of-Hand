"""SDK-independent Hold'em decisions using the existing preflop/equity policies."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import replace

from sleight_of_hand.engine.actions import ActionType
from sleight_of_hand.holdem import preflop
from sleight_of_hand.holdem.decision import Decision, DecisionState
from sleight_of_hand.holdem.equity import estimate_equity
from sleight_of_hand.holdem.hands import hand_class
from sleight_of_hand.holdem.opponent import OPEN, RESHOVE, ShoveModel
from sleight_of_hand.policy.heuristic import (
    DEFAULT_PARAMS,
    PolicyParams,
    action_probs_from_strength,
    sample_action,
)


def live_opponents(state: DecisionState) -> int:
    """Seat-indexed stacks include folded players; zero stacks can be all-in."""
    seats = set(range(len(state.opponent_stacks) + 1)) - {state.your_seat}
    folded = {
        entry.get("seat")
        for entry in state.action_history
        if entry.get("action") == "fold"
    }
    return len(seats - folded)


def raise_amount(state: DecisionState, aggression: float) -> int:
    """Use a pot-based raise-to target, bounded by the server's legal totals.

    The minimum already includes the current bet and minimum increment.
    Add up to a pot-sized amount above that minimum as aggression increases;
    never interpret the resulting amount as chips to add to a previous bet.
    """
    upper = state.max_raise
    lower = min(state.min_raise, upper)  # short all-in below the normal minimum
    extra = round((state.pot + min(state.to_call, state.your_stack)) * aggression)
    return max(lower, min(upper, lower + extra))


def betting_params(state: DecisionState, params: PolicyParams) -> PolicyParams:
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


class HoldemAgent:
    """One match's policy and shove model, using a caller-owned RNG.

    State is read directly, without normalizing cards, history mappings or
    seats. Lifecycle callers use ``shove_model.record_start`` and
    ``shove_model.observe_result`` with an explicit round key and hero seat.
    """

    def __init__(
        self,
        rng: random.Random,
        params: PolicyParams = DEFAULT_PARAMS,
        samples: int = 128,
        preflop_config: preflop.PreflopConfig = preflop.DEFAULT_PREFLOP,
        trace_preflop: bool = False,
    ) -> None:
        if not 1 <= samples <= 512:
            raise ValueError("samples must be between 1 and 512")
        self.params = params.clipped()
        self.rng = rng
        self.samples = samples
        self.preflop_config = preflop_config
        self.trace_preflop = trace_preflop
        self._warned_preflop = False
        self.shove_model = ShoveModel(preflop_config)

    def shove_estimate(self, context: preflop.PreflopContext) -> dict | None:
        """Adaptive shove-range estimate for a shove decision, else None."""
        if context.facing is not preflop.Facing.SHOVE:
            return None
        kind = RESHOVE if context.hero_acted else OPEN
        return self.shove_model.estimate(
            kind, preflop.stack_bucket(context.effective_stack_bb)
        )

    def preflop_decision(self, state: DecisionState) -> Decision | None:
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
        shove = self.shove_estimate(context)
        probs = preflop.action_distribution(
            context,
            label,
            self.preflop_config,
            shove["adaptive_width"] if shove else None,
        )
        choices = [a for a, p in probs.items() if p > 0]
        choice = self.rng.choices(choices, weights=[probs[a] for a in choices])[0]
        action = self.legal_preflop_decision(state, context, choice)
        self._trace(
            state,
            context,
            "ok" if action else "no_legal_action",
            label,
            probs,
            action,
            shove,
        )
        return action

    def legal_preflop_decision(
        self, state: DecisionState, context: preflop.PreflopContext, choice: str
    ) -> Decision | None:
        valid = set(state.valid_actions)
        if choice == preflop.RAISE:
            if "raise" in valid and state.max_raise > 0:
                return Decision(
                    "raise",
                    preflop.raise_to(
                        context, state.min_raise, state.max_raise, self.preflop_config
                    ),
                )
            if "all_in" in valid:
                return Decision("all_in")
        if choice == preflop.FOLD and "fold" in valid and "check" not in valid:
            return Decision("fold")
        if "check" in valid:
            return Decision("check")
        if "call" in valid:
            return Decision("call")
        return Decision("fold") if "fold" in valid else None

    def _trace(
        self,
        state: DecisionState,
        context: preflop.PreflopContext | None,
        reason: str,
        label: str | None = None,
        probs: dict[str, float] | None = None,
        action: Decision | None = None,
        shove: dict | None = None,
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
        if shove is not None:
            record["shove_model"] = {
                k: round(v, 4) if isinstance(v, float) else v for k, v in shove.items()
            }
        if label is not None:
            record["hand_class"] = label
        if probs is not None:
            record["probs"] = {a: round(p, 4) for a, p in probs.items()}
        if action is not None:
            record["action"] = action.action
            if action.action == "raise":
                record["raise_to"] = action.amount
        print(json.dumps(record, separators=(",", ":")), file=sys.stderr)

    def decide(
        self, state: DecisionState, *, equity_estimator=estimate_equity
    ) -> Decision:
        valid = set(state.valid_actions)
        if not valid:
            raise ValueError("ChipZen requested a decision without legal actions")
        passive = "check" if "check" in valid else "call" if "call" in valid else None
        mapped = {}
        if passive:
            mapped[ActionType.CALL] = (
                Decision("check") if passive == "check" else Decision("call")
            )
        if "fold" in valid:
            mapped[ActionType.FOLD] = Decision("fold")
        if "raise" in valid and state.max_raise > 0:
            mapped[ActionType.RAISE] = Decision(
                "raise", raise_amount(state, self.params.aggression)
            )
        elif "all_in" in valid:
            # Older SDK vocabulary: only emit this if the server offers it.
            mapped[ActionType.RAISE] = Decision("all_in")
        if not mapped:
            raise ValueError(f"no supported Hold'em actions: {sorted(valid)}")
        if len(mapped) == 1:
            return next(iter(mapped.values()))
        action = self.preflop_decision(state)
        if action is not None:
            return action

        try:
            strength = equity_estimator(
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
