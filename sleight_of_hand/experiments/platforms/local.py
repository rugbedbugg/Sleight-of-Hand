"""Local heads-up No-Limit Hold'em platform.

The repository's engine is Leduc-only, so this module is the local *platform*:
a minimal HU NLHE dealer that owns local reality (deck, blinds, betting,
showdown, payouts) and drives seats through the same Chipzen-shaped hook
sequence the official SDK uses (``on_match_start``, ``on_round_start``,
``decide`` with ``GameState.from_turn_request``, ``on_turn_result``,
``on_phase_change``, ``on_round_result``, ``on_match_end``).

Where the hosted protocol is ambiguous the dealer does not guess silently:
pot reporting, ``to_call`` capping, round-start stack timing and bet capping
are explicit platform configuration recorded with every run. Local results
therefore describe this dealer's conventions, not Chipzen's.
"""

from __future__ import annotations

import copy
import io
import os
import random
from contextlib import contextmanager, redirect_stderr, redirect_stdout

from chipzen import Action, GameState

from sleight_of_hand.holdem.equity import RANKS, SUITS, encode, rank_hand
from sleight_of_hand.holdem.hands import hand_class
from sleight_of_hand.holdem.preflop import PERCENTILES
from sleight_of_hand.policy.heuristic import PolicyParams

from ..model import Availability, OpponentRef
from ..spec import ExperimentSpec
from .base import Emit, MatchContext, MatchSummary, Platform, PlatformStatus

ADAPTER_VERSION = "local-nlhe/1"
PHASES = ("preflop", "flop", "turn", "river")
CONFIG_KEYS = {
    "small_blind",
    "big_blind",
    "starting_stack",
    "stack_mode",
    "pot_convention",
    "to_call_convention",
    "cap_bets_to_effective",
    "accounting_observer",
}
OBSERVER_ENV = (
    "SLEIGHT_CHIPZEN_ACCOUNTING_OBSERVER",
    "SLEIGHT_CHIPZEN_ACCOUNTING_STDOUT",
    "SLEIGHT_CHIPZEN_ACCOUNTING_LOG",
)


def full_deck() -> list[str]:
    return [rank + suit for rank in RANKS for suit in SUITS]


def check_config(config: dict) -> dict:
    if set(config) != CONFIG_KEYS:
        raise ValueError(
            f"local platform config needs exactly {sorted(CONFIG_KEYS)}; "
            f"got {sorted(config)}"
        )
    sb, bb, stack = config["small_blind"], config["big_blind"], config["starting_stack"]
    if not all(type(v) is int for v in (sb, bb, stack)) or not 0 < sb <= bb < stack:
        raise ValueError("invalid blinds or starting stack")
    if config["stack_mode"] not in {"reset", "carry"}:
        raise ValueError("stack_mode must be reset or carry")
    if config["pot_convention"] not in {"committed", "contestable"}:
        raise ValueError("pot_convention must be committed or contestable")
    if config["to_call_convention"] not in {"owed", "capped"}:
        raise ValueError("to_call_convention must be owed or capped")
    for key in ("cap_bets_to_effective", "accounting_observer"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    return config


# --- the dealer ----------------------------------------------------------------


class Hand:
    """One heads-up hand. Seat ``button`` posts the small blind."""

    def __init__(self, number, button, stacks, deck, config, round_id):
        self.number, self.button, self.round_id = number, button, round_id
        self.config = config
        self.bb = config["big_blind"]
        self.start_stacks = list(stacks)
        self.stack = list(stacks)
        self.holes = {0: deck[0:2], 1: deck[2:4]}
        self.runout = deck[4:9]
        self.board: list[str] = []
        self.street = [0, 0]
        self.total = [0, 0]
        self.history: list[dict] = []
        self.phase = "preflop"
        self.folded: int | None = None
        self.acted: set[int] = set()
        self.closed: set[int] = set()  # may call/fold but not re-raise
        self.last_increment = self.bb
        big = 1 - button
        self._post(button, "post_small_blind", config["small_blind"])
        self._post(big, "post_big_blind", self.bb)

    def _post(self, seat, action, amount):
        paid = min(amount, self.stack[seat])
        self._pay(seat, paid)
        self.history.append(self._entry(seat, action, paid))

    def _entry(self, seat, action, amount):
        return {
            "seat": seat,
            "action": action,
            "amount": amount,
            "phase": self.phase,
            "is_timeout": False,
        }

    def _pay(self, seat, amount):
        self.stack[seat] -= amount
        self.street[seat] += amount
        self.total[seat] += amount

    # --- reporting conventions ------------------------------------------------

    def unmatched(self) -> int:
        """Chips one player has committed that the other can never match."""
        return max(
            max(0, self.total[s] - (self.total[1 - s] + self.stack[1 - s]))
            for s in (0, 1)
        )

    def pot(self) -> int:
        committed = self.total[0] + self.total[1]
        if self.config["pot_convention"] == "contestable":
            return committed - self.unmatched()
        return committed

    # --- legality -------------------------------------------------------------

    def owed(self, seat) -> int:
        return max(0, self.street[1 - seat] - self.street[seat])

    def needs_action(self, seat) -> bool:
        if self.folded is not None or self.stack[seat] == 0:
            return False
        other = 1 - seat
        if self.stack[other] == 0 and self.street[seat] >= self.street[other]:
            return False  # nothing left to respond to
        return self.owed(seat) > 0 or seat not in self.acted

    def legal(self, seat) -> dict:
        other, owed = 1 - seat, self.owed(seat)
        valid = ["fold", "call"] if owed > 0 else ["check"]
        top = self.street[seat] + self.stack[seat]
        if self.config["cap_bets_to_effective"]:
            top = min(top, self.street[other] + self.stack[other])
        low = self.street[other] + max(self.last_increment, self.bb)
        can_raise = (
            self.stack[seat] > owed
            and self.stack[other] > 0
            and seat not in self.closed
            and top > self.street[other]
        )
        if can_raise:
            valid.append("raise")
        to_call = owed
        if self.config["to_call_convention"] == "capped":
            to_call = min(owed, self.stack[seat])
        return {
            "valid_actions": valid,
            "to_call": to_call,
            "min_raise": min(low, top) if can_raise else 0,
            "max_raise": top if can_raise else 0,
        }

    # --- actions --------------------------------------------------------------

    def apply(self, seat, action: str, amount: int) -> dict:
        """Apply a legal action; illegal ones are rejected like the server.

        A rejected action becomes the SDK's safe fallback (check, else fold)
        and is reported so the run records it.
        """
        legal = self.legal(seat)
        rejected = None
        if action == "all_in":
            action, amount = (
                ("raise", legal["max_raise"])
                if "raise" in legal["valid_actions"]
                else ("call", 0)
            )
        ok = action in legal["valid_actions"] and (
            action != "raise"
            or (
                type(amount) is int
                and legal["min_raise"] <= amount <= legal["max_raise"]
            )
        )
        if not ok:
            rejected = {"action": action, "amount": amount}
            action = "check" if "check" in legal["valid_actions"] else "fold"
            amount = 0
        other = 1 - seat
        if action == "fold":
            self.folded = seat
            paid = 0
        elif action == "check":
            paid = 0
        elif action == "call":
            paid = min(self.owed(seat), self.stack[seat])
            self._pay(seat, paid)
        else:  # raise to ``amount`` on this street
            increment = amount - self.street[other]
            paid = amount - self.street[seat]
            self._pay(seat, paid)
            if increment >= self.last_increment:
                self.last_increment = increment
                self.acted = set()
                self.closed = set()
            else:  # a short all-in does not reopen raising for prior actors
                self.closed |= self.acted
        self.acted.add(seat)
        recorded = amount if action == "raise" else paid
        self.history.append(self._entry(seat, action, recorded))
        return {
            "seat": seat,
            "action": action,
            "amount": recorded,
            "rejected": rejected,
        }

    def next_street(self) -> bool:
        """Advance; False once the hand needs no more betting decisions."""
        index = PHASES.index(self.phase)
        if index == 3:
            return False
        self.phase = PHASES[index + 1]
        self.board = self.runout[: (3, 4, 5)[index]]
        self.street = [0, 0]
        self.acted, self.closed = set(), set()
        self.last_increment = self.bb
        return True

    def settle(self) -> dict:
        totals = list(self.total)
        payouts = [0, 0]
        showdown = []
        if self.folded is not None:
            winner = 1 - self.folded
            payouts[winner] = totals[0] + totals[1]
            winners = [winner]
        else:
            self.board = list(self.runout)
            low = min(totals)
            for seat in (0, 1):  # return uncalled chips
                payouts[seat] += totals[seat] - low
            ranks = {
                s: rank_hand([encode(c) for c in self.holes[s] + self.board])
                for s in (0, 1)
            }
            showdown = [
                {"seat": s, "hole_cards": list(self.holes[s]), "rank": list(ranks[s])}
                for s in (0, 1)
            ]
            contested = 2 * low
            if ranks[0] == ranks[1]:
                winners = [0, 1]
                half = contested // 2
                payouts[0] += half
                payouts[1] += half
                payouts[1 - self.button] += contested - 2 * half  # odd chip: BB
            else:
                winner = 0 if ranks[0] > ranks[1] else 1
                winners = [winner]
                payouts[winner] += contested
        final = [self.stack[s] + payouts[s] for s in (0, 1)]
        if sum(final) != sum(self.start_stacks):
            raise AssertionError("chip conservation violated")
        return {
            "hand_number": self.number,
            "winner_seats": winners,
            "pot": totals[0] + totals[1],
            "payouts": [{"seat": s, "amount": payouts[s]} for s in (0, 1)],
            "showdown": showdown,
            "action_history": copy.deepcopy(self.history),
            "stacks": final,
            "board": list(self.board),
        }


# --- seats -----------------------------------------------------------------------


class ScriptedSeat:
    """A Chipzen-hook-compatible scripted opponent; hooks default to no-ops."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.seat: int | None = None

    def on_match_start(self, message):
        for seat in message.get("seats", []):
            if seat.get("is_self"):
                self.seat = seat["seat"]

    def on_round_start(self, message): ...
    def on_turn_result(self, message): ...
    def on_phase_change(self, message): ...
    def on_round_result(self, message): ...
    def on_match_end(self, message): ...

    @staticmethod
    def passive(state: GameState) -> Action:
        return Action.check() if "check" in state.valid_actions else Action.call()


class CallingStation(ScriptedSeat):
    def decide(self, state):
        return self.passive(state)


class RandomLegal(ScriptedSeat):
    def decide(self, state):
        choice = self.rng.choice(state.valid_actions)
        if choice == "raise":
            top = min(state.max_raise, state.min_raise + state.pot)
            return Action.raise_to(
                self.rng.randint(state.min_raise, max(top, state.min_raise))
            )
        return Action(action=choice)


def strength_percentile(hole: list[str]) -> float:
    """Combo-weighted percentile of the hand class (0 = strongest)."""
    return PERCENTILES[hand_class(hole)][0]


class TightAggressive(ScriptedSeat):
    """A simple fixed TAG: card-strength thresholds, pot-fraction sizing."""

    def decide(self, state):
        hole = [str(c) for c in state.hole_cards]
        valid = state.valid_actions
        if state.phase == "preflop":
            top = strength_percentile(hole)
            facing = state.to_call > 0 and len(
                [h for h in state.action_history if h["action"] == "raise"]
            )
            if not facing:
                if "raise" in valid and top < 30.0:
                    return Action.raise_to(
                        max(state.min_raise, min(state.max_raise, 5 * state.to_call))
                    )
                return (
                    self.passive(state)
                    if "check" in valid or top < 55.0
                    else Action.fold()
                )
            if "raise" in valid and top < 5.0:
                return Action.raise_to(min(state.max_raise, state.min_raise * 2))
            return Action.call() if top < 15.0 else Action.fold()
        made = rank_hand([encode(c) for c in hole + [str(c) for c in state.board]])
        if made[0] >= 1:  # a pair or better
            if "raise" in valid:
                target = state.min_raise + (2 * state.pot) // 3
                return Action.raise_to(
                    max(state.min_raise, min(state.max_raise, target))
                )
            return self.passive(state)
        if "check" in valid:
            return Action.check()
        return Action.fold()


def street_commit(state: GameState, seat: int) -> int:
    """Chips ``seat`` has put in on the current street, from public history."""
    total = 0
    for entry in state.action_history:
        if entry.get("phase") != state.phase or entry.get("seat") != seat:
            continue
        if entry.get("action") == "raise":
            total = entry["amount"]  # raise-to totals
        elif entry.get("action") in ("call", "post_small_blind", "post_big_blind"):
            total += entry["amount"]
    return total


class AccountingProbe(ScriptedSeat):
    """Engineers one covering postflop bet without tripping the observer early.

    The observer captures only the first hand where SOH faces a call of at
    least its stack (or the opponent has nothing behind), so before the target
    the probe never bets SOH's whole stack and never commits its own. It
    gathers chips with small raises and small value bets, and once it leads by
    ``margin`` it stops risking chips. The target is any hand where SOH has
    the button (so the probe acts first postflop) and the probe started with
    at least ``margin`` more chips: it calls into the flop without anyone
    all-in, then bets the maximum legal amount as the street's first action.
    After firing it only checks or folds.
    """

    def __init__(self, rng, margin: int):
        super().__init__(rng)
        self.margin = margin
        self.fired = False
        self.leading = False
        self.target = False

    def on_round_start(self, message):
        state = message["state"]
        stacks, hero = state["stacks"], self.seat
        self.leading = stacks[hero] >= stacks[1 - hero] + self.margin
        self.target = not self.fired and self.leading and state["dealer_seat"] != hero

    def _safe_raise(self, state, target: int):
        """A raise-to below both SOH's reach and our own all-in."""
        villain = 1 - self.seat
        reach = street_commit(state, villain) + state.opponent_stacks[0]
        target = min(target, state.max_raise - 1, reach - 1)
        if "raise" in state.valid_actions and target >= state.min_raise:
            return Action.raise_to(target)
        return None

    @staticmethod
    def _quiet(state):
        return Action.check() if "check" in state.valid_actions else Action.fold()

    def _affordable(self, state, limit: float) -> bool:
        return 0 < state.to_call < state.your_stack and state.to_call <= limit

    def decide(self, state):
        if self.fired:
            return self._quiet(state)
        if self.target:
            if state.phase == "preflop":
                if state.to_call == 0:
                    return Action.check()
                if (
                    self._affordable(state, state.your_stack)
                    and state.opponent_stacks[0]
                ):
                    return Action.call()  # reach the flop, nobody all-in
                return Action.fold()
            first = not any(h["phase"] == state.phase for h in state.action_history)
            if first and "raise" in state.valid_actions:
                self.fired = True  # first to act, no contribution: cover SOH
                return Action.raise_to(state.max_raise)
            return self._quiet(state)
        if self.leading:
            return self._quiet(state)  # protect the lead until SOH's button
        if state.phase == "preflop":
            if state.to_call <= self.margin // 2:
                raised = self._safe_raise(state, state.min_raise)
                if raised:
                    return raised
            if state.to_call == 0:
                return Action.check()
            # Sticky: a fold-prone probe is simply bullied out of every pot.
            if self._affordable(state, state.your_stack / 5):
                return Action.call()
            return Action.fold()
        hole = [str(c) for c in state.hole_cards]
        made = rank_hand([encode(c) for c in hole + [str(c) for c in state.board]])
        if made[0] >= 1:
            if state.to_call == 0:
                raised = self._safe_raise(state, state.min_raise + state.pot // 3)
                if raised:
                    return raised
            if self._affordable(state, min(state.pot, state.your_stack / 3)):
                return Action.call()
        return self._quiet(state)


@contextmanager
def _observer_env(enabled: bool):
    saved = {k: os.environ.get(k) for k in OBSERVER_ENV}
    for key in OBSERVER_ENV:
        os.environ.pop(key, None)
    if enabled:
        os.environ["SLEIGHT_CHIPZEN_ACCOUNTING_OBSERVER"] = "1"
        os.environ["SLEIGHT_CHIPZEN_ACCOUNTING_STDOUT"] = "1"
    try:
        yield
    finally:
        for key, value in saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value


def build_soh(policy_config: dict, seed: int, observer: bool = False):
    """The canonical adapter, configured only through approved surfaces."""
    from bots.chipzen.bot import SleightOfHandBot

    from .priors import memory_for

    with _observer_env(observer):
        bot = SleightOfHandBot(
            params=PolicyParams(**policy_config.get("params", {})),
            seed=seed,
            samples=policy_config.get("samples", 128),
            opponent_memory=memory_for(policy_config),
        )
    if not policy_config.get("opponent_memory", True):
        bot.opponent_memory = None  # the adapter's supported "no memory" state
    return bot


SCRIPTED = {
    "calling_station": CallingStation,
    "random_legal": RandomLegal,
    "tag_simple": TightAggressive,
}


def opponent_seat(ref: OpponentRef, seed: int, config: dict):
    rng = random.Random(seed)
    if ref.kind == "script" and ref.key in SCRIPTED:
        return SCRIPTED[ref.key](rng)
    if ref.kind == "probe" and ref.key == "accounting_cover":
        return AccountingProbe(rng, margin=2 * config["big_blind"])
    if ref.kind == "soh" and ref.key == "canonical":
        return build_soh({}, seed)
    raise ValueError(f"unknown local opponent {ref.qualified}")


# --- the platform ------------------------------------------------------------------


class LocalPlatform(Platform):
    name = "local"
    adapter_version = ADAPTER_VERSION

    def status(self, spec: ExperimentSpec | None = None) -> PlatformStatus:
        return PlatformStatus(Availability.AVAILABLE, "in-process dealer")

    def metadata(self) -> dict:
        import chipzen

        return {
            "adapter_version": ADAPTER_VERSION,
            "chipzen_sdk": chipzen.__version__,
            "dealer": "heads-up NLHE; button posts the small blind",
            "round_start_stacks": "before_blinds",
            "action_history_amounts": "raise-to totals per street; calls add chips",
            "hook_order": "SDK order; GameState via GameState.from_turn_request",
            "rejected_actions": "SDK safe fallback: check, else fold",
            "tie_odd_chip": "big blind",
        }

    def prepare(self, spec: ExperimentSpec) -> None:
        check_config(spec.platform_config)
        if spec.game_variant != "nlhe_hu":
            raise ValueError("local platform plays nlhe_hu only")
        for ref in spec.opponent_cohort:
            opponent_seat(ref, 0, spec.platform_config)

    def play_match(self, context: MatchContext, emit: Emit) -> MatchSummary:
        spec = context.spec
        config = check_config(spec.platform_config)
        seeds = context.seeds
        # Arm-independent, so paired arms see byte-identical inputs.
        match_id = f"local-{spec.experiment_id}-m{context.match_index:06d}"
        soh = build_soh(
            spec.policy_config, seeds["soh"], observer=config["accounting_observer"]
        )
        opponent = opponent_seat(context.opponent, seeds["opponent"], config)
        seats = {0: soh, 1: opponent}
        deck_rng = random.Random(seeds["deck"])
        emit(
            {
                "kind": "match_meta",
                "soh_seat": 0,
                "opponent": context.opponent.qualified,
                "seeds": dict(seeds),
                "match_id": match_id,
            }
        )

        def call(seat, hook, message):
            # Capture SOH's own output (warnings, opt-in observer frames).
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                value = getattr(seats[seat], hook)(message)
            if seat == 0 and (out.getvalue() or err.getvalue()):
                emit(
                    {
                        "kind": "seat_output",
                        "seat": seat,
                        "hook": hook,
                        "stdout": out.getvalue(),
                        "stderr": err.getvalue(),
                    }
                )
            return value

        def broadcast(hook, message):
            emit({"kind": "message", "to": [0, 1], "message": message})
            for seat in (0, 1):
                call(seat, hook, copy.deepcopy(message))

        for seat in (0, 1):
            start = {
                "type": "match_start",
                "match_id": match_id,
                "seats": [
                    {
                        "seat": s,
                        "participant_id": f"{match_id}-p{s}",
                        "is_self": s == seat,
                    }
                    for s in (0, 1)
                ],
                "game_config": {
                    "variant": "nlhe",
                    "starting_stack": config["starting_stack"],
                    "small_blind": config["small_blind"],
                    "big_blind": config["big_blind"],
                    "ante": 0,
                    "total_hands": spec.stopping.hands_per_match,
                },
            }
            emit({"kind": "message", "to": [seat], "message": start})
            call(seat, "on_match_start", start)

        stacks = [config["starting_stack"]] * 2
        net = [0, 0]  # cumulative chips won; stacks reset in "reset" mode
        played, ended = 0, "complete"
        for number in range(1, spec.stopping.hands_per_match + 1):
            if config["stack_mode"] == "reset":
                stacks = [config["starting_stack"]] * 2
            elif min(stacks) < config["big_blind"]:
                ended = "bust"
                break
            deck = full_deck()
            deck_rng.shuffle(deck)
            button = (number + context.match_index) % 2
            round_id = f"{match_id}-r{number:05d}"
            hand = Hand(number, button, stacks, deck, config, round_id)
            emit(
                {
                    "kind": "deal",
                    "hand": number,
                    "button": button,
                    "stacks": list(stacks),
                    "holes": {str(s): hand.holes[s] for s in (0, 1)},
                    "runout": hand.runout,
                }
            )
            for seat in (0, 1):
                message = {
                    "type": "round_start",
                    "match_id": match_id,
                    "round_id": round_id,
                    "round_number": number,
                    "state": {
                        "hand_number": number,
                        "dealer_seat": button,
                        "your_hole_cards": list(hand.holes[seat]),
                        "stacks": list(stacks),
                    },
                }
                emit({"kind": "message", "to": [seat], "message": message})
                call(seat, "on_round_start", message)
            self._betting(hand, seats, call, broadcast, emit, match_id, first=button)
            while hand.folded is None and hand.next_street():
                broadcast(
                    "on_phase_change",
                    {
                        "type": "phase_change",
                        "match_id": match_id,
                        "round_id": round_id,
                        "state": {"phase": hand.phase, "board": list(hand.board)},
                    },
                )
                self._betting(hand, seats, call, broadcast, emit, match_id, 1 - button)
            result = hand.settle()
            broadcast(
                "on_round_result",
                {
                    "type": "round_result",
                    "match_id": match_id,
                    "round_id": round_id,
                    "round_number": number,
                    "result": result,
                },
            )
            for seat in (0, 1):
                net[seat] += result["stacks"][seat] - hand.start_stacks[seat]
            stacks = result["stacks"]
            played = number
        broadcast(
            "on_match_end",
            {
                "type": "match_end",
                "match_id": match_id,
                "reason": ended,
                "results": [
                    {"seat": s, "score": stacks[s], "net_chips": net[s]} for s in (0, 1)
                ],
            },
        )
        return MatchSummary(
            match_index=context.match_index,
            hands=played,
            platform_ids={"match_id": match_id},
            ended=ended,
        )

    @staticmethod
    def _betting(hand, seats, call, broadcast, emit, match_id, first):
        seat = first
        while hand.folded is None:
            if not hand.needs_action(seat):
                if not hand.needs_action(1 - seat):
                    return
                seat = 1 - seat
                continue
            legal = hand.legal(seat)
            request = {
                "type": "turn_request",
                "match_id": match_id,
                "round_id": hand.round_id,
                "request_id": f"{hand.round_id}-a{len(hand.history)}",
                "seat": seat,
                "valid_actions": legal["valid_actions"],
                "state": {
                    "hand_number": hand.number,
                    "phase": hand.phase,
                    "board": list(hand.board),
                    "your_hole_cards": list(hand.holes[seat]),
                    "pot": hand.pot(),
                    "your_stack": hand.stack[seat],
                    "opponent_stacks": [hand.stack[1 - seat]],
                    "your_seat": seat,
                    "dealer_seat": hand.button,
                    "to_call": legal["to_call"],
                    "min_raise": legal["min_raise"],
                    "max_raise": legal["max_raise"],
                    "action_history": copy.deepcopy(hand.history),
                },
            }
            emit({"kind": "message", "to": [seat], "message": request})
            state = GameState.from_turn_request(
                request, your_seat=seat, dealer_seat=hand.button
            )
            action = call(seat, "decide", state)
            chosen = {
                "action": action.action,
                "amount": action.amount if action.action == "raise" else 0,
            }
            emit(
                {
                    "kind": "action",
                    "seat": seat,
                    "request_id": request["request_id"],
                    **chosen,
                }
            )
            applied = hand.apply(seat, chosen["action"], chosen["amount"])
            if applied["rejected"] is not None:
                emit({"kind": "rejected", "seat": seat, **applied["rejected"]})
            broadcast(
                "on_turn_result",
                {
                    "type": "turn_result",
                    "match_id": match_id,
                    "round_id": hand.round_id,
                    "is_timeout": False,
                    "details": {
                        "seat": seat,
                        "action": applied["action"],
                        "amount": applied["amount"],
                        "pot": hand.pot(),
                    },
                },
            )
            seat = 1 - seat
