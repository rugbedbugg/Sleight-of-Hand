"""Independent integer-chip model of the deliberately restricted M1 tree."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    history: tuple[str, ...] = ()
    contributions: tuple[int, int] = (0, 0)
    player: int = 0
    terminal: bool = False
    winner: int | None = None

    @property
    def key(self):
        return "/".join(self.history) or "root"


def actions(game, node):
    if node.terminal:
        return []
    p = node.player
    high = max(node.contributions)
    owed = high - node.contributions[p]
    stack = game["stacks"][p]
    if owed:
        result = [("c", "call", high, owed), ("f", "fold", None, 0)]
        if high < stack:
            result.append(
                (f"r{stack - high}", "raise", stack, stack - node.contributions[p])
            )
    else:
        result = [("c", "check", None, 0)]
        for target in (game["pot"] // 2, stack):
            result.append(
                (f"b{target}", "raise", target, target - node.contributions[p])
            )
    return [
        {
            "id": token,
            "action": verb,
            "raise_to": target if verb == "raise" else None,
            "pay": pay,
            "all_in": pay > 0 and node.contributions[p] + pay == stack,
        }
        for token, verb, target, pay in result
    ]


def advance(game, node, token):
    available = {a["id"]: a for a in actions(game, node)}
    if token not in available:
        raise ValueError("illegal public sequence")
    action = available[token]
    paid = list(node.contributions)
    paid[node.player] += action["pay"]
    terminal = action["action"] in ("fold", "call") or (
        token == "c" and node.history == ("c",)
    )
    return Node(
        node.history + (token,),
        tuple(paid),
        1 - node.player,
        terminal,
        1 - node.player if action["action"] == "fold" else None,
    )


def nodes(game):
    result = []

    def visit(node):
        result.append(node)
        if len(result) > 64:
            raise ValueError("public tree exceeds node cap")
        for action in actions(game, node):
            visit(advance(game, node, action["id"]))

    visit(Node())
    return result


def decision(game, node, hand):
    """The complete query contract; positions are OOP=0 and IP/button=1."""
    return {
        "phase": "river",
        "board": game["board"],
        "hero_position": node.player,
        "hole_cards": list(hand),
        "history": list(node.history),
        "pot": game["pot"] + sum(node.contributions),
        "stacks": [s - c for s, c in zip(game["stacks"], node.contributions)],
        "street_contributions": list(node.contributions),
        "to_call": max(node.contributions) - node.contributions[node.player],
        "legal_actions": actions(game, node),
    }
