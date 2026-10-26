"""Independent finite-tree EV/BR check using SOH's separate hand evaluator.

At a hero node maximize AFTER summing compatible opponent hands, so the
best response never gets to condition its action on hidden opponent cards.
"""

from sleight_of_hand.holdem.equity import encode, rank_hand

from .tree import Node, actions, advance


def evaluate(game, profile):
    ranges = game["ranges"]
    ranks = [
        [rank_hand([encode(c) for c in h["cards"] + game["board"]]) for h in r]
        for r in ranges
    ]
    joint = [
        (i, j, a["weight"] * b["weight"])
        for i, a in enumerate(ranges[0])
        for j, b in enumerate(ranges[1])
        if set(a["cards"]).isdisjoint(b["cards"])
    ]
    mass = sum(w for _, _, w in joint)

    def value(player, hand, node, opp_reach, best_response):
        if node.terminal:
            total = 0.0
            for other, weight in enumerate(opp_reach):
                if node.winner is None:
                    ours, theirs = ranks[player][hand], ranks[1 - player][other]
                    share = 1.0 if ours > theirs else 0.0 if ours < theirs else 0.5
                else:
                    share = float(node.winner == player)
                total += weight * (
                    share * (game["pot"] + sum(node.contributions))
                    - node.contributions[player]
                )
            return total
        legal = actions(game, node)
        if node.player == player:
            values = [
                value(
                    player, hand, advance(game, node, a["id"]), opp_reach, best_response
                )
                for a in legal
            ]
            if best_response:
                return max(values)
            row = profile[str(player)][node.key]["strategy"][hand]
            return sum(p * v for p, v in zip(row, values))
        rows = profile[str(1 - player)][node.key]["strategy"]
        return sum(
            value(
                player,
                hand,
                advance(game, node, a["id"]),
                [w * rows[j][k] for j, w in enumerate(opp_reach)],
                best_response,
            )
            for k, a in enumerate(legal)
        )

    ev, br = [], []
    for p in (0, 1):
        totals = [0.0, 0.0]
        for i in range(len(ranges[p])):
            reach = [0.0] * len(ranges[1 - p])
            for h0, h1, weight in joint:
                own, other = (h0, h1) if p == 0 else (h1, h0)
                if own == i:
                    reach[other] = weight / mass
            for k, best in enumerate((False, True)):
                totals[k] += value(p, i, Node(), reach, best)
        ev.append(totals[0])
        br.append(totals[1])
    return {
        "ev_chips": ev,
        "best_response_chips": br,
        "exploitability_chips": (sum(br) - game["pot"]) / 2,
        "compatible_deals": len(joint),
        "joint_weight": mass,
    }
