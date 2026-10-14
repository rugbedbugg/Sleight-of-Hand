"""Resolution-V fractional-factorial screening of fixed policy parameters.

A designed experiment, not reinforcement learning or adaptive search: one
predeclared batch of immutable full parameter vectors (the 2^(5-1) half
fraction plus the canonical centre) is compared against canonical, then at
most one held-out confirmation. Nothing here learns, trains or promotes.
"""

from __future__ import annotations

from decimal import Decimal
from itertools import combinations, product

from .campaign import (
    COMPLETIONS,
    ROLES,
    SCREENING,
    Campaign,
    ReviewRequired,
    common,
    fixed_horizon,
    in_domain,
    number,
)
from .model import canonical, exact, name, sha256
from .spec import PARAM_NAMES, SeedPolicy

VERSION = 1
ALGORITHM = {"kind": SCREENING, "version": VERSION}
OBJECTIVE = "paired-bb-per-100-bonferroni-vs-canonical"
SEED_DERIVATION = "sha256-campaign-phase-v1"
PHASES = ("screening", "confirmation")
FIELDS = {
    "schema_version",
    "id",
    "governance",
    "purpose",
    "campaign_seed",
    "template",
    "policy_revision",
    "fixed_policy",
    "depends_on",
    "algorithm",
    "factors",
    "design",
    "screening_stopping",
    "confirmation_stopping",
    "candidate_family",
    "diagnostic_family",
    "objective",
    "max_generated_vectors",
    "resources",
    "created_at",
    "seed_derivation",
}
DESIGN = {
    "type": "regular-two-level-fractional-factorial",
    "enumeration": "lexicographic-base-factors-low-first",
}
CANDIDATE_FAMILY = {
    "comparator": "baseline",
    "comparisons": "each-treatment-vs-baseline",
    "correction": "bonferroni",
    "family_alpha": 0.05,
    "selection": "largest-positive-corrected-improvement-then-arm-name",
}
DIAGNOSTIC_FAMILY = {
    "estimator": "two-level-effect-2-over-runs-per-matched-block",
    "correction": "bonferroni",
    "family_alpha": 0.05,
    "binding": False,
}
ORDER = {1: "main-effect", 2: "two-factor", 3: "three-factor", 4: "four-factor"}
NO_QUALIFIER = "no multivariate treatment cleared the corrected canonical comparison"
SELECTED = (
    "screening selected one corrected-detectable treatment for held-out confirmation"
)


def _canonical_defaults():
    from sleight_of_hand.policy.heuristic import DEFAULT_PARAMS

    return dict(DEFAULT_PARAMS.__dict__)


def load(path, value) -> Campaign:
    exact(value, FIELDS)
    template = common(path, value, VERSION)
    exact(value["algorithm"], {"kind", "version"})
    if type(value["algorithm"]["version"]) is not int or value["algorithm"] != (
        ALGORITHM
    ):
        raise ValueError("unsupported screening algorithm")
    if value["objective"] != OBJECTIVE or value["seed_derivation"] != SEED_DERIVATION:
        raise ValueError("unsupported objective or seed derivation")
    _dependencies(value["depends_on"])
    symbols = _factors(value["factors"])
    runs = _design(value["design"], symbols)
    for key in ("screening_stopping", "confirmation_stopping"):
        fixed_horizon(value[key], template)
    family = exact(value["candidate_family"], set(CANDIDATE_FAMILY) | {"size"})
    if {k: v for k, v in family.items() if k != "size"} != CANDIDATE_FAMILY or (
        type(family["size"]) is not int or family["size"] != runs
    ):
        raise ValueError("candidate family must be every treatment vs canonical")
    effects = words(symbols)
    diagnostics = exact(
        value["diagnostic_family"], set(DIAGNOSTIC_FAMILY) | {"effects", "size"}
    )
    if (
        {k: diagnostics[k] for k in DIAGNOSTIC_FAMILY} != DIAGNOSTIC_FAMILY
        or diagnostics["effects"] != effects
        or type(diagnostics["size"]) is not int
        or diagnostics["size"] != len(effects)
    ):
        raise ValueError("diagnostic family must be all main and two-factor effects")
    # Screening arms plus a reserved two-arm confirmation; never more.
    if (
        type(value["max_generated_vectors"]) is not int
        or value["max_generated_vectors"] != runs + 1 + 2
    ):
        raise ValueError("invalid screening budget")
    result = Campaign(canonical(value).decode("ascii"), template)
    defaults = _canonical_defaults()
    if {k: number(v) for k, v in result.initial.items()} != {
        k: number(v) for k, v in defaults.items()
    }:
        raise ReviewRequired("screening centre differs from canonical PolicyParams")
    for row in design(result):
        validate_vector(result, row["vector"])
    seed_blocks(result)  # reject collisions before any execution
    return result


def _dependencies(deps):
    if type(deps) is not list:
        raise ValueError("invalid campaign dependencies")
    seen = set()
    for d in deps:
        exact(d, {"id", "status", "classification"})
        name(d["id"], "dependency ID")
        if d["id"] in seen:
            raise ValueError("duplicate campaign dependency")
        seen.add(d["id"])
        if d["status"] != "COMPLETE" or not (
            d["classification"] is None or d["classification"] in COMPLETIONS
        ):
            raise ValueError("impossible dependency requirement")


def _factors(factors):
    if type(factors) is not list or len(factors) != len(PARAM_NAMES):
        raise ValueError("screening needs one factor per PolicyParam")
    symbols, parameters = [], set()
    for f in factors:
        exact(f, {"symbol", "parameter", "low", "center", "high"})
        symbol, key = f["symbol"], f["parameter"]
        if type(symbol) is not str or len(symbol) != 1 or not "A" <= symbol <= "Z":
            raise ValueError("factor symbols are single capital letters")
        if symbol in symbols:
            raise ValueError("duplicate factor symbol")
        if key not in PARAM_NAMES or key in parameters:
            raise ValueError("unsupported or duplicate policy parameter")
        low, center, high = (number(f[k]) for k in ("low", "center", "high"))
        if not low < center < high or center * 2 != low + high:
            raise ValueError("factor levels must be low < centre < high, centred")
        if not in_domain(key, low) or not in_domain(key, high):
            raise ValueError("factor level outside the policy domain")
        symbols.append(symbol)
        parameters.add(key)
    if symbols != sorted(symbols):
        raise ValueError("factors must be declared in symbol order")
    return symbols


def _design(value, symbols):
    exact(
        value,
        set(DESIGN)
        | {"runs", "base_factors", "generators", "defining_relation", "resolution"},
    )
    if {k: value[k] for k in DESIGN} != DESIGN:
        raise ValueError("unsupported factorial design")
    base = value["base_factors"]
    if type(base) is not list or base != symbols[: len(base)] or len(base) < 2:
        raise ValueError("base factors must be the leading declared symbols")
    generators = value["generators"]
    if type(generators) is not list or len(generators) != 1:
        raise ValueError("v1 supports exactly one generator (a half fraction)")
    generator = exact(generators[0], {"factor", "product"})
    if base + [generator["factor"]] != symbols or generator["product"] != base:
        raise ValueError("generator must reference exactly the declared factors")
    word = "".join(symbols)
    if value["defining_relation"] != [word]:
        raise ValueError("defining relation differs from the generator")
    runs = 2 ** len(base)
    if (
        type(value["runs"]) is not int
        or value["runs"] != runs
        or type(value["resolution"]) is not int
        or value["resolution"] != len(word)
        or len(word) < 5
    ):
        raise ValueError("design must be a Resolution V half fraction")
    return runs


def factor_symbols(campaign):
    return [f["symbol"] for f in campaign.config["factors"]]


def words(symbols):
    """Predeclared effect family: every main effect then every two-factor pair."""
    return list(symbols) + ["".join(p) for p in combinations(symbols, 2)]


def alias(campaign, word):
    """Under I = (all factors), a word is aliased with its complement."""
    relation = campaign.config["design"]["defining_relation"][0]
    return "".join(s for s in relation if s not in word)


def arm(row, vector):
    return f"treatment-{row:02d}-{sha256(canonical(vector))[:12]}"


def design(campaign):
    """Deterministic rows: base factors in lexicographic -1/+1 order; E = A*B*C*D."""
    c = campaign.config
    levels = {f["symbol"]: f for f in c["factors"]}
    base = c["design"]["base_factors"]
    generated = c["design"]["generators"][0]
    rows = []
    for index, signs in enumerate(product((-1, 1), repeat=len(base))):
        sign = dict(zip(base, signs))
        value = 1
        for symbol in generated["product"]:
            value *= sign[symbol]
        sign[generated["factor"]] = value
        vector = {
            levels[s]["parameter"]: levels[s]["low" if v < 0 else "high"]
            for s, v in sign.items()
        }
        rows.append(
            {"row": index, "signs": sign, "vector": vector, "arm": arm(index, vector)}
        )
    return rows


def column(rows, word):
    out = []
    for row in rows:
        value = 1
        for symbol in word:
            value *= row["signs"][symbol]
        out.append(value)
    return out


def contrasts(campaign):
    """Analysis contrasts: effect = (2/runs) * sum(sign_i * y_i) per block."""
    rows = design(campaign)
    names = {f["symbol"]: f["parameter"] for f in campaign.config["factors"]}
    scale = Decimal(2) / len(rows)
    result = []
    for word in words(factor_symbols(campaign)):
        aliased = alias(campaign, word)
        note = (
            f"{ORDER[len(word)]} screening contrast "
            + " x ".join(names[s] for s in word)
            + f", aliased with {ORDER[len(aliased)]} term {aliased}"
        )
        result.append(
            {
                "label": word,
                "weights": {
                    r["arm"]: float(scale * s) for r, s in zip(rows, column(rows, word))
                },
                "note": note,
            }
        )
    return result


def validate_vector(campaign, vector):
    exact(vector, set(PARAM_NAMES))
    allowed = [campaign.initial] + [r["vector"] for r in design(campaign)]
    if canonical(vector) not in {canonical(v) for v in allowed}:
        raise ValueError("vector is not canonical or a declared design row")
    for key, value in vector.items():
        if not in_domain(key, number(value)):
            raise ValueError("candidate outside approved bounds")


def seed_blocks(campaign):
    """One block per phase from [id, seed, phase]; collisions fail closed."""
    c = campaign.config
    template = campaign.template.seed_policy
    streams = {
        template.derive(role, match)
        for match in range(campaign.template.stopping.max_matches)
        for role in ROLES
    }
    result = {}
    for phase in PHASES:
        material = canonical([c["id"], c["campaign_seed"], phase])
        seed = int(sha256(material)[:16], 16) & ((1 << 63) - 1)
        if seed == template.base_seed or seed in result.values():
            raise ValueError("seed block overlap")
        result[f"{phase}-0"] = seed
        stopping = c[f"{phase}_stopping"]
        for match in range(stopping["max_matches"]):
            for role in ROLES:
                stream = SeedPolicy(seed).derive(role, match)
                if stream in streams:
                    raise ValueError("screening/confirmation stream overlap")
                streams.add(stream)
    return result


def vectors(campaign, state):
    """(arm, vector) pairs for the next batch: screening once, then confirmation."""
    if state["stop_reason"] is not None:
        selected = next(
            (r["arm"] for r in design(campaign) if r["vector"] == state["incumbent"]),
            None,
        )
        if selected is None:
            raise ValueError("confirmation needs a selected design treatment")
        return [("baseline", campaign.initial), (selected, state["incumbent"])]
    if state["generation"]:
        raise ValueError("screening runs exactly once")
    return [("baseline", campaign.initial)] + [
        (r["arm"], r["vector"]) for r in design(campaign)
    ]


def stop_reason(campaign, state):
    if not state["generation"]:
        return None
    return NO_QUALIFIER if state["incumbent"] == campaign.initial else SELECTED


def selection(campaign, state, stratum, item):
    """Corrected baseline comparisons select; factorial contrasts only explain."""
    confirmation = state["pending"]["phase"] == "confirmation"
    treatments = sorted(s.arm for s in item.specs if s.arm != "baseline")
    expected = {("baseline", a) for a in treatments}
    comparisons = stratum["comparisons"]
    if (
        len(comparisons) != len(expected)
        or {tuple(c["arms"]) for c in comparisons} != expected
    ):
        raise ValueError("comparison family differs from the predeclared family")
    qualified = []
    for comparison in comparisons:
        correction = comparison.get("multiplicity", {})
        if (
            correction.get("method") != "bonferroni"
            or correction.get("family_size") != len(expected)
            or correction.get("family_alpha") != 0.05
        ):
            raise ValueError("unapproved statistical comparison rule")
        if (
            comparison["paired_difference"]["matches"]
            != item.specs[0].stopping.max_matches
        ):
            raise ValueError("incomplete paired evidence")
        recommendation = comparison["recommendation"]
        if recommendation["verdict"] not in {
            "DIFFERENCE_DETECTED",
            "NO_DIFFERENCE_DETECTED",
        }:
            raise ValueError("batch has no valid statistical verdict")
        treatment = comparison["arms"][1]
        if recommendation["verdict"] == "DIFFERENCE_DETECTED":
            higher = recommendation.get("higher_arm")
            if higher not in comparison["arms"]:
                raise ValueError("invalid higher arm")
            # arms are [baseline, treatment]; the paired mean is treatment - baseline
            improvement = comparison["paired_difference"]["mean_bb_per_100"]
            if (higher == treatment) != (improvement > 0):
                raise ValueError("verdict and improvement disagree")
            if higher == treatment:
                qualified.append((improvement, treatment))
    qualified.sort(key=lambda pair: (-pair[0], pair[1]))
    winner = qualified[0][1] if qualified else "baseline"
    vector = next(s.policy_config["params"] for s in item.specs if s.arm == winner)
    result = {
        "selected_arm": winner,
        "incumbent": vector,
        "comparisons": comparisons,
        "qualified": qualified,
        "family": {
            "comparator": "baseline",
            "size": len(expected),
            "method": "bonferroni",
            "family_alpha": 0.05,
        },
        "reason": (
            "largest positive paired bb/100 improvement over canonical among "
            "corrected detectable higher treatments; ties by arm name"
            if qualified
            else NO_QUALIFIER
        ),
    }
    if confirmation:
        if "contrasts" in stratum:
            raise ValueError("confirmation has no factorial diagnostics")
        verdict = comparisons[0]["recommendation"]
        result["classification"] = (
            "PROMISING"
            if qualified
            else "REJECTED"
            if verdict["verdict"] == "DIFFERENCE_DETECTED"
            else "INCONCLUSIVE"
        )
        # Confirmation classifies the selected treatment; it never reselects.
        result["incumbent"] = state["incumbent"]
        return result
    result["diagnostics"] = _diagnostics(campaign, stratum, item)
    return result


def _diagnostics(campaign, stratum, item):
    """Validate the report's predeclared contrasts; they never select anything."""
    declared = contrasts(campaign)
    found = stratum.get("contrasts")
    if (
        type(found) is not list
        or [{k: c.get(k) for k in ("label", "weights", "note")} for c in found]
        != declared
    ):
        raise ValueError("factorial diagnostics differ from the declared design")
    blocks = item.specs[0].stopping.max_matches
    for c in found:
        correction = c.get("multiplicity", {})
        if (
            c["blocks"] != blocks
            or c.get("binding") is not False
            or correction.get("method") != "bonferroni"
            or correction.get("family_size") != len(declared)
            or correction.get("family_alpha") != 0.05
        ):
            raise ValueError("incomplete or uncorrected factorial diagnostics")
    return [
        {
            k: c[k]
            for k in ("label", "note", "blocks", "mean", "se", "ci95", "multiplicity")
        }
        | {"binding": False}
        for c in found
    ]
