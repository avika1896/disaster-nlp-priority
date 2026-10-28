"""
Step 6 - Actionability.

CrisisMMD has an `informative` label and a `humanitarian_category` label but NO
"actionable" label. Actionability is therefore DERIVED from the humanitarian
category with the mapping below. It is a definition we chose, not a dataset
label, and the report must describe it that way.

Mapping (edit the sets to change the definition):
  actionable      : injured_or_dead_people, affected_individuals,
                    infrastructure_and_utility_damage,
                    rescue_volunteering_or_donation_effort,
                    rare_humanitarian (= missing_or_found_people + vehicle_damage,
                    merged in Step 7a; the raw names are listed too)
  not actionable  : not_humanitarian, other_relevant_information

Two uses:
  actionable_from_label(label)   - turns a DATASET category into the derived label
  actionability(category_probs)  - turns MODEL probabilities into a decision
"""

ACTIONABLE_CATEGORIES = {
    "injured_or_dead_people",
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "rescue_volunteering_or_donation_effort",
    "rare_humanitarian",
    "missing_or_found_people",
    "vehicle_damage",
}
NOT_ACTIONABLE_CATEGORIES = {"not_humanitarian", "other_relevant_information"}
KNOWN_CATEGORIES = ACTIONABLE_CATEGORIES | NOT_ACTIONABLE_CATEGORIES


def actionable_from_label(category):
    """True / False for a known category name, None if the name is unknown."""
    if category in ACTIONABLE_CATEGORIES:
        return True
    if category in NOT_ACTIONABLE_CATEGORIES:
        return False
    return None


def actionability(category_probs: dict, threshold: float = 0.5) -> dict:
    """Probability mass on actionable categories (probabilities are renormalised)."""
    unknown = set(category_probs) - KNOWN_CATEGORIES
    if unknown:
        raise ValueError(f"Unknown categories (add them to actionability.py): {sorted(unknown)}")
    total = sum(category_probs.values())
    if total <= 1e-9:
        return {"is_actionable": False, "actionable_probability": 0.0}
    p = sum(v for k, v in category_probs.items() if k in ACTIONABLE_CATEGORIES) / total
    return {"is_actionable": p >= threshold, "actionable_probability": round(p, 4)}
