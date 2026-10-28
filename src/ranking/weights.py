"""
Hand-set weights for the priority score (Step 7b).

IMPORTANT FOR THE REPORT / VIVA:
The dataset has NO priority labels, so these numbers are expert-judgement
choices, not learned values. They are kept in this one file so they can be
inspected, argued about and tuned. Say this explicitly in the paper.

Score for one tweet (0-100):

    priority = 100 * p_informative *
               ( W_CATEGORY * category_score
               + W_CASUALTY * casualty_score
               + W_NEED     * need_score )

  p_informative  : Step 5 transformer probability that the tweet is informative.
                   It multiplies everything, so non-informative tweets sink.
  category_score : expected CATEGORY_WEIGHT under the Step 7a transformer's
                   probabilities, over the humanitarian classes only
                   (renormalised, so it does not double-count informativeness).
  casualty_score : log-scaled largest casualty count found by the extractor.
  need_score     : the strongest need tag found by the extractor.
"""

COMPONENT_WEIGHTS = {"category": 0.60, "casualty": 0.25, "need": 0.15}
assert abs(sum(COMPONENT_WEIGHTS.values()) - 1.0) < 1e-9

NOT_HUMANITARIAN = "not_humanitarian"

# Class names must match the Step 7a model's labels exactly.
CATEGORY_WEIGHT = {
    "injured_or_dead_people": 1.0,
    "affected_individuals": 0.9,
    "rare_humanitarian": 0.7,   # missing/found people + vehicle damage, merged in 7a
    "infrastructure_and_utility_damage": 0.7,
    "rescue_volunteering_or_donation_effort": 0.5,
    "other_relevant_information": 0.2,
    NOT_HUMANITARIAN: 0.0,
}

# Tags produced by src/information_extraction/extractor.py.
# These mean "the tweet mentions X", not "X is requested" (see extractor limits).
NEED_WEIGHT = {
    "explicit_request": 1.0,
    "medical": 1.0,
    "shelter": 0.8,
    "water": 0.8,
    "food": 0.7,
    "power_utilities": 0.6,
    "rescue": 0.6,
    "evacuation": 0.6,
    "donations": 0.3,
}

# A casualty count of this size or larger gives casualty_score = 1.0.
CASUALTY_SATURATION = 1000

# Tier cut-offs are arbitrary starting points. score_dataset.py reports how
# many tweets fall in each tier so they can be tuned.
HIGH_THRESHOLD = 40.0
MEDIUM_THRESHOLD = 30.0

# ---------------------------------------------------------------------------
# Step 9 - urgency (an ENGINEERED feature, not a dataset label).
# CrisisMMD has no urgency labels. urgency.py counts indicator words and turns
# them into a 0-100 score with the numbers below. They are hand-set starting
# points, kept here so they can be changed and compared in experiments.
# ---------------------------------------------------------------------------
URGENCY_BASE = {"high": 75.0, "medium": 45.0, "low": 10.0, "none": 0.0}
URGENCY_EXTRA_PER_INDICATOR = 5.0      # added for each extra high/medium indicator
URGENCY_HIGH_CUTOFF = 70.0             # urgency_level HIGH at or above this
URGENCY_MEDIUM_CUTOFF = 40.0           # urgency_level MEDIUM at or above this

# ---------------------------------------------------------------------------
# Step 10 - multi-factor priority score (priority_score.py).
# Hand-set, NOT scientifically established, NOT learned (no priority labels).
#   priority = 100 * p_informative * sum(weight * component)
# ---------------------------------------------------------------------------
PRIORITY_V2_WEIGHTS = {
    "urgency": 0.30,
    "human_impact": 0.30,
    "assistance": 0.15,
    "damage": 0.15,
    "location": 0.10,
}
assert abs(sum(PRIORITY_V2_WEIGHTS.values()) - 1.0) < 1e-9

# How much each humanitarian category counts as "human impact" (0-1).
HUMAN_IMPACT_CATEGORY_WEIGHT = {
    "injured_or_dead_people": 1.0,
    "affected_individuals": 0.9,
    "rare_humanitarian": 0.7,      # missing/found people + vehicle damage (merged in 7a)
}

# Tier cut-offs for the 0-100 score. CHOSEN, not learned: set from the VALIDATION
# split's score quantiles (about the top 10% = HIGH, the next 20% = MEDIUM) using
# src/ranking/score_dataset_v2.py. No priority labels exist, so these are a
# transparent convention. Check the TEST split afterwards; do not tune on it.
PRIORITY_V2_HIGH = 22.0
PRIORITY_V2_MEDIUM = 12.0
