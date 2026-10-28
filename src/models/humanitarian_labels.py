"""
Label handling for the humanitarian_category task (Step 7a).

In train.csv, vehicle_damage has 38 rows and missing_or_found_people has 24.
That is too few for a transformer to learn them separately, so they are merged
into one class. This is a documented, deliberate choice: the report must say
that these two categories are NOT distinguished by the model.
"""

RARE_CATEGORIES = {"vehicle_damage", "missing_or_found_people"}
MERGED_NAME = "rare_humanitarian"


def merge_label(label):
    return MERGED_NAME if label in RARE_CATEGORIES else label