"""
Gazetteer for rule-based location extraction (Step 6).

Seeded from the places that appear in the CrisisMMD events present in this
project's data (see disaster_event values) plus places seen in sample tweets.
It is deliberately small and editable. Extend it after reading the coverage
report from run_extraction.py. Nothing here is learned from labels.
"""

# disaster_event value (from the processed CSVs) -> region(s) the event affected.
# Used only as a clearly-labelled fallback (`event_region`), never mixed into
# text-extracted locations.
EVENT_REGION = {
    "hurricane_harvey": ["Texas"],
    "hurricane_irma": ["Florida", "Caribbean"],
    "hurricane_maria": ["Puerto Rico"],
    "california_wildfires": ["California"],
    "mexico_earthquake": ["Mexico"],
    "srilanka_floods": ["Sri Lanka"],
    "iraq_iran_earthquake": ["Iraq", "Iran"],
}

PLACES = [
    # Harvey
    "Texas", "Houston", "Rockport", "Corpus Christi", "Galveston", "Beaumont",
    "Port Arthur", "Louisiana", "Dallas", "Austin", "San Antonio",
    # Irma
    "Florida", "Florida Keys", "Miami", "Tampa", "Naples", "Jacksonville",
    "Charleston", "Barbuda", "Antigua", "Anguilla", "Saint Martin", "St. Martin",
    "Virgin Islands", "Bahamas", "Cuba", "Dominican Republic", "Caribbean",
    "Georgia", "South Carolina",
    # Maria
    "Puerto Rico", "San Juan", "Vieques", "Culebra", "Dominica", "Guadeloupe",
    # California wildfires
    "California", "Napa", "Sonoma", "Santa Rosa", "Los Angeles", "Anaheim",
    # Mexico earthquake
    "Mexico", "Mexico City", "Puebla", "Morelos", "Oaxaca", "Chiapas",
    # Sri Lanka floods
    "Sri Lanka", "Colombo", "Galle", "Matara", "Ratnapura", "Kalutara",
    # Iraq-Iran earthquake
    "Iraq", "Iran", "Kermanshah", "Sarpol-e Zahab", "Halabja", "Sulaymaniyah",
    "Kurdistan",
    # Generic
    "United States", "U.S.", "USA",
]