"""Tests for Step 6 extractor. Cases are real tweets from the processed data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "information_extraction"))
from extractor import extract, normalize_text  # noqa: E402


def deaths(text):
    return sorted(c["count"] for c in extract(text)["casualties"] if c["type"] == "deaths")


def test_count_then_word_variants():
    assert deaths("At least 217 dead after powerful earthquake hits central Mexico - https://t.co/E53Quc2lz2") == [217]
    assert deaths("RT @HuffPost: 3 deaths linked to animal-borne bacteria confirmed in puerto rico") == [3]
    assert deaths("#California #wildfires: 10 killed in 'unprecedented' wine country blaze.") == [10]
    assert deaths("Massive wildfires sweeping through parts of California have so far left 17 people dead.") == [17]
    assert deaths("California Fires Leave 32 Dead, a Vast Landscape Charred") == [32]
    assert deaths("245 Dead As School Collapse During Massive Earthquake In Mexico") == [245]


def test_death_toll_phrase_and_false_positives():
    # "2020 death toll - 1000X" must not yield 2020 or 1000
    t = "Current #Harvey death toll: 50. 2020 death toll - 1000X more due to deadly water &amp; toxins?"
    assert deaths(t) == [50]


def test_big_non_casualty_number_ignored():
    t = "Hurricane Irma track update: Dominican Republic &amp; Cuba next - 9 dead &amp; 49 MILLION at risk"
    assert deaths(t) == [9]


def test_number_words_and_thousands_separator():
    assert deaths("three killed in blast") == [3]
    assert deaths("1,200 dead after quake") == [1200]


def test_injured_and_missing_types():
    r = extract("40 injured and 5 missing after the quake")["casualties"]
    assert {"type": "injured", "count": 40} in r
    assert {"type": "missing", "count": 5} in r


def test_no_casualties_in_plain_tweet():
    r = extract("How Hurricane Irma Might Impact Your Travel Plans https://t.co/Vt0242S8I0")
    assert r["casualties"] == [] and r["max_deaths"] is None


def test_needs_tags():
    r = extract("#USCG brings packs of bottled water to #Vieques in response to the damage caused by #HurricaneMaria")
    assert "water" in r["needs"]
    r2 = extract("Thousands flee as wildfires ravage northern California; 10 killed")
    assert "evacuation" in r2["needs"]


def test_vague_need_phrase_not_tagged():
    r = extract("Fastest Hurricane to reach Category 5. We need to collectively wake tf up")
    assert r["needs"] == []


def test_locations_gazetteer():
    assert "Vieques" in extract("#USCG brings packs of bottled water to #Vieques")["locations"]
    assert "Houston" in extract("3 Reasons Houston Was A 'Sitting Duck' For Harvey Flooding")["locations"]
    assert "Rockport" in extract("RT @zachrogers03: Prayers needed in Rockport, we took quite a blow #Harvey")["locations"]


def test_longest_match_wins():
    locs = extract("Dominican Republic and Cuba next")["locations"]
    assert "Dominican Republic" in locs and "Dominica" not in locs


def test_candidate_locations_exclude_known_and_stopwords():
    r = extract("Relief arrives in Tacloban City after the storm; help in Hurricane Zone")
    assert "Tacloban City" in r["candidate_locations"]
    assert not any(c.startswith("Hurricane") for c in r["candidate_locations"])


def test_event_region_fallback_is_separate():
    r = extract("Thoughts with everyone affected", disaster_event="hurricane_maria")
    assert r["locations"] == [] and r["event_region"] == ["Puerto Rico"]


def test_normalize_handles_html_urls_mojibake_and_non_strings():
    assert normalize_text("Save 8\u00c2 Goats &amp; hens https://t.co/x") == "Save 8 Goats & hens"
    assert normalize_text(None) == ""