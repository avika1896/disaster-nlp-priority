# Disaster Tweet Priority

Reads disaster tweets and ranks them by how urgently they need a response.

For each tweet the system:
1. Predicts if it is informative and which humanitarian category it belongs to.
2. Extracts casualties, needs and locations.
3. Combines these into a priority score and a tier (high, medium or low).

A FastAPI backend serves the results and a single-page dashboard shows the ranked queue.

## How it works

| Part | What it does |
|---|---|
| Preprocessing | Cleans tweet text (`src/preprocessing/text_cleaner.py`). |
| Baseline model | TF-IDF features with scikit-learn linear models (Logistic Regression and Linear SVM). Logistic Regression was chosen. |
| Transformer model | DistilBERT (`distilbert-base-uncased`) fine-tuned for sequence classification with Hugging Face Transformers. |
| Extraction | Rule-based extraction of casualties, needs and locations (`src/information_extraction/extractor.py`, small hand-made place list in `gazetteer.py`). |
| Location | `location.py` picks one place or returns `Unknown`. IDRISI was not used. |
| Disaster type | `disaster_type.py`: transparent keyword rules (hurricane, earthquake, flood, wildfire, tornado, tsunami, landslide, volcano), `Unknown` if no keyword. |
| Actionability | `actionability.py`: derived from the humanitarian category with a documented mapping. |
| Urgency | `src/ranking/urgency.py`: engineered word-indicator score (0-100) with the reasons. Not a dataset label. |
| Priority scoring | `src/ranking/priority_score.py`: 0-100 score from urgency, human impact, assistance, damage and location confidence, multiplied by the informative probability. Weights are hand-set in `weights.py`. Returns HIGH / MEDIUM / LOW with reasons. |
| Pipeline | `src/pipeline.py` combines everything into one structured result per post. |
| API | FastAPI app in `backend/` (`POST /analyze` is the main endpoint). |
| Dashboard | Single file, `frontend/index.html`. |

## Data

16,058 unique tweets, each with an informative label and a humanitarian category.

| Split | Tweets |
|---|---|
| Train | 10,312 |
| Validation | 2,210 |
| Test | 2,210 |
| Event hold-out test | 1,326 |

The event hold-out test uses two events that never appear in training: `iraq_iran_earthquake` and `srilanka_floods`. No tweet appears in more than one split.

Dataset: CrisisMMD (raw files in `data/raw/crisismmd_datasplit_all`).

## Results

All numbers come from the report files in `models/` and `data/processed/`.

### Task 1: informative or not informative (2 classes)

| Test set | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| Test (2,210) | Baseline (TF-IDF + Logistic Regression) | 0.774 | 0.729 |
| Test (2,210) | Transformer | 0.808 | 0.764 |
| Event hold-out (1,326) | Baseline | 0.762 | 0.761 |
| Event hold-out (1,326) | Transformer | 0.885 | 0.884 |

Source: `models/transformer/step5_comparison_report.json`.

### Task 2: humanitarian category (7 classes)

Validation set (2,210 tweets):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Baseline | 0.632 | 0.534 |
| Transformer | 0.679 | 0.625 |

The weakest classes are `affected_individuals` (F1 0.446) and `rare_humanitarian` (F1 0.444, only 8 validation tweets).
Source: `models/transformer/humanitarian/step7a_comparison_report.json`.
Test set (2,210 tweets):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Baseline | 0.622 | 0.550 |
| Transformer | 0.679 | 0.615 |

### Actionability (derived label)

CrisisMMD has no actionable label. Actionable is derived from the humanitarian category (mapping in `src/information_extraction/actionability.py`: injured or dead, affected, infrastructure damage, rescue or donation, and the rare classes are actionable; not humanitarian and other relevant information are not). The reference is that mapping applied to the dataset category, so this measures how the 7-class model's mistakes cross the actionable line, not agreement with human judgement.

Test set (2,210 tweets): accuracy 0.815, macro-F1 0.804. For the actionable class: precision 0.693, recall 0.835, F1 0.758.
Source: `data/processed/step14_actionability_report.json` (run `python src/evaluation/evaluate_actionability.py`).

### Disaster type (keyword rules)

Run on all 16,058 tweets without the event name. The reference is the type derived from the event name (dataset metadata, not per-tweet annotation).

| Measure | Value |
|---|---|
| Tweets where a keyword was found (coverage) | 46.0% |
| Accuracy on those tweets | 90.0% |
| Accuracy over all tweets, `Unknown` counted as wrong | 41.4% |

Most tweets never name the disaster, so the detector says `Unknown` rather than guess. Hurricane Harvey tweets are the weakest (62.6% on covered tweets) because many mention flooding or tornado warnings. Source: `data/processed/step8_disaster_type_report.json` (run `python src/information_extraction/evaluate_disaster_type.py`).

### Priority score

There are no priority labels, so accuracy cannot be reported. What was checked:
- Tier mix on the held-out test split, with cut-offs chosen from validation quantiles: about 10% HIGH, 16% MEDIUM, 74% LOW (validation: 10% / 18% / 72%).
- Mean score by dataset category is in the expected order: injured or dead people highest (30.5), affected individuals (21.9), then lower for infrastructure damage, rescue or donation, other information, and not humanitarian lowest (3.4).
- These numbers use the saved top category as probability 1.0 (`src/ranking/score_dataset_v2.py`); the live API uses the full probabilities, so scores differ slightly.

Not measured yet (needs hand labels, see "Evaluate" below): agreement with human priority labels, and precision, recall and F1 of location, casualty and need extraction.

Extraction coverage (how often each rule fires, not accuracy) is in `data/processed/step6_extraction_report.json`.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Start the API (leave this terminal open):

```bash
python -m uvicorn backend.main:app --port 8000
```

Start the dashboard in a second terminal:

```bash
python -m http.server 5500 --directory frontend
```

Then open http://127.0.0.1:5500. The API docs are at http://127.0.0.1:8000/docs.

## Test

```bash
python -m pytest -q
```

## Evaluate

Measured automatically:
```bash
python src/evaluation/evaluate_actionability.py
python src/information_extraction/evaluate_disaster_type.py
python src/ranking/score_dataset_v2.py validation
python src/ranking/score_dataset_v2.py test
```

Needs a small hand-labelled sample first (nothing is reported until real labels exist):
```bash
# Priority: label 100 posts (guide: docs/priority_labelling_guide.md), then
python src/evaluation/prepare_priority_review.py
python src/evaluation/evaluate_priority.py

# Location, casualties, needs: fill the gold_* columns in
# data/processed/step6_extraction_sample_for_labelling.csv (format in the script header), then
python src/evaluation/evaluate_extraction.py
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/analyze` | Full analysis of one post. Body: `{"text": "..."}` (`disaster_event` is optional). |
| POST | `/analyze/batch` | Up to 100 posts, ranked highest priority first. |
| POST | `/score` | Score one tweet. Body: `{"text": "...", "disaster_event": "optional"}` |
| POST | `/score/batch` | Score up to 100 tweets. Results come back ranked, highest priority first. Body: `{"tweets": [...]}` |
| GET | `/health` | Shows whether the models are loaded. |
| GET | `/config` | Shows the scoring weights and tier thresholds. |

`/analyze` returns `is_informative`, `is_actionable`, `disaster_type`, `location`, `incident`, `assistance_required`, `affected_people`, `damage`, `damage_severity`, `urgency_score`, `priority_score`, `priority_level` (HIGH / MEDIUM / LOW) and `priority_reasons`, plus extra detail. Anything that cannot be found is `Unknown`. The older `/score` endpoints return `priority_score`, `priority_tier`, `components`, `predicted_category`, `category_probabilities` and an `extraction` block (`casualties`, `max_deaths`, `needs`, `locations`, `candidate_locations`, `event_region`).

## Limitations

- The priority weights are chosen by hand. They are not learned from data, because the dataset has no priority labels. The ranking is a transparent rule, not a trained model.
- Category prediction is much weaker than informative prediction (macro-F1 about 0.62 against 0.76), and small classes such as `rare_humanitarian` are unreliable.
- The main test set shares five events with the training set (california_wildfires, hurricane_harvey, hurricane_irma, hurricane_maria, mexico_earthquake), so its scores are likely optimistic. The event hold-out test is the fairer check on new events.
- Extraction is rule-based, so it can miss places or needs that are not in its word lists.
- Urgency is an engineered score from word lists, not a dataset label, and it matches words, not meaning (for example "not urgent" still matches "urgent").
- Priority tier cut-offs were chosen from validation quantiles, not learned. Agreement with human priority judgements has not been measured yet.
- Disaster type is keyword-based: it finds a type in only 46% of tweets and returns `Unknown` otherwise. The only flood event (`srilanka_floods`) is held out of training, so a trained flood classifier was not possible.
- Location uses a small hand-made place list. The IDRISI dataset suggested in the project brief was not used, so location accuracy is unmeasured until a labelled sample exists.
- Actionability is derived from humanitarian categories, not annotated by humans.
