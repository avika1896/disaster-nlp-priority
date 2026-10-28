# How the manually reviewed priority labels are created (Step 14)

CrisisMMD has no priority labels, and the priority score is our own hand-weighted
formula. To test whether the formula agrees with human judgement, a person labels
a small sample of posts, and `src/evaluation/evaluate_priority.py` compares the
labels with the system.

## Steps
1. Run `python src/evaluation/prepare_priority_review.py`. It writes
   `data/processed/priority_review_sheet.csv` (100 test posts, text only) and a
   separate key file. The sheet shows NO system score, so you cannot be biased by it.
2. Open the sheet and fill `human_priority` with exactly `HIGH`, `MEDIUM` or `LOW`.
3. Optional but recommended: a second person fills `human_priority_2`. The script
   then reports agreement between the two reviewers (Cohen's kappa).
4. Run `python src/evaluation/evaluate_priority.py`.

## Label definitions (judge from the text only)
- **HIGH**: the post reports or clearly implies people in immediate danger, people
  trapped, serious injuries or deaths, or an urgent request for rescue or medical
  help. A responder should act on it first.
- **MEDIUM**: the post reports damage, displacement, a shortage of essentials,
  blocked roads, or a specific need for help, but no immediate danger to life.
- **LOW**: general information, news, opinions, prayers, thanks, advertising,
  jokes, or anything unrelated to a needed response.

## Rules
- Do not look at model outputs or the scoring formula while labelling.
- Label what the text says, not what you guess about the situation.
- If two levels both fit, pick the one you could defend in the viva.
- Write down the date and who labelled; state in the report that labels are human
  judgements on a small sample, not ground truth.
