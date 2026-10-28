"""Scoring routes: POST /score and POST /score/batch."""

from fastapi import APIRouter, HTTPException, Request

from backend.schemas import BatchIn, BatchOut, ScoreOut, TweetIn
from backend.scorer import score_tweets

router = APIRouter(tags=["scoring"])


def _predictor(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    return predictor


@router.post("/score", response_model=ScoreOut, summary="Score one tweet")
def score_one(tweet: TweetIn, request: Request):
    return score_tweets(_predictor(request), [(tweet.text, tweet.disaster_event)])[0]


@router.post("/score/batch", response_model=BatchOut,
             summary="Score up to 100 tweets; results come back ranked, highest priority first")
def score_batch(batch: BatchIn, request: Request):
    results = score_tweets(_predictor(request), [(t.text, t.disaster_event) for t in batch.tweets])
    ranked = sorted(enumerate(results), key=lambda pair: -pair[1]["priority_score"])
    return {"count": len(results),
            "results": [{**r, "input_index": i} for i, r in ranked]}