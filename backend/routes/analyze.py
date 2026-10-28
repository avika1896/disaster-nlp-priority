"""Analysis routes (Step 12): POST /analyze and POST /analyze/batch.

/analyze takes {"text": "..."} (disaster_event is optional) and returns the
full structured result from src/pipeline.py. /score and /score/batch still
exist and are unchanged.
"""

import sys
from pathlib import Path

from fastapi import APIRouter, Request

from backend.routes.score import _predictor
from backend.schemas import AnalyzeOut, BatchAnalyzeOut, BatchIn, TweetIn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import analyze_many  # noqa: E402

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeOut, summary="Analyze one post")
def analyze_one(tweet: TweetIn, request: Request):
    return analyze_many(_predictor(request), [(tweet.text, tweet.disaster_event)])[0]


@router.post("/analyze/batch", response_model=BatchAnalyzeOut,
             summary="Analyze up to 100 posts; results come back ranked, highest priority first")
def analyze_batch(batch: BatchIn, request: Request):
    results = analyze_many(_predictor(request), [(t.text, t.disaster_event) for t in batch.tweets])
    ranked = sorted(enumerate(results), key=lambda pair: -pair[1]["priority_score"])
    return {"count": len(results), "results": [{**r, "input_index": i} for i, r in ranked]}
