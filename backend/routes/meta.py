"""Meta routes: GET /health and GET /config (exposes the hand-set weights)."""

import sys
from pathlib import Path

from fastapi import APIRouter, Request

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ranking"))
import weights  # noqa: E402

router = APIRouter(tags=["meta"])


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "models_loaded": getattr(request.app.state, "predictor", None) is not None}


@router.get("/config", summary="The hand-set scoring weights and tier thresholds")
def config():
    return {
        "note": "These weights are expert-judgement choices, not learned from data (no priority labels exist).",
        "component_weights": weights.COMPONENT_WEIGHTS,
        "category_weights": weights.CATEGORY_WEIGHT,
        "need_weights": weights.NEED_WEIGHT,
        "casualty_saturation": weights.CASUALTY_SATURATION,
        "tier_thresholds": {"high": weights.HIGH_THRESHOLD, "medium": weights.MEDIUM_THRESHOLD},
        # Used by POST /analyze (the multi-factor score in src/ranking/priority_score.py).
        "analysis_score": {
            "note": ("Hand-set, not learned. Urgency is an engineered feature, not a dataset label. "
                     "Tier cut-offs were chosen from validation score quantiles."),
            "priority_weights": weights.PRIORITY_V2_WEIGHTS,
            "human_impact_category_weights": weights.HUMAN_IMPACT_CATEGORY_WEIGHT,
            "urgency_base": weights.URGENCY_BASE,
            "urgency_cutoffs": {"high": weights.URGENCY_HIGH_CUTOFF, "medium": weights.URGENCY_MEDIUM_CUTOFF},
            "tier_thresholds": {"high": weights.PRIORITY_V2_HIGH, "medium": weights.PRIORITY_V2_MEDIUM},
        },
    }