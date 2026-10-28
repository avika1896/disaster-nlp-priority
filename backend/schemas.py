"""Request and response schemas for the disaster-tweet priority API (Step 8)."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class TweetIn(BaseModel):
    text: str = Field(..., max_length=1000, description="Raw tweet text (URLs, hashtags etc. are fine).")
    disaster_event: Optional[str] = Field(
        None, description="Optional event key, e.g. 'hurricane_maria'. Only used for the event_region fallback.")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be blank")
        return v


class BatchIn(BaseModel):
    tweets: List[TweetIn] = Field(..., min_length=1, max_length=100)


class Casualty(BaseModel):
    type: str
    count: int


class Extraction(BaseModel):
    casualties: List[Casualty]
    max_deaths: Optional[int]
    needs: List[str]
    locations: List[str]
    candidate_locations: List[str]
    event_region: List[str]


class Components(BaseModel):
    p_informative: float
    category: float
    casualty: float
    need: float


class ScoreOut(BaseModel):
    text: str
    clean_text: str
    priority_score: float
    priority_tier: str
    components: Components
    informative_probability: float
    predicted_category: Optional[str]
    category_probabilities: Dict[str, float]
    extraction: Extraction
    warning: Optional[str] = None


class RankedScore(ScoreOut):
    input_index: int


class BatchOut(BaseModel):
    count: int
    results: List[RankedScore]  # sorted by priority_score, highest first

# ---- Step 12: the /analyze endpoints (full structured result from src/pipeline.py) ----
class AnalyzeOut(BaseModel):
    text: str
    is_informative: bool
    is_actionable: bool
    disaster_type: str
    location: str
    incident: str
    assistance_required: str
    affected_people: bool
    damage: str
    damage_severity: str
    urgency_score: float
    priority_score: float
    priority_level: str
    priority_reasons: List[str]
    # extra detail used by the dashboard and the report
    informative_probability: float
    predicted_category: Optional[str]
    actionable_probability: float
    urgency_reason: List[str]
    location_confidence: float
    location_source: str
    assistance_all: List[str]
    components: Dict[str, float]
    extraction: Extraction
    warning: Optional[str] = None


class RankedAnalyze(AnalyzeOut):
    input_index: int


class BatchAnalyzeOut(BaseModel):
    count: int
    results: List[RankedAnalyze]  # sorted by priority_score, highest first
