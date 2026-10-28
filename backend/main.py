"""
Step 8 — FastAPI app.

Run from the project root:
    python -m uvicorn backend.main:app --port 8000
Then open http://127.0.0.1:8000/docs for the interactive API page.

create_app(predictor) lets tests inject a stand-in predictor. With no predictor
(the normal case) the two fine-tuned models are loaded once at startup.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.routes import analyze, meta, score  # noqa: E402


def create_app(predictor=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.predictor is None:
            from backend.scorer import TransformerPredictor
            print("Loading models ...")
            app.state.predictor = TransformerPredictor()
            print("Models loaded.")
        yield

    app = FastAPI(
        title="Disaster Tweet Priority API",
        description=("Scores disaster tweets for priority using two fine-tuned DistilBERT models, "
                     "rule-based information extraction and a transparent hand-weighted formula."),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.predictor = predictor
    # Local development only: lets a dashboard served from another port call the API.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(meta.router)
    app.include_router(score.router)
    app.include_router(analyze.router)
    return app


app = create_app()