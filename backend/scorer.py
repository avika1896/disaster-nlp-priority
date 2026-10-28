"""
Step 8 — The scoring pipeline behind the API.

score_tweets() is pure orchestration and takes any "predictor" object with
    predict(clean_texts) -> (list_of_p_informative, list_of_category_prob_dicts)
so the API can be tested without PyTorch. TransformerPredictor is the real one.

Per tweet:
  raw text -> src.preprocessing.text_cleaner.clean_text (same defaults as the
  Step 2 dataset build, because both transformers were trained on that output)
  -> two transformers -> Step 6 extractor (on the RAW text) -> Step 7 priority score
"""

import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src" / "ranking", ROOT / "src" / "information_extraction"):
    sys.path.insert(0, str(p))

from src.preprocessing.text_cleaner import clean_text  # noqa: E402
from extractor import extract  # noqa: E402
from priority import priority_score  # noqa: E402

INFORMATIVE_DIR = ROOT / "models" / "transformer" / "best"
HUMANITARIAN_DIR = ROOT / "models" / "transformer" / "humanitarian" / "best"
MAX_LEN = 64   # same as training (Step 5)

ZERO_COMPONENTS = {"p_informative": 0.0, "category": 0.0, "casualty": 0.0, "need": 0.0}


def score_tweets(predictor, tweets: List[Tuple[str, Optional[str]]]) -> List[dict]:
    """tweets: list of (raw_text, disaster_event_or_None). Returns one dict per tweet."""
    cleaned = [clean_text(t) for t, _ in tweets]
    usable = [i for i, c in enumerate(cleaned) if c.strip()]

    p_inf_by_i, cat_by_i = {}, {}
    if usable:
        p_list, c_list = predictor.predict([cleaned[i] for i in usable])
        for i, p, c in zip(usable, p_list, c_list):
            p_inf_by_i[i], cat_by_i[i] = float(p), c

    results = []
    for i, (raw, event) in enumerate(tweets):
        ext = extract(raw, event)
        if i not in p_inf_by_i:    # nothing left after cleaning (e.g. only a URL)
            results.append({
                "text": raw, "clean_text": cleaned[i], "priority_score": 0.0,
                "priority_tier": "low", "components": dict(ZERO_COMPONENTS),
                "informative_probability": 0.0, "predicted_category": None,
                "category_probabilities": {}, "extraction": ext,
                "warning": "No usable text after cleaning; scored 0.",
            })
            continue
        probs = {k: round(float(v), 4) for k, v in cat_by_i[i].items()}
        res = priority_score(p_inf_by_i[i], probs, ext)
        results.append({
            "text": raw, "clean_text": cleaned[i],
            "priority_score": res["score"], "priority_tier": res["tier"],
            "components": res["components"],
            "informative_probability": round(p_inf_by_i[i], 4),
            "predicted_category": max(probs, key=probs.get),
            "category_probabilities": probs, "extraction": ext, "warning": None,
        })
    return results


class TransformerPredictor:
    """Loads both fine-tuned DistilBERT models once and serves predictions."""

    def __init__(self, batch_size: int = 64):
        import torch  # imported here so tests without PyTorch can still import this module
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        for d in (INFORMATIVE_DIR, HUMANITARIAN_DIR):
            if not d.exists():
                raise FileNotFoundError(f"Trained model not found: {d}. Run Steps 5 and 7a first.")
        self._torch = torch
        self.batch_size = batch_size
        self._lock = threading.Lock()   # one inference at a time; simple and safe
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        def load(path):
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
            names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
            return tok, model, names

        self.inf_tok, self.inf_model, inf_names = load(INFORMATIVE_DIR)
        self.cat_tok, self.cat_model, self.cat_names = load(HUMANITARIAN_DIR)
        self.inf_idx = inf_names.index("informative")

    def _proba(self, tok, model, texts):
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = tok(texts[i:i + self.batch_size], truncation=True, max_length=MAX_LEN,
                          padding=True, return_tensors="pt").to(self.device)
                out.append(torch.softmax(model(**enc).logits, dim=-1).cpu())
        return torch.cat(out).numpy()

    def predict(self, clean_texts: List[str]):
        with self._lock:
            p_inf = self._proba(self.inf_tok, self.inf_model, clean_texts)[:, self.inf_idx]
            p_cat = self._proba(self.cat_tok, self.cat_model, clean_texts)
        cats = [dict(zip(self.cat_names, row.tolist())) for row in p_cat]
        return p_inf.tolist(), cats