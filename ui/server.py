"""FastAPI backend for the Bangla authorship-attribution demonstration.

The interface has one job: let someone who has not read the code watch the
system work and understand why the answer is trustworthy.  That shapes the API.

* ``/api/corpus`` lists every book with its role and one readable paragraph, so
  the claim "one book per author was held back" is something a visitor can
  check rather than something the page merely asserts.
* ``/api/compare`` runs one passage through three models at once.  A single
  verdict hides the interesting part; three side by side show where they agree
  and, more usefully, where they do not.
* ``/api/demo`` offers sample passages taken live from the held-out books.

Nothing here hard-codes an author, a book or a passage.  Everything is read
from ``corpus/`` at request time, so the page cannot drift from the data the
models were actually trained and tested on.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Locate repository root and import library
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banglastylo import config
from banglastylo.compare import ComparisonPanel, catalogue
from banglastylo.interface import FEATURE_GLOSS, Attributor
from banglastylo.normalize import sentence_split, word_tokenize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("banglastylo.ui")

app = FastAPI(
    title="Bangla Authorship Attribution",
    description="Three models, three authors, one held-out book each.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Both are loaded once at startup.  Each failure is isolated: the panel can
# serve two models if the third is missing, and the evidence view can be absent
# without taking the comparison down with it.
panel: ComparisonPanel | None = None
attributor: Attributor | None = None
load_error: str | None = None

try:
    panel = ComparisonPanel()
    logger.info("comparison panel: %s", panel.status())
except Exception as exc:
    load_error = f"comparison panel unavailable: {exc}"
    logger.exception("comparison panel failed to load")

try:
    attributor = Attributor.load()
except Exception as exc:  # noqa: BLE001 - the evidence view is optional
    logger.warning("evidence view unavailable: %s", exc)


class PassageRequest(BaseModel):
    text: str = Field(..., description="Bangla passage to attribute")
    top_k: int = Field(8, ge=1, le=50)


@app.get("/api/health")
def health():
    return {
        "status": "ready" if panel is not None else "degraded",
        "models": panel.status() if panel else {},
        "evidence_available": attributor is not None,
        "error": load_error,
        "authors": config.AUTHORS,
    }


@app.get("/api/corpus")
def corpus_listing():
    """Every book, its role, and a paragraph of it."""
    books = catalogue()
    if not books:
        raise HTTPException(
            status_code=503,
            detail=f"no corpus at {config.CORPUS_TXT}; "
                   "run scripts/01_build_corpus.py",
        )
    return {
        "authors": config.AUTHORS,
        "books": books,
        "n_train": sum(1 for b in books if b["role"] == "train"),
        "n_unseen": sum(1 for b in books if b["role"] == "unseen"),
    }


@app.get("/api/demo")
def get_demos():
    """One sample passage per author, taken from that author's held-out book."""
    demos = {}
    for b in catalogue():
        if b["role"] != "unseen" or not b["glimpse"]:
            continue
        demos[b["author"]] = {
            "title": f"{b['author_en']} — from “{b['book']}” (held out)",
            "author_hint": f"{b['author_en']} ({b['author_bn']})",
            "book": b["book"],
            "unseen": True,
            "text": b["glimpse"],
        }
    if not demos:
        raise HTTPException(status_code=503, detail="no held-out corpus found")
    return demos


@app.post("/api/compare")
def compare(p: PassageRequest):
    """Run one passage through all three demonstration models."""
    text = p.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Passage text cannot be empty.")
    if panel is None:
        raise HTTPException(status_code=503, detail=load_error or "models not loaded")

    verdicts = [v.__dict__ for v in panel.predict(text)]
    live = [v for v in verdicts if v["available"] and v["predicted"]]
    winners = {v["predicted"] for v in live}
    tokens = len(word_tokenize(text))

    return {
        "verdicts": verdicts,
        "authors": config.AUTHORS,
        "consensus": live[0]["predicted"] if len(winners) == 1 and live else None,
        "unanimous": len(winners) == 1 and len(live) > 1,
        "n_models": len(live),
        "tokens": tokens,
        "sentences": len(sentence_split(text)),
        # Below this the models disagree often enough that a single label would
        # overstate the evidence; the page says so rather than hiding it.
        "short_warning": tokens < 40,
    }


@app.get("/api/analytics/books")
def analytics_books():
    """Per-book profile: size, style statistics, distinctive words, cast."""
    from dataclasses import asdict

    from banglastylo.analytics import book_profiles

    try:
        return {"authors": config.AUTHORS,
                "books": [asdict(b) for b in book_profiles()]}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/analytics/features")
def analytics_features():
    """What the stylometric SVM looks at, family by family."""
    from banglastylo.analytics import feature_families

    return {"families": feature_families()}


@app.post("/api/analytics/represent")
def analytics_represent(p: PassageRequest):
    """Turn one passage into BoW, n-grams and the 29 structural measurements."""
    from banglastylo.analytics import representation_demo

    text = p.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Passage text cannot be empty.")
    return representation_demo(text)


@app.get("/api/results")
def results_table():
    """The scored models, straight out of the results table on disk."""
    import csv

    path = config.TABLES / "results_all.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"{path} not found")
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["accuracy"] = float(r["accuracy"])
        r["macro_f1"] = float(r["macro_f1"])
    return {"rows": rows, "chance": round(1 / len(config.AUTHORS), 4)}


@app.post("/api/predict")
def predict(p: PassageRequest):
    """The auditable view: predicted author plus the features that decided it."""
    text = p.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Passage text cannot be empty.")
    if attributor is None:
        raise HTTPException(status_code=503, detail="evidence models not loaded")

    try:
        result = attributor.predict(text, top_k=p.top_k)
        profile = Attributor.profile(text)
        result["profile"] = [
            {"key": k, "label": FEATURE_GLOSS.get(k, k.replace("_", " ")),
             "value": round(float(v), 4)}
            for k, v in profile.items()
        ]
        result["authors"] = config.AUTHORS
        return result
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


static_dir = ROOT / "ui" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ui.server:app", host="127.0.0.1", port=8000, reload=True)
