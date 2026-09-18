"""FastAPI backend for Bangla Authorship Attribution Web UI.

Serves the prediction API and static frontend assets.
Adheres strictly to the contract in prompts/UI_BUILD_PROMPT.md.
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
from banglastylo.interface import Attributor, DEMO_PASSAGE, FEATURE_GLOSS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("banglastylo.ui")

app = FastAPI(
    title="Bangla Authorship Attribution",
    description="Auditable style fingerprinting and author prediction for Bangla literature.",
    version="1.0.0",
)

# Enable CORS for local testing flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Attributor once at startup
attributor: Attributor | None = None
model_load_error: str | None = None

try:
    logger.info("Loading Attributor models from %s...", config.MODELS)
    attributor = Attributor.load()
    logger.info("Attributor models loaded successfully.")
except FileNotFoundError as err:
    model_load_error = (
        f"Trained models not found in {config.MODELS}. "
        "Please run notebook 04 (or scripts/run_notebooks.py 04) to train and cache them."
    )
    logger.error(model_load_error)
except Exception as exc:
    model_load_error = f"Error loading models: {exc}"
    logger.error(model_load_error, exc_info=True)


class PassageRequest(BaseModel):
    text: str = Field(..., description="Bangla passage text to analyze")
    top_k: int = Field(8, ge=1, le=50, description="Number of top features to return")


@app.get("/api/health")
def health():
    """Health check reporting model loading status."""
    return {
        "status": "ready" if attributor is not None else "degraded",
        "models_loaded": attributor is not None,
        "error": model_load_error,
        "artifacts_dir": str(config.MODELS),
    }


@app.get("/api/demo")
def get_demos():
    """Pre-canned demo passages covering expected test cases."""
    return {
        "tagore": {
            "title": "Rabindranath Tagore — Sadhu Register",
            "author_hint": "Rabindranath Tagore (রবীন্দ্রনাথ ঠাকুর)",
            "text": DEMO_PASSAGE,
        },
        "sarat": {
            "title": "Sarat Chandra Chattopadhyay — Dena-Paona Opening",
            "author_hint": "Sarat Chandra Chattopadhyay (শরৎচন্দ্র চট্টোপাধ্যায়)",
            "text": (
                "চণ্ডীগড়ে চণ্ডী বহু প্রাচীন দেবতা। কিংবদন্তী আছে রাজা বীরবাহুর কোন এক "
                "পূর্বপুরুষ কি একটা যুদ্ধ জয় করিয়া বারুই নদীর উপকূলে এই মন্দির স্থাপিত করেন, "
                "এবং পরবর্তীকালে কেবল ইহাকেই আশ্রয় করিয়া এই চণ্ডীগড় গ্রামখানি ধীরে ধীরে "
                "প্রতিষ্ঠিত হইয়া উঠিয়াছিল।"
            ),
        },
        "short": {
            "title": "Short Snippet (<80 tokens warning trigger)",
            "author_hint": "Short text for testing calibration warning",
            "text": (
                "বৃষ্টি থামিলে সে জানালার পাশে দাঁড়াইয়া দূরের আকাশের দিকে চাহিয়া রহিল।"
            ),
        },
    }


@app.post("/api/predict")
def predict(p: PassageRequest):
    """Attribute passage to one of the 5 authors and provide auditable evidence."""
    clean_text = p.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Passage text cannot be empty.")

    if attributor is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "Models are not loaded on the server.",
        )

    try:
        # Run prediction
        result = attributor.predict(clean_text, top_k=p.top_k)

        # Profile passage structural features
        profile_raw = Attributor.profile(clean_text)
        profile_glossed = [
            {
                "key": k,
                "label": FEATURE_GLOSS.get(k, k.replace("_", " ")),
                "value": round(float(v), 4),
            }
            for k, v in profile_raw.items()
        ]

        # Extract Sadhu-Chalit register metrics
        sadhu_rate = float(profile_raw.get("sadhu_rate", 0.0))
        chalit_rate = float(profile_raw.get("chalit_rate", 0.0))
        sadhu_chalit_ratio = float(profile_raw.get("sadhu_chalit_ratio", 0.0))

        # Include author metadata dictionary so UI never hardcodes keys
        active_keys = set(config.AUTHORS.keys())
        authors_meta = {}
        for k in active_keys:
            if k in config.AUTHORS:
                meta = config.AUTHORS[k]
                authors_meta[k] = {
                    "key": k,
                    "en": meta.get("en", k),
                    "bn": meta.get("bn", k),
                    "short": meta.get("short", k),
                    "years": meta.get("years", ""),
                }

        result["authors"] = authors_meta
        result["register"] = {
            "sadhu_rate": round(sadhu_rate, 4),
            "chalit_rate": round(chalit_rate, 4),
            "sadhu_chalit_ratio": round(sadhu_chalit_ratio, 4),
        }
        result["profile"] = profile_glossed

        return result
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


# Mount static assets
static_dir = ROOT / "ui" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ui.server:app", host="127.0.0.1", port=8000, reload=True)

