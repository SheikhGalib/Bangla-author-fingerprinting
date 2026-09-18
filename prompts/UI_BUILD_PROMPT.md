# Build a UI for the Bangla authorship attribution system

> **How to use this file:** paste the whole thing into your coding agent as the
> opening message, with the repository open. It is written to be self-contained
> — the agent should not need to read the notebooks or the report to build the
> UI correctly. Everything below marked **CONTRACT** is verified against the
> running code; treat it as fact rather than re-deriving it.

---

## 1. What you are building

A web UI for a **Bangla authorship attribution** system. A user pastes a Bangla
passage; the UI shows which of five 19th/20th-century authors most likely wrote
it, **and the evidence behind that prediction**.

The machine-learning work is finished and the models are trained. **Your job is
the interface only.** You are wrapping an existing Python API in a web front
end.

### The single most important design constraint

This system's whole point is that predictions are **auditable, not opaque**.
The original project proposal commits to returning "the predicted author
alongside the ranked stylistic features that most influenced the decision."

A UI that shows only the author's name is a failed UI, even if it looks good.
The evidence is the product. Design around it.

---

## 2. Hard rules

**Do:**
- Build only the UI layer. Everything ML lives in `src/banglastylo/`.
- Call the existing `Attributor` API (§4). It is the supported interface.
- Use the project's existing `.venv`.
- Show both models' answers and whether they agree (§3).
- Surface the caveats in §7 in the UI itself, not just in a README.

**Do not:**
- Retrain, re-fit, or re-run any notebook. The models are trained; loading
  takes ~2 seconds, training takes ~45 minutes.
- Modify anything in `src/banglastylo/`, `notebooks/`, or `scripts/`. If you
  genuinely need a change there, stop and say so rather than editing.
- Reimplement tokenisation, normalisation or feature extraction. Call the
  library.
- Commit anything into `artifacts/` or `data/`.
- Present the prediction with more confidence than §7 allows.

---

## 3. What the system actually does (so your UI doesn't misrepresent it)

Two independent models answer the same question, and the UI must show both.

**Generative** — one language model per author, trained only on that author's
text. It asks *"how surprised is Bankim's model by this passage?"* and picks
the least surprised. Reports **perplexity** per author; **lower is better**.
This is the more accurate model (83.4% vs 79.5% on held-out data).

**Discriminative** — one linear SVM trained to separate the five authors, over
stylometric features. Reports a **margin** over the runner-up and a per-feature
breakdown of exactly why.

They agree on **75.4%** of passages. **When they disagree, say so prominently
and treat the result as low confidence** — do not silently pick a winner. On
disagreements the generative model is right more often (84 cases to 56), but
neither is reliable there.

### The five authors

`predict()` returns one of these five keys. Display names, never raw keys:

| key | English | Bengali | Period |
|---|---|---|---|
| `vidyasagar` | Ishwar Chandra Vidyasagar | ঈশ্বরচন্দ্র বিদ্যাসাগর | 1820–1891 |
| `bankim` | Bankim Chandra Chattopadhyay | বঙ্কিমচন্দ্র চট্টোপাধ্যায় | 1838–1894 |
| `tagore` | Rabindranath Tagore | রবীন্দ্রনাথ ঠাকুর | 1861–1941 |
| `abanindranath` | Abanindranath Tagore | অবনীন্দ্রনাথ ঠাকুর | 1871–1951 |
| `sarat` | Sarat Chandra Chattopadhyay | শরৎচন্দ্র চট্টোপাধ্যায় | 1876–1938 |

Two are named Tagore and two Chattopadhyay — **never abbreviate to surname
alone.** `config.AUTHORS[key]["short"]` gives safe short labels
(`"R. Tagore"`, `"A. Tagore"`, `"Bankim"`, `"Sarat Ch."`, `"Vidyasagar"`).

---

## 4. CONTRACT — the Python API

```python
import sys
sys.path.insert(0, "src")
from banglastylo.interface import Attributor, DEMO_PASSAGE
from banglastylo import config

attributor = Attributor.load()          # ~2 s; do this ONCE at startup
result = attributor.predict(passage_text, top_k=8)
```

`Attributor.load()` reads `artifacts/models/svm_combined.pkl` and
`generative_char.pkl`. It raises `FileNotFoundError` if they are missing —
catch that at startup and show a clear message telling the user to run
notebook 04. **Load once and reuse**; never load per request.

`predict()` takes ~0.3 s per passage. It is CPU-bound and not thread-safe to
assume otherwise — serialise calls or use one worker.

### `predict()` return value

Verified by running it. Every value is a plain Python type and the whole dict
is `json.dumps`-safe.

```python
{
  "predicted":             "tagore",   # str — the headline answer
  "agree":                 True,       # bool — do the two models agree?
  "n_tokens":              26,         # int
  "n_sentences":           2,          # int
  "short_passage_warning": True,       # bool — True when n_tokens < 80

  "discriminative": {
    "predicted":  "tagore",
    "runner_up":  "sarat",
    "margin":     0.3859,              # float; bigger = more confident
    "scores":     {"tagore": 0.599, "sarat": 0.213, ...},   # all 5, higher=better

    # Top contributions. Each is the EXACT arithmetic w[i]*x[i], not an estimate.
    "evidence_for": [
      {"feature":      "ch:হা ",                      # internal id
       "gloss":        "character n-gram “হা␣”",      # DISPLAY THIS, not `feature`
       "contribution": 0.0940,                        # signed; + favours `predicted`
       "value":        1.4914},                       # the feature's value in this passage
      ...
    ],
    "evidence_against": [ ...same shape, negative contributions... ],

    # The same decomposition restricted to human-checkable features.
    # SEE §5 — this is the list you put in front of the user.
    "readable_for":     [ ...same shape... ],
    "readable_against": [ ...same shape... ]
  },

  "generative": {
    "predicted":  "tagore",
    "runner_up":  "sarat",
    "perplexity": {"tagore": 6.011, "sarat": 6.158, "bankim": 7.665,
                   "vidyasagar": 8.664, "abanindranath": 11.360},  # LOWER IS BETTER
    "evidence_for": [{"symbol": "া", "log_odds": 9.45}, ...]
  }
}
```

**`predicted` vs `discriminative.predicted`:** the top-level `predicted` is the
discriminative model's answer when available, falling back to the generative
one. When `agree` is `False` the two sub-blocks differ — show both.

### Other useful calls

```python
# Raw structural statistics for a passage, before any model weighs in.
# Returns {feature_name: float} — 29 entries.
profile = Attributor.profile(passage_text)

# Human-readable names for those statistics.
from banglastylo.interface import FEATURE_GLOSS
FEATURE_GLOSS["sadhu_chalit_ratio"]   # 'sadhu-to-chalit register ratio'

# Author metadata.
config.AUTHORS["tagore"]
# {'wikisource': ..., 'bn': 'রবীন্দ্রনাথ ঠাকুর', 'en': 'Rabindranath Tagore',
#  'short': 'R. Tagore', 'years': '1861-1941'}

# A built-in sample passage, for the "try an example" button.
from banglastylo.interface import DEMO_PASSAGE

# The exact plain-text report the CLI prints, if you want a "raw output" view.
from banglastylo.interface import render
render(result, attributor)   # -> str
```

There is a working reference implementation of the whole flow in **`app.py`**
(~75 lines). Read it before writing anything.

---

## 5. Designing the evidence display

This is where the UI earns its keep. Get this part right.

### Two evidence lists, and why

`evidence_for` is ranked by raw contribution, so **character n-grams take
almost every slot**. They carry the most weight but mean little to a reader —
"the trigram ায়ে" is not something anyone can check against the passage.

`readable_for` is the same decomposition filtered to three families a human
*can* verify: function words, structural statistics, and POS patterns.

**Lead with `readable_for`.** Put `evidence_for` behind a "show all features"
toggle for the technically curious. A user should be able to look at a piece of
evidence and then look at the passage and see it.

### Reading a `gloss`

| gloss looks like | means |
|---|---|
| `function word “তাহা”` | how often this function word appears |
| `POS pattern PRON–PART` | a part-of-speech sequence |
| `sadhu-to-chalit register ratio` | a structural statistic — **see below** |
| `character n-gram “হা␣”` | a character sequence; `␣` is a real space |

`␣` (U+2423) marks a word boundary inside character n-grams. Two n-grams that
differ only by a space are genuinely different features — do not strip it.

### The one feature worth explaining in the UI

Bangla prose of this period comes in two registers, and this is the most
interpretable signal in the whole system. Consider a tooltip or a small panel:

| | *sadhu* (সাধু, literary) | *chalit* (চলিত, colloquial) |
|---|---|---|
| "it happened" | হইল | হল |
| "having done" | করিয়া | করে |
| "his / her" | তাহার | তার |

Vidyasagar is the most sadhu of the five; Abanindranath Tagore the most chalit.
`Attributor.profile()` returns `sadhu_rate`, `chalit_rate` and
`sadhu_chalit_ratio` for any passage, so you can place the user's text on that
axis directly. It is a genuinely nice thing to show.

### Suggested layout

```
┌─────────────────────────────────────────────────────────────┐
│  [ Bangla text area — large, ~8 rows, RTL-safe LTR Bengali ] │
│  [ Try an example ]                   [ Attribute ] (primary)│
│  0 tokens                                                    │
├─────────────────────────────────────────────────────────────┤
│  RABINDRANATH TAGORE          রবীন্দ্রনাথ ঠাকুর  1861–1941   │
│  ✓ both models agree          ·  26 tokens, 2 sentences      │
│  ⚠ short passage — below ~80 tokens this is unreliable       │
├──────────────────────────────┬──────────────────────────────┤
│  WHY  (discriminative SVM)   │  HOW WELL EACH AUTHOR'S       │
│  runner-up: Sarat Ch.        │  MODEL FITS (perplexity,      │
│  margin 0.39                 │  lower = better)              │
│                              │                               │
│  Supporting:                 │  R. Tagore      6.01  ◀ best  │
│   +0.051 POS pattern         │  Sarat Ch.      6.16          │
│          PRON–PART           │  Bankim         7.67          │
│   +0.037 function word তাহা  │  Vidyasagar     8.66          │
│   +0.031 function word যাহা  │  A. Tagore     11.36          │
│                              │                               │
│  Against:                    │  (a horizontal bar chart is   │
│   −0.048 function word মাত্র  │   the obvious rendering)      │
│                              │                               │
│  [ show all features ]       │                               │
└──────────────────────────────┴──────────────────────────────┘
```

Visual guidance:
- Signed contributions read best as **diverging horizontal bars** — supporting
  one way, opposing the other, from a shared zero.
- Perplexity is **lower is better**, which inverts the usual intuition. Label
  it explicitly; don't rely on bar length alone to convey direction.
- `scores` (discriminative) is higher-is-better. The two panels run in opposite
  directions — make that unmissable.
- Bengali needs a font with Bengali coverage. Use **Noto Sans Bengali** from
  Google Fonts, or the vendored `reports/fonts/NotoSansBengali-Regular.ttf`.
  With a default system stack, Bengali conjuncts break.

### The disagreement state

When `agree` is `False`, this must be visually distinct — a warning banner, not
a footnote. Something like: *"The two models disagree (SVM says Tagore, the
language models say Sarat Chandra). Treat this attribution as low confidence."*
Show both answers with equal weight.

---

## 6. Suggested stack

Pick one; the first is recommended.

**FastAPI + a single static page.** `POST /api/predict {"text": "..."}` →
returns the `predict()` dict as JSON; serve `index.html` with vanilla JS or
Alpine. Minimal moving parts, and the JSON contract is already right.

**Streamlit** — `pip install streamlit`, ~80 lines, near-zero UI code.
Fastest to a demo; less control over layout. Fine if a demo is the goal.

**Gradio** — similar trade-off, with sharing built in.

Either way:
- Load the `Attributor` **once at startup**, not per request.
- Treat 2–3 s of first-load latency as expected and show a loading state.
- Validate: reject empty input; warn below 80 tokens rather than refusing.
- Handle the `FileNotFoundError` from `Attributor.load()` with a message that
  names the fix ("run notebook 04 to train the models").

Put your code in a **new top-level `ui/` directory**. Add UI-only dependencies
to `ui/requirements.txt`, not the project's root `requirements.txt`.

### Minimal FastAPI skeleton

```python
# ui/server.py
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banglastylo import config
from banglastylo.interface import Attributor, DEMO_PASSAGE

app = FastAPI()
attributor = Attributor.load()          # once, at import

class Passage(BaseModel):
    text: str

@app.post("/api/predict")
def predict(p: Passage):
    if not p.text.strip():
        raise HTTPException(400, "empty passage")
    result = attributor.predict(p.text, top_k=8)
    # hand the UI display names so it never has to know the key scheme
    result["authors"] = {
        k: config.AUTHORS[k] for k in result["generative"]["perplexity"]
    }
    return result

@app.get("/api/demo")
def demo():
    return {"text": DEMO_PASSAGE}

app.mount("/", StaticFiles(directory=ROOT / "ui" / "static", html=True))
```

Run with `../.venv/Scripts/python.exe -m uvicorn ui.server:app --reload` from
the repo root.

---

## 7. Honesty requirements — surface these in the UI

The system has real limits. A UI that hides them misrepresents the work. Each
of these should be visible somewhere in the interface, not buried in docs.

**It is a closed set of five authors.** Given a passage by anyone else — or
modern Bangla, or a news article — it will confidently name one of the five.
There is no "none of the above" and no abstain option. Say so near the result,
or in an always-visible "About" panel.

**Short passages are unreliable.** `short_passage_warning` is `True` below 80
tokens. Measured accuracy at 40 tokens: **55%** discriminative, **74%**
generative — against 83%/80% at 220 tokens. Show the warning when the flag is
set; consider a live token count as the user types.

**It is trained on literary prose from 1820–1951.** Modern Bangla, poetry,
social-media text and news are all out of domain. Results on them are not
meaningful.

**Accuracy is ~83%, not ~100%.** Roughly one passage in six is wrong. Never
render the prediction as certain. The margin, the perplexity spread and the
agreement flag are all available — use them to convey graded confidence.

A good pattern: a persistent, low-key "What this can and can't do" panel,
plus inline warnings when a specific flag fires.

---

## 8. Definition of done

- [ ] Paste Bangla text → get a prediction in under a second
- [ ] The five authors show with English **and** Bengali names, never bare keys
- [ ] `readable_for` evidence is the primary display; raw features behind a toggle
- [ ] Per-author perplexity is shown, labelled **lower is better**
- [ ] Agreement/disagreement is visually obvious
- [ ] The short-passage warning fires below 80 tokens
- [ ] Bengali renders correctly, conjuncts included (test with **ক্ষ ঞ্জ ন্ত্র হ্ম**)
- [ ] The closed-set and domain limits are visible in the UI
- [ ] Missing models produce a helpful error, not a stack trace
- [ ] Empty input is rejected cleanly
- [ ] Nothing under `src/`, `notebooks/`, `scripts/`, `data/` or `artifacts/`
      was modified
- [ ] `ui/README.md` says how to run it

### Test passages

The demo passage (`DEMO_PASSAGE`, sadhu register, expect Tagore):

> বৃষ্টি থামিলে সে জানালার পাশে দাঁড়াইয়া দূরের আকাশের দিকে তাকাইয়া রহিল। তাহার মনে
> হইল, এই পৃথিবীতে সুখ বলিয়া কিছু নাই; যাহা আছে তাহা কেবল প্রতীক্ষা মাত্র।

Sarat Chandra, opening of *দেনা-পাওনা* (expect `sarat`, both models agreeing):

> চণ্ডীগড়ে চণ্ডী বহু প্রাচীন দেবতা। কিংবদন্তী আছে রাজা বীরবাহুর কোন এক পূর্বপুরুষ কি একটা
> যুদ্ধ জয় করিয়া বারুই নদীর উপকূলে এই মন্দির স্থাপিত করেন, এবং পরবর্তীকালে কেবল ইহাকেই
> আশ্রয় করিয়া এই চণ্ডীগড় গ্রামখানি ধীরে ধীরে প্রতিষ্ঠিত হইয়া উঠিয়াছিল।

Also test: a two-word input (warning path), empty input (rejection path), and
a long English paragraph (out-of-domain — it will still confidently answer,
which is exactly the behaviour §7 asks you to communicate).

---

## 9. Background, if you want it

- `README.md` — what the project is, how to run the pipeline
- `reports/WALKTHROUGH.md` — **how to interpret the numbers**; §7 and §8 are
  the source of the honesty requirements above
- `reports/report.pdf` — the full report
- `app.py` — the reference CLI implementation of this exact flow
- `src/banglastylo/interface.py` — the API you are calling, with docstrings

You should not need to read the notebooks.
