# Authorship Attribution and Style Fingerprinting in Bangla Literature

KUET, Department of Computer Science and Engineering
CSE 4122 — Natural Language Processing Laboratory

| | |
|---|---|
| **Mohammad Abu Daud Sharif** | Roll 2107002 |
| **Sheikh Md. Galib Mahim** | Roll 2107020 |
| Section A, Lab Group A1 · Year 4, Term 1 · Session 2024-25 | |

Supervisors: Dr. K. M. Azharul Hasan (Professor), Md. Nazirulhasan Shawon
(Assistant Professor).

---

## Read these two first

| | |
|---|---|
| [`reports/report.pdf`](reports/report.pdf) | the project report — what was built, what was found |
| [`reports/WALKTHROUGH.md`](reports/WALKTHROUGH.md) | how to *interpret* it — what each number means, and which ones to distrust |

## What this is

Given an unseen Bangla passage, predict which of a closed set of authors wrote
it — from **how** it is written rather than **what** it says — and show the
evidence behind the prediction.

The project compares two attribution paradigms head to head on the same data:

* **Generative** — one Kneser–Ney n-gram language model per author, fit only on
  that author's text. A passage goes to whichever author's model assigns it the
  lowest perplexity.
* **Discriminative** — a linear SVM over stylometric and embedding features, and
  a fine-tuned BanglaBERT, both trained directly to tell the authors apart.

and asks a representation-level question alongside it: in a small-corpus
regime, do **pretrained** embeddings or embeddings **trained from scratch** on
the author corpus better isolate authorial style from topic?

## The one methodological commitment that matters

Splits are **work-disjoint**. Every book belongs to exactly one of train /
validation / test, so a classifier cannot win by recognising a novel's
character names — topic identification wearing style's clothes. Notebook 02
measures what that discipline costs by running the same model on a naive random
split; the gap is the size of the leak.

It is large: **0.993 on a random passage-level split against 0.787 work-disjoint**,
for the identical model on identical text. Every number in this repository is
from the work-disjoint split.

## Headline results

5 authors, 2,955 passages, 42 works, 682,607 tokens. Chance is 0.200.

| Model | Accuracy |
|---|---|
| **Generative char-5-gram LM** (one Kneser–Ney model per author) | **0.834** |
| SVM · 29 structural features only | 0.801 |
| SVM · all stylometric features | 0.795 |
| Fine-tuned BanglaBERT (2 epochs, CPU) | 0.775 |

The generative paradigm wins, and twenty-nine hand-designed scalars beat a
110M-parameter transformer at this compute budget. `WALKTHROUGH.md` §8 spells
out what those results do and do not license.

---

# Getting it running from a fresh clone

A clone has the **code, the notebooks, the figures and the report** — but not
the corpus or the trained models. Those are gitignored (they come to ~500 MB
and are fully reproducible). This section takes you from nothing to a working
system and a rebuilt report.

Budget **about 1.5 hours**, most of it unattended.

## Step 0 — prerequisites

| Need | Why | Check |
|---|---|---|
| **Python 3.10+** | the code uses `X \| None` annotations | `python --version` |
| **Internet** | the crawl hits Wikisource; BanglaBERT downloads from HuggingFace | |
| **~1.5 GB free disk** | ~500 MB project + ~500 MB HuggingFace cache in `~/.cache` | |
| **A LaTeX install with XeLaTeX** | only for step 5; everything else works without it | `xelatex --version` |

XeLaTeX specifically — not pdfLaTeX. The report uses `fontspec` to load a
Bengali font. On Windows install [MiKTeX](https://miktex.org/download); on
Linux `sudo apt install texlive-xetex texlive-latex-extra latexmk`.

This was built and run on **Python 3.14, Windows, CPU only**. No GPU is
required. Two consequences of 3.14 worth knowing: `gensim` and `bnlp-toolkit`
have no wheel for it and need a C toolchain, so **word2vec and the Bangla POS
tagger are implemented from scratch** in `src/banglastylo/`. Nothing in the
project depends on either package, so an older Python works too.

## Step 1 — environment (~3 min)

```bash
cd "NLP Project"
python -m venv .venv
```

Then, **Windows**:

```bash
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m ipykernel install --user --name banglastylo --display-name "Python (banglastylo)"
```

**macOS / Linux** — same thing, but the interpreter is `.venv/bin/python`.
Substitute that for `.venv/Scripts/python.exe` in every command below.

The kernel registration is not optional: `scripts/run_notebooks.py` executes
the notebooks against a kernel named `banglastylo`, and will fail without it.

Check it took:

```bash
.venv/Scripts/python.exe -c "import torch, sklearn, transformers; print('ok')"
```

## Step 2 — build the corpus (~30–45 min, unattended)

```bash
.venv/Scripts/python.exe scripts/01_build_corpus.py
```

This crawls Bengali Wikisource for 8 candidate authors and writes
`data/raw/corpus_raw.json` (~16 MB).

- **It is slow on purpose.** Wikimedia rate-limits anonymous API access and
  returns HTTP 429; the client waits 1.5 s between calls and backs off when
  throttled. `HTTP 429; sleeping 30s` in the output is normal, not an error.
- **Every response is cached** under `data/raw/cache/`. Interrupting with
  Ctrl-C is safe — re-running resumes and re-fetches nothing.
- Progress prints per book, e.g.
  `[sarat] দেবদাস … 109/110 proofread 146,814 chars`.

Done when you see a summary table of 8 authors and `wrote …/corpus_raw.json`.

## Step 3 — run the pipeline (~50 min, unattended)

```bash
.venv/Scripts/python.exe scripts/run_notebooks.py
```

Executes notebooks 01→06 in order, writing outputs back into the `.ipynb`
files, and stops at the first failure. Roughly:

| Notebook | Time | Produces |
|---|---|---|
| 01 corpus construction | 1 min | `data/processed/passages.jsonl` |
| 02 splits + POS tagger | 1 min | `data/splits/split.json`, leakage experiment |
| 03 embeddings | **~10 min** | trains word2vec, encodes all passages with BanglaBERT |
| 04 models + evaluation | **~45 min** | **fine-tunes BanglaBERT on CPU**, all result tables |
| 05 interpretability | 2 min | fingerprints, permutation importance |
| 06 interface | <1 min | interface demo |

Almost all of that 45 minutes is BanglaBERT fine-tuning on CPU (~0.42 s per
sample). It is **cached afterwards** — the checkpoint lands in
`artifacts/models/banglabert_finetuned/`, and re-running notebook 04 reloads it
and finishes in ~3 minutes.

To re-run just part of it:

```bash
.venv/Scripts/python.exe scripts/run_notebooks.py 04 05
```

Or open the notebooks in Jupyter and run them by hand — pick the
**"Python (banglastylo)"** kernel:

```bash
.venv/Scripts/python.exe -m jupyter lab
```

## Step 4 — check you reproduced the numbers

```bash
.venv/Scripts/python.exe -c "import pandas as pd; print(pd.read_csv('artifacts/tables/results_main.csv').sort_values('accuracy', ascending=False)[['model','accuracy']].head(4).to_string(index=False))"
```

You should get, to three decimals:

```
      Generative char-5-gram LM    0.834
          SVM · structural only    0.801
 SVM · stylometric + BanglaBERT    0.801
    SVM · funcword + structural    0.798
```

Everything is seeded (`SEED = 20242025` in `src/banglastylo/config.py`), so
these should match exactly. If they don't, the corpus differs — Wikisource is a
live wiki and pages get edited. `data/raw/provenance.json` is committed so you
can diff the per-book page counts against the run the report describes.

## Step 5 — rebuild the report (~1 min)

```bash
.venv/Scripts/python.exe scripts/make_report.py
cd reports && latexmk -xelatex report.tex
```

`make_report.py` regenerates `reports/numbers.tex`, which defines every figure
quoted in the report as a LaTeX macro. **No number in the report is typed by
hand** — change the pipeline, re-run this, and the prose updates. If an
artifact is missing the macro renders as a visible `??` and the script lists
what it could not find.

Output: `reports/report.pdf`, 17 pages.

## Step 6 — use it

```bash
.venv/Scripts/python.exe app.py --demo
.venv/Scripts/python.exe app.py --text "…বাংলা অনুচ্ছেদ…"
.venv/Scripts/python.exe app.py --file passage.txt
.venv/Scripts/python.exe app.py --file passage.txt --json
```

Needs `artifacts/models/svm_combined.pkl` and `generative_char.pkl`, both
written by notebook 04.

## If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `No such kernel named banglastylo` | Step 1's `ipykernel install` was skipped |
| `HTTP 429; sleeping 30s` repeatedly | Normal. Wikimedia throttling; it backs off and continues |
| Crawl looks stuck | Check `data/raw/cache/` is growing. Ctrl-C and re-run is safe |
| `no trained models in artifacts/models` | Run notebook 04 (step 3) |
| `UnicodeEncodeError: 'charmap'` | Windows console. Prefix with `PYTHONIOENCODING=utf-8`, or use the scripts, which set it themselves |
| `latexmk: command not found` | No LaTeX install — steps 1–4 and `app.py` still work without it |
| `Missing character` in the LaTeX log | The Bengali font did not load. It is vendored in `reports/fonts/`; make sure you ran `latexmk` **from inside `reports/`** |
| Notebook 04 takes forever | Expected on CPU, ~45 min. It is cached after the first run |
| Results differ in the 3rd decimal | Wikisource changed. Compare `data/raw/provenance.json` |

## Building a UI on top of this

See **`prompts/UI_BUILD_PROMPT.md`** — a self-contained brief (including the
exact `predict()` output schema) to hand to a coding agent. Don't retrain
anything; the UI is a thin layer over `banglastylo.interface.Attributor`.

---

## Layout

```
docs/            the original proposal, plus markitdown conversions in docs/md/
notebooks/       01-06, the runnable pipeline; run in order
src/banglastylo/ the library the notebooks call
  config.py        every path and hyper-parameter
  wikisource.py    cached, rate-limited Bengali Wikisource client
  corpus.py        crawl -> clean -> passage segmentation -> balance
  normalize.py     Bangla normalisation, sentence split, tokenisation
  postag.py        rule-based coarse Bangla POS tagger + UD validation
  stylometry.py    four feature families as sklearn transformers
  embeddings.py    skip-gram from scratch, frozen BanglaBERT, style-vs-topic
  models.py        Kneser-Ney LMs, generative attributor, SVM, BERT fine-tune
  splits.py        work-disjoint splitting and the leakage check
  evaluate.py      metrics, bootstrap CIs, confusion analysis, figures
  interpret.py     exact linear attribution, permutation importance
  interface.py     the auditable prediction front end
scripts/         corpus build, notebook generation and execution, report build
data/            raw crawl + cache, processed passages, split assignments
artifacts/       trained models, figures, result tables
reports/         report.tex, WALKTHROUGH.md, generated figures and tables
```

## Corpus

Public-domain prose from [Bengali Wikisource](https://bn.wikisource.org),
taken from the `পাতা:` (Page:) namespace and filtered to pages the Wikisource
community has marked **proofread** — level 1 pages are raw OCR and for Bangla
that OCR is frequently garbled into Devanagari look-alikes. All authors died
more than sixty years before this project; the texts are out of copyright.
`data/raw/provenance.json` records, per book, how many scanned pages existed
and how many survived the filter.

## Licence and attribution

Corpus text is from Bengali Wikisource (CC BY-SA 4.0). The POS tagger is
validated against
[`UD_Bengali-BRU`](https://github.com/UniversalDependencies/UD_Bengali-BRU)
(CC BY-SA 4.0). `csebuetnlp/banglabert` is used under its own licence.
