# AGENTS.md — working in this repository

Context for AI coding agents. Canonical file; `CLAUDE.md` points here.

---

## What this is

A Bangla authorship-attribution system for KUET CSE 4122 (NLP Laboratory).
Given a Bangla passage it predicts which of **three** authors wrote it —
Rabindranath Tagore, Kazi Nazrul Islam, Humayun Ahmed — and shows the evidence.

It compares a **generative** paradigm (one Kneser–Ney language model per
author, attribute to lowest perplexity) against a **discriminative** one
(Naive Bayes, softmax regression, a linear SVM over stylometric features, a
BiLSTM, a from-scratch Transformer encoder, and a fine-tuned BanglaBERT).

**Team:** Abu Daud Sharif (2107002), Sheikh Md. Galib Mahim (2107020).

---

## The corpus, and why it is shaped this way

Nine books from `ebanglalibrary.com`, three per author:

| Author | Training books | Held out |
|---|---|---|
| Rabindranath Tagore | গল্পগুচ্ছ, সাধনা | রহস্য সমগ্র |
| Kazi Nazrul Islam | কুহেলিকা, রিক্তের বেদন | ব্যথার দান |
| Humayun Ahmed | হিমু, বিপদ | হিমুর হাতে কয়েকটি নীলপদ্ম |

**The held-out book *is* the test set.** One book per author was designated
`unseen` before anything was downloaded, and no fitting step ever touches it.
Work-disjointness is therefore a property of how the corpus was collected, not
a claim about the splitting code. This is the single most important thing to
preserve.

The corpus is also written as plain UTF-8 `.txt` under `corpus/train/` and
`corpus/unseen/`, one file per book, so it can be read without running
anything and a passage can be copied out of a held-out book by hand.

`corpus/` and `data/` are gitignored. The books are third-party text collected
for coursework; don't commit them, and don't upload them anywhere public.

---

## Environment

```bash
.venv/Scripts/python.exe      # Windows — this is the interpreter to use
```

Python 3.14, CPU only locally. Always use the venv interpreter explicitly.

**Console encoding.** Windows defaults to cp1252 and raises
`UnicodeEncodeError` the moment Bengali is printed. Prefix ad-hoc commands:

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "..."
```

Scripts under `scripts/` set this themselves.

---

## Layout

```
src/banglastylo/
  config.py            paths, hyper-parameters, AUTHORS. Start here.
  ebangla.py           the book roster + cached scraper for ebanglalibrary
  overlap.py           train/unseen reprint detection  ← see invariant 2
  corpus.py            cleaning, passage segmentation, balancing
  normalize.py         Bangla normalisation, sentence split, tokenisation
  postag.py            rule-based coarse Bangla POS tagger + UD validation
  stylometry.py        4 feature families as sklearn transformers
  classifiers.py       from-scratch Naive Bayes, softmax regression, edit distance
  neural.py            from-scratch BiLSTM + Transformer encoder
  embeddings.py        skip-gram from scratch, frozen BanglaBERT
  models.py            Kneser-Ney LMs, generative attributor, SVM, BERT fine-tune
  splits.py            split_by_role (the one in use) + leakage check
  evaluate.py          metrics, bootstrap CIs, confusion analysis, figures
  interpret.py         exact linear attribution, permutation importance
  interface.py         the auditable prediction front end
  compare.py           the 3 demo models side by side + corpus catalogue
  analytics.py         per-book stats, distinctive words, recurring names

scripts/01_build_corpus.py     download → de-overlap → write corpus/*.txt
scripts/02_make_splits.py      segment into passages, fix train/val/test
scripts/03_train_local.py      every CPU model + ablations
scripts/04_collect_results.py  merge local + GPU results
scripts/kaggle_run.py          push / status / pull the GPU job
scripts/make_kaggle_notebook.py  generates the presentation notebook
kaggle/bangla_authorship_complete.ipynb   GENERATED - the notebook to show
kaggle/train_gpu.py            the remote job (BanglaBERT, BiLSTM, Transformer)

corpus/train|unseen/  readable .txt, one per book  (gitignored)
docs/LAB_COVERAGE.md  which lab concept lives where
ui/                   FastAPI + static frontend
  static/index.html     demo: corpus, pipeline, 3-model comparison
  static/books.html     per-book analytics, generated from corpus/
  static/methods.html   code, explanation and run output per step
  static/book_notes.json  editable form/genre notes, read at page load
```

---

## Invariants — do not break these

**1. The `unseen` books never enter any fitting step.** Not the vocabulary, not
the TF-IDF `fit`, not the word2vec, not the BERT tokenizer training. `role` on
`Passage` marks them; `splits.split_by_role` routes them to test and nothing
else. `splits.leakage_report` re-checks it and is run inside
`02_make_splits.py` and `03_train_local.py` — keep both calls.

**2. Overlap between training and held-out books is removed before training.**
Bengali publishers reissue stories in later compilations. Tagore's *রহস্য
সমগ্র* (held out) reprints six stories that also appear in *গল্পগুচ্ছ*
(training). `overlap.py` detects this from the text and drops the offending
chapters **from the training side**; the held-out book is never edited. Losing
this step silently invalidates every Tagore number.

**3. `SEED = 20242025`, everywhere.** All results reproducible; a re-run
produces identical numbers.

**4. Training is unbalanced and class-weighted; the test set is balanced.**
Down-sampling Tagore to Humayun's size would discard most of the training
material. The test set *is* balanced (179/author) so accuracy is directly
readable against a 1/3 chance rate.

**5. No `gensim`, no `bnlp-toolkit`.** Neither has a Python 3.14 wheel and both
need an unavailable C toolchain. Word2Vec, the POS tagger, Naive Bayes and
softmax regression are from scratch for that reason — it is deliberate,
documented, and is also what the labs asked for.

**6. Report numbers are generated, never typed.** `scripts/make_report.py`
writes `reports/numbers.tex` from `artifacts/tables/`. A missing macro renders
as a visible `??` by design.

---

## Commands

```bash
.venv/Scripts/python.exe scripts/01_build_corpus.py    # ~5 min cold, 0 cached
.venv/Scripts/python.exe scripts/02_make_splits.py     # seconds
.venv/Scripts/python.exe scripts/03_train_local.py     # ~2 min
.venv/Scripts/python.exe scripts/kaggle_run.py push    # GPU job
.venv/Scripts/python.exe scripts/kaggle_run.py status
.venv/Scripts/python.exe scripts/kaggle_run.py pull
.venv/Scripts/python.exe scripts/04_collect_results.py
.venv/Scripts/python.exe scripts/05_report_artifacts.py
.venv/Scripts/python.exe scripts/make_report.py

# The single presentation notebook (generated, then run on Kaggle GPU)
.venv/Scripts/python.exe scripts/make_kaggle_notebook.py
.venv/Scripts/python.exe scripts/kaggle_run.py push-notebook
.venv/Scripts/python.exe scripts/kaggle_run.py status-notebook
.venv/Scripts/python.exe scripts/kaggle_run.py pull-notebook   # -> notebooks/

# The interface: demo + /books.html + /methods.html
.venv/Scripts/python.exe -m uvicorn ui.server:app --port 8000

.venv/Scripts/python.exe -m ruff check src/ scripts/ ui/ app.py   # must pass clean
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe app.py --demo
```

---

## Gotchas found the hard way

Each of these cost real time.

**ebanglalibrary slugs are not consistently normalised.** One book's URL spells
য় **decomposed** (`য` + nukta, `%af%bc`) where every other spells it
precomposed (`%9f`). They are visually identical and the site 404s on the wrong
one. Always harvest a book URL from a page, never retype it.

**Chapter `<p>` tags are not always direct children of `.entry-content`.** A
minority of pages nest them one level deeper. Taking only direct children
returns *nothing* for those, silently — it cost two chapters of ব্যথার দান.
`_content_block` falls back to descendants.

**Shingle stride must be 1 in `overlap.py`.** At stride 5 two printings of the
same story fall out of phase — a single extra token shifts every later window —
and identical stories scored 0.10–0.32 containment, under the threshold. At
stride 1 the same pairs score 0.67–0.99 and everything else scores exactly
0.000.

**Kaggle does not mount a dataset at its slug.** This one mounted at
`/kaggle/input/datasets/...`, not `/kaggle/input/bangla-authorship`. Never
hard-code the input path in a kernel; `train_gpu.find_input()` searches
recursively and prints the actual mounts on failure.

**A kernel pushed immediately after a dataset version sees no data.** Wait for
the dataset to report `ready`.

**A stale checkpoint fails silently and convincingly.** `artifacts/models/` held
a BanglaBERT from the earlier five-author study; the UI loaded it and returned
fluent, high-confidence predictions naming authors the corpus no longer has.
`compare.ComparisonPanel._roster_matches` rejects any checkpoint whose class
list is not exactly `config.AUTHORS`. Keep that check.

**Notebook cells are authored inside raw triple-quoted strings.** An escaped
double quote survives into the generated code verbatim and breaks the cell, so
docstrings inside notebook code use triple *single* quotes.
`make_kaggle_notebook.py` is the source of truth; never hand-edit the `.ipynb`.

**`latexmk` can fail silently.** Check `report.log` for lines starting with `!`
and confirm the PDF's mtime moved. Grepping for "Output written" happily
reports success on a stale build.

**LaTeX macro names cannot contain digits.** `\bestF1` parses as `\bestF` plus
a stray `1`. Hence `bestFone`.

**`char_wb` n-grams carry word-boundary spaces.** Two n-grams differing only by
a leading/trailing space are different features. The interface renders the
space as `␣` (U+2423) — don't strip it.

**sklearn returns `numpy.str_`.** A `str` subclass, so it mostly works but
surprises `type(x) is str`. `interpret.py` coerces on the way out.

---

## Code conventions

- `from __future__ import annotations` at the top of every module.
- Module docstrings explain **why**, not what — especially non-obvious choices.
- Comments earn their place by explaining a decision or a trap.
- Type hints on public functions. `X | None`, `list[str]`, `dict[str, int]`.
- Feature transformers are sklearn-compatible so ablations are block swaps.
- Feature names are namespaced by family: `fw:`, `st:`, `ch:`, `pos:`. The
  prefix is load-bearing — `permutation_importance_grouped` splits on it.
- Ruff must pass clean.

---

## Things that look wrong but are not

- **Accuracy is ~0.97, far above the old five-author study's 0.79.** Three
  authors instead of five, and they span 1861–2012 with very different
  registers. Part of what is being detected is era and orthography, not
  authorial habit alone. Say so; don't quietly present it as pure style.
- **The TF-IDF word-unigram baseline scores 0.968, nearly the best.** Vocabulary
  is doing a lot of work. That is a finding to report, not a bug to hide.
- **POS tagger at 87.2%** is scored on `UD_Bengali-BRU`: 56 sentences. Small on
  purpose — it is the only Bengali UD treebank.
- **The generative model beats the fine-tuned transformer.** Consistent with
  the earlier study, and the headline finding.

---

## What not to do

- Don't retrain to test a UI or docs change — load the pickles.
- Don't add a dependency without checking it has a Python 3.14 wheel.
- Don't edit files under `artifacts/`, `data/` or `corpus/` by hand.
- Don't hard-code result numbers into prose, LaTeX or Markdown.
- Don't commit `artifacts/models/`, `corpus/` or `data/`.
- Don't overstate the results. Closed set of three authors, literary prose.
