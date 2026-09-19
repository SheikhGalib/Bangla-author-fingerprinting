# CLAUDE.md

**Read [`AGENTS.md`](AGENTS.md) first — it is the canonical context file for
this repository.** It covers the layout, the invariants that must not be
broken, the commands, and a list of gotchas that each cost real time to find.

This file holds only the Claude Code-specific notes that do not belong in a
tool-neutral file.

---

## Quick orientation

Bangla authorship attribution for KUET CSE 4122. Predicts which of **three**
famous authors — Tagore, Nazrul, Humayun Ahmed — wrote a passage, and shows the
evidence. Compares a generative paradigm (per-author Kneser–Ney LMs) against a
discriminative one (Naive Bayes, softmax regression, SVM, BiLSTM, a
from-scratch Transformer, and a fine-tuned BanglaBERT).

The pipeline has been run end-to-end. `reports/report.pdf`, `corpus/` and the
tables under `artifacts/tables/` are finished deliverables.

`docs/LAB_COVERAGE.md` maps each lab concept to the file that implements it —
read it before claiming a technique is or isn't covered.

## The interpreter

```bash
.venv/Scripts/python.exe        # Windows
```

Always explicit; never rely on a bare `python`. Prefix with
`PYTHONIOENCODING=utf-8` for anything that prints Bengali, or the Windows
console raises `UnicodeEncodeError` on cp1252.

## The corpus is third-party text

`corpus/` holds nine books downloaded for coursework. It is gitignored, and so
is `.env` (which carries the Kaggle token). Don't commit either, don't paste
book text into commit messages or documentation, and don't make the Kaggle
dataset public — `kaggle_run.py` creates it private deliberately.

## Long-running commands

| Command | Time | Notes |
|---|---|---|
| `scripts/01_build_corpus.py` | ~5 min cold, seconds cached | ~150 pages at a 1.5 s delay |
| `scripts/03_train_local.py` | ~2 min | all CPU models + ablation |
| `scripts/kaggle_run.py push` | ~20 min remote | poll with `status`, then `pull` |
| `scripts/05_report_artifacts.py` | ~3 min | refits models for the leakage experiment |

Use `run_in_background: true` for the Kaggle wait; an `until` loop on
`kaggle_run.py status` grepping for `COMPLETE|ERROR|CANCEL` gives one
notification rather than repeated polling.

## Writing files

Bash heredocs into `python -` work fine for short scripts. Writing **large
Python or LaTeX files via heredoc has failed in this repo**. Use Write/Edit for
anything substantial, and verify the file landed.

In-place string replacement has silently no-op'd here when the pattern didn't
match exactly — LaTeX table rows in particular. **Always `assert` the match**
before writing, or swap by line number.

## Verifying your work

- **Lint must pass:** `.venv/Scripts/python.exe -m ruff check src/ scripts/ app.py ui/`
- **LaTeX can fail silently.** After `latexmk`, check `report.log` for lines
  starting with `!`, confirm the PDF's mtime moved, and grep for
  `Undefined control sequence`. Grepping only for "Output written" will happily
  report success on a stale build.
- **Reading the PDF** with the Read tool (`pages: "1-4"`) is the reliable way
  to confirm a figure, table or the Bengali font rendered.
- **A missing macro is visible, not silent.** `make_report.py` prints which
  ones are missing and renders them as `??` in the PDF.
- **Reproducibility is checkable:** everything is seeded (`SEED = 20242025`).

## Before you change anything

The invariants in `AGENTS.md` are not stylistic preferences. The held-out
books never entering a fitting step, the anthology de-overlap, the fixed seed,
and the balanced-test/weighted-train asymmetry are each load-bearing for the
reported results. If a change touches one of them, say so explicitly rather
than making it quietly.
