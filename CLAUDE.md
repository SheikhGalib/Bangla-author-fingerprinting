# CLAUDE.md

**Read [`AGENTS.md`](AGENTS.md) first — it is the canonical context file for
this repository.** It covers the layout, the invariants that must not be
broken, the commands, and a list of gotchas that each cost real time to find.

This file holds only the Claude Code-specific notes that do not belong in a
tool-neutral file. Keeping the substance in one place stops the two from
drifting apart.

---

## Quick orientation

Bangla authorship attribution for KUET CSE 4122. Predicts which of five
public-domain authors wrote a passage, and shows the evidence. Compares a
generative paradigm (per-author Kneser–Ney language models) against a
discriminative one (SVM + fine-tuned BanglaBERT).

The pipeline has been run end-to-end. `reports/report.pdf` and the executed
notebooks are finished deliverables — treat them as such rather than as
work-in-progress.

## The interpreter

```bash
.venv/Scripts/python.exe        # Windows
```

Always explicit; never rely on a bare `python`. Prefix with
`PYTHONIOENCODING=utf-8` for anything that prints Bengali, or the Windows
console raises `UnicodeEncodeError` on cp1252.

## Long-running commands

Several stages take tens of minutes. Run them with `run_in_background: true`
and poll the log rather than blocking a foreground call:

| Command | Time | Notes |
|---|---|---|
| `scripts/01_build_corpus.py` | 30–45 min | rate-limited; HTTP 429 backoff messages are normal |
| `scripts/run_notebooks.py` | ~50 min cold | notebook 04 fine-tunes BanglaBERT on CPU |
| `scripts/run_notebooks.py 04` | ~3 min warm | reloads the cached checkpoint |

Before triggering a cold notebook 04, check whether
`artifacts/models/banglabert_finetuned/classes.txt` exists — if it does, the
notebook reloads rather than retraining, and you save 40 minutes.

## Writing files

Bash heredocs into `python -` work fine for short scripts. Writing **large
Python or LaTeX files via heredoc has failed in this repo** (quoting/parse
errors that silently skip the write). Use the Write and Edit tools for anything
substantial, and verify the file landed before moving on.

Similarly, in-place `sed`/`python -c` string replacement against source files
has silently no-op'd here when the pattern didn't match exactly. If you use
that approach, assert the match or check the result — don't assume.

## Verifying your work

- **Lint must pass:** `.venv/Scripts/python.exe -m ruff check src/ scripts/ app.py`
- **LaTeX can fail silently.** After `latexmk`, check `report.log` for lines
  starting with `!` and confirm the PDF's mtime actually moved. Grepping only
  for "Output written" will happily report success on a stale build.
- **Reading the PDF** with the Read tool (`pages: "1-4"`) is the reliable way
  to confirm a figure or table rendered as intended.
- **Reproducibility is checkable:** everything is seeded (`SEED = 20242025`).
  A full re-run produces byte-identical numbers. If it doesn't, something
  changed — investigate rather than shrug.

## Before you change anything

The invariants in `AGENTS.md` are not stylistic preferences. Work-disjoint
splitting, the fixed seed, train-only fitting, and the proofread-page filter
are each load-bearing for the reported results. If a change touches one of
them, say so explicitly rather than making it quietly.
