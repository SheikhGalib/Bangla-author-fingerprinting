"""Generate ``reports/numbers.tex`` from the run's artifacts.

Every number in the report body comes through a macro defined here, so the
prose can never quietly go stale relative to the last run: change the pipeline,
re-run this, and the report updates.  If an artifact is missing the macro is
defined as a visible ``??`` rather than silently omitted, so a hole in the
report is obvious at a glance rather than being read as a real figure.

LaTeX control sequences are letters only -- a macro named ``bestF1`` parses as
``bestF`` followed by a stray ``1`` and fails with a baffling error.  Hence
``bestFone``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from banglastylo import config

T = config.TABLES
macros: dict[str, str] = {}


def put(name: str, value) -> None:
    macros[name] = str(value)


def pct(x: float, places: int = 1) -> str:
    return f"{100 * float(x):.{places}f}\\%"


def tex_escape(s: str) -> str:
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def read_csv(name: str) -> pd.DataFrame | None:
    p = T / name
    return pd.read_csv(p) if p.exists() else None


def read_json(name: str):
    p = T / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------------------------------------------------------------------
# Configuration the prose quotes
# ---------------------------------------------------------------------------
put("CharOrder", config.CHAR_LM_ORDER)
put("WordOrder", config.WORD_LM_ORDER)
put("PassageTokens", config.PASSAGE_TOKENS)
put("MinPassageTokens", config.MIN_PASSAGE_TOKENS)
put("WtvDim", config.W2V_DIM)
put("TopFunctionWords", config.TOP_FUNCTION_WORDS)
put("PretrainedModel", tex_escape(config.PRETRAINED_MODEL))
put("Seed", config.SEED)
put("nAuthors", len(config.AUTHORS))
put("chanceAcc", f"{1 / len(config.AUTHORS):.3f}")
put("chancePct", pct(1 / len(config.AUTHORS)))

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
stats = read_csv("corpus_stats.csv")
if stats is not None:
    put("nPassages", f"{int(stats['passages'].sum()):,}")
    put("nWorks", int(stats["works"].sum()))
    put("nChapters", int(stats["chapters"].sum()))
    put("nTokens", f"{int(stats['tokens'].sum()):,}")
    put("meanPassageTokens", f"{stats['mean_passage_tokens'].mean():.0f}")
    longest = stats.sort_values("mean_sent_len", ascending=False).iloc[0]
    shortest = stats.sort_values("mean_sent_len").iloc[0]
    put("longestSentAuthor", tex_escape(longest["author"]))
    put("longestSentLen", f"{longest['mean_sent_len']:.1f}")
    put("shortestSentAuthor", tex_escape(shortest["author"]))
    put("shortestSentLen", f"{shortest['mean_sent_len']:.1f}")

prov = config.RAW / "provenance.json"
if prov.exists():
    rows = json.loads(prov.read_text(encoding="utf-8"))
    put("nBooks", len(rows))
    put("nTrainBooks", sum(1 for r in rows if r["role"] == "train"))
    put("nUnseenBooks", sum(1 for r in rows if r["role"] == "unseen"))
    put("corpusChars", f"{sum(r['chars'] for r in rows):,}")
    dropped = sum(len(r.get("chapters_removed", [])) for r in rows)
    put("nOverlapDropped", dropped)

overlap = config.PROCESSED / "overlap_report.json"
if overlap.exists():
    hits = json.loads(overlap.read_text(encoding="utf-8"))
    put("nOverlapHits", len(hits))
    if hits:
        lo = min(h["containment"] for h in hits)
        hi = max(h["containment"] for h in hits)
        put("overlapRange", f"{lo:.2f}--{hi:.2f}")
        put("overlapTrainBook", tex_escape(hits[0]["train_work"]))
        put("overlapUnseenBook", tex_escape(hits[0]["unseen_work"]))

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
comp = read_csv("split_composition.csv")
if comp is not None:
    for part in ("train", "val", "test"):
        put(f"n{part.capitalize()}", f"{int(comp[part].sum()):,}")
    put("testPerAuthor", int(comp["test"].iloc[0]))

leak = read_json("leakage_experiment.json")
if leak:
    put("randomSplitAcc", f"{leak['random_split_accuracy']:.3f}")
    put("workSplitAcc", f"{leak['work_disjoint_accuracy']:.3f}")
    put("leakGap", pct(leak["leakage_inflation"]))
    put("leakGapAbs", f"{leak['leakage_inflation']:.3f}")

# ---------------------------------------------------------------------------
# POS tagger
# ---------------------------------------------------------------------------
pos = read_json("postag_eval.json")
if pos:
    put("posAcc", pct(pos["accuracy"]))
    put("posAccRaw", f"{pos['accuracy']:.3f}")
    put("posTokens", pos["n_tokens"])
    put("posSentences", pos["n_sentences"])

# ---------------------------------------------------------------------------
# Main results
# ---------------------------------------------------------------------------
res = read_csv("results_all.csv")
if res is None:
    res = read_csv("results_local.csv")
if res is not None:
    res = res.sort_values("accuracy", ascending=False).reset_index(drop=True)
    best = res.iloc[0]
    put("bestModel", tex_escape(best["model"]))
    put("bestAcc", f"{best['accuracy']:.3f}")
    put("bestAccPct", pct(best["accuracy"]))
    put("bestFone", f"{best['macro_f1']:.3f}")
    put("nModels", len(res))

    def find(fragment: str):
        m = res[res["model"].str.contains(fragment, case=False, regex=False)]
        return m.iloc[0] if len(m) else None

    for key, fragment in [
        ("genChar", "Kneser-Ney LM (char)"),
        ("genWord", "Kneser-Ney LM (word)"),
        ("svmAll", "SVM (all stylometric)"),
        ("svmFunc", "funcword only"),
        ("svmStruct", "structural only"),
        ("svmPos", "posngram only"),
        ("svmChar", "charngram only"),
        ("naiveBayes", "Naive Bayes (word BoW)"),
        ("naiveBayesChar", "Naive Bayes (char n-gram)"),
        ("softmax", "Softmax regression"),
        ("tfidfBaseline", "TF-IDF words"),
        ("bertFT", "BanglaBERT (fine-tuned)"),
        ("bilstm", "BiLSTM"),
        ("transScratch", "Transformer (from scratch)"),
    ]:
        row = find(fragment)
        if row is not None:
            put(f"{key}Acc", f"{row['accuracy']:.3f}")
            put(f"{key}AccPct", pct(row["accuracy"]))
            put(f"{key}Fone", f"{row['macro_f1']:.3f}")

    # Best of each paradigm, so the comparison sentence writes itself.
    gen = res[res["family"] == "generative"]
    disc = res[res["family"].str.startswith("discriminative", na=False)]
    if len(gen):
        put("bestGenModel", tex_escape(gen.iloc[0]["model"]))
        put("bestGenAcc", f"{gen.iloc[0]['accuracy']:.3f}")
    if len(disc):
        put("bestDiscModel", tex_escape(disc.iloc[0]["model"]))
        put("bestDiscAcc", f"{disc.iloc[0]['accuracy']:.3f}")

# ---------------------------------------------------------------------------
# Paradigm agreement and length sensitivity
# ---------------------------------------------------------------------------
agree = read_csv("paradigm_agreement.csv")
if agree is not None:
    d = dict(zip(agree["case"], agree["n"]))
    total = int(sum(d.values()))
    n_agree = int(d.get("agree, both correct", 0) + d.get("agree, both wrong", 0))
    put("paradigmAgreement", pct(n_agree / max(total, 1)))
    put("bothCorrect", int(d.get("agree, both correct", 0)))
    put("bothWrong", int(d.get("agree, both wrong", 0)))
    put("discOnlyCorrect", int(d.get("disagree, discriminative correct", 0)))
    put("genOnlyCorrect", int(d.get("disagree, generative correct", 0)))

curve = read_csv("length_curve.csv")
if curve is not None:
    lo, hi = curve.iloc[0], curve.iloc[-1]
    put("lenShort", int(lo["tokens"]))
    put("lenLong", int(hi["tokens"]))
    put("accShortGen", f"{lo['generative']:.3f}")
    put("accLongGen", f"{hi['generative']:.3f}")
    put("accShortDisc", f"{lo['discriminative']:.3f}")
    put("accLongDisc", f"{hi['discriminative']:.3f}")

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
KNOWN = [
    "CharOrder", "WordOrder", "PassageTokens", "MinPassageTokens", "WtvDim",
    "TopFunctionWords",
    "PretrainedModel", "Seed", "nAuthors", "chanceAcc", "chancePct",
    "nPassages", "nWorks", "nChapters", "nTokens", "meanPassageTokens",
    "longestSentAuthor", "longestSentLen", "shortestSentAuthor",
    "shortestSentLen", "nBooks", "nTrainBooks", "nUnseenBooks", "corpusChars",
    "nOverlapDropped", "nOverlapHits", "overlapRange", "overlapTrainBook",
    "overlapUnseenBook", "nTrain", "nVal", "nTest", "testPerAuthor",
    "randomSplitAcc", "workSplitAcc", "leakGap", "leakGapAbs",
    "posAcc", "posAccRaw", "posTokens", "posSentences",
    "bestModel", "bestAcc", "bestAccPct", "bestFone", "nModels",
    "bestGenModel", "bestGenAcc", "bestDiscModel", "bestDiscAcc",
    "genCharAcc", "genWordAcc", "svmAllAcc", "svmFuncAcc", "svmStructAcc", "svmPosAcc", "svmCharAcc", "naiveBayesAcc", "naiveBayesCharAcc", "softmaxAcc", "tfidfBaselineAcc", "bertFTAcc", "bilstmAcc", "transScratchAcc",
    "genCharFone", "genWordFone", "svmAllFone", "svmFuncFone", "svmStructFone", "svmPosFone", "svmCharFone", "naiveBayesFone", "naiveBayesCharFone", "softmaxFone", "tfidfBaselineFone", "bertFTFone", "bilstmFone", "transScratchFone",
    "paradigmAgreement", "bothCorrect", "bothWrong", "discOnlyCorrect",
    "genOnlyCorrect", "lenShort", "lenLong", "accShortGen", "accLongGen",
    "accShortDisc", "accLongDisc",
]

lines = ["% Generated by scripts/make_report.py -- do not edit by hand.", ""]
for name in sorted(set(macros) | set(KNOWN)):
    value = macros.get(name, r"\textbf{??}")
    lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

out = config.REPORTS / "numbers.tex"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")

missing = [n for n in KNOWN if n not in macros]
print(f"wrote {out} with {len(macros)} macros")
if missing:
    print("MISSING (rendered as ?? in the report):", ", ".join(missing))
