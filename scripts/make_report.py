"""Generate ``reports/numbers.tex`` from the run's artifacts.

Every number in the report body comes through a macro defined here, so the
prose can never quietly go stale relative to the last run: change the pipeline,
re-run this, and the report updates.  If an artifact is missing the macro is
defined as a visible ``??`` rather than silently omitted, so a hole in the
report is obvious at a glance.
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


def put_config() -> None:
    """Hyper-parameters the prose quotes, taken from config rather than retyped."""
    macros.update(
        CharOrder=str(config.CHAR_LM_ORDER),
        WordOrder=str(config.WORD_LM_ORDER),
        FTEpochs=str(config.FT_EPOCHS),
        FTMaxLen=str(config.FT_MAX_LEN),
        WtvDim=str(config.W2V_DIM),
        WtvWindow=str(config.W2V_WINDOW),
        WtvNegative=str(config.W2V_NEGATIVE),
        PassageTokens=str(config.PASSAGE_TOKENS),
        PretrainedModel=config.PRETRAINED_MODEL.replace("_", r"\_"),
        Seed=str(config.SEED),
    )


def put(name: str, value) -> None:
    macros[name] = str(value)


def pct(x: float, places: int = 1) -> str:
    return f"{100 * float(x):.{places}f}\\%"


def read_csv(name: str) -> pd.DataFrame | None:
    p = T / name
    return pd.read_csv(p) if p.exists() else None


def read_json(name: str):
    p = T / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


put_config()

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
stats = read_csv("corpus_stats.csv")
if stats is not None:
    put("nAuthors", len(stats))
    put("nPassages", f"{int(stats['passages'].sum()):,}")
    put("nWorks", int(stats["works"].sum()))
    put("nTokens", f"{int(stats['tokens'].sum()):,}")
    put("passagesPerAuthor", f"{int(stats['passages'].iloc[0]):,}")
    put("meanPassageTokens", f"{stats['mean_passage_tokens'].mean():.0f}")
    put("chanceAcc", f"{1 / len(stats):.3f}")
    put("chancePct", pct(1 / len(stats)))
    put("earliestAuthor", stats.sort_values("years")["author"].iloc[0])

prov = config.RAW / "provenance.json"
if prov.exists():
    rows = [r for rs in json.loads(prov.read_text(encoding="utf-8")).values()
            for r in rs]
    seen = sum(r["n_pages"] for r in rows)
    kept = sum(r["n_proofread"] for r in rows)
    put("scanPagesSeen", f"{seen:,}")
    put("scanPagesProofread", f"{kept:,}")
    put("proofreadRate", pct(kept / max(seen, 1)))
    put("booksTouched", len(rows))

# ---------------------------------------------------------------------------
# Splits and leakage
# ---------------------------------------------------------------------------
comp = read_csv("split_composition.csv")
if comp is not None:
    tot = comp[["train", "val", "test"]].to_numpy().sum()
    for part in ("train", "val", "test"):
        put(f"{part}Frac", pct(comp[part].sum() / tot))
        put(f"n{part.capitalize()}", f"{int(comp[part].sum()):,}")

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
    per = pos["per_tag"]
    for tag in ("PRON", "VERB", "NOUN", "PUNCT", "ADP", "PART"):
        if tag in per:
            put(f"posRecall{tag.capitalize()}", pct(per[tag]["recall"], 0))

# ---------------------------------------------------------------------------
# Main results
# ---------------------------------------------------------------------------
res = read_csv("results_main.csv")
if res is not None:
    res = res.sort_values("accuracy", ascending=False)
    best = res.iloc[0]
    put("bestModel", best["model"].replace("·", "--"))
    put("bestAcc", f"{best['accuracy']:.3f}")
    put("bestAccPct", pct(best["accuracy"]))
    put("bestFone", f"{best['macro_f1']:.3f}")
    put("bestCI", f"[{best['acc_lo']:.3f}, {best['acc_hi']:.3f}]")

    def find(fragment: str):
        m = res[res["model"].str.contains(fragment, case=False, regex=False)]
        return m.iloc[0] if len(m) else None

    for key, fragment in [
        ("genChar", "Generative char"),
        ("genWord", "Generative word"),
        ("svmAll", "all stylometric"),
        ("svmFunc", "function words only"),
        ("svmStruct", "structural only"),
        ("svmPos", "POS n-grams only"),
        ("svmChar", "char n-grams only"),
        ("svmScratchMean", "scratch w2v (mean"),
        ("svmScratchSif", "scratch w2v (SIF"),
        ("svmBert", "BanglaBERT frozen"),
        ("bertFT", "Fine-tuned BanglaBERT"),
        ("comboScratch", "stylometric + scratch"),
        ("comboBert", "stylometric + BanglaBERT"),
    ]:
        row = find(fragment)
        if row is not None:
            put(f"{key}Acc", f"{row['accuracy']:.3f}")
            put(f"{key}AccPct", pct(row["accuracy"]))
            put(f"{key}Fone", f"{row['macro_f1']:.3f}")

# ---------------------------------------------------------------------------
# Style vs topic
# ---------------------------------------------------------------------------
svt = read_csv("style_vs_topic.csv")
if svt is not None:
    svt = svt.rename(columns={svt.columns[0]: "representation"})
    for _, r in svt.iterrows():
        key = ("Scratch" if "scratch" in r["representation"].lower() else "Bert")
        if "mean" in r["representation"]:
            key = "ScratchMean"
        elif "SIF" in r["representation"]:
            key = "ScratchSif"
        put(f"sil{key}Author", f"{r['silhouette_author']:.3f}")
        put(f"sil{key}Work", f"{r['silhouette_work']:.3f}")
        put(f"sil{key}Ratio", f"{r['style_topic_ratio']:.2f}")

# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------
imp = read_csv("permutation_importance.csv")
if imp is not None:
    for _, r in imp.iterrows():
        name = {"ch": "Char", "fw": "Func", "st": "Struct",
                "pos": "Pos"}.get(r["family"], r["family"].capitalize())
        put(f"perm{name}Drop", f"{r['mean_drop']:.3f}")
    top = imp.sort_values("mean_drop", ascending=False).iloc[0]
    put("permTopFamily", {"ch": "character $n$-grams", "fw": "function words",
                          "st": "structural statistics",
                          "pos": "POS $n$-grams"}.get(top["family"],
                                                      top["family"]))

agree = read_csv("paradigm_agreement.csv")
if agree is not None:
    agree.columns = ["case", "n"]
    d = dict(zip(agree["case"], agree["n"]))
    total = sum(d.values())
    n_agree = d.get("agree, both correct", 0) + d.get("agree, both wrong", 0)
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
    put("accShortDisc", f"{lo['discriminative']:.3f}")
    put("accLongDisc", f"{hi['discriminative']:.3f}")
    put("accShortGen", f"{lo['generative']:.3f}")
    put("accLongGen", f"{hi['generative']:.3f}")

# ---------------------------------------------------------------------------
# Fingerprints: who is most sadhu, who most chalit
# ---------------------------------------------------------------------------
fp = T / "style_fingerprints.csv"
if fp.exists():
    d = pd.read_csv(fp, index_col=0)
    if "sadhu_chalit_ratio" in d.columns:
        s = d["sadhu_chalit_ratio"].sort_values()
        put("mostChalit", config.AUTHORS.get(s.index[0], {}).get("en", s.index[0]))
        put("mostSadhu", config.AUTHORS.get(s.index[-1], {}).get("en", s.index[-1]))

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
# LaTeX control sequences are letters only: a name like ``estF1`` parses as
# ``estF`` followed by a stray ``1``.  Hence ``Fone`` rather than ``F1``.
KNOWN = [
    "nAuthors", "nPassages", "nWorks", "nTokens", "passagesPerAuthor",
    "meanPassageTokens", "chanceAcc", "chancePct", "scanPagesSeen",
    "scanPagesProofread", "proofreadRate", "booksTouched",
    "trainFrac", "valFrac", "testFrac", "nTrain", "nVal", "nTest",
    "randomSplitAcc", "workSplitAcc", "leakGap", "leakGapAbs",
    "posAcc", "posAccRaw", "posTokens", "posSentences",
    "bestModel", "bestAcc", "bestAccPct", "bestFone", "bestCI",
    "permTopFamily", "paradigmAgreement", "bothCorrect", "bothWrong",
    "discOnlyCorrect", "genOnlyCorrect", "mostSadhu", "mostChalit",
    "CharOrder", "WordOrder", "FTEpochs", "FTMaxLen", "PretrainedModel",
    "genCharAcc", "genWordAcc", "svmAllAcc", "svmFuncAcc", "svmStructAcc",
    "svmPosAcc", "svmCharAcc", "svmScratchMeanAcc", "svmScratchSifAcc",
    "svmBertAcc", "bertFTAcc", "comboScratchAcc", "comboBertAcc",
    "silScratchMeanAuthor", "silScratchMeanWork", "silScratchMeanRatio",
    "silScratchSifAuthor", "silScratchSifWork", "silScratchSifRatio",
    "silBertAuthor", "silBertWork", "silBertRatio",
    "permCharDrop", "permFuncDrop", "permStructDrop", "permPosDrop",
    "lenShort", "lenLong", "accShortDisc", "accLongDisc",
    "accShortGen", "accLongGen",
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
