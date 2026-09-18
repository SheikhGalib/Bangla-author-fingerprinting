"""Author the project's Jupyter notebooks.

The notebooks are the deliverable; this script is the source they are generated
from, so that a change to a shared preamble does not have to be repeated in six
JSON files by hand.  Run it, then execute the notebooks.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)

KERNEL = {
    "kernelspec": {
        "display_name": "Python (banglastylo)",
        "language": "python",
        "name": "banglastylo",
    },
    "language_info": {"name": "python", "version": "3.14"},
}

PREAMBLE = """\
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Bengali glyphs need a font that has them; fall back silently if absent.
for _f in ("Nirmala UI", "Vrinda", "Shonar Bangla", "Noto Sans Bengali"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.25})
pd.set_option("display.width", 160, "display.max_columns", 60)

from banglastylo import config
print("project root:", ROOT)
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.rstrip("\n").split("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.rstrip("\n").split("\n")}


def write(name: str, cells: list[dict]) -> None:
    nb = {"cells": cells, "metadata": KERNEL,
          "nbformat": 4, "nbformat_minor": 5}
    # nbformat wants each source line to keep its newline except the last.
    for c in nb["cells"]:
        c["source"] = [l + "\n" for l in c["source"][:-1]] + c["source"][-1:]
    (NB / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print("wrote", NB / name)


# ===========================================================================
# 01 — corpus
# ===========================================================================
nb01 = [
    md("""\
# 01 · Corpus construction

**Authorship Attribution and Style Fingerprinting in Bangla Literature**
KUET CSE 4122 — NLP Laboratory

This notebook turns Bengali Wikisource into a balanced, work-labelled,
passage-level corpus, and documents every decision that shaped it. The
proposal commits to treating corpus construction as *part of the methodology,
not a black box*, so the filtering statistics are reported rather than assumed.

Three things decide the quality of everything downstream:

1. **Proofreading filter.** Wikisource marks each scanned page with a quality
   level. Level 1 is raw OCR, and for Bangla that OCR is frequently garbled
   into Devanagari look-alikes. Only pages at level ≥ 3 (community-proofread)
   are admitted.
2. **Bangla-ratio filter.** Lines whose characters are less than 88 % in the
   Bangla Unicode block are dropped — this removes running headers, English
   epigraphs, page numbers and the residue of bad OCR in one rule.
3. **Work labelling.** Every passage records the book it came from, because
   the train/test split in notebook 02 is made over *works*, not passages."""),
    code(PREAMBLE),
    md("""\
## 1.1 Run the crawl

`build_raw_corpus` is cached on disk: if `data/raw/corpus_raw.json` already
exists from `scripts/01_build_corpus.py`, this cell just loads it."""),
    code("""\
from banglastylo import corpus

raw_path = config.RAW / "corpus_raw.json"
if raw_path.exists():
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    print("loaded cached crawl")
else:
    raw = corpus.build_raw_corpus()

summary = pd.DataFrame([
    {"author": a,
     "works": len(w),
     "characters": sum(len(t) for t in w.values())}
    for a, w in raw.items()
]).sort_values("characters", ascending=False)
summary["author_en"] = summary["author"].map(lambda k: config.AUTHORS[k]["en"])
summary"""),
    md("""\
## 1.2 What the proofreading filter cost

`provenance.json` records, for every scanned book, how many pages existed and
how many survived the quality filter. This is the number to quote when
someone asks how much of Wikisource is actually usable."""),
    code("""\
prov_path = config.RAW / "provenance.json"
prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {}

rows = [dict(author=a, **r) for a, rs in prov.items() for r in rs]
prov_df = pd.DataFrame(rows)
if len(prov_df):
    prov_df["kept_frac"] = prov_df["n_proofread"] / prov_df["n_pages"].clip(lower=1)
    print(f"{len(prov_df)} scanned books touched; "
          f"{prov_df['n_pages'].sum():,} scan pages seen, "
          f"{prov_df['n_proofread'].sum():,} proofread "
          f"({prov_df['n_proofread'].sum() / max(prov_df['n_pages'].sum(),1):.1%})")
    display(prov_df.groupby("author")[["n_pages", "n_proofread", "n_chars"]]
            .sum().sort_values("n_chars", ascending=False))"""),
    md("""\
## 1.3 Segment into passages

A **passage** is ~220 tokens cut on sentence boundaries — long enough for
sentence-length statistics to mean something, short enough to give a few
hundred independent samples per author.

Near-duplicate passages are removed with an 8-gram shingle overlap test.
This matters more than it looks: Wikisource hosts several editions of the same
novel, and without this filter the "same" text would appear on both sides of a
split."""),
    code("""\
passages = corpus.build_passages(raw)
print(f"\\n{len(passages):,} passages before balancing")

df = pd.DataFrame([p.__dict__ for p in passages])
display(df.groupby("author").agg(
    passages=("passage_id", "count"),
    works=("work", "nunique"),
    mean_tokens=("n_tokens", "mean"),
    mean_sentences=("n_sentences", "mean"),
).sort_values("passages", ascending=False))"""),
    md("""\
## 1.4 Balance

An author is dropped from the study for too few passages *or* too few distinct
works. The second criterion is the binding one: a one-book author cannot take
part in a work-disjoint split at all, because every passage of theirs would
land in the same split. Who was dropped is printed rather than glossed over.

The survivors are down-sampled to the smallest of them, walking each author's
works round-robin so that balancing does not silently reduce an author to a
single book."""),
    code("""\
balanced, dropped = corpus.balance(passages)
corpus.save_passages(balanced)
if dropped:
    pd.DataFrame(dropped).to_csv(config.TABLES / "dropped_authors.csv", index=False)
    display(pd.DataFrame(dropped).assign(
        author_en=lambda d: d["author"].map(lambda k: config.AUTHORS[k]["en"])))

bdf = pd.DataFrame([p.__dict__ for p in balanced])
n_authors = bdf['author'].nunique()
print(f"\\n{len(balanced):,} passages, {n_authors} authors, "
      f"chance accuracy = {1/n_authors:.3f}")
display(bdf.groupby("author").agg(
    passages=("passage_id", "count"),
    works=("work", "nunique"),
    tokens=("n_tokens", "sum"),
))"""),
    md("## 1.5 Corpus figures"),
    code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

order = sorted(bdf["author"].unique())
names = [config.AUTHORS[a]["short"] for a in order]

axes[0].bar(names, [(bdf.author == a).sum() for a in order], color="#2F5D62")
axes[0].set_title("passages per author (balanced)")
axes[0].tick_params(axis="x", rotation=40)

axes[1].bar(names, [bdf.loc[bdf.author == a, "work"].nunique() for a in order],
            color="#7FA8A0")
axes[1].set_title("distinct works per author")
axes[1].tick_params(axis="x", rotation=40)

for a, n in zip(order, names):
    sub = bdf.loc[bdf.author == a, "n_tokens"]
    axes[2].hist(sub, bins=30, histtype="step", lw=1.4, label=n)
axes[2].set_title("passage length (tokens)")
axes[2].legend(fontsize=7)

fig.tight_layout()
fig.savefig(config.FIGURES / "corpus_overview.png", dpi=200)
fig.savefig(config.REPORT_FIGURES / "corpus_overview.png", dpi=200)
plt.show()"""),
    md("## 1.6 A sample passage from each author"),
    code("""\
for a in order:
    row = bdf[bdf.author == a].iloc[0]
    print("=" * 90)
    print(f"{config.AUTHORS[a]['en']}  ({config.AUTHORS[a]['bn']}, "
          f"{config.AUTHORS[a]['years']})")
    print(f"work: {row['work'][:80]}")
    print("-" * 90)
    print(row["text"][:380].replace("\\n", " "), "…")
print("=" * 90)"""),
    md("""\
## 1.7 Corpus statistics table for the report"""),
    code("""\
stats = bdf.groupby("author").agg(
    passages=("passage_id", "count"),
    works=("work", "nunique"),
    tokens=("n_tokens", "sum"),
    mean_passage_tokens=("n_tokens", "mean"),
)
stats["author"] = [config.AUTHORS[a]["en"] for a in stats.index]
stats["years"] = [config.AUTHORS[a]["years"] for a in stats.index]
stats = stats[["author", "years", "works", "passages", "tokens",
               "mean_passage_tokens"]]
stats.to_csv(config.TABLES / "corpus_stats.csv", index=False)

tex = [r"\\begin{tabular}{llrrrr}", r"\\toprule",
       r"Author & Period & Works & Passages & Tokens & Mean tokens/passage \\\\",
       r"\\midrule"]
for _, r in stats.iterrows():
    tex.append(f"{r['author']} & {r['years']} & {r['works']} & {r['passages']} & "
               f"{r['tokens']:,} & {r['mean_passage_tokens']:.0f} \\\\\\\\")
tex += [r"\\midrule",
        f"\\\\textbf{{Total}} & & {stats['works'].sum()} & {stats['passages'].sum()} & "
        f"{stats['tokens'].sum():,} & \\\\\\\\",
        r"\\bottomrule", r"\\end{tabular}"]
(config.TABLES / "corpus_stats.tex").write_text("\\n".join(tex), encoding="utf-8")
stats"""),
]

# ===========================================================================
# 02 — splits + POS tagger validation
# ===========================================================================
nb02 = [
    md("""\
# 02 · Work-disjoint splits, and validating the Bangla POS tagger

Two pieces of methodology that decide whether the later numbers can be
believed at all.

**Splitting.** If passages are split at random, passages from the same novel
land on both sides. A classifier can then win by recognising a character name
— topic identification wearing style's clothes. Splitting over *works* removes
that route. The cost is that the realised split fractions drift from the
targets, because books differ in size; that drift is reported below rather
than hidden.

**POS tagger.** The proposal calls for POS n-grams "from a Bangla tagger".
There is no usable off-the-shelf one here (`stanza` ships no Bengali POS
model; `bnlp-toolkit` needs `gensim`, which has no Python 3.14 wheel), so the
tagger is written from scratch and **scored against the Universal
Dependencies Bengali treebank** so the reader can discount the POS features
appropriately."""),
    code(PREAMBLE),
    md("## 2.1 Build the split"),
    code("""\
from banglastylo import corpus, splits

passages = corpus.load_passages()
split = splits.split_by_work(passages)
splits.save(split)

print("split sizes:", split.counts())
total = sum(split.counts().values())
for k, v in split.counts().items():
    print(f"  {k:5s} {v:5d}  ({v/total:.1%})")"""),
    md("""\
### The leakage check

This is not decoration. It asserts the invariant the whole evaluation rests
on: no work, and no passage, appears in two splits."""),
    code("""\
report = splits.leakage_report(split)
print("clean:", report["clean"])
for k, v in report["overlaps"].items():
    print(f"  {k:24s} {v}")
assert report["clean"], "LEAKAGE: a work or passage spans two splits"
print("\\nno work and no passage crosses a split boundary.")"""),
    md("## 2.2 Per-author split composition"),
    code("""\
desc = pd.DataFrame(splits.describe(split)).T
desc["author"] = [config.AUTHORS[a]["en"] for a in desc.index]
desc = desc[["author", "train", "val", "test",
             "works_train", "works_val", "works_test"]]
desc.to_csv(config.TABLES / "split_composition.csv")
desc"""),
    code("""\
fig, ax = plt.subplots(figsize=(9, 3.6))
names = [config.AUTHORS[a]["short"] for a in desc.index]
bottom = np.zeros(len(desc))
for part, colour in (("train", "#2F5D62"), ("val", "#7FA8A0"), ("test", "#C9A227")):
    vals = desc[part].to_numpy(dtype=float)
    ax.bar(names, vals, bottom=bottom, label=part, color=colour)
    bottom += vals
ax.set_ylabel("passages")
ax.set_title("work-disjoint split composition per author")
ax.tick_params(axis="x", rotation=35)
ax.legend()
fig.tight_layout()
fig.savefig(config.FIGURES / "split_composition.png", dpi=200)
fig.savefig(config.REPORT_FIGURES / "split_composition.png", dpi=200)
plt.show()"""),
    md("""\
## 2.3 How much does work-disjoint splitting actually matter?

Worth measuring rather than asserting. The same model is trained twice — once
on the honest work-disjoint split, once on a naive random split of the same
passages. The gap is the size of the topic-leak the discipline buys away."""),
    code("""\
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from banglastylo.models import build_svm_pipeline

Xtr, ytr = splits.xy(split.train)
Xte, yte = splits.xy(split.test)

pipe = build_svm_pipeline(families=("funcword", "structural", "charngram"))
pipe.fit(Xtr, ytr)
acc_honest = accuracy_score(yte, pipe.predict(Xte))

texts = [p.text for p in passages]
labels = [p.author for p in passages]
rXtr, rXte, rytr, ryte = train_test_split(
    texts, labels, test_size=0.25, random_state=config.SEED, stratify=labels)
pipe_naive = build_svm_pipeline(families=("funcword", "structural", "charngram"))
pipe_naive.fit(rXtr, rytr)
acc_naive = accuracy_score(ryte, pipe_naive.predict(rXte))

print(f"random passage-level split (leaky) : {acc_naive:.4f}")
print(f"work-disjoint split (honest)       : {acc_honest:.4f}")
print(f"inflation from topic leakage       : {acc_naive - acc_honest:+.4f}")

json.dump({"random_split_accuracy": acc_naive,
           "work_disjoint_accuracy": acc_honest,
           "leakage_inflation": acc_naive - acc_honest},
          open(config.TABLES / "leakage_experiment.json", "w"), indent=2)"""),
    md("""\
## 2.4 Validating the POS tagger

Scored against `UD_Bengali-BRU`, the only Universal Dependencies treebank for
Bengali — 56 sentences, test-only, which is why it can be used for evaluation
but not for training.

Three UD distinctions are deliberately not attempted and are folded before
scoring: `AUX`→`VERB` (Bangla auxiliaries are ordinary verbs in ordinary verb
forms), `PROPN`→`NOUN` (needs a name gazetteer that does not exist for
19th-century Bangla), `SCONJ`→`CCONJ` (the stylometric feature of interest is
conjunction density, not subtype). Each fold loses information rather than
inventing it."""),
    code("""\
from banglastylo import postag
import urllib.request

ud_path = config.RAW / "bn_bru-ud-test.conllu"
if not ud_path.exists():
    url = ("https://raw.githubusercontent.com/UniversalDependencies/"
           "UD_Bengali-BRU/master/bn_bru-ud-test.conllu")
    ud_path.write_text(urllib.request.urlopen(url, timeout=60)
                       .read().decode("utf-8"), encoding="utf-8")

ev = postag.evaluate_on_ud(ud_path)
print(f"UD_Bengali-BRU: {ev['n_sentences']} sentences, {ev['n_tokens']} tokens")
print(f"token accuracy: {ev['accuracy']:.4f}")
json.dump(ev, open(config.TABLES / "postag_eval.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

pd.DataFrame(ev["per_tag"]).T.assign(
    support=lambda d: d["support"].astype(int)).sort_values("support",
                                                            ascending=False)"""),
    code("""\
print("most frequent tagger confusions (gold -> predicted):")
for (gold, pred), n in ev["top_confusions"]:
    print(f"  {gold:6s} -> {pred:6s}  {n:3d}")"""),
    md("""\
### The tagger on the proposal's own worked example

The proposal's example sentence claims a `NOUN–VERB–ADP–NOUN` pattern and
picks out the postpositions পাশে and দিকে and the pronoun সে. Running the
tagger on it is the cheapest possible sanity check that it recovers what the
proposal says it should."""),
    code("""\
from banglastylo.normalize import normalize_text, word_tokenize

sentence = normalize_text(
    "বৃষ্টি থামলে সে জানালার পাশে দাঁড়িয়ে দূরের আকাশের দিকে তাকিয়ে রইল।")
toks = word_tokenize(sentence, keep_punct=True)
tags = postag.tag(toks)
pd.DataFrame({"token": toks, "tag": tags}).T"""),
]

# ===========================================================================
# 03 — embeddings
# ===========================================================================
nb03 = [
    md("""\
# 03 · Embeddings — pretrained versus trained from scratch

The proposal's representation-level question: *in a small-corpus regime, do
pretrained embeddings or embeddings trained from scratch on the author corpus
better capture stylistic signal rather than general semantics?*

Two regimes:

| | trained on | dimension | knows |
|---|---|---|---|
| **from scratch** | this corpus' training split only | 200 | 19th-century literary Bangla, both registers |
| **BanglaBERT** | ~27 GB of modern Bangla | 768 | modern Bangla semantics |

Both are fit or applied **without ever seeing test text**: the skip-gram model
is trained on the training split alone.

The comparison is not just downstream accuracy. `style_vs_topic_score`
measures how well each representation separates *authors* versus how well it
separates *individual books* — a representation that mostly encodes topic
separates books much better than authors, because each book has its own
subject matter."""),
    code(PREAMBLE),
    code("""\
from banglastylo import corpus, splits, embeddings

passages = corpus.load_passages()
split = splits.load(passages)
Xtr, ytr = splits.xy(split.train)
Xva, yva = splits.xy(split.val)
Xte, yte = splits.xy(split.test)
works_te = [p.work for p in split.test]
print(f"train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}")"""),
    md("""\
## 3.1 Skip-gram with negative sampling, from scratch

Written out in PyTorch rather than imported: `gensim` has no Python 3.14
wheel here, and writing it makes the training regime explicit — Mikolov
subsampling, a 3/4-power noise distribution, and a dynamic window radius."""),
    code("""\
w2v_path = config.MODELS / "w2v_scratch.npz"
if w2v_path.exists():
    w2v = embeddings.SkipGramWord2Vec.load(w2v_path)
    print(f"loaded cached model: {len(w2v.index_to_word):,} words × "
          f"{w2v.vectors.shape[1]}d")
else:
    w2v = embeddings.SkipGramWord2Vec()
    w2v.fit(Xtr)
    w2v.save(w2v_path)
print("vocabulary:", f"{len(w2v.index_to_word):,}")"""),
    md("""\
### Does it look like it learned Bangla?

Nearest neighbours for a few probe words. Function words should attract other
function words, register markers should attract their own register — that is
the property the stylometric use of these vectors depends on."""),
    code("""\
probes = ["তাহার", "করিয়া", "মানুষ", "রাজা", "কহিলেন", "হইল", "নারী", "গ্রাম"]
for p in probes:
    nb = w2v.most_similar(p, k=6)
    if nb:
        print(f"{p:10s} → " + ", ".join(f"{w} ({s:.2f})" for w, s in nb))
    else:
        print(f"{p:10s} → (out of vocabulary)")"""),
    md("## 3.2 Pretrained BanglaBERT, frozen"),
    code("""\
import numpy as np

bert_cache = config.MODELS / "banglabert_vectors.npz"
if bert_cache.exists():
    d = np.load(bert_cache)
    Btr, Bva, Bte = d["train"], d["val"], d["test"]
    print("loaded cached BanglaBERT vectors")
else:
    enc = embeddings.BanglaBertEmbedder()
    print("encoding train…"); Btr = enc.encode(Xtr, verbose=True)
    print("encoding val…");   Bva = enc.encode(Xva)
    print("encoding test…");  Bte = enc.encode(Xte, verbose=True)
    np.savez_compressed(bert_cache, train=Btr, val=Bva, test=Bte)
    del enc
print("BanglaBERT passage vectors:", Btr.shape, Bva.shape, Bte.shape)"""),
    md("""\
## 3.3 Pooling word vectors into passage vectors

SIF weighting (`a / (a + p(w))`) damps frequent words. For a *semantic* task
that helps; for a *stylometric* one it is a real trade-off, because the
frequent function words SIF damps are exactly what classical stylometry
relies on. Both poolings are computed so the report can show the difference
instead of assuming it."""),
    code("""\
Wtr_sif = embeddings.average_vectors(Xtr, w2v, weighting="sif")
Wva_sif = embeddings.average_vectors(Xva, w2v, weighting="sif")
Wte_sif = embeddings.average_vectors(Xte, w2v, weighting="sif")

Wtr_mean = embeddings.average_vectors(Xtr, w2v, weighting="mean")
Wva_mean = embeddings.average_vectors(Xva, w2v, weighting="mean")
Wte_mean = embeddings.average_vectors(Xte, w2v, weighting="mean")

np.savez_compressed(config.MODELS / "w2v_vectors.npz",
                    train_sif=Wtr_sif, val_sif=Wva_sif, test_sif=Wte_sif,
                    train_mean=Wtr_mean, val_mean=Wva_mean, test_mean=Wte_mean)
print("scratch-embedding passage vectors:", Wtr_sif.shape)"""),
    md("""\
## 3.4 Style or topic? The diagnostic

Two silhouette scores on the same vectors: one with authors as labels, one
with individual works as labels. The **ratio** is the quantity the proposal's
question turns on."""),
    code("""\
diag = {}
for name, V in [("scratch-w2v (SIF)",  Wte_sif),
                ("scratch-w2v (mean)", Wte_mean),
                ("BanglaBERT (frozen)", Bte)]:
    diag[name] = embeddings.style_vs_topic_score(V, yte, works_te)

diag_df = pd.DataFrame(diag).T
diag_df.to_csv(config.TABLES / "style_vs_topic.csv")
diag_df"""),
    code("""\
fig, ax = plt.subplots(figsize=(7.6, 3.4))
y = np.arange(len(diag_df))
ax.barh(y - 0.19, diag_df["silhouette_author"], height=0.36,
        label="clusters by AUTHOR (style)", color="#2F5D62")
ax.barh(y + 0.19, diag_df["silhouette_work"], height=0.36,
        label="clusters by WORK (topic)", color="#C9A227")
ax.set_yticks(y, diag_df.index, fontsize=9)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("silhouette score (cosine)")
ax.set_title("Does the representation encode style or topic?", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(config.FIGURES / "style_vs_topic.png", dpi=200)
fig.savefig(config.REPORT_FIGURES / "style_vs_topic.png", dpi=200)
plt.show()"""),
    md("""\
## 3.5 What the embedding spaces look like

PCA to two dimensions, coloured by author. Not evidence on its own — PCA
discards most of the variance — but it makes the difference between the two
regimes visible at a glance."""),
    code("""\
from sklearn.decomposition import PCA

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
order = sorted(set(yte))
colours = plt.cm.tab10(np.linspace(0, 1, len(order)))

for ax, (name, V) in zip(axes, [("scratch-w2v (SIF)", Wte_sif),
                                ("scratch-w2v (mean)", Wte_mean),
                                ("BanglaBERT (frozen)", Bte)]):
    P = PCA(n_components=2, random_state=config.SEED).fit_transform(V)
    for a, c in zip(order, colours):
        m = np.array(yte) == a
        ax.scatter(P[m, 0], P[m, 1], s=8, alpha=0.6, color=c,
                   label=config.AUTHORS[a]["short"])
    ax.set_title(name, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
axes[-1].legend(fontsize=7, markerscale=1.6, loc="best")
fig.suptitle("Test-split passages in each embedding space (PCA)", fontsize=11)
fig.tight_layout()
fig.savefig(config.FIGURES / "embedding_pca.png", dpi=200)
fig.savefig(config.REPORT_FIGURES / "embedding_pca.png", dpi=200)
plt.show()"""),
]

# ===========================================================================
# 04 — models + evaluation
# ===========================================================================
nb04 = [
    md("""\
# 04 · Generative versus discriminative attribution, and the ablations

The centre of the project. Everything is evaluated on the **same
work-disjoint test split**, so the rows of the final table are comparable.

**Generative.** One n-gram language model per author, fit only on that
author's training text, smoothed with interpolated Kneser–Ney. A passage goes
to whichever author's model assigns it the lowest perplexity. The model never
sees the other authors — it learns what this author's text looks like, not
how they differ.

**Discriminative.** All authors' features go into one classifier trained to
tell them apart: a linear SVM over stylometric and/or embedding features, and
a fine-tuned BanglaBERT.

The ablations answer the proposal's two comparison questions: which feature
families matter, and pretrained versus from-scratch embeddings."""),
    code(PREAMBLE),
    code("""\
from banglastylo import corpus, splits, evaluate, embeddings
from banglastylo.models import (GenerativeAttributor, build_svm_pipeline,
                                build_svm, BertClassifier)

passages = corpus.load_passages()
split = splits.load(passages)
Xtr, ytr = splits.xy(split.train)
Xva, yva = splits.xy(split.val)
Xte, yte = splits.xy(split.test)

n_authors = len(set(ytr))
chance = 1 / n_authors
results = evaluate.ResultTable(chance=chance)
print(f"{n_authors} authors · train {len(Xtr)} · val {len(Xva)} · test {len(Xte)}")
print(f"chance accuracy = {chance:.4f}")"""),
    md("""\
## 4.1 Generative: per-author language models

Character models first. Bangla is morphologically rich and its inflection is
suffixal, so a character model sees inside the word — it picks up the
register difference between করিয়াছিলেন and করেছিলেন without either form
having been seen whole."""),
    code("""\
gen_char = GenerativeAttributor(level="char", order=config.CHAR_LM_ORDER)
gen_char.fit(Xtr, ytr)
pred_gen_char = gen_char.predict(Xte)
res_gen_char = results.add(
    f"Generative char-{config.CHAR_LM_ORDER}-gram LM", "characters",
    yte, pred_gen_char, "per-author Kneser-Ney LM, lowest perplexity wins")"""),
    code("""\
gen_word = GenerativeAttributor(level="word", order=config.WORD_LM_ORDER)
gen_word.fit(Xtr, ytr)
pred_gen_word = gen_word.predict(Xte)
res_gen_word = results.add(
    f"Generative word-{config.WORD_LM_ORDER}-gram LM", "words",
    yte, pred_gen_word, "per-author Kneser-Ney LM, lowest perplexity wins")"""),
    md("""\
### The perplexity table itself

The generative decision is a comparison of perplexities, so it is worth
looking at the actual numbers for a handful of test passages rather than only
the argmax."""),
    code("""\
sample = list(range(0, len(Xte), max(len(Xte) // 6, 1)))[:6]
ppl = gen_char.perplexity_table([Xte[i] for i in sample])
ppl_df = pd.DataFrame(ppl, columns=[config.AUTHORS[a]["short"]
                                    for a in gen_char.classes_])
ppl_df.insert(0, "true author",
              [config.AUTHORS[yte[i]]["short"] for i in sample])
ppl_df.style.background_gradient(cmap="Blues_r", axis=1,
                                 subset=ppl_df.columns[1:])"""),
    md("""\
## 4.2 Discriminative: linear SVM ablations

Each row isolates one feature family or one combination, so the table answers
"what is doing the work?" rather than only "how well does it do?"."""),
    code("""\
svm_ablations = {
    "SVM · function words only":        ("funcword",),
    "SVM · structural only":            ("structural",),
    "SVM · POS n-grams only":           ("posngram",),
    "SVM · char n-grams only":          ("charngram",),
    "SVM · funcword + structural":      ("funcword", "structural"),
    "SVM · all stylometric":            ("funcword", "structural",
                                         "charngram", "posngram"),
}

pipelines = {}
for name, fams in svm_ablations.items():
    pipe = build_svm_pipeline(families=fams)
    pipe.fit(Xtr, ytr)
    pipelines[name] = pipe
    results.add(name, "+".join(fams), yte, pipe.predict(Xte))"""),
    md("""\
## 4.3 Discriminative: embedding features

The proposal's pretrained-versus-from-scratch comparison, held under
otherwise identical conditions — same classifier, same split, same
evaluation."""),
    code("""\
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

V = np.load(config.MODELS / "w2v_vectors.npz")
B = np.load(config.MODELS / "banglabert_vectors.npz")

embedding_sets = {
    "SVM · scratch w2v (SIF pooled)":  (V["train_sif"],  V["test_sif"]),
    "SVM · scratch w2v (mean pooled)": (V["train_mean"], V["test_mean"]),
    "SVM · BanglaBERT frozen":         (B["train"],      B["test"]),
}

emb_models = {}
for name, (Etr, Ete) in embedding_sets.items():
    clf = make_pipeline(StandardScaler(), build_svm())
    clf.fit(Etr, ytr)
    emb_models[name] = clf
    results.add(name, "embeddings", yte, clf.predict(Ete))"""),
    md("""\
### Stylometric + embedding, combined

The proposal's headline discriminative configuration: concatenate the
stylometric features with the embedding features and feed the lot to one
SVM."""),
    code("""\
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler
from banglastylo.stylometry import StylometricFeatures

feat = StylometricFeatures(("funcword", "structural", "charngram", "posngram"))
feat.fit(Xtr)
Str, Ste = feat.transform(Xtr), feat.transform(Xte)

for label, (Etr, Ete) in [("scratch w2v", (V["train_mean"], V["test_mean"])),
                          ("BanglaBERT",  (B["train"],      B["test"]))]:
    scale = StandardScaler().fit(Etr)
    Ctr = sparse.hstack([Str, sparse.csr_matrix(scale.transform(Etr))]).tocsr()
    Cte = sparse.hstack([Ste, sparse.csr_matrix(scale.transform(Ete))]).tocsr()
    clf = make_pipeline(MaxAbsScaler(), build_svm())
    clf.fit(Ctr, ytr)
    results.add(f"SVM · stylometric + {label}", "combined", yte, clf.predict(Cte))"""),
    md("""\
## 4.4 Discriminative: fine-tuned BanglaBERT

The heaviest model in the comparison, and the one with the most outside
knowledge. Worth noting what it is being asked to do: BanglaBERT was
pretrained on *modern* Bangla, and this corpus is 19th- and early-20th-century
literary prose in a register the pretraining data barely contains."""),
    code("""\
import time

# Fine-tuning costs ~40 minutes on CPU, so a re-run reloads the checkpoint
# rather than paying for it again just to redraw a figure.
bert_dir = config.MODELS / "banglabert_finetuned"
if (bert_dir / "classes.txt").exists():
    bert = BertClassifier.load(bert_dir)
    print(f"loaded cached fine-tuned model from {bert_dir.name}")
else:
    t0 = time.time()
    bert = BertClassifier()
    bert.fit(Xtr, ytr, val=(Xva, yva))
    print(f"fine-tuning took {(time.time()-t0)/60:.1f} min on {bert.device}")
    bert.save(bert_dir)

pred_bert = bert.predict(Xte)
res_bert = results.add("Fine-tuned BanglaBERT", "pretrained transformer",
                       yte, pred_bert)"""),
    md("## 4.5 The results table"),
    code("""\
results.save("results_main")
df = results.frame().sort_values("accuracy", ascending=False)
tex = results.to_latex(
    "results_main",
    caption=("Closed-set authorship attribution on the held-out, work-disjoint "
             "test split. Intervals are 95\\\\% percentile bootstrap over test "
             "passages."),
    tex_label="tab:results")
df[["model", "family", "accuracy", "acc_lo", "acc_hi", "macro_f1"]]"""),
    code("""\
fig = evaluate.plot_ablation(
    results.frame(), path=config.FIGURES / "ablation.png",
    title="Generative vs discriminative attribution, and feature ablations")
plt.show()"""),
    md("""\
## 4.6 Confusion analysis

The proposal asks specifically about confusion between *stylistically similar*
authors, so the confusion matrices matter more than the headline accuracy."""),
    code("""\
best_name = df.iloc[0]["model"]
print("best model overall:", best_name)

# One matrix per *paradigm*, not "best" and "generative" — those can be the
# same model, and two identical heatmaps side by side would say nothing.
fig = evaluate.plot_confusion(
    res_gen_char, f"Generative char-{config.CHAR_LM_ORDER}-gram LM",
    path=config.FIGURES / "confusion_generative.png")
plt.show()

disc_name = "SVM · all stylometric"
fig = evaluate.plot_confusion(results.full[disc_name],
                              "Discriminative: SVM, all stylometric",
                              path=config.FIGURES / "confusion_discriminative.png")
plt.show()

if best_name not in (disc_name, f"Generative char-{config.CHAR_LM_ORDER}-gram LM"):
    fig = evaluate.plot_confusion(results.full[best_name], best_name,
                                  path=config.FIGURES / "confusion_best.png")
    plt.show()
else:
    print("(best model is one of the two above; no third matrix drawn)")"""),
    code("""\
print("most-confused author pairs\\n")
shown = dict.fromkeys([best_name,
                       f"Generative char-{config.CHAR_LM_ORDER}-gram LM",
                       "SVM · all stylometric"])
for name in shown:
    print(name)
    for a, b, n, rate in evaluate.confusable_pairs(results.full[name], top_k=5):
        print(f"   {config.AUTHORS[a]['en']:32s} <-> "
              f"{config.AUTHORS[b]['en']:32s} {n:4d} errors  ({rate:.3f})")
    print()"""),
    md("""\
### Where the two paradigms disagree

Agreement rate and, when they disagree, which one is right. This is the
comparison the proposal asks for, made concrete."""),
    code("""\
pred_disc = pipelines["SVM · all stylometric"].predict(Xte)
pred_gen  = pred_gen_char
yte_arr = np.asarray(yte)

agree = pred_disc == pred_gen
both_right = agree & (pred_disc == yte_arr)
both_wrong = agree & (pred_disc != yte_arr)
disc_right = (~agree) & (pred_disc == yte_arr)
gen_right  = (~agree) & (pred_gen  == yte_arr)
neither    = (~agree) & (pred_disc != yte_arr) & (pred_gen != yte_arr)

breakdown = pd.Series({
    "agree, both correct":        both_right.sum(),
    "agree, both wrong":          both_wrong.sum(),
    "disagree, discriminative correct": disc_right.sum(),
    "disagree, generative correct":     gen_right.sum(),
    "disagree, neither correct":        neither.sum(),
})
print(f"paradigms agree on {agree.mean():.1%} of test passages\\n")
print(breakdown.to_string())
breakdown.to_csv(config.TABLES / "paradigm_agreement.csv")"""),
    md("## 4.7 Save the models the interface needs"),
    code("""\
import pickle
from banglastylo.interface import Attributor

gen_char.save(config.MODELS / "generative_char.pkl")
with open(config.MODELS / "svm_combined.pkl", "wb") as fh:
    pickle.dump(pipelines["SVM · all stylometric"], fh)
print("saved:", sorted(p.name for p in config.MODELS.glob("*")))"""),
]

# ===========================================================================
# 05 — interpretability
# ===========================================================================
nb05 = [
    md("""\
# 05 · Interpretability and style fingerprints

The proposal requires predictions to be *auditable rather than opaque*. Three
mechanisms, each answering a different question:

| question | mechanism |
|---|---|
| why *this* prediction, for *this* passage? | exact linear decomposition `w·x` |
| which feature *families* carry the model? | grouped permutation importance |
| what does each author *look like*? | structural fingerprint in z-scores |

For a linear SVM the contribution of feature *i* is exactly `w[c,i] · x[i]`.
That is the same quantity SHAP estimates for a linear model — computed
exactly rather than sampled, which is why no SHAP dependency appears here."""),
    code(PREAMBLE),
    code("""\
import pickle
from banglastylo import corpus, splits, interpret, evaluate

passages = corpus.load_passages()
split = splits.load(passages)
Xtr, ytr = splits.xy(split.train)
Xte, yte = splits.xy(split.test)

with open(config.MODELS / "svm_combined.pkl", "rb") as fh:
    pipe = pickle.load(fh)
from banglastylo.models import GenerativeAttributor
gen = GenerativeAttributor.load(config.MODELS / "generative_char.pkl")
print("models loaded")"""),
    md("""\
## 5.1 Which feature families carry the model?

Shuffle a whole family's columns together and measure what accuracy costs.
Model-agnostic, and asked at the level the ablation question is asked at."""),
    code("""\
imp = interpret.permutation_importance_grouped(pipe, Xte, yte, n_repeats=5)
imp.to_csv(config.TABLES / "permutation_importance.csv", index=False)
imp"""),
    code("""\
FAMILY_LABEL = {"fw": "function words", "st": "structural statistics",
                "ch": "character n-grams", "pos": "POS n-grams"}
fig, ax = plt.subplots(figsize=(7.2, 3.2))
d = imp.sort_values("mean_drop")
ax.barh([FAMILY_LABEL.get(f, f) for f in d["family"]], d["mean_drop"],
        xerr=d["sd_drop"], color="#2F5D62", ecolor="#999")
ax.set_xlabel("accuracy lost when this family is shuffled")
ax.set_title("Grouped permutation importance (test split)", fontsize=10)
fig.tight_layout()
fig.savefig(config.FIGURES / "permutation_importance.png", dpi=200)
fig.savefig(config.REPORT_FIGURES / "permutation_importance.png", dpi=200)
plt.show()"""),
    md("""\
## 5.2 Style fingerprints

Each author's mean structural feature, in corpus standard deviations. The
`sadhu_chalit_ratio` column is the one to read first: Bangla prose of this
period is written in either the literary সাধু register (হইল, করিয়া, তাহার) or
the colloquial চলিত one (হল, করে, তার), and where an author sits on that axis
is the most interpretable single fact about their style."""),
    code("""\
sig = interpret.family_signature([p.text for p in passages],
                                 [p.author for p in passages])
sig.to_csv(config.TABLES / "style_fingerprints.csv")
fig = interpret.plot_fingerprint(sig, path=config.FIGURES / "fingerprints.png")
plt.show()"""),
    md("""\
The z-scored heatmap above is easiest to read at a glance; the raw units below
are what the report quotes, since "mean sentence length 18.4 tokens" is a
statement about Bangla prose and "+1.7 sd" is a statement about this corpus."""),
    code("""\
show = ["sadhu_rate", "chalit_rate", "sadhu_chalit_ratio", "sent_len_mean",
        "word_len_mean", "type_token_ratio", "quote_rate", "comma_rate"]

from banglastylo.stylometry import StructuralFeatures, STRUCTURAL_NAMES
S = StructuralFeatures().fit([p.text for p in passages]).transform(
    [p.text for p in passages])
raw = pd.DataFrame(S, columns=STRUCTURAL_NAMES)
raw["author"] = [p.author for p in passages]
means = raw.groupby("author")[show].mean()
means.index = [config.AUTHORS[a]["en"] for a in means.index]
means.round(3)"""),
    md("""\
## 5.3 What each author's SVM row weights

Corpus-wide, not per passage: the features whose presence most pushes a
passage *towards* each author."""),
    code("""\
top = interpret.top_features_per_author(pipe, top_k=10)
from banglastylo.interface import gloss
top["reads as"] = top["feature"].map(gloss)
top.to_csv(config.TABLES / "top_features_per_author.csv", index=False)

for a in sorted(top["author"].unique()):
    sub = top[(top.author == a) & (top.direction == "for")].head(8)
    print(f"\\n{config.AUTHORS[a]['en']}")
    for _, r in sub.iterrows():
        print(f"   {r['weight']:+7.3f}  {r['reads as']}")"""),
    md("""\
## 5.4 Auditing single predictions

The two paradigms explain themselves differently, and both explanations are
shown. The discriminative one decomposes `w·x`; the generative one reports the
log-odds each character contributed to the winner over the runner-up."""),
    code("""\
from banglastylo.interface import Attributor, render

attributor = Attributor(svm_pipeline=pipe, generative=gen)

rng = np.random.default_rng(config.SEED)
for i in rng.choice(len(Xte), size=3, replace=False):
    print("#" * 80)
    print(f"TRUE AUTHOR: {config.AUTHORS[yte[i]]['en']}")
    print("PASSAGE:", Xte[i][:220].replace("\\n", " "), "…")
    print(render(attributor.predict(Xte[i]), attributor))
    print()"""),
    md("""\
### A worked example in the proposal's own format

The proposal walks through one constructed sentence. Here is the same shape of
output, produced by the real system."""),
    code("""\
from banglastylo.interface import DEMO_PASSAGE
print(render(attributor.predict(DEMO_PASSAGE), attributor))"""),
    md("""\
## 5.5 Where the model fails, and why

Errors are more informative than successes. These are the test passages the
model got wrong with the *highest* confidence — the ones whose style genuinely
points the wrong way."""),
    code("""\
scores = pipe.decision_function(Xte)
pred = pipe.predict(Xte)
classes = list(pipe.named_steps["clf"].classes_)
margin = np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2]

wrong = np.where(pred != np.asarray(yte))[0]
wrong = wrong[np.argsort(-margin[wrong])][:4]
for i in wrong:
    print("=" * 82)
    print(f"true: {config.AUTHORS[yte[i]]['en']:32s} "
          f"predicted: {config.AUTHORS[pred[i]]['en']}  (margin {margin[i]:.2f})")
    print(Xte[i][:260].replace("\\n", " "), "…")
    exp = interpret.linear_contributions(pipe, Xte[i], top_k=6)
    for f, c, v in exp["for_prediction"]:
        print(f"    {c:+7.3f}  {gloss(f)}")
print("=" * 82)"""),
    md("""\
## 5.6 How much text does the system actually need?

Accuracy as a function of passage length. The practical question a user of the
interface will ask — how long does my passage have to be before the answer
means anything?"""),
    code("""\
from sklearn.metrics import accuracy_score
from banglastylo.normalize import word_tokenize

lengths = [40, 60, 90, 130, 180, 220]
rows = []
for L in lengths:
    trunc = [" ".join(word_tokenize(t)[:L]) for t in Xte]
    rows.append({
        "tokens": L,
        "discriminative": accuracy_score(yte, pipe.predict(trunc)),
        "generative": accuracy_score(yte, gen.predict(trunc)),
    })
curve = pd.DataFrame(rows)
curve.to_csv(config.TABLES / "length_curve.csv", index=False)

fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.plot(curve["tokens"], curve["discriminative"], "o-",
        color="#2F5D62", label="discriminative (SVM)")
ax.plot(curve["tokens"], curve["generative"], "s--",
        color="#C9A227", label="generative (char LM)")
ax.axhline(1/len(set(yte)), ls=":", color="#B04A3F", label="chance")
ax.set_xlabel("passage length given to the model (tokens)")
ax.set_ylabel("test accuracy")
ax.set_title("How much text does attribution need?", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(config.FIGURES / "length_curve.png", dpi=200)
fig.savefig(config.REPORT_FIGURES / "length_curve.png", dpi=200)
plt.show()
curve"""),
]

# ===========================================================================
# 06 — interface demo
# ===========================================================================
nb06 = [
    md("""\
# 06 · The interface

> *"A minimal input–output interface accepts a pasted Bangla passage and
> returns the predicted author alongside the ranked stylistic features that
> most influenced the decision, so predictions remain auditable rather than
> opaque."* — project proposal, §5

Two front ends over the same `Attributor`: this notebook (with a text box if
`ipywidgets` is available) and `app.py` on the command line."""),
    code(PREAMBLE),
    code("""\
from banglastylo.interface import Attributor, render, DEMO_PASSAGE

attributor = Attributor.load()
print("loaded models for:",
      ", ".join(config.AUTHORS[a]["en"]
                for a in attributor.svm.named_steps["clf"].classes_))"""),
    md("## 6.1 On the built-in demo passage"),
    code("""print(render(attributor.predict(DEMO_PASSAGE), attributor))"""),
    md("""\
## 6.2 Interactive

Type or paste a Bangla passage and press **Attribute**. Longer is better —
see the length curve in notebook 05; below about 80 tokens the estimate is
noisy and the interface says so."""),
    code("""\
try:
    import ipywidgets as W
    from IPython.display import display, clear_output

    box = W.Textarea(value=DEMO_PASSAGE, placeholder="বাংলা অনুচ্ছেদ…",
                     layout=W.Layout(width="100%", height="150px"))
    button = W.Button(description="Attribute", button_style="success")
    out = W.Output()

    def on_click(_):
        with out:
            clear_output()
            text = box.value.strip()
            if not text:
                print("paste a passage first")
                return
            print(render(attributor.predict(text), attributor))

    button.on_click(on_click)
    display(W.VBox([box, button, out]))
    on_click(None)
except ImportError:
    print("ipywidgets not installed — use the cell below instead")"""),
    code("""\
passage = DEMO_PASSAGE  # <-- paste your own passage here
print(render(attributor.predict(passage), attributor))"""),
    md("""\
## 6.3 The same thing from the command line

```
.venv/Scripts/python.exe app.py --demo
.venv/Scripts/python.exe app.py --text "…বাংলা অনুচ্ছেদ…"
.venv/Scripts/python.exe app.py --file passage.txt --json
```"""),
    md("""\
## 6.4 The structural profile of a passage

Useful when the two paradigms disagree: the raw numbers behind the
attribution, before any model has weighed in."""),
    code("""\
prof = attributor.profile(passage)
from banglastylo.interface import FEATURE_GLOSS
pd.DataFrame({
    "feature": [FEATURE_GLOSS.get(k, k.replace("_", " ")) for k in prof],
    "value": list(prof.values()),
}).set_index("feature").round(3)"""),
]


if __name__ == "__main__":
    write("01_corpus_construction.ipynb", nb01)
    write("02_splits_and_pos_tagger.ipynb", nb02)
    write("03_embeddings.ipynb", nb03)
    write("04_models_and_evaluation.ipynb", nb04)
    write("05_interpretability.ipynb", nb05)
    write("06_interface.ipynb", nb06)
