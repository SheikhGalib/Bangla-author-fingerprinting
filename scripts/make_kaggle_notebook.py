"""Generate the single presentation notebook that runs on Kaggle.

    .venv/Scripts/python.exe scripts/make_kaggle_notebook.py

Why generated rather than hand-written: a notebook is JSON, and hand-editing
JSON with embedded Bengali and embedded Python is how cells quietly get
corrupted.  Writing it from a list of (markdown, code) pairs keeps the source
reviewable and makes regeneration free.

The notebook is written for a reader who will be *walked through it out loud*.
Every code cell is preceded by plain-English prose explaining what it is about
to do and why, and prints something small and legible rather than a wall of
numbers.  It is one continuous story: corpus, the trap we found, the split,
what the text looks like as numbers, four models, and what the comparison
means.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

OUT = ROOT / "kaggle" / "bangla_authorship_complete.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ===========================================================================
md(r"""
# Bangla Authorship Attribution — who wrote this passage?

**KUET CSE 4122 · NLP Laboratory** · Abu Daud Sharif (2107002), Sheikh Md. Galib Mahim (2107020)

---

## The question

Given a paragraph of Bangla prose, can a machine tell **which of three authors wrote it**?

| | Author | Lived |
|---|---|---|
| ১ | রবীন্দ্রনাথ ঠাকুর — Rabindranath Tagore | 1861–1941 |
| ২ | কাজী নজরুল ইসলাম — Kazi Nazrul Islam | 1899–1976 |
| ৩ | হুমায়ূন আহমেদ — Humayun Ahmed | 1948–2012 |

## The honest part

Anyone can get a high score on this task by cheating accidentally. If paragraphs from the
*same novel* appear in both training and testing, the computer just memorises the character
names and wins. That is **topic detection**, not style detection.

So we did this instead: for each author we picked **3 books**, trained on **2**, and
**locked the third away before downloading anything**. That third book is the exam paper.
Later in this notebook we show what happens if you *don't* do this — the score goes to a
suspicious **100%**.

## What this notebook contains

1. The corpus, book by book
2. A trap we found in it — and how we caught it
3. Splitting the data honestly
4. What Bangla text looks like as numbers
5. Four models, from simple counting to a pretrained transformer
6. Confusion matrices for each
7. The leakage experiment
8. What it all means
""")

code(r"""
# Standard setup. Everything is seeded so this notebook reproduces exactly.
import json, math, os, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 20242025
random.seed(SEED); np.random.seed(SEED)
pd.set_option("display.width", 130)
plt.rcParams["figure.dpi"] = 110

# Bengali needs a font that can draw it, or every label becomes a box.
import matplotlib.font_manager as fm
_bn = [f for f in fm.findSystemFonts() if re.search(r"(NotoSansBengali|Nirmala|Lohit|Mukti)", f, re.I)]
if _bn:
    fm.fontManager.addfont(_bn[0])
    plt.rcParams["font.family"] = fm.FontProperties(fname=_bn[0]).get_name()
    print("Bengali font:", plt.rcParams["font.family"])
else:
    print("No Bengali font found — Bengali labels in plots will show as boxes.")
    print("(All the numbers are still correct; we use English labels in plots.)")

print("seed:", SEED)
""")

# ---------------------------------------------------------------- 1. corpus
md(r"""
---
# 1. The corpus

The data arrives as a Kaggle dataset: passages already cut from the books, plus the
train/validation/test assignment decided on our own machine. The notebook does **not**
re-split anything — it uses exactly the split the rest of the project used, so every number
here is comparable with the report.
""")

code(r"""
def find_input():
    '''Find the attached dataset. Kaggle does not always mount it at its slug.'''
    root = Path("/kaggle/input")
    for c in sorted(root.rglob("passages.jsonl")):
        return c.parent
    raise SystemExit(f"dataset not found; mounted: {[str(p) for p in root.glob('*')]}")

IN = find_input()
sys.path.insert(0, str(IN / "src"))

# Use the project's own Bangla tokeniser, not str.split(). It separates
# punctuation and folds digits to a placeholder. The BiLSTM below must see
# exactly the tokens the rest of the project uses, or its score here will
# not match the score in the report.
from banglastylo.normalize import word_tokenize

print("input:", IN)
print("tokeniser check:", word_tokenize("সে বলিল, আজ ৫টা বাজে।"))

passages = [json.loads(l) for l in (IN / "passages.jsonl").open(encoding="utf-8") if l.strip()]
assign   = json.loads((IN / "split.json").read_text(encoding="utf-8"))
by_id    = {p["passage_id"]: p for p in passages}

AUTHORS = {
    "tagore":  {"bn": "রবীন্দ্রনাথ ঠাকুর", "en": "Rabindranath Tagore"},
    "nazrul":  {"bn": "কাজী নজরুল ইসলাম",  "en": "Kazi Nazrul Islam"},
    "humayun": {"bn": "হুমায়ূন আহমেদ",      "en": "Humayun Ahmed"},
}

print(f"{len(passages):,} passages total")
""")

md(r"""
### What a "passage" is

We cut each book into chunks of about **120 words**, always ending on a sentence boundary
and never running across a chapter break. 120 words is roughly one paragraph — which is
what someone would realistically paste into a demo.
""")

code(r"""
rows = []
for p in passages:
    rows.append({"author": p["author"], "book": p["work"], "role": p["role"],
                 "chapter": p["chapter"], "tokens": p["n_tokens"], "sents": p["n_sentences"]})
df = pd.DataFrame(rows)

book_table = (df.groupby(["author", "role", "book"])
                .agg(chapters=("chapter", "nunique"),
                     passages=("tokens", "size"),
                     words=("tokens", "sum"))
                .reset_index()
                .sort_values(["author", "role"]))
book_table["author"] = book_table["author"].map(lambda k: AUTHORS[k]["en"])
print("THE NINE BOOKS  (role 'unseen' = the locked-away exam paper)\n")
display(book_table)
""")

md(r"""
### The three authors do not write alike

Before any machine learning, look at the raw style numbers. Sentence length alone nearly
separates the three: Tagore writes long literary sentences, Humayun Ahmed writes short
dialogue-driven ones.
""")

code(r"""
style = (df.groupby("author")
           .agg(passages=("tokens", "size"),
                words=("tokens", "sum"),
                avg_words_per_sentence=("tokens", "sum"))
           .reset_index())
# words per sentence = total words / total sentences
tot = df.groupby("author").agg(w=("tokens", "sum"), s=("sents", "sum"))
style["avg_words_per_sentence"] = (tot["w"] / tot["s"]).round(1).values
style["author"] = style["author"].map(lambda k: AUTHORS[k]["en"])
display(style)

fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(style["author"], style["avg_words_per_sentence"], color=["#2F5D62", "#7A9E9F", "#B04A3F"])
ax.set_xlabel("average words per sentence")
ax.set_title("Sentence length separates the three authors on its own")
for i, v in enumerate(style["avg_words_per_sentence"]):
    ax.text(v + 0.2, i, str(v), va="center", fontsize=9)
plt.tight_layout(); plt.show()
""")

# --------------------------------------------------------------- 2. the trap
md(r"""
---
# 2. The trap we found — and why it matters

Bengali publishers **reprint short stories in later collections**. Our held-out Tagore book,
রহস্য সমগ্র, turned out to be an anthology that reprints stories also in গল্পগুচ্ছ — one of
the books we were *training* on.

If we had missed this, the "unseen" exam paper would have contained pages the model had
already memorised word for word, and every Tagore number in the project would have been
worthless.

**Checking the titles is not enough** — the same story appears with different punctuation.
So we compare the *text*: chop each chapter into overlapping 10-word windows, hash them, and
ask what fraction of one chapter's windows also appear in the other.
""")

code(r"""
def shingles(text, n=10, stride=1):
    '''Set of hashed 10-word windows.

    stride=1 is essential. At stride 5 the two printings fall out of step — one extra
    word early on shifts every later window — and two copies of the SAME story scored
    only 0.10-0.32, slipping under the threshold.
    '''
    w = text.split()
    return {hash(" ".join(w[i:i+n])) for i in range(0, max(1, len(w)-n), stride)}

def containment(a, b):
    '''Fraction of a that also appears in b. Not Jaccard: a short story inside a big
    anthology has a low Jaccard score just because the anthology is bigger.'''
    return len(a & b) / len(a) if a else 0.0

# Rebuild per-chapter text for Tagore's two relevant books
chap_text = defaultdict(list)
for p in passages:
    chap_text[(p["work"], p["chapter"])].append(p["text"])
chap_text = {k: " ".join(v) for k, v in chap_text.items()}

train_ch = {k: v for k, v in chap_text.items() if k[0] == "গল্পগুচ্ছ"}
unseen_ch = {k: v for k, v in chap_text.items() if k[0] == "রহস্য সমগ্র"}
un_sh = {k: shingles(v) for k, v in unseen_ch.items()}

scores = []
for k, v in train_ch.items():
    a = shingles(v)
    best = max(((containment(a, b), k2[1]) for k2, b in un_sh.items()), default=(0, ""))
    scores.append((best[0], k[1], best[1]))
scores.sort(reverse=True)

print("Training chapter  ->  closest held-out chapter      containment")
print("-" * 72)
for s, t, u in scores[:8]:
    print(f"{t[:28]:30s} -> {u[:24]:26s} {s:.3f}")
print(f"\nlowest of all {len(scores)} training chapters: {scores[-1][0]:.3f}")
""")

md(r"""
**Look at the gap.** The reprinted stories score 0.6–1.0; everything else scores 0.000.
There is nothing in between, so the threshold is not a delicate judgement call.

Those chapters were removed **from the training side**. We never edit the held-out book —
the promise that you can open any page of it and know the model hasn't seen it is worth
more than the training text it costs.

> The passages in this notebook are already de-overlapped, which is why the scores above
> are computed on what survived.
""")

# ---------------------------------------------------------------- 3. split
md(r"""
---
# 3. Splitting the data honestly

- **Train** — passages from the 6 training books.
- **Validation** — held out from the training books *by whole chapter*, so we never tune on
  half a chapter whose other half was trained on.
- **Test** — the 3 locked-away books, balanced to the same count per author so that random
  guessing scores exactly 1/3.
""")

code(r"""
split = {k: [by_id[i] for i in v if i in by_id] for k, v in assign.items()}
for name, part in split.items():
    print(f"{name:6s} {len(part):5d} passages")

comp = pd.DataFrame([
    {"author": AUTHORS[a]["en"],
     "train": sum(1 for p in split["train"] if p["author"] == a),
     "val":   sum(1 for p in split["val"]   if p["author"] == a),
     "test":  sum(1 for p in split["test"]  if p["author"] == a),
     "test book": next((p["work"] for p in split["test"] if p["author"] == a), "")}
    for a in AUTHORS])
display(comp)

# The guarantee, verified rather than asserted:
tr_books = {p["work"] for p in split["train"]} | {p["work"] for p in split["val"]}
te_books = {p["work"] for p in split["test"]}
print("books shared between train/val and test:", tr_books & te_books, "<- must be empty")

Xtr = [p["text"] for p in split["train"]]; ytr = [p["author"] for p in split["train"]]
Xva = [p["text"] for p in split["val"]];   yva = [p["author"] for p in split["val"]]
Xte = [p["text"] for p in split["test"]];  yte = [p["author"] for p in split["test"]]
CLASSES = sorted(set(ytr))
print("\nchance accuracy =", round(1/len(CLASSES), 4))
""")

# ------------------------------------------------------- 4. representations
md(r"""
---
# 4. Turning Bangla text into numbers

A classifier cannot read. Everything below is a different answer to *"what numbers should
represent this paragraph?"* — and the choice matters more than the classifier on top.

### 4a. Bag of Words — just count the words
Order is thrown away completely. "রহিম করিম কে মারল" and "করিম রহিম কে মারল" look identical.
""")

code(r"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sample = Xte[0][:220]
print("A passage from a held-out book:\n")
print(sample, "...\n")

cv_demo = CountVectorizer(token_pattern=r"\S+")
bow = cv_demo.fit_transform([sample])
vocab = cv_demo.get_feature_names_out()
counts = bow.toarray()[0]
top = sorted(zip(counts, vocab), reverse=True)[:10]
print("Bag of Words (top 10 by count):")
for c, w in top:
    print(f"   {w:20s} {c}")
""")

md(r"""
### 4b. TF-IDF — count, but weight rare words higher

A word used by everybody (আর, এবং) tells you nothing. A word used by *one* author often and
the others rarely is gold. TF-IDF does exactly that arithmetic:

$$\text{tf-idf}(w, d) = \underbrace{\frac{\text{count of } w \text{ in } d}{\text{words in } d}}_{\text{common here?}} \times \underbrace{\log\frac{\text{number of documents}}{\text{documents containing } w}}_{\text{rare elsewhere?}}$$
""")

code(r"""
tfidf_demo = TfidfVectorizer(token_pattern=r"\S+", max_features=20000)
M = tfidf_demo.fit_transform(Xtr)
feat = np.array(tfidf_demo.get_feature_names_out())

print("Highest average TF-IDF word per author (i.e. most characteristic):\n")
for a in CLASSES:
    idx = [i for i, y in enumerate(ytr) if y == a]
    mean = np.asarray(M[idx].mean(axis=0)).ravel()
    best = mean.argsort()[::-1][:8]
    print(f"{AUTHORS[a]['en']:22s} " + "  ".join(feat[b] for b in best))
""")

md(r"""
### 4c. N-grams — small windows of order

A **bigram** is two adjacent words, a **trigram** three. They recover a little of the word
order that Bag-of-Words throws away. **Character** n-grams are especially useful for Bangla,
because Bangla marks tense and case with *suffixes* — so a character window can see that
করিয়াছিলেন and করেছিলেন are the same verb in two registers, which a word model cannot.
""")

code(r"""
words = sample.split()
print("word bigrams :", ["  ".join(g) for g in zip(words, words[1:])][:6])
print("word trigrams:", ["  ".join(g) for g in zip(words, words[1:], words[2:])][:4])

s = re.sub(r"\s+", " ", sample)
tri = Counter(s[i:i+3] for i in range(len(s)-2))
print("\nmost common character 3-grams (␣ = space):")
print("   " + "   ".join(f"'{g.replace(' ', '␣')}'×{n}" for g, n in tri.most_common(10)))
""")

# --------------------------------------------------------- 5. helper: eval
md(r"""
---
# 5. Four models

We compare four, arranged so each one can do something the previous cannot:

| # | Model | What it adds |
|---|---|---|
| 1 | Naive Bayes | counts words |
| 2 | TF-IDF + SVM | + weights informative words |
| 3 | BiLSTM | + reads word **order** |
| 4 | BanglaBERT | + arrives already knowing Bangla |

First, one shared scoring function so all four are judged identically.
""")

code(r"""
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

RESULTS = {}

def evaluate(name, y_true, y_pred, note=""):
    '''Score a model, draw its confusion matrix, and remember the result.'''
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    RESULTS[name] = {"accuracy": acc, "macro_f1": f1, "note": note,
                     "y_pred": list(y_pred)}

    print(f"=== {name} ===")
    print(f"accuracy  {acc:.4f}     macro F1  {f1:.4f}     (chance {1/len(CLASSES):.3f})\n")
    print(classification_report(y_true, y_pred,
                                target_names=[AUTHORS[c]["en"] for c in CLASSES], digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    names = [AUTHORS[c]["en"].split()[-1] for c in CLASSES]
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true author")
    ax.set_title(f"{name}\nacc {acc:.3f}", fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.colorbar(im, fraction=0.046); plt.tight_layout(); plt.show()
    return acc
""")

# -------------------------------------------------------------- model 1: NB
md(r"""
## Model 1 — Naive Bayes, written from scratch

The oldest idea here, and still competitive. It asks: **under which author's word-frequency
model is this paragraph most likely?** Then Bayes' rule turns that around into
"which author most likely wrote it".

Two details that matter:

- **Everything in log space.** Multiplying a few hundred probabilities underflows to zero in
  floating point long before the comparison matters, so we add logs instead of multiplying.
- **Add-α smoothing.** A word an author never used has probability zero, which would veto
  that author on a *single* unseen word. Adding a small α to every count prevents that veto.
""")

code(r"""
class MultinomialNaiveBayes:
    '''P(author | words) via Bayes' rule, in log space.'''

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # log P(author) — the prior, straight from how many passages each wrote
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        self.log_prior_ = np.log(counts / counts.sum())

        # log P(word | author) — total count of each word per author, smoothed
        feat = np.zeros((len(self.classes_), X.shape[1]))
        for i, c in enumerate(self.classes_):
            feat[i] = np.asarray(X[y == c].sum(axis=0)).ravel()
        feat += self.alpha
        self.log_lik_ = np.log(feat / feat.sum(axis=1, keepdims=True))
        return self

    def predict(self, X):
        # score = log P(author) + Σ count(word) · log P(word | author)
        return self.classes_[(X @ self.log_lik_.T + self.log_prior_).argmax(axis=1)]

cv = CountVectorizer(token_pattern=r"\S+", max_features=20000)
Ktr, Kte = cv.fit_transform(Xtr), cv.transform(Xte)   # fitted on TRAIN only
print("vocabulary:", Ktr.shape[1], "words\n")

nb = MultinomialNaiveBayes(alpha=0.2).fit(Ktr, ytr)
evaluate("Naive Bayes (word counts)", yte, nb.predict(Kte))
""")

md(r"""
### What Naive Bayes learned — the actual evidence

Because the model is just log-probabilities per word, we can ask it directly: *which words
push hardest toward each author?* This is the model's reasoning, not an approximation of it.
""")

code(r"""
vocab = np.array(cv.get_feature_names_out())
for i, c in enumerate(nb.classes_):
    others = [j for j in range(len(nb.classes_)) if j != i]
    edge = nb.log_lik_[i] - nb.log_lik_[others].max(axis=0)   # log-odds vs best rival
    frequent = np.asarray(Ktr.sum(axis=0)).ravel() > 40       # ignore rare noise
    edge = np.where(frequent, edge, -np.inf)
    print(f"{AUTHORS[c]['en']:22s} " + "  ".join(vocab[k] for k in edge.argsort()[::-1][:9]))
""")

# ------------------------------------------------------------ model 2: SVM
md(r"""
## Model 2 — TF-IDF + linear SVM

Same words, but weighted by TF-IDF, and separated by a **maximum-margin boundary** instead
of by Bayes' rule. An SVM draws the dividing line that leaves the widest possible gap
between the classes.

This model is also our **topic control**. Word unigrams are the most topic-sensitive
representation available, so whatever it scores tells us how much of this task can be solved
by vocabulary alone — a number we take seriously in the conclusions.
""")

code(r"""
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

svm = make_pipeline(
    TfidfVectorizer(token_pattern=r"\S+", max_features=20000),
    LinearSVC(random_state=SEED, dual="auto", max_iter=5000),
).fit(Xtr, ytr)

evaluate("TF-IDF + SVM", yte, svm.predict(Xte))
""")

# --------------------------------------------------------- model 3: BiLSTM
md(r"""
## Model 3 — BiLSTM, trained from scratch

The first model that can see **word order**. An LSTM reads the sentence one word at a time,
carrying a memory; *bi*-directional means we read it forwards and backwards and join the two.

### The architecture

```
   পাসেজ:  [ w₁   w₂   w₃  ...  wₙ ]
              ↓    ↓    ↓        ↓
  Embedding  every word → a 200-number vector        (learned from scratch)
              ↓    ↓    ↓        ↓
      LSTM →  ─────────────────────→   forward memory
      LSTM ←  ←─────────────────────   backward memory
              ↓    ↓    ↓        ↓
    Pooling   average + maximum over all positions   (padding masked out)
                          ↓
      Head    Linear(768 → 3)  →  one score per author
```

**Why mask the padding?** Passages differ in length, so short ones get filled with blanks to
make a rectangular batch. If we average over the blanks too, a short passage has its signal
watered down in proportion to how short it is. Masking excludes them.

**Why average+max instead of the last state?** An author's habits are spread through the
whole paragraph. The final LSTM state is dominated by the last clause — the one place the
style signal is *not* concentrated.
""")

code(r"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
torch.manual_seed(SEED)

PAD, UNK, MAXLEN = 0, 1, 256

def build_vocab(texts, max_size=30000, min_count=2):
    c = Counter(w for t in texts for w in word_tokenize(t))
    v = {"<pad>": PAD, "<unk>": UNK}
    for w, n in c.most_common(max_size - 2):
        if n < min_count: break
        v[w] = len(v)
    return v

def encode(texts, vocab, maxlen=MAXLEN):
    out = np.full((len(texts), maxlen), PAD, dtype=np.int64)
    for r, t in enumerate(texts):
        ids = [vocab.get(w, UNK) for w in word_tokenize(t)][:maxlen]
        out[r, :len(ids)] = ids
    return out

# Vocabulary from TRAINING text only — building it over the test books would
# leak which words those books contain.
vocab = build_vocab(Xtr)
print("vocabulary:", len(vocab), "words")

lbl = {a: i for i, a in enumerate(CLASSES)}
tr_x = torch.from_numpy(encode(Xtr, vocab)); tr_y = torch.tensor([lbl[a] for a in ytr])
va_x = torch.from_numpy(encode(Xva, vocab)); va_y = torch.tensor([lbl[a] for a in yva])
te_x = torch.from_numpy(encode(Xte, vocab))
""")

code(r"""
class BiLSTM(nn.Module):
    def __init__(self, n_vocab, n_classes, emb=200, hidden=192, layers=2, dropout=0.3):
        super().__init__()
        self.emb  = nn.Embedding(n_vocab, emb, padding_idx=PAD)
        # Initialise the table at N(0, 0.1), matching banglastylo.neural.
        # PyTorch's default is N(0, 1) - ten times wider - and on a corpus this
        # small that difference alone is worth several accuracy points.
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.emb.weight[PAD].fill_(0.0)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 4, n_classes)   # ×2 directions ×2 poolings

    def forward(self, x):
        mask = (x != PAD).unsqueeze(-1)              # True where a real word sits
        h, _ = self.lstm(self.emb(x))
        h = h.masked_fill(~mask, 0.0)
        mean = h.sum(1) / mask.sum(1).clamp(min=1)   # average, padding excluded
        mx   = h.masked_fill(~mask, -1e9).max(1).values
        return self.head(self.drop(torch.cat([mean, mx], dim=1)))

model = BiLSTM(len(vocab), len(CLASSES)).to(DEV)
print(model)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\ntrainable parameters: {n_params:,}")
""")

code(r"""
EPOCHS, BATCH = 25, 32
dl = DataLoader(TensorDataset(tr_x, tr_y), batch_size=BATCH, shuffle=True)
opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3,
                                            total_steps=EPOCHS*len(dl), pct_start=0.2)
lossf = nn.CrossEntropyLoss()

def predict_torch(m, X, bs=64):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(m(X[i:i+bs].to(DEV)).cpu().numpy())
    return np.vstack(out)

hist, best, best_state = [], -1, None
t0 = time.time()
for ep in range(EPOCHS):
    model.train(); tot = 0
    for xb, yb in dl:
        xb, yb = xb.to(DEV), yb.to(DEV)
        loss = lossf(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        tot += float(loss) * len(yb)
    va_acc = accuracy_score(yva, [CLASSES[i] for i in predict_torch(model, va_x).argmax(1)])
    hist.append({"epoch": ep+1, "loss": tot/len(tr_y), "val_acc": va_acc})
    if va_acc > best:
        best, best_state = va_acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if (ep+1) % 5 == 0 or ep == 0:
        print(f"epoch {ep+1:3d}/{EPOCHS}  loss {tot/len(tr_y):.4f}  val_acc {va_acc:.4f}")

model.load_state_dict(best_state)
print(f"\ntrained in {time.time()-t0:.0f}s; best validation accuracy {best:.4f}")
""")

code(r"""
h = pd.DataFrame(hist)
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot(h.epoch, h.loss, color="#B04A3F"); ax[0].set_title("training loss"); ax[0].set_xlabel("epoch")
ax[1].plot(h.epoch, h.val_acc, color="#2F5D62"); ax[1].set_title("validation accuracy"); ax[1].set_xlabel("epoch")
ax[1].set_ylim(0, 1.02)
plt.tight_layout(); plt.show()

bilstm_pred = [CLASSES[i] for i in predict_torch(model, te_x).argmax(1)]
evaluate("BiLSTM (from scratch)", yte, bilstm_pred)
""")

md(r"""
> **Notice the gap.** Validation accuracy is high, test accuracy is much lower. That is not a
> bug — validation comes from the *training books* (different chapters), so it measures
> generalising across chapters. The test set is **different books entirely**, which is a far
> harder and much more honest question.
""")

# ----------------------------------------------------- model 4: BanglaBERT
md(r"""
## Model 4 — BanglaBERT, fine-tuned

Everything so far learned Bangla from our six books alone. BanglaBERT arrives having already
read a very large amount of Bangla, and we only adapt it to our task. This is the difference
that matters.

### The architecture, stage by stage

```
 Stage 1  6 training books              (the 3 held-out books leave here…)
              ↓
 Stage 2  passages of ~120 words
              ↓
 Stage 3  WordPiece tokenizer   [CLS] + sub-words + [SEP], padded to 256
              ↓                  sub-words: an unknown word still breaks into known pieces
 Stage 4  BanglaBERT encoder    12 transformer layers, ALREADY PRETRAINED
              ↓                  self-attention: every word looks at every other word
 Stage 5  classification head   the [CLS] vector (768 numbers) → Linear(768 → 3)
              ↓
 Stage 6  weighted cross-entropy + AdamW    (weights because our classes are unbalanced)
              ↓
 Stage 7  keep the best epoch by validation accuracy
              ↓
 Stage 8  TEST on the 3 held-out books      (…and come back only here)
```

**Why class weights?** We deliberately did not throw away Tagore's extra material, so he has
~4× Humayun Ahmed's passages. Without reweighting, the model is rewarded for simply guessing
Tagore more often.
""")

code(r"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "csebuetnlp/banglabert"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CLASSES)).to(DEV)

print("A passage becomes sub-word pieces:\n")
pieces = tok.tokenize(Xte[0][:90])
print("  ", pieces[:22], "...")
print(f"\n  {len(Xte[0][:90].split())} words  ->  {len(pieces)} sub-word tokens")
print("\n('##' marks a piece glued onto the previous one — this is how an unseen")
print(" inflected form still decomposes into pieces the model knows.)")
""")

code(r"""
MAXLEN_B, EPOCHS_B, BATCH_B = 256, 4, 16
torch.manual_seed(SEED)

enc = tok(Xtr, padding="max_length", truncation=True, max_length=MAXLEN_B, return_tensors="pt")
ds  = TensorDataset(enc["input_ids"], enc["attention_mask"],
                    torch.tensor([lbl[a] for a in ytr]))
dlb = DataLoader(ds, batch_size=BATCH_B, shuffle=True)

cnt = np.array([ytr.count(c) for c in CLASSES], dtype=float)
w   = torch.tensor(cnt.sum() / (len(CLASSES) * cnt), dtype=torch.float32, device=DEV)
print("class weights:", {c: round(float(x), 3) for c, x in zip(CLASSES, w)})

opt_b = torch.optim.AdamW(bert.parameters(), lr=2e-5)
sch_b = torch.optim.lr_scheduler.OneCycleLR(opt_b, max_lr=2e-5,
                                            total_steps=EPOCHS_B*len(dlb), pct_start=0.1)
lossb = nn.CrossEntropyLoss(weight=w)

def bert_predict(texts, bs=64):
    bert.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i+bs], padding=True, truncation=True,
                    max_length=MAXLEN_B, return_tensors="pt").to(DEV)
            out.append(bert(**e).logits.cpu().numpy())
    return np.vstack(out)

t0, best_b, best_sb, hist_b = time.time(), -1, None, []
for ep in range(EPOCHS_B):
    bert.train(); tot = 0
    for ids, am, yb in dlb:
        ids, am, yb = ids.to(DEV), am.to(DEV), yb.to(DEV)
        loss = lossb(bert(input_ids=ids, attention_mask=am).logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
        opt_b.step(); sch_b.step(); opt_b.zero_grad()
        tot += float(loss) * len(yb)
    va = accuracy_score(yva, [CLASSES[i] for i in bert_predict(Xva).argmax(1)])
    hist_b.append({"epoch": ep+1, "loss": tot/len(ytr), "val_acc": va})
    print(f"epoch {ep+1}/{EPOCHS_B}  loss {tot/len(ytr):.4f}  val_acc {va:.4f}")
    if va > best_b:
        best_b, best_sb = va, {k: v.cpu().clone() for k, v in bert.state_dict().items()}

bert.load_state_dict(best_sb)
print(f"\nfine-tuned in {(time.time()-t0)/60:.1f} min; best validation {best_b:.4f}")
""")

code(r"""
bert_logits = bert_predict(Xte)
bert_pred = [CLASSES[i] for i in bert_logits.argmax(1)]
evaluate("BanglaBERT (fine-tuned)", yte, bert_pred)
""")

# ------------------------------------------------------------ 6. comparison
md(r"""
---
# 6. Putting the four side by side
""")

code(r"""
comp = (pd.DataFrame([{"model": k, "accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
                      for k, v in RESULTS.items()])
          .sort_values("accuracy", ascending=False).reset_index(drop=True))
display(comp.style.format({"accuracy": "{:.4f}", "macro_f1": "{:.4f}"}))

order = ["Naive Bayes (word counts)", "TF-IDF + SVM",
         "BiLSTM (from scratch)", "BanglaBERT (fine-tuned)"]
order = [m for m in order if m in RESULTS]
vals  = [RESULTS[m]["accuracy"] for m in order]

fig, ax = plt.subplots(figsize=(7.5, 3.4))
bars = ax.bar(range(len(order)), vals,
              color=["#7A9E9F", "#7A9E9F", "#B04A3F", "#2F5D62"])
ax.axhline(1/len(CLASSES), ls="--", color="grey", lw=1)
ax.text(len(order)-0.4, 1/len(CLASSES)+0.012, "chance", color="grey", fontsize=8)
ax.set_xticks(range(len(order)),
              ["Naive Bayes\n(counts words)", "TF-IDF + SVM\n(+ weighting)",
               "BiLSTM\n(+ word order)", "BanglaBERT\n(+ pretraining)"][:len(order)],
              fontsize=8)
ax.set_ylabel("accuracy on held-out books"); ax.set_ylim(0, 1.05)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.3f}", ha="center", fontsize=9, weight="bold")
ax.set_title("Each model adds a capability — but more capability is not more accuracy")
plt.tight_layout(); plt.show()
""")

md(r"""
### Read that chart carefully — it does not do what you expect

1. **Counting words is already nearly enough.** Naive Bayes is within a hair of the SVM.
2. **Adding word order made it WORSE.** The BiLSTM is the weakest of the four. It is the more
   capable architecture, but it starts from random numbers and must learn Bangla *and* the
   authorship task from six books. The bag-of-words models need to learn no language at all.
3. **Pretraining is what actually pays.** BanglaBERT and the BiLSTM are both neural sequence
   models on identical data; the only real difference is that one already knew Bangla.

> **On a small corpus, what a model already knows beats what it is able to represent.**
""")

# ---------------------------------------------------------- 7. the leakage
md(r"""
---
# 7. The experiment that matters most

Now we show what happens if you split the data the easy, wrong way: pool every passage,
shuffle, and cut 75/25. Passages from the *same book* then land on both sides.
""")

code(r"""
allp = split["train"] + split["val"] + split["test"]
idx = list(range(len(allp))); random.Random(SEED).shuffle(idx)
cut = int(0.75 * len(idx))
tr = [allp[i] for i in idx[:cut]]; te = [allp[i] for i in idx[cut:]]

leaky = make_pipeline(
    TfidfVectorizer(token_pattern=r"\S+", max_features=20000),
    LinearSVC(random_state=SEED, dual="auto", max_iter=5000),
).fit([p["text"] for p in tr], [p["author"] for p in tr])

leak_acc = accuracy_score([p["author"] for p in te],
                          leaky.predict([p["text"] for p in te]))
honest_acc = RESULTS["TF-IDF + SVM"]["accuracy"]

print(f"  random passage split (the WRONG way) : {leak_acc:.4f}")
print(f"  book-disjoint split  (our way)       : {honest_acc:.4f}")
print(f"  inflation from leakage               : {leak_acc - honest_acc:+.4f}")

fig, ax = plt.subplots(figsize=(5.5, 2.8))
ax.barh(["Random passage split\n(wrong)", "Held-out books\n(ours)"],
        [leak_acc, honest_acc], color=["#B04A3F", "#2F5D62"])
ax.set_xlim(0, 1.08); ax.set_xlabel("accuracy")
for i, v in enumerate([leak_acc, honest_acc]):
    ax.text(v + 0.012, i, f"{v:.3f}", va="center", weight="bold")
ax.set_title("The same model, scored two ways")
plt.tight_layout(); plt.show()
""")

md(r"""
### A perfect score is a symptom, not an achievement

The wrong protocol gives essentially **100%**. It looks wonderful and means nothing: the
classifier recognises the novel from its character names, then reports that novel's author.

This is the single most important thing in the project. A reported accuracy is only as
meaningful as the split behind it, and **any Bangla attribution result that does not say
which protocol produced it cannot be checked.**
""")

# ----------------------------------------------------- 8. how much text
md(r"""
---
# 8. How much text does the model need?

Practical question for the live demo: if someone pastes in one sentence, should we believe
the answer?
""")

code(r"""
curve = []
for n in (10, 20, 40, 60, 80, 120):
    trimmed, truth = [], []
    for p in split["test"]:
        w = word_tokenize(p["text"])
        if len(w) >= n:
            trimmed.append(" ".join(w[:n])); truth.append(p["author"])
    if len(trimmed) < 30:
        continue
    curve.append({"words shown": n, "n passages": len(trimmed),
                  "TF-IDF+SVM": accuracy_score(truth, svm.predict(trimmed)),
                  "BanglaBERT": accuracy_score(truth, [CLASSES[i] for i in bert_predict(trimmed).argmax(1)])})
cdf = pd.DataFrame(curve)
display(cdf.style.format({"TF-IDF+SVM": "{:.3f}", "BanglaBERT": "{:.3f}"}))

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(cdf["words shown"], cdf["TF-IDF+SVM"], "o-", label="TF-IDF + SVM", color="#7A9E9F")
ax.plot(cdf["words shown"], cdf["BanglaBERT"], "o-", label="BanglaBERT", color="#2F5D62")
ax.axhline(1/len(CLASSES), ls="--", color="grey", lw=1)
ax.set_xlabel("words shown to the model"); ax.set_ylabel("accuracy")
ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Accuracy grows with the amount of text")
plt.tight_layout(); plt.show()
""")

# ------------------------------------------------------------- 9. try it
md(r"""
---
# 9. Try it on anything

Paste any Bangla paragraph below and all the trained models will vote.
""")

code(r"""
def who_wrote(text):
    print("passage:", text[:110].replace("\n", " "), "...\n")
    probs = {}

    k = cv.transform([text])
    probs["Naive Bayes"] = nb.predict(k)[0]
    probs["TF-IDF + SVM"] = svm.predict([text])[0]
    probs["BiLSTM"] = CLASSES[predict_torch(model, torch.from_numpy(encode([text], vocab))).argmax(1)[0]]

    lg = bert_predict([text])[0]
    e = np.exp(lg - lg.max()); p = e / e.sum()
    probs["BanglaBERT"] = CLASSES[int(p.argmax())]

    for m, a in probs.items():
        print(f"   {m:16s} -> {AUTHORS[a]['en']}")
    print("\n   BanglaBERT confidence: " +
          "  ".join(f"{AUTHORS[c]['en'].split()[-1]} {v:.1%}" for c, v in zip(CLASSES, p)))
    if len(set(probs.values())) == 1:
        print("\n   >>> all four agree")
    else:
        print("\n   >>> the models disagree — worth showing, not hiding")

# A passage from a held-out book that nothing was trained on:
who_wrote(Xte[5])
""")

# ------------------------------------------------------------ 10. wrap up
md(r"""
---
# 10. What we found

**1. The evaluation protocol matters more than the model.**
The same classifier scores ~1.00 with a random split and ~0.97 with held-out books. A
protocol that returns a perfect score cannot distinguish a model that learned style from one
that memorised a novel.

**2. Corpora need checking, not trusting.**
One of our held-out books silently reprinted six stories from a training book. Titles did not
reveal it; comparing the text did — and only after we fixed the detection method itself.

**3. Capability is not accuracy.**
Adding word order (BiLSTM) made things worse than plain word counting. Adding external
pretraining (BanglaBERT) made them clearly better. With six books, what the model already
knows matters more than what it can represent.

**4. Simple methods are strong here.**
Naive Bayes — Bayes' rule and a smoothed word count, about twenty lines — lands within three
points of a fine-tuned 110-million-parameter transformer.

### Honest limitations

- **Closed set of three.** Give it a fourth author and it will still confidently name one of
  our three.
- **Period is mixed up with authorship.** Our authors span 1861–2012 and differ in spelling
  convention and register, not only in personal style. Some of the accuracy is the model
  detecting *when* a text was written.
- **Vocabulary does a lot of the work.** The topic-sensitive TF-IDF baseline scores nearly as
  high as everything else, so we cannot claim the models have isolated pure style.
- **The corpus is small.** Six training books. The BiLSTM result is a direct consequence.

### Lab concepts used

Lab 1 regex cleaning, tokenisation, danda-aware sentence splitting · Lab 2 Bag-of-Words,
TF-IDF, n-grams, smoothing · Lab 3 Naive Bayes from scratch, word embeddings ·
Lab 4 BiLSTM, padding masks, pooling · Lab 5 transformers, self-attention, fine-tuning
""")

code(r"""
out = Path("/kaggle/working")
comp.to_csv(out / "notebook_results.csv", index=False)
print("saved:", out / "notebook_results.csv")
display(comp.style.format({"accuracy": "{:.4f}", "macro_f1": "{:.4f}"}))
print("\nDone.")
""")


# ===========================================================================
def build() -> dict:
    cells = []
    for kind, text in CELLS:
        lines = text.split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "source": source,
                          "execution_count": None, "outputs": []})
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    n_md = sum(1 for k, _ in CELLS if k == "markdown")
    n_code = sum(1 for k, _ in CELLS if k == "code")
    print(f"wrote {OUT}")
    print(f"  {n_md} markdown cells, {n_code} code cells")
