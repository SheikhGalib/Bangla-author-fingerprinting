# Walkthrough — how to read this project

*Authorship Attribution and Style Fingerprinting in Bangla Literature*
KUET CSE 4122 · Abu Daud Sharif (2107002), Sheikh Md. Galib Mahim (2107020)

This document is for someone who has the repository open and wants to know what
each piece is for, what each number means, and — most importantly — **which
numbers to distrust**. The report (`report.pdf`) argues the findings; this
explains how to interpret them.

---

## 1. The one-paragraph version

Give the system a Bangla passage; it names which of five authors wrote it, and
shows why. It does this two ways at once: a **generative** method (one language
model per author, pick the one least surprised by the passage) and a
**discriminative** method (one classifier trained to tell the five apart). The
generative method wins, 0.834 to 0.795. The whole thing rests on a corpus split
by *book*, not by passage — a distinction worth 20.7 accuracy points.

---

## 2. The mental model

```
Wikisource                                            "Who wrote this?"
    │                                                        │
    ▼                                                        ▼
[ crawl ] ──► [ clean ] ──► [ cut into passages ] ──► [ WORK-DISJOINT SPLIT ]
                                                             │
                                    ┌────────────────────────┴───────────┐
                                    ▼                                    ▼
                        GENERATIVE                            DISCRIMINATIVE
                   one LM per author                   one classifier for all
                   "does this look like               "what separates these
                    Bankim's text?"                     five from each other?"
                                    │                                    │
                                    └────────────────┬───────────────────┘
                                                     ▼
                                      predicted author + ranked evidence
```

Everything to the right of the split is only meaningful *because* of the split.
That is the single most important thing to understand about this project.

---

## 3. Why the split is the whole ballgame

Suppose you cut *Devdas* into 200-word passages and randomly put 150 in train
and 50 in test. A classifier sees "Parvati", "Chandramukhi" and the village
name in training and again at test. It scores 99%. It has learned **which novel
a passage came from**, and you have written that down as a style result.

We measured it. Same model, same features, same text:

| Split | Accuracy |
|---|---|
| Random, passage-level | **0.993** |
| Work-disjoint (every book wholly in one split) | **0.787** |
| **Difference** | **+0.207** |

**How to use this:** when you read *any* authorship-attribution accuracy — ours,
a paper's, a blog post's — the first question is which regime produced it. A
number above ~0.95 on a small literary corpus usually means passage-level
splitting. The gap is bigger than the gap between any two systems in our own
results table, so it dominates every comparison it touches.

Notebook 02 §2.3 runs this experiment. `splits.leakage_report()` asserts the
invariant rather than trusting it.

---

## 4. The corpus, and what to be suspicious of

**Five authors, 2,955 passages, 42 works, 682,607 tokens**, balanced at 591
passages each.

| Author | Period | Works | Character |
|---|---|---|---|
| Ishwar Chandra Vidyasagar | 1820–1891 | 12 | most *sadhu* (literary register) |
| Bankim Chandra Chattopadhyay | 1838–1894 | 6 | novelist, short sentences |
| Rabindranath Tagore | 1861–1941 | 14 | most heterogeneous material |
| Sarat Chandra Chattopadhyay | 1876–1938 | 5 | novelist, dialogue-heavy |
| Abanindranath Tagore | 1871–1951 | 5 | most *chalit* (colloquial) |

### What to trust
- **Text quality is high.** Only Wikisource pages the community marked
  *proofread* were used (7,158 of 7,480 scanned pages, 95.7%). Unproofread
  Bangla OCR is frequently garbled into Devanagari look-alikes and would have
  poisoned everything.
- **No duplicate editions.** An 8-gram shingle test removes near-duplicates;
  Wikisource hosts multiple printings of the same novel.

### What to be suspicious of
- **Prose only, by accident.** Verse lines are short, so poetry books fell out
  of the ≥30-character line filter. It happens to be the right control — you
  cannot compare a poet's verse to a novelist's prose and call the difference
  authorship — but nobody designed it that way. It was noticed afterwards.
- **Editorial style is inside "style".** These are specific printed editions.
  A publisher's house punctuation is in the signal and cannot be separated out.
- **Three authors were dropped**, and one for an instructive reason. Wikisource
  lists *বৌদ্ধগান ও দোহা* under Haraprasad Shastri, but that is his *edition* of
  the Charyapada — 10th-century Old Bengali. An author page links what a person
  edited, not only what they wrote. It is excluded by name in `config.py`.

---

## 5. Reading the results table

`artifacts/tables/results_main.csv`, Table 1 in the report. Chance is 0.200.

| Rank | Model | Accuracy |
|---|---|---|
| 1 | **Generative char-5-gram LM** | **0.834** |
| 2 | SVM · structural only *(29 features)* | 0.801 |
| 2 | SVM · stylometric + BanglaBERT | 0.801 |
| 5 | SVM · all stylometric | 0.795 |
| 7 | SVM · char n-grams only *(3000 features)* | 0.778 |
| 8 | Fine-tuned BanglaBERT *(110M params)* | 0.775 |
| 14 | SVM · POS n-grams only | 0.611 |

### How to read the intervals
Every row carries a 95% bootstrap interval, e.g. 0.834 [0.806, 0.861]. **Rows
whose intervals overlap heavily are not distinguishable.** Ranks 2–6 (0.795 to
0.801) are one cluster, not five findings. The generative model's lead over the
SVM cluster is at the edge of significance; its lead over POS-only is not in
doubt.

### The three real findings

**1. Generative beats discriminative (0.834 vs 0.795).** Counterintuitive,
because the generative model trains under a handicap — each author's LM never
sees the other authors. Why it still wins: it scores every one of ~1,200
characters in the passage, while the SVM first compresses the passage into
feature counts and throws away whatever that projection does not preserve.
Confirming evidence: the *word*-level generative model scores only 0.768. The
six-point gap is the cost of not seeing inside the word, in a language where
register and case live in suffixes.

**2. Twenty-nine hand-designed numbers beat a 110M-parameter transformer**
(0.801 vs 0.775). Read this carefully — see §8 on what it does *not* mean.

**3. From-scratch embeddings hold more style per unit of topic.** See §6.

---

## 6. Reading the style-vs-topic diagnostic

Downstream accuracy conflates the representation with the classifier, so there
is a direct measurement: cluster the same test vectors twice, once labelled by
**author**, once by **individual work**.

| Representation | by author | by work | ratio |
|---|---|---|---|
| Scratch skip-gram (SIF) | 0.264 | 0.320 | 0.82 |
| Scratch skip-gram (mean) | 0.255 | 0.320 | 0.80 |
| Frozen BanglaBERT | 0.143 | 0.201 | 0.71 |

**How to read it.** The from-scratch model, trained on 1,821 passages of this
one corpus, separates authors *nearly twice as well* as BanglaBERT, which was
pretrained on ~27 GB of Bangla. That is the proposal's hypothesis confirmed.

**How not to over-read it.** Two things:

- **Every ratio is below 1.** All three representations separate *books* better
  than *authors*. Embedding-space clustering remains substantially topical no
  matter how it was obtained. This is not a clean win for anybody; it is a
  reason the work-disjoint split exists.
- **BanglaBERT was pretrained on modern Bangla.** Our corpus is 19th-century
  literary prose in a register that pretraining data barely contains. This is
  evidence about *that domain gap*, not a general verdict on pretraining.

---

## 7. Reading the interpretability outputs

There are three, and they answer different questions. Confusing them is the
most likely misreading of this project.

### 7a. Ablation table — "what can this family do alone?"
Train on one family only. Structural alone: 0.801. Char n-grams alone: 0.778.

### 7b. Permutation importance — "what does this family add, given the others?"
Shuffle one family's columns in the *combined* model. Char n-grams: −0.570.
Structural: −0.008.

**These look contradictory and are not.** Structural features are strong alone
and worth nothing on the margin, because character n-grams already encode
most of the same information in a different form. Permutation importance under
redundancy credits the larger, more expressive family and zeroes the smaller
one. **Quoting only the permutation numbers would misrepresent the model** — it
would suggest the interpretable features are useless, when in fact they are the
strongest single family in isolation.

### 7c. Style fingerprints — "what does each author look like?"
`artifacts/tables/style_fingerprints.csv`, per author, in corpus standard
deviations. The column to read first is `sadhu_chalit_ratio`.

Bangla prose of this period is written in one of two registers:

| | *sadhu* (সাধু, literary) | *chalit* (চলিত, colloquial) |
|---|---|---|
| "it happened" | হইল | হল |
| "having done" | করিয়া | করে |
| "his/her" | তাহার | তার |

Vidyasagar sits furthest towards sadhu (+0.65 sd), Abanindranath Tagore
furthest towards chalit (−0.81 sd). **Neither fact was supplied to the model**
— it is recovered from the text, and it matches what literary history says.
That agreement is the best available evidence that the feature is measuring
something real rather than an artefact.

### 7d. Single-prediction audit
For a linear SVM, feature *i* contributes exactly `w[c,i] × x[i]`. This is the
same quantity SHAP estimates for a linear model, computed exactly rather than
sampled — which is why there is no SHAP dependency.

The interface prints **two** rankings from that decomposition:
- *strongest evidence overall* — character n-grams take nearly every slot;
- *strongest evidence you can check by eye* — function words, structural
  statistics and POS patterns only.

The second list exists because "the trigram ায়ে" is not something a reader can
verify against the passage, whereas "this text is in the sadhu register" is.
**Use the second list when judging whether the model is right for the right
reason.**

---

## 8. What these results do *not* license

Being precise here matters more than the headline numbers.

**"Transformers are bad at Bangla stylometry."** No. Our fine-tuned BanglaBERT
ran **2 epochs at 192 tokens on CPU**, about 40 minutes. That is a
compute-constrained lower bound. A longer schedule on a GPU with 512-token
inputs would very likely score higher. What the comparison supports is: *a
transformer did not beat 29 hand-designed scalars at this budget.*

**"The generative paradigm is better for authorship attribution."** It is
better *here* — five authors, ~590 passages each, work-disjoint, Bangla prose.
Per-author LMs need per-author data; the paradigm gets harder as authors are
added and each one's slice thins.

**"87.2% POS accuracy means the tagger is good."** It is measured on
`UD_Bengali-BRU`, which is **56 sentences, 320 tokens** — a tiny evaluation
set with wide error bars. It also folds three UD distinctions (AUX→VERB,
PROPN→NOUN, SCONJ→CCONJ), so it is scored on an easier task than a full UD
tagger. Treat it as "roughly right, coarse", which is consistent with POS
n-grams being the weakest feature family (0.611).

**"The system can tell you who wrote an arbitrary Bangla passage."** It is a
**closed set of five**. Given text by a sixth author it will confidently name
one of the five. There is no "none of the above".

**"It works on any passage."** At 40 tokens the discriminative model is at
0.551 and the generative at 0.744. The interface warns below 80 tokens. Below
that, sentence-length and lexical-richness statistics are estimated from two or
three sentences and are mostly noise.

---

## 9. Reading the confusion matrices

The two paradigms fail in *structurally different* ways — this is more
informative than either accuracy.

| | Generative char-5-gram | SVM, all stylometric |
|---|---|---|
| Worst pair | Bankim ↔ R. Tagore (19.4%) | R. Tagore ↔ Vidyasagar (31.1%) |
| R. Tagore ↔ Vidyasagar | 2.4% | 31.1% |

**The SVM's dominant error:** 36% of Tagore's passages are called Vidyasagar.
They sit on the same side of the aggregate structural axes the SVM reads — both
sadhu-leaning, both long-worded — even though their character-level texture is
distinct. The SVM only sees the aggregate.

**The generative model's dominant error:** it over-predicts Tagore. His own
recall is fine (0.83), but a quarter of Bankim's passages and a third of
Sarat's land in his model. Tagore contributes 14 works spanning essays, fiction
and lectures — the most heterogeneous training material of the five — so his
language model is the broadest, and **a broad model wins ambiguous passages by
not being surprised by anything**. This is the standard failure mode of
per-author likelihood attribution.

**Practical consequence:** when the two paradigms disagree, the interface says
so. That disagreement is a real signal, and the generative model is right more
often when they split (84 passages to 56).

---

## 10. Where to look for what

| Question | File |
|---|---|
| How was the corpus built, and what did filtering cost? | `notebooks/01_corpus_construction.ipynb` |
| Is the split honest? How much does that cost? | `notebooks/02_splits_and_pos_tagger.ipynb` §2.1–2.3 |
| How good is the POS tagger, really? | `notebooks/02_...ipynb` §2.4 |
| Pretrained vs from-scratch embeddings | `notebooks/03_embeddings.ipynb` |
| All model comparisons and ablations | `notebooks/04_models_and_evaluation.ipynb` |
| Why did it predict *that*? | `notebooks/05_interpretability.ipynb` |
| Try it on your own passage | `notebooks/06_interface.ipynb` or `app.py` |
| Every number in the report, as LaTeX macros | `reports/numbers.tex` |

Reading order for a reviewer with 20 minutes: report §1 (the leakage box) →
Table 1 → the three Findings boxes → the confusion figure → §10 Limitations.

---

## 11. Things a reader might reasonably challenge

Written down because they are the questions we would ask.

1. **"Char n-grams could still be topic."** True. A character 4-gram can
   fragment a proper noun. Work-disjoint splitting limits this but does not
   eliminate it, since an author's recurring vocabulary spans their books. The
   structural-only result (0.801, which contains no lexical content at all) is
   the cleanest evidence that real style signal exists here.

2. **"Balancing to 591 threw away data."** Yes — Abanindranath had 938
   passages. Balanced classes make accuracy and macro-F₁ directly interpretable
   against a flat 0.200 chance. The discarded passages are still in
   `data/processed/passages.jsonl` if someone wants to redo it unbalanced.

3. **"One random seed."** Correct, seed 20242025 throughout. The bootstrap
   intervals capture test-set sampling noise but *not* seed-to-seed variation in
   training. Differences under ~0.02 should not be read as real.

4. **"The POS tagger was written by the same people who used it."** Yes, and
   that is why it is scored against an external gold standard rather than
   self-assessed. `UD_Bengali-BRU` is used for evaluation precisely because it
   is too small to train on — there was no way to tune towards it.

5. **"Why not just use a bigger LLM?"** A fair question, and out of scope here:
   the proposal specified this comparison, and the environment is CPU-only. It
   would also not answer the question the project asks, which is about *which
   signal* carries authorship, not about maximising a leaderboard number.
