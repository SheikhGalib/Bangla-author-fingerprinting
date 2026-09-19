# Lab concept coverage

Where each idea from the CSE 4122 labs appears in this project, with the file
and symbol to open. Concepts that were *deliberately* not used are listed at the
end with the reason, because leaving them out was a decision rather than an
oversight.

The short version: every lab contributes at least one model or transform that
is scored in the results table. Nothing here is a demonstration bolted on to
tick a box — if a technique appears, it competes.

---

## Lab 1 — Text preprocessing and spelling correction

| Lab concept | Where it lives | Notes |
|---|---|---|
| Regex cleaning | `ebangla.chapter_text`, `normalize._EDITORIAL` | Strips HTML, page furniture, wiki templates |
| Tokenization | `normalize.word_tokenize` | Bangla-block aware; digits fold to `<num>` |
| Sentence segmentation | `normalize.sentence_split` | Splits on **danda** `।`, not the full stop |
| Stop-word handling | `stylometry.FunctionWordFeatures` | **Inverted**: function words are the signal, not noise |
| Edit distance (Levenshtein) | `classifiers.edit_distance` | Two-row DP |
| Spelling normalisation | `classifiers.nearest_vocabulary_word` | Maps near-miss spellings onto the vocabulary |

The stop-word inversion is the point worth making to a reader. Topic
classification deletes function words because they carry no topic. Authorship
attribution keeps them for exactly that reason: a word that says nothing about
*what* a passage is about is free to say something about *who* wrote it.

## Lab 2 — Text representation and n-gram language modelling

| Lab concept | Where it lives | Notes |
|---|---|---|
| Bag of Words (counts) | `03_train_local.py`, `CountVectorizer` | Feeds Naive Bayes |
| TF-IDF | `stylometry.CharNgramFeatures`; TF-IDF word baseline | Two different uses |
| N-gram language model | `models.InterpolatedKneserNeyLM` | Char 5-gram and word 3-gram |
| MLE probabilities → smoothing | same | Goes past add-one to **interpolated Kneser–Ney** |
| Perplexity | `InterpolatedKneserNeyLM.perplexity` | The generative attribution rule |

The lab builds a bigram model on raw MLE counts and shows it assigning zero
probability to unseen bigrams. Kneser–Ney is the principled fix for exactly
that failure, and it is what makes the generative attributor work at all: an
author model that returns probability zero for any unseen n-gram cannot rank
candidate authors.

## Lab 3 — Word embeddings and foundational classifiers

| Lab concept | Where it lives | Notes |
|---|---|---|
| Skip-gram Word2Vec **from scratch** | `embeddings.SkipGramWord2Vec` | NumPy, negative sampling, no `gensim` |
| Cosine similarity / nearest words | `SkipGramWord2Vec.most_similar` | |
| Weighted embedding pooling | `embeddings.average_vectors` | Plain mean and SIF weighting |
| Naive Bayes **from scratch** | `classifiers.MultinomialNaiveBayes` | Log-space, add-alpha smoothing |
| Logistic regression **from scratch** | `classifiers.SoftmaxRegression` | Binary → multinomial generalisation |

The lab derives binary logistic regression with a sigmoid and binary
cross-entropy. `SoftmaxRegression` is that derivation carried to *K* authors:
sigmoid becomes softmax, the loss becomes categorical, and the gradient keeps
the same `X.T @ (predicted − actual)` shape it had in the binary case.

## Lab 4 — PyTorch sequence models

| Lab concept | Where it lives | Notes |
|---|---|---|
| Bidirectional recurrent classifier | `neural.BiLSTMClassifier` | 2-layer BiLSTM |
| Pretrained embeddings into `nn.Embedding` | `neural._embedding_matrix` | Seeded from **our own** from-scratch Word2Vec |
| Padding + masking | `BiLSTMClassifier.forward` | Padding excluded from pooling, not just from loss |
| Sequence labelling (POS) | `postag.py` | Rule-based tagger, scored on `UD_Bengali-BRU` |

The lab loads Google's Word2Vec into the embedding layer. Doing that here would
import knowledge from outside the corpus and muddy the comparison, so the
embedding table is seeded from the skip-gram vectors trained in Lab 3's module
on the training split only. Same mechanism, no leakage.

## Lab 5 — Transformers

| Lab concept | Where it lives | Notes |
|---|---|---|
| Encoder-only Transformer **from scratch** | `neural.TransformerClassifier` | 3 layers, 4 heads |
| Sinusoidal positional encoding | `TransformerClassifier._build.sinusoid` | |
| `src_key_padding_mask` | `Net.forward` | |
| Mean pooling over tokens | `Net.forward` | Masked, so short passages are not diluted |
| Pretrained transformer, fine-tuned | `models.BertClassifier` | BanglaBERT, **multi-class head** |

`BertClassifier` is a genuine multi-class classifier — `num_labels` is the
number of authors and `predict_proba` returns a probability per author — so the
system answers "which of these authors does this most resemble?" directly,
alongside the per-author language models which answer it by comparing
perplexities. The two paradigms give independent answers to the same question,
which is the project's central comparison.

---

## Deliberately not used

| Concept | Why not |
|---|---|
| Porter stemming / WordNet lemmatisation | Both are English-specific. No Bangla lemmatiser has a Python 3.14 wheel here, and character n-grams already capture the suffix morphology a stemmer would normalise away — which for *authorship* is signal we want to keep, not strip. |
| Seq2Seq / machine translation | The task is single-label classification. There is no target sequence to generate. |
| Shannon guessing game / LM text generation | The per-author Kneser–Ney models could generate text, but two of the three authors are in copyright and a high-order character model trained on a small corpus reproduces long verbatim spans. Not worth the risk for a demo. |
| `gensim`, `bnlp-toolkit` | No Python 3.14 wheel and no C toolchain available. This is why Word2Vec and the POS tagger are written from scratch — it is the reason those implementations exist. |

## Coverage at a glance

| Lab | Concepts used | Scored in the results table |
|---|---|---|
| 1 — Preprocessing | 6 / 7 | underpins every model |
| 2 — Representation & n-grams | 5 / 5 | Kneser–Ney LM (char, word), TF-IDF baseline |
| 3 — Embeddings & classifiers | 5 / 5 | Naive Bayes, softmax regression |
| 4 — Sequence models | 4 / 5 | BiLSTM |
| 5 — Transformers | 5 / 5 | Transformer from scratch, BanglaBERT |
