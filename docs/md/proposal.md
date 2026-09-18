| Authorship  |                | Attribution |           | and Style  | Fingerprinting |
| ----------- | -------------- | ----------- | --------- | ---------- | -------------- |
|             |                |             | in Bangla | Literature |                |
|             |                |             | Project   | Proposal   |                |
| 1. Overview | and Motivation |             |           |            |                |
Authorship attribution asks: given an unseen passage, which author (from a closed candidate set)
most likely wrote it, based on how something is written rather than what it says. Bangla literature
spans stylistically distinct classical and modern voices, yet computational stylometry for Bangla re-
mains comparatively underexplored, and per-author corpora are typically small. This project builds
a multi-author Bangla stylometric pipeline that combines interpretable stylometric features with word
embeddings, and directly compares a generative attribution paradigm against a discriminative one
— while also asking a representation-level question: in a small-corpus regime, do pretrained embed-
dings or embeddings trained from scratch on the author corpus better capture stylistic signal rather
| than general | semantics? |     |     |     |     |
| ------------ | ---------- | --- | --- | --- | --- |
2. Objectives
• Curate and document a cleaned, public-domain, multi-author Bangla corpus.
• Extract stylometric features: function-word frequencies, sentence-length distributions, POS n-
| gram | statistics. |     |     |     |     |
| ---- | ----------- | --- | --- | --- | --- |
• Compare pretrained vs. from-scratch embeddings for isolating stylistic signal from semantic con-
tent.
• Implement and compare a generative per-author language-model attributor against a discrimina-
| tive | SVM / fine-tuned | BERT | classifier. |     |     |
| ---- | ---------------- | ---- | ----------- | --- | --- |
• Analyze confusion between stylistically similar authors and interpret the features driving each
| model’s | predictions. |     |     |     |     |
| ------- | ------------ | --- | --- | --- | --- |
• Ship a simple interface: paste a passage, receive the predicted author and the top features behind
| the decision. |     |     |     |     |     |
| ------------- | --- | --- | --- | --- | --- |
3. Methodology
| 3.1 Corpus | construction |     |     |     |     |
| ---------- | ------------ | --- | --- | --- | --- |
Public-domainworksarecollectedacrossclassicalandmodernBanglaauthors, withsentencesegmenta-
tion, normalization, deduplication, and removal of front/back matter. Sampling is balanced per author
to limit corpus-size skew, and the full collection-and-cleaning process is documented as part of the
| methodology, | not treated | as a black | box. |     |     |
| ------------ | ----------- | ---------- | ---- | --- | --- |
| 3.2 Feature  | engineering |            |      |     |     |
Stylometric: function-word frequency vectors (pronouns, postpositions, particles), sentence-length
distribution statistics (mean, variance, skew), and POS n-grams from a Bangla tagger.
Embeddings: (a) pretrained Bangla embeddings (e.g. fastText / BanglaBERT) and (b) embeddings
trained from scratch on the author corpus, evaluated on how well each isolates author style versus
topical content.
| 3.3 Two | attribution | paradigms |     |     |     |
| ------- | ----------- | --------- | --- | --- | --- |
Generative: a per-author language model is fit on each author’s text; an unseen passage is attributed
to the author whose model assigns it the highest likelihood (lowest perplexity).
Discriminative: stylometric and embedding features are concatenated and fed to an SVM, and sepa-
rately used to fine-tune a BERT classifier; both are trained directly to discriminate between authors.

|     |     |     | generative: | highest likelihood |     |
| --- | --- | --- | ----------- | ------------------ | --- |
Per-author
language model
|     | Multi-author  | Stylometric | +        | Predicted        | author + |
| --- | ------------- | ----------- | -------- | ---------------- | -------- |
|     | Bangla corpus | embedding   | features | top contributing | features |
SVM / fine-tuned
BERT classifier
|            |         |     | discriminative: | direct classification |     |
| ---------- | ------- | --- | --------------- | --------------------- | --- |
| 3.4 Worked | example |     |                 |                       |     |
Input passage (constructed example): ??(cid:523) ????? ?? ??????? ???? ??????? ??? ?? ?????? ???? ??????
????
Extracted signal: sentence length =12 words · function words ????, (postpositions), (pronoun) ·
|             |                       |     |     | ???? | ??  |
| ----------- | --------------------- | --- | --- | ---- | --- |
| POS pattern | NOUN--VERB--ADP--NOUN |     |     |      |     |
Generative model: log-likelihood highest under Author B’s language model.
Discriminative model: SVM assigns Author B probability 0.71, driven mostly by postposition frequency
| and mean | sentence length. |     |     |     |     |
| -------- | ---------------- | --- | --- | --- | --- |
Output: Predicted author: Author B — top features: postposition rate, mean sentence length,
| NOUN--VERB--ADP | rate. |     |     |     |     |
| --------------- | ----- | --- | --- | --- | --- |
| 4. Evaluation   | Plan  |     |     |     |     |
• Metrics: accuracy and macro-F1 (per-author class imbalance), with confusion-matrix analysis
| focused | on stylistically | close authors. |     |     |     |
| ------- | ---------------- | -------------- | --- | --- | --- |
• Ablations: stylometric-only vs. embedding-only vs. combined features; pretrained vs. from-
scratch embeddings.
• Interpretability: feature-importance / SHAP for the SVM; likelihood-contribution breakdown
for the generative models; comparison of which feature families drive each paradigm’s correct and
| incorrect | predictions. |     |     |     |     |
| --------- | ------------ | --- | --- | --- | --- |
5. Interface
A minimal input–output interface accepts a pasted Bangla passage and returns the predicted author
alongsidetherankedstylisticfeaturesthatmostinfluencedthedecision, sopredictionsremainauditable
rather than opaque.
| 6. Expected | Outcomes |     |     |     |     |
| ----------- | -------- | --- | --- | --- | --- |
A documented, reusable multi-author Bangla stylometric corpus; an empirical comparison of genera-
tive and discriminative attribution for a morphologically rich, low-resource language; and evidence on
whether pretrained or from-scratch embeddings better isolate authorial style in small-corpus settings.
