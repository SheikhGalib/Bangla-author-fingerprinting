"""Stage 3: train and score every model that does not need a GPU.

    .venv/Scripts/python.exe scripts/03_train_local.py

Covers both paradigms at the classical end:

* **generative** -- one interpolated Kneser-Ney language model per author,
  attributing a passage to whichever author's model finds it least surprising
  (character-level and word-level);
* **discriminative** -- Naive Bayes and softmax regression written from
  scratch, plus a linear SVM over the four stylometric feature families, with
  a per-family ablation.

The GPU models (fine-tuned BanglaBERT, BiLSTM, from-scratch Transformer) train
on Kaggle via ``scripts/kaggle_run.py`` and are merged into the same table by
``scripts/04_collect_results.py``.  Both halves read the identical split file,
which is what makes the merged table a comparison rather than two experiments.

Writes ``artifacts/tables/results_local.csv`` and the fitted models under
``artifacts/models/``.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from banglastylo import config, corpus, models, splits
from banglastylo.classifiers import MultinomialNaiveBayes, SoftmaxRegression
from banglastylo.stylometry import StylometricFeatures

RESULTS: list[dict] = []


def record(name: str, family: str, y_true, y_pred, notes: str = "") -> None:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    RESULTS.append({"model": name, "family": family, "accuracy": acc,
                    "macro_f1": f1, "notes": notes})
    print(f"  -> {name:34s} acc={acc:.4f}  macroF1={f1:.4f}   {notes}", flush=True)


def main() -> int:
    passages = corpus.load_passages()
    split = splits.load(passages)
    xtr, ytr = splits.xy(split.train)
    # The CPU models do no early stopping, so validation is only reported
    # here for the split summary; the labels themselves are not needed.
    xva, _ = splits.xy(split.val)
    xte, yte = splits.xy(split.test)
    print(f"train={len(xtr)}  val={len(xva)}  test={len(xte)}  "
          f"classes={sorted(set(ytr))}")
    print(f"chance = {1 / len(set(yte)):.3f}\n")

    rep = splits.leakage_report(split)
    if not rep["clean"]:
        print("LEAKAGE DETECTED — refusing to train", file=sys.stderr)
        return 1
    print("leakage check: clean\n")

    # --- generative: Kneser-Ney language models ----------------------------
    for level in ("char", "word"):
        print(f"=== generative: {level}-level Kneser-Ney ===")
        t0 = time.time()
        gen = models.GenerativeAttributor(level=level)
        gen.fit(xtr, ytr)
        record(f"Kneser-Ney LM ({level})", "generative",
               yte, gen.predict(xte), f"order {gen.order}, {time.time() - t0:.0f}s")
        with (config.MODELS / f"generative_{level}.pkl").open("wb") as fh:
            pickle.dump(gen, fh)

    # --- discriminative: Naive Bayes over word Bag-of-Words -----------------
    # Words, not characters, and the choice is pedagogical rather than
    # numerical.  Character n-grams score higher on Bangla because case and
    # tense are suffixes, so they see inside a word and recognise করিয়াছিলেন
    # and করেছিলেন as one verb in two registers.  But the evidence they produce
    # ("the 3-character sequence ␣হা favours Nazrul") cannot be checked against
    # the passage by eye, whereas "this word is 4x more likely under Humayun"
    # can.  Word BoW is also exactly the representation Lab 2 and Lab 3 built.
    # The character variant is still scored below so the cost is on the record.
    #
    # Counts, not TF-IDF: multinomial NB is defined over counts, and feeding it
    # continuous weights quietly changes the model it is supposed to be.
    print("\n=== discriminative: Naive Bayes over word BoW (from scratch) ===")
    wv = CountVectorizer(token_pattern=r"\S+", max_features=20000)
    Xtr = wv.fit_transform(xtr)          # fitted on train only
    Xte = wv.transform(xte)
    nb = MultinomialNaiveBayes(alpha=0.2).fit(Xtr, np.array(ytr))
    record("Naive Bayes (word BoW)", "discriminative", yte, nb.predict(Xte),
           f"{Xtr.shape[1]} word features, alpha=0.2")
    with (config.MODELS / "naive_bayes_word.pkl").open("wb") as fh:
        pickle.dump({"vectorizer": wv, "model": nb}, fh)

    print("\n=== reference only: the same model over character n-grams ===")
    cv = CountVectorizer(analyzer="char_wb", ngram_range=config.CHAR_NGRAM_RANGE,
                         max_features=config.CHAR_NGRAM_MAX_FEATURES)
    Ctr = cv.fit_transform(xtr)
    nb_char = MultinomialNaiveBayes(alpha=0.2).fit(Ctr, np.array(ytr))
    record("Naive Bayes (char n-gram)", "reference", yte,
           nb_char.predict(cv.transform(xte)),
           f"{Ctr.shape[1]} char features, alpha=0.2")

    # --- discriminative: softmax regression over stylometry ----------------
    print("\n=== discriminative: softmax regression (from scratch) ===")
    feats = StylometricFeatures()
    Ftr = feats.fit_transform(xtr)
    Fte = feats.transform(xte)
    Ftr = np.asarray(Ftr.todense() if hasattr(Ftr, "todense") else Ftr)
    Fte = np.asarray(Fte.todense() if hasattr(Fte, "todense") else Fte)
    scale = np.abs(Ftr).max(axis=0)
    scale[scale == 0] = 1.0
    lr = SoftmaxRegression(lr=0.5, epochs=300, verbose=True).fit(
        Ftr / scale, np.array(ytr))
    record("Softmax regression (stylometric)", "discriminative",
           yte, lr.predict(Fte / scale), f"{Ftr.shape[1]} features, 300 epochs")

    # --- discriminative: SVM, all families then ablated --------------------
    print("\n=== discriminative: linear SVM ===")
    families = ("funcword", "structural", "charngram", "posngram")
    pipe = models.build_svm_pipeline(families=families)
    pipe.fit(xtr, ytr)
    record("SVM (all stylometric)", "discriminative", yte, pipe.predict(xte),
           f"{len(families)} families")
    # `svm_combined.pkl` is the name `interface.Attributor.load` looks for.
    with (config.MODELS / "svm_combined.pkl").open("wb") as fh:
        pickle.dump(pipe, fh)

    print("\n=== ablation: one family at a time ===")
    for fam in families:
        p = models.build_svm_pipeline(families=(fam,))
        p.fit(xtr, ytr)
        record(f"SVM ({fam} only)", "ablation", yte, p.predict(xte), "")

    # --- baseline: TF-IDF word unigrams, the Lab 2 representation ----------
    print("\n=== baseline: TF-IDF word unigrams + SVM ===")
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    bow = make_pipeline(
        TfidfVectorizer(max_features=20000, token_pattern=r"\S+"),
        LinearSVC(random_state=config.SEED, dual="auto", max_iter=5000))
    bow.fit(xtr, ytr)
    record("TF-IDF words + SVM", "baseline", yte, bow.predict(xte),
           "topic-sensitive by design")
    with (config.MODELS / "tfidf_svm.pkl").open("wb") as fh:
        pickle.dump(bow, fh)

    # --- save ---------------------------------------------------------------
    import pandas as pd
    df = pd.DataFrame(RESULTS).sort_values("accuracy", ascending=False)
    config.TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TABLES / "results_local.csv", index=False)
    (config.TABLES / "results_local.json").write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 78}\nlocal results (test = three unseen books)\n{'=' * 78}")
    print(df.to_string(index=False,
                       formatters={"accuracy": "{:.4f}".format,
                                   "macro_f1": "{:.4f}".format}))
    print(f"\nwrote {config.TABLES / 'results_local.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
