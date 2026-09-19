"""Side-by-side attribution from the three models the demonstration shows.

The project trains a dozen models; this module exposes exactly three, chosen so
that a five-minute explanation covers a real progression rather than a list:

1. **TF--IDF word unigrams + SVM** -- counts words, weights the informative
   ones, draws a straight boundary.  No word order at all.
2. **BiLSTM** -- reads the passage left-to-right and right-to-left, so word
   *order* is available to it.  Trained from random initialisation on this
   corpus alone.
3. **BanglaBERT, fine-tuned** -- the same task, but starting from an encoder
   already pretrained on a large Bangla corpus.

Read in that order the three answer one question: how much does each added
capability -- order, then pretraining -- actually buy?  Showing them beside
each other on the same passage makes the answer concrete instead of a table.

Every model is loaded once at import and run on CPU.  Loading is individually
guarded: a missing checkpoint disables that one panel rather than taking the
whole interface down, because the UI is often run before the GPU job has been
pulled.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

#: Where the Kaggle job's checkpoints land after ``kaggle_run.py pull``.
KAGGLE_DIR = config.ARTIFACTS / "kaggle"


@dataclass
class Verdict:
    """One model's answer, plus enough context to explain it on screen."""

    key: str
    name: str
    stage: str                     # the one-line "what this model can see"
    predicted: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    margin: float = 0.0            # top probability minus runner-up
    available: bool = True
    error: str | None = None


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class ComparisonPanel:
    """Loads the three demonstration models and runs a passage through each."""

    #: Order is the explanation: each row adds one capability to the row above.
    SPEC = (
        ("tfidf_svm", "TF-IDF + SVM",
         "Counts words. No word order."),
        ("bilstm", "BiLSTM (from scratch)",
         "Reads word order, both directions. Learned only from our 6 books."),
        ("banglabert", "BanglaBERT (fine-tuned)",
         "Pretrained on a large Bangla corpus, then fine-tuned on our 6 books."),
    )

    def __init__(self) -> None:
        self.errors: dict[str, str] = {}
        self.tfidf = self._load_tfidf()
        self.bilstm = self._load_bilstm()
        self.bert = self._load_bert()

    # -- loading -----------------------------------------------------------
    def _load_tfidf(self):
        path = config.MODELS / "tfidf_svm.pkl"
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception as exc:  # noqa: BLE001 - one bad file must not break the page
            self.errors["tfidf_svm"] = f"{type(exc).__name__}: {exc}"
            return None

    @staticmethod
    def _roster_matches(classes) -> bool:
        """Is this checkpoint trained on the authors currently configured?

        This check exists because the alternative is silent and convincing.
        ``artifacts/models/`` still holds a BanglaBERT from an earlier
        five-author version of this project; loading it produced fluent,
        high-confidence predictions naming authors the corpus no longer
        contains.  A stale checkpoint must fail loudly, so the class list is
        compared against ``config.AUTHORS`` before the model is accepted.
        """
        return set(map(str, classes)) == set(config.AUTHORS)

    def _load_bilstm(self):
        from .neural import BiLSTMClassifier

        for path in (config.MODELS / "bilstm", KAGGLE_DIR / "bilstm"):
            if not (path / "model.pt").exists():
                continue
            try:
                model = BiLSTMClassifier.load(path, device="cpu")
            except Exception as exc:  # noqa: BLE001 - see above
                self.errors["bilstm"] = f"{type(exc).__name__}: {exc}"
                continue
            if not self._roster_matches(model.classes_):
                self.errors["bilstm"] = (
                    f"stale checkpoint at {path}: trained on "
                    f"{sorted(model.classes_)}, expected {sorted(config.AUTHORS)}")
                continue
            return model
        self.errors.setdefault(
            "bilstm", "no checkpoint; run scripts/kaggle_run.py pull")
        return None

    def _load_bert(self):
        for path in (config.MODELS / "banglabert_finetuned",
                     KAGGLE_DIR / "banglabert_finetuned"):
            if not (path / "config.json").exists():
                continue
            try:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                classes = (path / "classes.txt").read_text(
                    encoding="utf-8").split()
                if not self._roster_matches(classes):
                    self.errors["banglabert"] = (
                        f"stale checkpoint at {path}: trained on "
                        f"{sorted(classes)}, expected {sorted(config.AUTHORS)}")
                    continue
                tok = AutoTokenizer.from_pretrained(str(path))
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(path)).eval()
                return {"tok": tok, "model": model, "classes": classes,
                        "torch": torch}
            except Exception as exc:  # noqa: BLE001 - see above
                self.errors["banglabert"] = f"{type(exc).__name__}: {exc}"
                continue
        self.errors.setdefault(
            "banglabert", "no checkpoint; run scripts/kaggle_run.py pull")
        return None

    # -- prediction --------------------------------------------------------
    @staticmethod
    def _finish(v: Verdict, classes, scores: np.ndarray) -> Verdict:
        order = np.argsort(scores)[::-1]
        v.predicted = str(classes[order[0]])
        v.scores = {str(c): round(float(s), 4) for c, s in zip(classes, scores)}
        v.margin = round(float(scores[order[0]] - scores[order[1]]), 4) \
            if len(scores) > 1 else 1.0
        return v

    def _tfidf_verdict(self, text: str) -> Verdict:
        v = Verdict("tfidf_svm", *self.SPEC[0][1:])
        if self.tfidf is None:
            v.available, v.error = False, self.errors.get("tfidf_svm")
            return v
        # LinearSVC has no predict_proba; its decision function is turned into
        # comparable numbers with a softmax.  These are relative confidences for
        # display, not calibrated probabilities, and are labelled as such.
        d = self.tfidf.decision_function([text])[0]
        return self._finish(v, self.tfidf.classes_, _softmax(np.asarray(d)))

    def _bilstm_verdict(self, text: str) -> Verdict:
        v = Verdict("bilstm", *self.SPEC[1][1:])
        if self.bilstm is None:
            v.available, v.error = False, self.errors.get("bilstm")
            return v
        p = self.bilstm.predict_proba([text])[0]
        return self._finish(v, self.bilstm.classes_, np.asarray(p))

    def _bert_verdict(self, text: str) -> Verdict:
        v = Verdict("banglabert", *self.SPEC[2][1:])
        if self.bert is None:
            v.available, v.error = False, self.errors.get("banglabert")
            return v
        torch = self.bert["torch"]
        enc = self.bert["tok"]([text], truncation=True, max_length=256,
                               return_tensors="pt")
        with torch.no_grad():
            logits = self.bert["model"](**enc).logits[0].numpy()
        return self._finish(v, self.bert["classes"], _softmax(logits))

    def predict(self, text: str) -> list[Verdict]:
        """Run the passage through all three, in explanation order."""
        return [self._tfidf_verdict(text),
                self._bilstm_verdict(text),
                self._bert_verdict(text)]

    def status(self) -> dict:
        return {
            "tfidf_svm": self.tfidf is not None,
            "bilstm": self.bilstm is not None,
            "banglabert": self.bert is not None,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# The corpus catalogue the interface displays
# ---------------------------------------------------------------------------
def _first_paragraph(path: Path, min_chars: int = 300) -> str:
    """A representative paragraph from a book file, skipping the header.

    Read from disk at request time rather than stored anywhere: the interface
    shows what is actually in ``corpus/``, so the glimpse cannot drift from the
    text the models were trained and tested on.
    """
    header = ("=", "-", "বই", "লেখক", "অধ্যায়", "অক্ষর", "উৎস", "ভূমিকা")
    body = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith(header)]
    for line in body[2:]:
        if len(line) >= min_chars:
            return line
    return next((ln for ln in body if len(ln) > 80), "")


def catalogue(glimpse_chars: int = 700) -> list[dict]:
    """Every book in the corpus, with its role and a readable excerpt.

    This is what lets the interface show, rather than assert, that one book per
    author was held back: the same listing names the training books and the
    held-out one, and offers a paragraph of each.
    """
    out: list[dict] = []
    for role, root in (("train", config.CORPUS_TRAIN),
                       ("unseen", config.CORPUS_UNSEEN)):
        if not root.exists():
            continue
        for key, meta in config.AUTHORS.items():
            folder = root / meta["en"]
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.txt")):
                text = path.read_text(encoding="utf-8")
                out.append({
                    "author": key,
                    "author_en": meta["en"],
                    "author_bn": meta["bn"],
                    "book": path.stem,
                    "role": role,
                    "chars": len(text),
                    "chapters": text.count("\n--- "),
                    "glimpse": _first_paragraph(path)[:glimpse_chars],
                })
    return out
