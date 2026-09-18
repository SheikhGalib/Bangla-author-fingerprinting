"""The two attribution paradigms the proposal sets against each other.

Generative
    One language model per author, fit only on that author's training text.  An
    unseen passage goes to whichever author's model assigns it the highest
    average log-probability, i.e. the lowest perplexity.  The model never sees
    the other authors during training — it learns "what this author's text looks
    like", not "how this author differs from the others".

Discriminative
    Features from all authors go into one classifier trained to tell them apart.
    Two are provided: a linear SVM over stylometric and/or embedding features,
    and a fine-tuned BanglaBERT.

:class:`InterpolatedKneserNeyLM` is written out rather than imported because the
per-token log-probability breakdown it exposes is what makes the generative side
interpretable — the report shows *which tokens* pushed a passage towards one
author, which a black-box LM API would not give.
"""
from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import config
from .normalize import word_tokenize

BOS = "<s>"
EOS = "</s>"


# ---------------------------------------------------------------------------
# Interpolated Kneser-Ney language model
# ---------------------------------------------------------------------------
class InterpolatedKneserNeyLM:
    """Interpolated Kneser–Ney over an arbitrary symbol sequence.

    Works for characters or words: the caller decides what a "token" is.

    Kneser–Ney is the right smoothing here rather than add-one or Laplace
    because of its *continuation* probability.  The unigram fallback asks not
    "how often did this token occur?" but "after how many distinct contexts did
    it occur?".  For stylometry that is exactly the useful question: a token
    that appears often but only ever in one fixed phrase is a topic artefact,
    while a token appearing after many different contexts is a habit.
    """

    def __init__(self, order: int = 3, discount: float = 0.75):
        self.order = order
        self.discount = discount
        self.ngram_counts: list[dict[tuple, int]] = []
        self.context_totals: list[dict[tuple, int]] = []
        self.context_types: list[dict[tuple, int]] = []
        self.continuation: dict[str, int] = {}
        self.n_bigram_types = 0
        self.vocab: set[str] = set()

    # -- training ----------------------------------------------------------
    def fit(self, sequences: list[list[str]]) -> InterpolatedKneserNeyLM:
        n = self.order
        self.ngram_counts = [defaultdict(int) for _ in range(n + 1)]
        self.context_totals = [defaultdict(int) for _ in range(n + 1)]
        self.context_types = [defaultdict(int) for _ in range(n + 1)]
        continuation_ctx: dict[str, set[tuple]] = defaultdict(set)
        bigram_types: set[tuple] = set()

        for seq in sequences:
            padded = [BOS] * (n - 1) + list(seq) + [EOS]
            self.vocab.update(seq)
            for k in range(1, n + 1):
                counts = self.ngram_counts[k]
                totals = self.context_totals[k]
                types = self.context_types[k]
                for i in range(n - 1, len(padded)):
                    gram = tuple(padded[i - k + 1 : i + 1])
                    ctx = gram[:-1]
                    if counts[gram] == 0:
                        types[ctx] += 1
                    counts[gram] += 1
                    totals[ctx] += 1
                    if k == 2:
                        continuation_ctx[gram[1]].add(gram[0])
                        bigram_types.add(gram)

        self.continuation = {w: len(s) for w, s in continuation_ctx.items()}
        self.n_bigram_types = max(len(bigram_types), 1)
        self.vocab.add(EOS)
        self.ngram_counts = [dict(d) for d in self.ngram_counts]
        self.context_totals = [dict(d) for d in self.context_totals]
        self.context_types = [dict(d) for d in self.context_types]
        return self

    # -- scoring -----------------------------------------------------------
    def _unigram_prob(self, token: str) -> float:
        # Continuation probability, backing off to a uniform floor for OOV.
        cont = self.continuation.get(token, 0)
        floor = 1.0 / (len(self.vocab) + 1)
        return max(cont / self.n_bigram_types, 1e-3 * floor)

    def _prob(self, gram: tuple[str, ...], k: int) -> float:
        if k == 1:
            return self._unigram_prob(gram[-1])
        ctx = gram[:-1]
        total = self.context_totals[k].get(ctx, 0)
        lower = self._prob(gram[1:], k - 1)
        if total == 0:
            return lower
        count = self.ngram_counts[k].get(gram, 0)
        n_types = self.context_types[k].get(ctx, 0)
        d = self.discount
        higher = max(count - d, 0.0) / total
        lam = d * n_types / total
        return higher + lam * lower

    def token_logprobs(self, seq: list[str]) -> list[tuple[str, float]]:
        """Per-token log2-probabilities — the generative interpretability hook."""
        n = self.order
        padded = [BOS] * (n - 1) + list(seq) + [EOS]
        out: list[tuple[str, float]] = []
        for i in range(n - 1, len(padded)):
            gram = tuple(padded[i - n + 1 : i + 1])
            p = max(self._prob(gram, n), 1e-12)
            out.append((padded[i], math.log2(p)))
        return out

    def log_prob(self, seq: list[str]) -> float:
        return sum(lp for _, lp in self.token_logprobs(seq))

    def perplexity(self, seq: list[str]) -> float:
        lps = self.token_logprobs(seq)
        if not lps:
            return float("inf")
        return 2 ** (-sum(lp for _, lp in lps) / len(lps))


# ---------------------------------------------------------------------------
# Generative attributor
# ---------------------------------------------------------------------------
def char_seq(text: str) -> list[str]:
    return list(text)


def word_seq(text: str) -> list[str]:
    return word_tokenize(text, keep_punct=True)


class GenerativeAttributor:
    """One language model per author; attribute by lowest perplexity.

    ``level='char'`` builds character n-gram models, ``level='word'`` word
    n-gram models.  Character models are usually stronger for morphologically
    rich languages because they see inside the word: Bangla's case clitics and
    verb inflections are suffixes, so a character model picks up the register
    difference between করিয়াছিলেন and করেছিলেন without needing either form to
    have been seen whole.
    """

    def __init__(
        self,
        level: str = "char",
        order: int | None = None,
        discount: float = 0.75,
    ):
        self.level = level
        self.order = order or (
            config.CHAR_LM_ORDER if level == "char" else config.WORD_LM_ORDER
        )
        self.discount = discount
        self.models: dict[str, InterpolatedKneserNeyLM] = {}
        self.classes_: list[str] = []

    def _seq(self, text: str) -> list[str]:
        return char_seq(text) if self.level == "char" else word_seq(text)

    def fit(self, texts: list[str], authors: list[str], verbose: bool = True):
        by_author: dict[str, list[list[str]]] = defaultdict(list)
        for t, a in zip(texts, authors):
            by_author[a].append(self._seq(t))
        self.classes_ = sorted(by_author)
        for a in self.classes_:
            lm = InterpolatedKneserNeyLM(self.order, self.discount)
            lm.fit(by_author[a])
            self.models[a] = lm
            if verbose:
                print(f"  [{self.level}-{self.order}gram] {a:16s} "
                      f"{len(by_author[a]):>4} passages, "
                      f"{len(lm.vocab):>7,} symbol types", flush=True)
        return self

    def decision_function(self, texts: list[str]) -> np.ndarray:
        """Average log2-probability per token under each author's model."""
        scores = np.zeros((len(texts), len(self.classes_)), dtype=np.float64)
        for r, text in enumerate(texts):
            seq = self._seq(text)
            for c, a in enumerate(self.classes_):
                lps = self.models[a].token_logprobs(seq)
                scores[r, c] = sum(lp for _, lp in lps) / max(len(lps), 1)
        return scores

    def predict(self, texts: list[str]) -> np.ndarray:
        s = self.decision_function(texts)
        return np.array([self.classes_[i] for i in s.argmax(1)])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Softmax over per-token average log-likelihood.

        This is a *ranking-preserving* convenience, not a calibrated posterior;
        the report treats it as a confidence ordering only.
        """
        s = self.decision_function(texts)
        n_tokens = np.array([max(len(self._seq(t)), 1) for t in texts])[:, None]
        logits = s * np.sqrt(n_tokens)     # sharpen with passage length
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(1, keepdims=True)

    def perplexity_table(self, texts: list[str]) -> np.ndarray:
        return 2 ** (-self.decision_function(texts))

    def explain(self, text: str, top_k: int = 15) -> dict:
        """Tokens that most separate the winning author from the runner-up."""
        seq = self._seq(text)
        per_author = {
            a: self.models[a].token_logprobs(seq) for a in self.classes_
        }
        avg = {a: sum(lp for _, lp in v) / max(len(v), 1)
               for a, v in per_author.items()}
        ranked = sorted(avg, key=avg.get, reverse=True)
        best, runner = ranked[0], ranked[1] if len(ranked) > 1 else ranked[0]
        deltas = [
            (tok, lp_best - lp_run)
            for (tok, lp_best), (_, lp_run) in zip(
                per_author[best], per_author[runner]
            )
        ]
        # Aggregate by symbol so a repeated habit shows up as one strong entry.
        agg: Counter[str] = Counter()
        for tok, d in deltas:
            agg[tok] += d
        return {
            "predicted": best,
            "runner_up": runner,
            "avg_logprob": avg,
            "perplexity": {a: 2 ** (-v) for a, v in avg.items()},
            "for_prediction": agg.most_common(top_k),
            "against_prediction": agg.most_common()[: -top_k - 1 : -1],
        }

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> GenerativeAttributor:
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ---------------------------------------------------------------------------
# Discriminative: linear SVM
# ---------------------------------------------------------------------------
def build_svm(C: float = 1.0, seed: int = config.SEED):
    """Linear SVM with a probability calibration wrapper.

    ``LinearSVC`` is preferred over an RBF kernel for two reasons: it scales to
    the several-thousand-dimensional sparse feature space, and its coefficients
    are directly readable as per-feature evidence, which the interpretability
    requirement depends on.
    """
    from sklearn.svm import LinearSVC

    return LinearSVC(C=C, random_state=seed, dual="auto", max_iter=5000)


def build_svm_pipeline(
    families: tuple[str, ...] = ("funcword", "structural", "charngram", "posngram"),
    C: float = 1.0,
    seed: int = config.SEED,
):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MaxAbsScaler

    from .stylometry import StylometricFeatures

    return Pipeline(
        [
            ("features", StylometricFeatures(families=families)),
            ("scale", MaxAbsScaler()),
            ("clf", build_svm(C=C, seed=seed)),
        ]
    )


# ---------------------------------------------------------------------------
# Discriminative: fine-tuned BanglaBERT
# ---------------------------------------------------------------------------
class BertClassifier:
    """Fine-tune ``csebuetnlp/banglabert`` for closed-set author attribution."""

    def __init__(
        self,
        model_name: str = config.PRETRAINED_MODEL,
        max_len: int = config.FT_MAX_LEN,
        epochs: int = config.FT_EPOCHS,
        batch_size: int = config.FT_BATCH,
        lr: float = config.FT_LR,
        seed: int = config.SEED,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = device or ("cuda" if __import__("torch").cuda.is_available()
                                 else "cpu")
        self.classes_: list[str] = []

    def fit(self, texts: list[str], authors: list[str],
            val: tuple[list[str], list[str]] | None = None, verbose: bool = True):
        import os

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # PyTorch defaults to physical cores; on this CPU-only machine the extra
        # threads are worth roughly a third of the wall-clock time.
        torch.set_num_threads(os.cpu_count() or 4)
        torch.manual_seed(self.seed)
        self.classes_ = sorted(set(authors))
        label_of = {a: i for i, a in enumerate(self.classes_)}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.classes_)
        ).to(self.device)

        enc = self.tokenizer(texts, padding="max_length", truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        y = torch.tensor([label_of[a] for a in authors])
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        n_steps = self.epochs * len(dl)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, total_steps=n_steps, pct_start=0.1
        )

        for epoch in range(self.epochs):
            self.model.train()
            running, seen = 0.0, 0
            for step, (ids, mask, yy) in enumerate(dl):
                ids, mask, yy = ids.to(self.device), mask.to(self.device), yy.to(self.device)
                out = self.model(input_ids=ids, attention_mask=mask, labels=yy)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                running += float(out.loss.detach()) * yy.numel()
                seen += yy.numel()
                if verbose and step % 25 == 0:
                    print(f"    epoch {epoch + 1} step {step}/{len(dl)} "
                          f"loss={running / max(seen, 1):.4f}", flush=True)
            msg = f"  epoch {epoch + 1}/{self.epochs} train_loss={running / seen:.4f}"
            if val is not None:
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(val[1], self.predict(val[0]))
                msg += f"  val_acc={acc:.4f}"
            if verbose:
                print(msg, flush=True)
        return self

    def _logits(self, texts: list[str]) -> np.ndarray:
        import torch

        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = texts[i : i + 32]
                enc = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.max_len,
                                     return_tensors="pt").to(self.device)
                out.append(self.model(**enc).logits.cpu().numpy())
        return np.vstack(out)

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.array([self.classes_[i] for i in self._logits(texts).argmax(1)])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        z = self._logits(texts)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "classes.txt").write_text("\n".join(self.classes_), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> BertClassifier:
        """Reload a fine-tuned checkpoint instead of paying for it again.

        Fine-tuning takes ~40 minutes on CPU, so a notebook re-run should not
        repeat it just to redraw a figure.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        obj = cls(**kwargs)
        obj.classes_ = (path / "classes.txt").read_text(
            encoding="utf-8"
        ).split("\n")
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path).to(
            obj.device
        )
        obj.model.eval()
        return obj
