"""Two word-representation regimes, for the proposal's central comparison.

The proposal asks whether, in a small-corpus regime, **pretrained** embeddings
or embeddings **trained from scratch on the author corpus** better capture
*stylistic* signal rather than general semantics.  This module supplies both
sides of that comparison:

``SkipGramWord2Vec``
    Skip-gram with negative sampling, written directly in PyTorch.  (``gensim``
    has no Python 3.14 wheel here, and writing it out makes the training
    regime — subsampling, negative distribution, epochs — explicit rather than
    hidden behind a library default.)  It is fit on the **training split only**,
    so no test text ever reaches the representation.

``BanglaBertEmbedder``
    Frozen ``csebuetnlp/banglabert``, mean-pooled over the last hidden layer.
    Pretrained on ~27 GB of modern Bangla — orders of magnitude more text than
    this corpus, but modern text, and pretrained for semantics.

Passage vectors are built by **SIF-style weighted averaging** (Arora et al.,
2017): each token is weighted ``a / (a + p(w))``, which damps the very frequent
words.  For a *stylometric* task that is a real trade-off worth naming — the
frequent function words SIF damps are exactly the ones classical stylometry
relies on — so :func:`average_vectors` can also be called with plain unweighted
averaging, and the report compares the two.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import config
from .normalize import word_tokenize


# ---------------------------------------------------------------------------
# From-scratch skip-gram with negative sampling
# ---------------------------------------------------------------------------
class SkipGramWord2Vec:
    """Skip-gram / negative sampling word2vec, trained from scratch."""

    def __init__(
        self,
        dim: int = config.W2V_DIM,
        window: int = config.W2V_WINDOW,
        negative: int = config.W2V_NEGATIVE,
        min_count: int = config.W2V_MIN_COUNT,
        epochs: int = config.W2V_EPOCHS,
        batch_size: int = config.W2V_BATCH,
        lr: float = config.W2V_LR,
        subsample_t: float = 1e-4,
        seed: int = config.SEED,
        device: str | None = None,
    ):
        self.dim = dim
        self.window = window
        self.negative = negative
        self.min_count = min_count
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.subsample_t = subsample_t
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab: dict[str, int] = {}
        self.index_to_word: list[str] = []
        self.vectors: np.ndarray | None = None
        self.word_freq: dict[str, float] = {}

    # -- vocabulary --------------------------------------------------------
    def _build_vocab(self, corpus: list[list[str]]) -> None:
        counts = Counter(t for doc in corpus for t in doc)
        kept = [(w, c) for w, c in counts.most_common() if c >= self.min_count]
        self.index_to_word = [w for w, _ in kept]
        self.vocab = {w: i for i, w in enumerate(self.index_to_word)}
        self.counts = np.array([c for _, c in kept], dtype=np.float64)
        total = self.counts.sum()
        self.word_freq = {
            w: self.counts[i] / total for i, w in enumerate(self.index_to_word)
        }
        # word2vec's 3/4-power unigram noise distribution
        noise = self.counts**0.75
        self.noise_probs = torch.tensor(noise / noise.sum(), dtype=torch.float)
        # Mikolov subsampling keep-probability
        f = self.counts / total
        self.keep_prob = np.minimum(
            1.0, (np.sqrt(f / self.subsample_t) + 1) * (self.subsample_t / f)
        )

    # -- pair generation ---------------------------------------------------
    def _pairs(self, corpus: list[list[str]], rng: random.Random):
        centers: list[int] = []
        contexts: list[int] = []
        for doc in corpus:
            ids = [self.vocab[t] for t in doc if t in self.vocab]
            ids = [i for i in ids if rng.random() < self.keep_prob[i]]
            for pos, center in enumerate(ids):
                # Dynamic window: word2vec samples the effective radius, which
                # up-weights near context words without a separate weight term.
                w = rng.randint(1, self.window)
                lo, hi = max(0, pos - w), min(len(ids), pos + w + 1)
                for j in range(lo, hi):
                    if j == pos:
                        continue
                    centers.append(center)
                    contexts.append(ids[j])
        return (
            torch.tensor(centers, dtype=torch.long),
            torch.tensor(contexts, dtype=torch.long),
        )

    # -- training ----------------------------------------------------------
    def fit(self, texts: list[str], verbose: bool = True) -> SkipGramWord2Vec:
        torch.manual_seed(self.seed)
        rng = random.Random(self.seed)
        corpus = [word_tokenize(t) for t in texts]
        self._build_vocab(corpus)
        V = len(self.index_to_word)
        if verbose:
            n_tok = sum(len(d) for d in corpus)
            print(f"  vocabulary {V:,} types from {n_tok:,} tokens "
                  f"(min_count={self.min_count})")

        emb_in = nn.Embedding(V, self.dim).to(self.device)
        emb_out = nn.Embedding(V, self.dim).to(self.device)
        nn.init.uniform_(emb_in.weight, -0.5 / self.dim, 0.5 / self.dim)
        nn.init.zeros_(emb_out.weight)
        opt = torch.optim.Adam(
            list(emb_in.parameters()) + list(emb_out.parameters()), lr=self.lr
        )
        noise = self.noise_probs.to(self.device)

        for epoch in range(self.epochs):
            centers, contexts = self._pairs(corpus, rng)
            perm = torch.randperm(centers.numel())
            centers, contexts = centers[perm], contexts[perm]
            total_loss = 0.0
            n_batches = 0
            for i in range(0, centers.numel(), self.batch_size):
                c = centers[i : i + self.batch_size].to(self.device)
                o = contexts[i : i + self.batch_size].to(self.device)
                if c.numel() == 0:
                    continue
                neg = torch.multinomial(
                    noise, c.numel() * self.negative, replacement=True
                ).view(c.numel(), self.negative)

                v_c = emb_in(c)                        # (B, d)
                v_o = emb_out(o)                       # (B, d)
                v_n = emb_out(neg)                     # (B, k, d)

                pos_score = (v_c * v_o).sum(-1)
                neg_score = torch.bmm(v_n, v_c.unsqueeze(-1)).squeeze(-1)
                loss = (
                    -nn.functional.logsigmoid(pos_score).mean()
                    - nn.functional.logsigmoid(-neg_score).sum(-1).mean()
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss)
                n_batches += 1
            if verbose:
                print(f"  epoch {epoch + 1}/{self.epochs}  "
                      f"pairs={centers.numel():,}  "
                      f"loss={total_loss / max(n_batches, 1):.4f}")

        self.vectors = emb_in.weight.detach().cpu().numpy()
        return self

    # -- use ---------------------------------------------------------------
    def __getitem__(self, word: str) -> np.ndarray | None:
        i = self.vocab.get(word)
        return None if i is None else self.vectors[i]

    def most_similar(self, word: str, k: int = 10) -> list[tuple[str, float]]:
        v = self[word]
        if v is None:
            return []
        M = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-9)
        q = v / (np.linalg.norm(v) + 1e-9)
        sims = M @ q
        order = np.argsort(-sims)[: k + 1]
        return [
            (self.index_to_word[i], float(sims[i]))
            for i in order
            if self.index_to_word[i] != word
        ][:k]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            vectors=self.vectors,
            words=np.array(self.index_to_word, dtype=object),
            counts=self.counts,
        )

    @classmethod
    def load(cls, path: str | Path) -> SkipGramWord2Vec:
        d = np.load(path, allow_pickle=True)
        m = cls()
        m.vectors = d["vectors"]
        m.index_to_word = list(d["words"])
        m.vocab = {w: i for i, w in enumerate(m.index_to_word)}
        m.counts = d["counts"]
        total = m.counts.sum()
        m.word_freq = {w: m.counts[i] / total for i, w in enumerate(m.index_to_word)}
        return m


# ---------------------------------------------------------------------------
# Passage vectors from word vectors
# ---------------------------------------------------------------------------
def average_vectors(
    texts: list[str],
    model: SkipGramWord2Vec,
    weighting: str = "sif",
    sif_a: float = 1e-3,
) -> np.ndarray:
    """Pool word vectors into one vector per passage.

    ``weighting='sif'`` applies ``a / (a + p(w))``, damping frequent words;
    ``weighting='mean'`` is a plain average, which *keeps* the function-word
    mass that stylometry cares about.  The report runs both.
    """
    dim = model.vectors.shape[1]
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for r, text in enumerate(texts):
        toks = [t for t in word_tokenize(text) if t in model.vocab]
        if not toks:
            continue
        idx = np.array([model.vocab[t] for t in toks])
        if weighting == "sif":
            p = np.array([model.word_freq[t] for t in toks])
            w = sif_a / (sif_a + p)
        else:
            w = np.ones(len(toks))
        w = w / w.sum()
        out[r] = (model.vectors[idx] * w[:, None]).sum(0)
    return out


# ---------------------------------------------------------------------------
# Pretrained BanglaBERT
# ---------------------------------------------------------------------------
class BanglaBertEmbedder:
    """Frozen pretrained encoder, mean-pooled over real (non-pad) tokens."""

    def __init__(
        self,
        model_name: str = config.PRETRAINED_MODEL,
        max_len: int = config.BERT_MAX_LEN,
        device: str | None = None,
        batch_size: int = 16,
    ):
        import os

        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(os.cpu_count() or 4)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str], verbose: bool = False) -> np.ndarray:
        vecs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_len, return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc).last_hidden_state       # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            vecs.append(pooled.cpu().numpy())
            if verbose and (i // self.batch_size) % 20 == 0:
                print(f"    encoded {i + len(batch)}/{len(texts)}", flush=True)
        return np.vstack(vecs).astype(np.float32)


# ---------------------------------------------------------------------------
# Diagnostic: does a representation encode style or topic?
# ---------------------------------------------------------------------------
def style_vs_topic_score(
    vectors: np.ndarray, authors: list[str], works: list[str]
) -> dict:
    """Compare how strongly a representation clusters by author vs by work.

    Both are silhouette scores on the same vectors.  A representation that
    captures *style* should separate authors at least as well as it separates
    individual books; one that mostly captures *topic* separates books much
    better than authors, because each book has its own subject matter.  The
    ratio is the number the proposal's representation-level question actually
    turns on.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder

    a = LabelEncoder().fit_transform(authors)
    w = LabelEncoder().fit_transform(works)
    s_author = float(silhouette_score(vectors, a, metric="cosine"))
    s_work = float(silhouette_score(vectors, w, metric="cosine"))
    return {
        "silhouette_author": s_author,
        "silhouette_work": s_work,
        "style_topic_ratio": s_author / s_work if abs(s_work) > 1e-9 else math.nan,
    }
