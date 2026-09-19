"""Neural sequence classifiers trained from scratch: BiLSTM and Transformer.

These sit between the two extremes the project already had.  On one side the
n-gram and stylometric models see no word order beyond a fixed window; on the
other, fine-tuned BanglaBERT arrives with 110M pretrained parameters and the
advantage is impossible to attribute.  A BiLSTM and a small Transformer encoder
trained here, from random initialisation, on this corpus alone, are the honest
middle: they can model order, and they know nothing that was not in the
training books.

Both are deliberately built out of the pieces rather than pulled from a model
hub -- the embedding table, the positional encoding, the padding mask and the
pooling step are each written out, because each one is a place where a sequence
classifier quietly goes wrong.

The padding mask is the subtle one.  Passages differ in length, so batches are
padded; without a mask the model averages real tokens together with padding and
a short passage gets its signal diluted in proportion to how short it is.  Both
classes below mask padding out of attention *and* out of the pooling average.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import config
from .normalize import word_tokenize

PAD, UNK = 0, 1

#: Sequence length in tokens.  Passages are ~220 tokens by construction
#: (``config.PASSAGE_TOKENS``); 256 covers almost all of them without padding
#: the batch out to a length only a handful of passages reach.
MAX_LEN = 256

#: Vocabulary cap.  Bangla is heavily inflected, so the tail is long and mostly
#: singletons; keeping everything would spend most of the embedding table on
#: words seen once in one book.
MAX_VOCAB = 30_000


def build_vocab(texts: list[str], max_size: int = MAX_VOCAB,
                min_count: int = 2) -> dict[str, int]:
    """Frequency-ranked vocabulary with reserved ``PAD`` and ``UNK`` slots.

    Fitted on the training texts only -- a vocabulary built over the test book
    would leak which words that book contains, which is a real if subtle form
    of the topic leakage this project is about.
    """
    from collections import Counter

    counts = Counter()
    for t in texts:
        counts.update(word_tokenize(t))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for word, n in counts.most_common(max_size - 2):
        if n < min_count:
            break
        vocab[word] = len(vocab)
    return vocab


def encode(texts: list[str], vocab: dict[str, int],
           max_len: int = MAX_LEN) -> np.ndarray:
    """Tokenise, map to ids, truncate and right-pad to a fixed length."""
    out = np.full((len(texts), max_len), PAD, dtype=np.int64)
    for r, text in enumerate(texts):
        ids = [vocab.get(t, UNK) for t in word_tokenize(text)][:max_len]
        out[r, : len(ids)] = ids
    return out


def _embedding_matrix(vocab: dict[str, int], dim: int, w2v=None,
                      seed: int = config.SEED) -> np.ndarray:
    """Random table, overwritten by from-scratch word2vec rows where available.

    Seeding the table with vectors the project trained itself (notebook 03) is
    the Lab 4 pattern of loading pretrained embeddings into an embedding layer
    -- except the "pretrained" vectors here come from this corpus, so nothing
    external enters the experiment.
    """
    rng = np.random.default_rng(seed)
    mat = rng.normal(0.0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    mat[PAD] = 0.0
    if (w2v is not None and getattr(w2v, "vectors", None) is not None
            and w2v.vectors.shape[1] == dim):
        hits = 0
        for word, i in vocab.items():
            j = w2v.vocab.get(word)
            if j is not None:
                mat[i] = w2v.vectors[j]
                hits += 1
        print(f"    seeded {hits}/{len(vocab)} embedding rows from word2vec")
    return mat


class _TorchClassifier:
    """Shared training loop, so the two architectures differ only in the model.

    Model selection is on validation accuracy and the best state is restored at
    the end.  Without that, the reported score depends on where the last epoch
    happened to land, which on a corpus this size is a swing of several points.
    """

    def __init__(
        self,
        epochs: int = 12,
        batch_size: int = 32,
        lr: float = 1e-3,
        seed: int = config.SEED,
        device: str | None = None,
    ):
        import torch

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_: list[str] = []
        self.vocab: dict[str, int] = {}
        self.history_: list[dict] = []

    def _build(self, n_vocab: int, n_classes: int, emb: np.ndarray):
        raise NotImplementedError

    def fit(self, texts: list[str], authors: list[str],
            val: tuple[list[str], list[str]] | None = None,
            w2v=None, verbose: bool = True):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.classes_ = sorted(set(authors))
        label_of = {a: i for i, a in enumerate(self.classes_)}
        self.vocab = build_vocab(texts)

        emb = _embedding_matrix(self.vocab, self.emb_dim, w2v, self.seed)
        self.model = self._build(len(self.vocab), len(self.classes_), emb)
        self.model.to(self.device)

        X = torch.from_numpy(encode(texts, self.vocab))
        y = torch.tensor([label_of[a] for a in authors])
        dl = DataLoader(TensorDataset(X, y), batch_size=self.batch_size,
                        shuffle=True, drop_last=False)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, total_steps=self.epochs * len(dl), pct_start=0.2)
        loss_fn = torch.nn.CrossEntropyLoss()

        best_acc, best_state = -1.0, None
        self.history_ = []
        for epoch in range(self.epochs):
            self.model.train()
            running, seen = 0.0, 0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                running += float(loss.detach()) * yb.numel()
                seen += yb.numel()

            row = {"epoch": epoch + 1, "train_loss": running / max(seen, 1)}
            if val is not None:
                from sklearn.metrics import accuracy_score
                acc = float(accuracy_score(val[1], self.predict(val[0])))
                row["val_acc"] = acc
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
            self.history_.append(row)
            if verbose:
                extra = f"  val_acc={row['val_acc']:.4f}" if val is not None else ""
                print(f"  epoch {row['epoch']:>3}/{self.epochs} "
                      f"loss={row['train_loss']:.4f}{extra}", flush=True)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f"  restored best checkpoint (val_acc={best_acc:.4f})")
        return self

    def _logits(self, texts: list[str]) -> np.ndarray:
        import torch

        self.model.eval()
        X = torch.from_numpy(encode(texts, self.vocab))
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                out.append(self.model(X[i : i + 64].to(self.device)).cpu().numpy())
        return np.vstack(out)

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.array([self.classes_[i] for i in self._logits(texts).argmax(1)])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        z = self._logits(texts)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    #: Constructor arguments that define the architecture, so a checkpoint can
    #: be rebuilt without the caller having to remember how it was configured.
    ARCH_KEYS: tuple[str, ...] = ()

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        (path / "meta.json").write_text(json.dumps({
            "classes": self.classes_,
            "vocab": self.vocab,
            "kind": type(self).__name__,
            "history": self.history_,
            "arch": {k: getattr(self, k) for k in self.ARCH_KEYS},
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, device: str | None = None):
        """Rebuild a trained classifier from a directory written by ``save``.

        The architecture is taken from the checkpoint's ``arch`` block when it
        is present and from the class defaults when it is not -- checkpoints
        written before ``arch`` was recorded still load, and a mismatch between
        the two shows up immediately as a state-dict shape error rather than as
        quietly wrong predictions.
        """
        import torch

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        obj = cls(device=device or "cpu", **meta.get("arch", {}))
        obj.classes_ = meta["classes"]
        obj.vocab = meta["vocab"]
        obj.history_ = meta.get("history", [])

        # A zero table: the real weights arrive with the state dict below.
        emb = np.zeros((len(obj.vocab), obj.emb_dim), dtype=np.float32)
        obj.model = obj._build(len(obj.vocab), len(obj.classes_), emb)
        state = torch.load(path / "model.pt", map_location=obj.device)
        obj.model.load_state_dict(state)
        obj.model.to(obj.device).eval()
        return obj


# ---------------------------------------------------------------------------
# Lab 4: recurrent
# ---------------------------------------------------------------------------
class BiLSTMClassifier(_TorchClassifier):
    """Two-layer bidirectional LSTM with masked mean-plus-max pooling.

    Concatenating the mean and the max of the hidden states, rather than taking
    the final one, matters for this task.  An author's habits are spread evenly
    through a passage; the final hidden state of a 220-token sequence is
    dominated by its last clause, which is the one place the style signal is
    *not* concentrated.
    """

    emb_dim = config.W2V_DIM
    ARCH_KEYS = ("hidden", "layers", "dropout")

    def __init__(self, hidden: int = 192, layers: int = 2,
                 dropout: float = 0.3, **kw):
        super().__init__(**kw)
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout

    def _build(self, n_vocab: int, n_classes: int, emb: np.ndarray):
        import torch
        from torch import nn

        hidden, layers, dropout = self.hidden, self.layers, self.dropout

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(n_vocab, emb.shape[1], padding_idx=PAD)
                self.emb.weight.data.copy_(torch.from_numpy(emb))
                self.lstm = nn.LSTM(
                    emb.shape[1], hidden, num_layers=layers, batch_first=True,
                    bidirectional=True, dropout=dropout if layers > 1 else 0.0)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(hidden * 4, n_classes)

            def forward(self, x):
                mask = (x != PAD).unsqueeze(-1)            # [B, T, 1]
                h, _ = self.lstm(self.emb(x))              # [B, T, 2H]
                h = h.masked_fill(~mask, 0.0)
                denom = mask.sum(1).clamp(min=1)
                mean = h.sum(1) / denom                    # padding excluded
                mx = h.masked_fill(~mask, -1e9).max(1).values
                return self.head(self.drop(torch.cat([mean, mx], dim=1)))

        return Net()


# ---------------------------------------------------------------------------
# Lab 5: transformer encoder, from scratch
# ---------------------------------------------------------------------------
class PositionalEncoding:
    """Namespace holder; the module is built inside the classifier."""


class TransformerClassifier(_TorchClassifier):
    """Encoder-only Transformer with sinusoidal positions and masked pooling.

    This is the Lab 5 architecture at corpus scale: embed, add a positional
    signal, run self-attention blocks, mean-pool the token representations and
    classify.  It is small on purpose -- four heads, three layers -- because a
    few thousand passages cannot support anything larger, and the report's
    point is the comparison against pretrained BanglaBERT rather than a bid to
    beat it.

    Self-attention is permutation-invariant, so without the positional term the
    model would see a bag of words and the whole architecture would be an
    expensive way to average embeddings.
    """

    emb_dim = 192
    ARCH_KEYS = ("d_model", "nhead", "layers", "ff", "dropout")

    def __init__(self, d_model: int = 192, nhead: int = 4, layers: int = 3,
                 ff: int = 384, dropout: float = 0.2, **kw):
        super().__init__(**kw)
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers
        self.ff = ff
        self.dropout = dropout
        self.emb_dim = d_model

    def _build(self, n_vocab: int, n_classes: int, emb: np.ndarray):
        import torch
        from torch import nn

        d_model, nhead, layers = self.d_model, self.nhead, self.layers
        ff, dropout = self.ff, self.dropout

        def sinusoid(max_len: int, dim: int) -> torch.Tensor:
            pe = torch.zeros(max_len, dim)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, dim, 2).float()
                            * (-math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe.unsqueeze(0)                          # [1, T, D]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(n_vocab, d_model, padding_idx=PAD)
                self.emb.weight.data.copy_(torch.from_numpy(emb))
                self.register_buffer("pe", sinusoid(MAX_LEN, d_model))
                self.drop = nn.Dropout(dropout)
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=ff,
                    dropout=dropout, batch_first=True, norm_first=True)
                self.enc = nn.TransformerEncoder(layer, num_layers=layers)
                self.norm = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, n_classes)

            def forward(self, x):
                pad = x == PAD                              # True where padding
                h = self.emb(x) * math.sqrt(d_model) + self.pe[:, : x.size(1)]
                h = self.enc(self.drop(h), src_key_padding_mask=pad)
                keep = (~pad).unsqueeze(-1).float()
                pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
                return self.head(self.norm(pooled))

        return Net()
