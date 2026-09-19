"""From-scratch Naive Bayes and softmax regression.

Both classifiers exist in scikit-learn, and the SVM in :mod:`models` is a
scikit-learn pipeline.  These are written out in NumPy anyway for one reason:
they are the two classifiers the course derived by hand, and a derivation is
only believed once the arithmetic it implies actually runs.  The versions here
are the lecture equations scaled up from a toy sentiment set to a real feature
matrix -- same maths, same variable names where it helps, no library shortcuts.

Naive Bayes is *generative*: it models ``P(features | author)`` and applies
Bayes' rule.  Softmax regression is *discriminative*: it fits the decision
boundary ``P(author | features)`` directly.  The project's headline comparison
is exactly this axis, so having a matched pair at the simplest possible level
is worth the code.

Both follow the scikit-learn ``fit``/``predict``/``predict_proba`` protocol so
that :mod:`evaluate` can score them beside everything else without special
cases.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from . import config


def _as_array(X):
    """Accept a sparse matrix or a dense array; return something indexable."""
    return X if sparse.issparse(X) else np.asarray(X, dtype=np.float64)


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes with Laplace (add-alpha) smoothing.

    Works on non-negative count-like features -- character n-gram counts or
    TF-IDF weights both behave sensibly.  Everything is computed in log space:
    a passage contributes a few hundred feature terms, and the product of that
    many probabilities underflows float64 long before the comparison matters.

    The smoothing is the part worth stating.  Any n-gram an author never used
    has zero empirical probability, which would veto that author on a single
    unseen feature no matter how much the rest of the passage agrees.  Adding
    ``alpha`` to every count is what stops one absent n-gram from carrying a
    veto it has not earned.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: np.ndarray | None = None

    def fit(self, X, y) -> MultinomialNaiveBayes:
        X = _as_array(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # log P(c) -- the prior, straight from the class frequencies.
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=np.float64)
        self.log_prior_ = np.log(counts / counts.sum())

        # log P(w | c) -- summed feature mass per class, then add-alpha.
        feat = np.zeros((n_classes, n_features), dtype=np.float64)
        for i, c in enumerate(self.classes_):
            rows = X[y == c]
            total = rows.sum(axis=0)
            feat[i] = np.asarray(total).ravel()
        feat += self.alpha
        self.log_likelihood_ = np.log(feat / feat.sum(axis=1, keepdims=True))
        return self

    def joint_log_likelihood(self, X) -> np.ndarray:
        """``log P(c) + sum_w count(w) * log P(w | c)`` for every class."""
        X = _as_array(X)
        return np.asarray(X @ self.log_likelihood_.T) + self.log_prior_

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.joint_log_likelihood(X).argmax(axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        z = self.joint_log_likelihood(X)
        z = z - z.max(axis=1, keepdims=True)      # stabilise before exp
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    """Multinomial logistic regression trained by mini-batch gradient descent.

    The binary logistic regression from the lab generalises to ``K`` authors by
    replacing the sigmoid with a softmax and the binary cross-entropy with its
    categorical form.  The gradient keeps the same shape it had in the binary
    case -- ``X.T @ (predicted - actual)`` -- which is the detail worth seeing
    survive the generalisation.

    L2 regularisation is applied to the weights but deliberately not to the
    bias: penalising the intercept would bias the model against classes that
    are simply more common, which is a different thing from overfitting.
    """

    def __init__(
        self,
        lr: float = 0.5,
        epochs: int = 200,
        batch_size: int = 256,
        l2: float = 1e-4,
        seed: int = config.SEED,
        verbose: bool = False,
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed
        self.verbose = verbose
        self.classes_: np.ndarray | None = None
        self.history_: list[float] = []

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y) -> SoftmaxRegression:
        X = _as_array(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        index = {c: i for i, c in enumerate(self.classes_)}
        n, d = X.shape
        k = len(self.classes_)

        # One-hot targets.
        Y = np.zeros((n, k), dtype=np.float64)
        Y[np.arange(n), [index[v] for v in y]] = 1.0

        rng = np.random.default_rng(self.seed)
        self.W = np.zeros((d, k), dtype=np.float64)
        self.b = np.zeros(k, dtype=np.float64)

        self.history_ = []
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = X[idx]
                yb = Y[idx]
                m = xb.shape[0]

                # Forward.
                logits = np.asarray(xb @ self.W) + self.b
                p = self._softmax(logits)

                # Cross-entropy, guarded against log(0).
                total += -np.sum(yb * np.log(np.clip(p, 1e-12, None)))

                # Backward: the binary gradient, one column per class.
                diff = (p - yb) / m
                gW = np.asarray(xb.T @ diff) + self.l2 * self.W
                gb = diff.sum(axis=0)

                self.W -= self.lr * gW
                self.b -= self.lr * gb

            loss = total / n
            self.history_.append(loss)
            if self.verbose and (epoch % 20 == 0 or epoch == self.epochs - 1):
                print(f"    epoch {epoch + 1:>4}/{self.epochs}  loss={loss:.4f}",
                      flush=True)
        return self

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(_as_array(X) @ self.W) + self.b

    def predict_proba(self, X) -> np.ndarray:
        return self._softmax(self.decision_function(X))

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.decision_function(X).argmax(axis=1)]


# ---------------------------------------------------------------------------
# Lab 1: edit distance
# ---------------------------------------------------------------------------
def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, the standard two-row dynamic program.

    Kept because the corpus needs it: the two sources spell the same Bangla
    word in more than one way (``ন`` vs ``ণ``, ``ি`` vs ``ী``, optional hasanta),
    and near-identical spellings would otherwise become two vocabulary entries
    and dilute the very counts the stylometry depends on.  Only the previous
    row is ever read, so the table is two rows rather than ``len(a) x len(b)``.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,                      # deletion
                cur[j - 1] + 1,                   # insertion
                prev[j - 1] + (ca != cb),         # substitution
            ))
        prev = cur
    return prev[-1]


def nearest_vocabulary_word(
    word: str, vocabulary: list[str], max_distance: int = 2
) -> tuple[str, int] | None:
    """Closest in-vocabulary spelling of ``word``, or ``None`` if none is near.

    Candidates are pre-filtered on length before the distance is computed: a
    word cannot be within ``max_distance`` edits of one whose length differs by
    more than that, and skipping those makes the scan over a large vocabulary
    cheap enough to run inside the interface.
    """
    best: tuple[str, int] | None = None
    for cand in vocabulary:
        if abs(len(cand) - len(word)) > max_distance:
            continue
        d = edit_distance(word, cand)
        if d <= max_distance and (best is None or d < best[1]):
            best = (cand, d)
            if d == 1:
                break
    return best
