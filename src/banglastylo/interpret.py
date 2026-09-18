"""Making a prediction auditable.

The proposal's interface requirement is that a prediction come back "alongside
the ranked stylistic features that most influenced the decision, so predictions
remain auditable rather than opaque".  Three mechanisms serve that here:

``linear_contributions``
    For a linear SVM the exact arithmetic behind one prediction is
    ``w · x + b``.  The contribution of feature *i* to class *c* is
    ``w[c, i] * x[i]`` — not an approximation, the actual decomposition of the
    score.  This is the same quantity SHAP would estimate for a linear model,
    computed exactly instead of sampled, which is why no SHAP dependency is
    needed here.

``permutation_importance_grouped``
    Model-agnostic, and grouped by *feature family* rather than by individual
    feature, which is the level the ablation question is asked at: shuffle every
    character-n-gram column together and see what accuracy costs.

``family_signature``
    Not an explanation of one prediction but of one *author*: the structural
    features on which that author's passages sit furthest from the corpus mean,
    in standard deviations.  This is what the report's "style fingerprint"
    figure draws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def _dense_row(X, r: int) -> np.ndarray:
    return np.asarray(X[r].todense()).ravel() if sparse.issparse(X) else np.asarray(X[r])


# ---------------------------------------------------------------------------
# Exact linear attribution
# ---------------------------------------------------------------------------
def linear_contributions(
    pipeline, text: str, top_k: int = 15, against: bool = True
) -> dict:
    """Decompose one linear-SVM decision into per-feature contributions."""
    feats = pipeline.named_steps["features"]
    scaler = pipeline.named_steps.get("scale")
    clf = pipeline.named_steps["clf"]

    X = feats.transform([text])
    if scaler is not None:
        X = scaler.transform(X)
    x = _dense_row(X, 0)
    names = np.asarray(feats.get_feature_names_out())

    W = clf.coef_
    b = clf.intercept_
    # sklearn hands back numpy.str_, which is a str subclass but surprises
    # callers that check `type(x) is str` or serialise strictly.  Everything
    # leaving this module is a plain Python type.
    classes = [str(c) for c in clf.classes_]

    if W.shape[0] == 1:                       # binary case
        W = np.vstack([-W[0], W[0]])
        b = np.array([-b[0], b[0]])

    scores = W @ x + b
    winner = int(np.argmax(scores))
    runner = int(np.argsort(scores)[-2]) if len(scores) > 1 else winner

    contrib = (W[winner] - W[runner]) * x     # evidence for winner over runner
    order = np.argsort(-contrib)

    out = {
        "predicted": classes[winner],
        "runner_up": classes[runner],
        "margin": float(scores[winner] - scores[runner]),
        "scores": {classes[i]: float(scores[i]) for i in range(len(classes))},
        "for_prediction": [
            (str(names[i]), float(contrib[i]), float(x[i]))
            for i in order[:top_k]
            if contrib[i] > 0
        ],
    }
    if against:
        out["against_prediction"] = [
            (str(names[i]), float(contrib[i]), float(x[i]))
            for i in order[::-1][:top_k]
            if contrib[i] < 0
        ]
    return out


def top_features_per_author(pipeline, top_k: int = 12) -> pd.DataFrame:
    """The features each author's SVM row weights most heavily, corpus-wide."""
    feats = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    names = np.asarray(feats.get_feature_names_out())
    rows = []
    for i, author in enumerate(clf.classes_):
        w = clf.coef_[i]
        for j in np.argsort(-w)[:top_k]:
            rows.append({"author": author, "feature": names[j],
                         "weight": float(w[j]), "direction": "for"})
        for j in np.argsort(w)[:top_k]:
            rows.append({"author": author, "feature": names[j],
                         "weight": float(w[j]), "direction": "against"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grouped permutation importance
# ---------------------------------------------------------------------------
def permutation_importance_grouped(
    pipeline, texts: list[str], y: list[str], n_repeats: int = 5,
    seed: int = 0, verbose: bool = True
) -> pd.DataFrame:
    """Accuracy drop when a whole feature family is shuffled.

    Runs on the already-transformed matrix so the expensive feature extraction
    happens once rather than once per repeat.
    """
    from sklearn.metrics import accuracy_score

    feats = pipeline.named_steps["features"]
    scaler = pipeline.named_steps.get("scale")
    clf = pipeline.named_steps["clf"]

    X = feats.transform(texts)
    if scaler is not None:
        X = scaler.transform(X)
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    names = np.asarray(feats.get_feature_names_out())
    prefixes = np.array([n.split(":", 1)[0] for n in names])

    base = accuracy_score(y, clf.predict(X))
    rng = np.random.default_rng(seed)
    rows = []
    for fam in sorted(set(prefixes)):
        cols = np.where(prefixes == fam)[0]
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, cols] = Xp[perm][:, cols]
            drops.append(base - accuracy_score(y, clf.predict(Xp)))
        rows.append({"family": fam, "n_features": len(cols),
                     "mean_drop": float(np.mean(drops)),
                     "sd_drop": float(np.std(drops)), "baseline": base})
        if verbose:
            print(f"  {fam:10s} {len(cols):>5} features  "
                  f"accuracy drop {np.mean(drops):+.4f}", flush=True)
    return pd.DataFrame(rows).sort_values("mean_drop", ascending=False)


# ---------------------------------------------------------------------------
# Per-author style fingerprint
# ---------------------------------------------------------------------------
def family_signature(texts: list[str], authors: list[str]) -> pd.DataFrame:
    """Each author's mean structural feature, in corpus standard deviations."""
    from .stylometry import STRUCTURAL_NAMES, StructuralFeatures

    S = StructuralFeatures().fit(texts).transform(texts)
    mu, sd = S.mean(0), S.std(0) + 1e-9
    Z = (S - mu) / sd
    df = pd.DataFrame(Z, columns=STRUCTURAL_NAMES)
    df["author"] = authors
    return df.groupby("author").mean()


def plot_fingerprint(sig: pd.DataFrame, features: list[str] | None = None,
                     path=None, title: str = "Structural style fingerprints"):
    import matplotlib.pyplot as plt

    from . import config

    features = features or [
        "sadhu_chalit_ratio", "sent_len_mean", "sent_len_sd", "word_len_mean",
        "type_token_ratio", "hapax_rate", "comma_rate", "danda_rate",
        "quote_rate", "conjunct_rate",
    ]
    features = [f for f in features if f in sig.columns]
    M = sig[features].to_numpy()
    lim = float(np.abs(M).max()) or 1.0

    fig, ax = plt.subplots(figsize=(0.85 * len(features) + 3.0,
                                    0.5 * len(sig) + 2.0))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    names = [config.AUTHORS.get(a, {}).get("short", a) for a in sig.index]
    ax.set_yticks(range(len(sig)), names, fontsize=8)
    ax.set_xticks(range(len(features)), features, rotation=40, ha="right", fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j]) > 0.62 * lim else "black")
    ax.set_title(f"{title}\n(author mean, in corpus standard deviations)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, shrink=0.85)
    fig.tight_layout()
    if path:
        from pathlib import Path

        fig.savefig(path, dpi=200, bbox_inches="tight")
        fig.savefig(config.REPORT_FIGURES / Path(path).name, dpi=200,
                    bbox_inches="tight")
    return fig
