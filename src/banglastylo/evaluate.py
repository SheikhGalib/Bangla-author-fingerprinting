"""Metrics, confusion analysis, ablation tables and figures.

Accuracy alone hides the thing the proposal actually asks about — *which*
authors get confused with *which*.  So every evaluation here returns the
confusion matrix alongside the headline numbers, and
:func:`confusable_pairs` ranks author pairs by how often they are mistaken for
each other in either direction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

_AUTHOR_LABEL = {k: v["en"] for k, v in config.AUTHORS.items()}
_AUTHOR_SHORT = {k: v.get("short", v["en"]) for k, v in config.AUTHORS.items()}


def label(author_key: str) -> str:
    return _AUTHOR_LABEL.get(author_key, author_key)


def short(author_key: str) -> str:
    """Axis-length name.

    Surnames alone will not do: two authors in this corpus are Tagore and two
    are Chattopadhyay, so surname labels collide silently on plots and fail
    loudly as duplicate DataFrame columns.
    """
    return _AUTHOR_SHORT.get(author_key, author_key)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def score(y_true, y_pred, classes: list[str] | None = None) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    classes = classes or sorted(set(y_true) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "classes": classes,
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(classes)
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }


def bootstrap_ci(
    y_true, y_pred, n: int = 2000, seed: int = config.SEED, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for accuracy, so single numbers carry error bars."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    m = len(y_true)
    accs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        accs[i] = (y_true[idx] == y_pred[idx]).mean()
    return float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))


def confusable_pairs(result: dict, top_k: int = 8) -> list[tuple[str, str, int, float]]:
    """Author pairs ranked by symmetric confusion rate."""
    classes = result["classes"]
    cm = np.array(result["confusion"], dtype=float)
    rows = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            n_wrong = cm[i, j] + cm[j, i]
            n_total = cm[i].sum() + cm[j].sum()
            rows.append(
                (classes[i], classes[j], int(n_wrong),
                 float(n_wrong / n_total) if n_total else 0.0)
            )
    return sorted(rows, key=lambda r: -r[3])[:top_k]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_confusion(
    result: dict, title: str, path: str | Path | None = None, normalize: bool = True
):
    import matplotlib.pyplot as plt

    classes = result["classes"]
    cm = np.array(result["confusion"], dtype=float)
    if normalize:
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)

    # Two of these are printed side by side at half text width, so the tick
    # labels have to survive being scaled to ~45 % — hence the generous sizes.
    fig, ax = plt.subplots(figsize=(1.05 * len(classes) + 2.4,
                                    0.92 * len(classes) + 1.9))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())
    names = [short(c) for c in classes]
    ax.set_xticks(range(len(classes)), names, rotation=35, ha="right", fontsize=11)
    ax.set_yticks(range(len(classes)), names, fontsize=11)
    ax.set_xlabel("predicted author", fontsize=11)
    ax.set_ylabel("true author", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                        ha="center", va="center", fontsize=10,
                        color="white" if v > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        fig.savefig(config.REPORT_FIGURES / Path(path).name, dpi=200,
                    bbox_inches="tight")
    return fig


def plot_ablation(df: pd.DataFrame, path: str | Path | None = None, title: str = ""):
    import matplotlib.pyplot as plt

    d = df.sort_values("accuracy")
    # Kept wide and short: at full text width a taller figure runs into the
    # page footer, and there is nothing in the vertical direction worth the room.
    fig, ax = plt.subplots(figsize=(8.4, 0.30 * len(d) + 1.4))
    y = np.arange(len(d))
    ax.barh(y - 0.19, d["accuracy"], height=0.36, label="accuracy", color="#2F5D62")
    ax.barh(y + 0.19, d["macro_f1"], height=0.36, label="macro-F1", color="#A8C6BE")
    ax.set_yticks(y, d["model"], fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.axvline(d["chance"].iloc[0], ls="--", lw=1, color="#B04A3F",
               label=f"chance ({d['chance'].iloc[0]:.2f})")
    ax.set_xlabel("score on the held-out test split")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        fig.savefig(config.REPORT_FIGURES / Path(path).name, dpi=200,
                    bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Result bookkeeping
# ---------------------------------------------------------------------------
class ResultTable:
    """Accumulates one row per experiment and writes CSV + LaTeX for the report."""

    def __init__(self, chance: float):
        self.chance = chance
        self.rows: list[dict] = []
        self.full: dict[str, dict] = {}

    def add(self, name: str, family: str, y_true, y_pred, notes: str = "") -> dict:
        res = score(y_true, y_pred)
        lo, hi = bootstrap_ci(y_true, y_pred)
        self.rows.append(
            {
                "model": name,
                "family": family,
                "accuracy": res["accuracy"],
                "acc_lo": lo,
                "acc_hi": hi,
                "macro_f1": res["macro_f1"],
                "chance": self.chance,
                "notes": notes,
            }
        )
        self.full[name] = res
        print(f"  {name:44s} acc={res['accuracy']:.4f} "
              f"[{lo:.3f}, {hi:.3f}]  macroF1={res['macro_f1']:.4f}", flush=True)
        return res

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, stem: str) -> None:
        df = self.frame()
        df.to_csv(config.TABLES / f"{stem}.csv", index=False)
        (config.TABLES / f"{stem}_full.json").write_text(
            json.dumps(self.full, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"wrote {config.TABLES / (stem + '.csv')}")

    def to_latex(self, stem: str, caption: str, tex_label: str) -> str:
        df = self.frame().sort_values("accuracy", ascending=False)
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llcc}",
            r"\toprule",
            r"Model & Feature family & Accuracy (95\% CI) & Macro-F1 \\",
            r"\midrule",
        ]
        for _, r in df.iterrows():
            lines.append(
                f"{_tex(r['model'])} & {_tex(r['family'])} & "
                f"{r['accuracy']:.3f} \\;[{r['acc_lo']:.3f}, {r['acc_hi']:.3f}] & "
                f"{r['macro_f1']:.3f} \\\\"
            )
        lines += [
            r"\midrule",
            f"Chance & --- & {self.chance:.3f} & {self.chance:.3f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{tex_label}}}",
            r"\end{table}",
        ]
        tex = "\n".join(lines)
        (config.TABLES / f"{stem}.tex").write_text(tex, encoding="utf-8")
        (config.REPORTS / "tables").mkdir(exist_ok=True)
        (config.REPORTS / "tables" / f"{stem}.tex").write_text(tex, encoding="utf-8")
        return tex


def _tex(s: str) -> str:
    for a, b in (("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#")):
        s = str(s).replace(a, b)
    return s
