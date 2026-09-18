"""The auditable prediction interface.

Paste a Bangla passage, get back the predicted author *and the evidence*: the
ranked stylistic features that moved the decision, from both paradigms at once.

Both attributors are consulted deliberately.  When the discriminative and
generative models agree, the answer is worth more than either alone; when they
disagree, saying so is more honest than picking a winner silently, and the
disagreement itself is diagnostic — it usually means the passage is short or the
two authors share a register.

Used from :mod:`app` (command line) and directly from the notebooks.
"""
from __future__ import annotations

import pickle
import textwrap
from pathlib import Path

from . import config
from .interpret import linear_contributions
from .normalize import normalize_text, sentence_split, word_tokenize
from .stylometry import STRUCTURAL_NAMES, StructuralFeatures

# Human-readable glosses for the structural features, so the output reads as
# an explanation rather than as variable names.
FEATURE_GLOSS = {
    "sent_len_mean": "mean sentence length (tokens)",
    "sent_len_sd": "sentence-length variability",
    "sent_len_skew": "sentence-length skew",
    "sent_len_median": "median sentence length",
    "sent_len_iqr": "sentence-length spread (IQR)",
    "word_len_mean": "mean word length (characters)",
    "long_word_rate": "share of words over 8 characters",
    "type_token_ratio": "type-token ratio",
    "hapax_rate": "share of once-only words",
    "yule_k": "Yule's K (vocabulary richness)",
    "danda_rate": "danda (।) rate",
    "comma_rate": "comma rate",
    "semicolon_rate": "semicolon rate",
    "question_rate": "question-mark rate",
    "quote_rate": "quotation rate (dialogue)",
    "sadhu_rate": "sadhu (literary register) marker rate",
    "chalit_rate": "chalit (colloquial register) marker rate",
    "sadhu_chalit_ratio": "sadhu-to-chalit register ratio",
    "conjunct_rate": "conjunct-consonant density",
    "punct_rate": "overall punctuation rate",
}


def gloss(feature: str) -> str:
    """Turn an internal feature name into something a reader can act on."""
    kind, _, rest = feature.partition(":")
    if kind == "fw":
        return f"function word “{rest}”"
    if kind == "st":
        return FEATURE_GLOSS.get(rest, rest.replace("_", " "))
    if kind == "ch":
        # char_wb n-grams carry word-boundary spaces, and two n-grams that
        # differ only in a leading or trailing space are different features.
        # Rendering the space makes them tell apart on screen.
        return f"character n-gram “{rest.replace(' ', '␣')}”"
    if kind == "pos":
        return f"POS pattern {rest.replace(' ', '–')}"
    return feature


#: Families whose features a human can act on directly.  Character n-grams are
#: excluded not because they are unimportant — they carry most of the weight —
#: but because "the trigram ␣হা" is not an observation a reader can check
#: against the passage the way "this text is in the sadhu register" is.
INTERPRETABLE_FAMILIES = ("fw", "st", "pos")


class Attributor:
    """Bundles the trained SVM and generative models behind one call."""

    def __init__(self, svm_pipeline=None, generative=None, author_meta=None):
        self.svm = svm_pipeline
        self.generative = generative
        self.author_meta = author_meta or config.AUTHORS

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls, directory: str | Path = config.MODELS) -> Attributor:
        directory = Path(directory)
        svm = gen = None
        p = directory / "svm_combined.pkl"
        if p.exists():
            with open(p, "rb") as fh:
                svm = pickle.load(fh)
        p = directory / "generative_char.pkl"
        if p.exists():
            with open(p, "rb") as fh:
                gen = pickle.load(fh)
        if svm is None and gen is None:
            raise FileNotFoundError(
                f"no trained models in {directory}; run notebook 04 first"
            )
        return cls(svm, gen)

    def save(self, directory: str | Path = config.MODELS) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.svm is not None:
            with open(directory / "svm_combined.pkl", "wb") as fh:
                pickle.dump(self.svm, fh)
        if self.generative is not None:
            with open(directory / "generative_char.pkl", "wb") as fh:
                pickle.dump(self.generative, fh)

    # -- prediction --------------------------------------------------------
    def name(self, key: str) -> str:
        meta = self.author_meta.get(key, {})
        return meta.get("en", key)

    def bengali_name(self, key: str) -> str:
        return self.author_meta.get(key, {}).get("bn", key)

    def predict(self, passage: str, top_k: int = 8) -> dict:
        text = normalize_text(passage)
        tokens = word_tokenize(text)
        sents = sentence_split(text)
        out: dict = {
            "n_tokens": len(tokens),
            "n_sentences": len(sents),
            "short_passage_warning": len(tokens) < 80,
        }

        if self.svm is not None:
            # Ask for a deep ranking so that the interpretable families still
            # have entries after character n-grams take the top slots.
            exp = linear_contributions(self.svm, text, top_k=400)

            def rows(items):
                return [
                    {"feature": f, "gloss": gloss(f), "contribution": c, "value": v}
                    for f, c, v in items
                ]

            def interpretable(items):
                return [t for t in items
                        if t[0].split(":", 1)[0] in INTERPRETABLE_FAMILIES]

            out["discriminative"] = {
                "predicted": exp["predicted"],
                "runner_up": exp["runner_up"],
                "margin": exp["margin"],
                "scores": exp["scores"],
                "evidence_for": rows(exp["for_prediction"][:top_k]),
                "evidence_against": rows(
                    exp.get("against_prediction", [])[:top_k]
                ),
                # The same decomposition, restricted to features a reader can
                # verify against the passage by eye.
                "readable_for": rows(
                    interpretable(exp["for_prediction"])[:top_k]
                ),
                "readable_against": rows(
                    interpretable(exp.get("against_prediction", []))[:top_k]
                ),
            }

        if self.generative is not None:
            g = self.generative.explain(text, top_k=top_k)
            out["generative"] = {
                "predicted": g["predicted"],
                "runner_up": g["runner_up"],
                "perplexity": g["perplexity"],
                "evidence_for": [
                    {"symbol": s, "log_odds": d} for s, d in g["for_prediction"]
                ],
            }

        d = out.get("discriminative", {}).get("predicted")
        gg = out.get("generative", {}).get("predicted")
        out["agree"] = (d is not None and gg is not None and d == gg)
        out["predicted"] = d or gg
        return out

    # -- structural profile of the passage itself --------------------------
    @staticmethod
    def profile(passage: str) -> dict[str, float]:
        text = normalize_text(passage)
        v = StructuralFeatures().fit([text]).transform([text])[0]
        return {n: float(v[i]) for i, n in enumerate(STRUCTURAL_NAMES)}


# ---------------------------------------------------------------------------
# Terminal rendering
# ---------------------------------------------------------------------------
def render(result: dict, attributor: Attributor, width: int = 78) -> str:
    lines: list[str] = []
    bar = "=" * width
    lines.append(bar)
    pred = result["predicted"]
    lines.append(f"PREDICTED AUTHOR : {attributor.name(pred)}  "
                 f"({attributor.bengali_name(pred)})")
    lines.append(f"passage          : {result['n_tokens']} tokens, "
                 f"{result['n_sentences']} sentences")
    if result["short_passage_warning"]:
        lines.append("  ! short passage — stylometric estimates are noisy below "
                     "~80 tokens")
    lines.append(bar)

    d = result.get("discriminative")
    if d:
        lines.append("")
        lines.append("DISCRIMINATIVE (linear SVM over stylometric features)")
        lines.append(f"  predicted : {attributor.name(d['predicted'])}")
        lines.append(f"  runner-up : {attributor.name(d['runner_up'])}  "
                     f"(margin {d['margin']:.3f})")
        lines.append("  strongest evidence overall "
                     f"(for {attributor.name(d['predicted'])} over "
                     f"{attributor.name(d['runner_up'])}):")
        for e in d["evidence_for"][:6]:
            lines.append(f"    {e['contribution']:+8.4f}  {e['gloss']}")
        if d.get("readable_for"):
            lines.append("  strongest evidence you can check by eye:")
            for e in d["readable_for"][:6]:
                lines.append(f"    {e['contribution']:+8.4f}  {e['gloss']}"
                             f"   [value {e['value']:.3f}]")
        if d["evidence_against"]:
            lines.append("  evidence pointing the other way:")
            for e in d["evidence_against"][:4]:
                lines.append(f"    {e['contribution']:+8.4f}  {e['gloss']}")

    g = result.get("generative")
    if g:
        lines.append("")
        lines.append("GENERATIVE (per-author character n-gram language models)")
        lines.append(f"  predicted : {attributor.name(g['predicted'])}")
        ppl = sorted(g["perplexity"].items(), key=lambda kv: kv[1])
        lines.append("  perplexity under each author's model (lower = better fit):")
        for a, p in ppl:
            mark = " <-- best" if a == g["predicted"] else ""
            lines.append(f"    {attributor.name(a):32s} {p:9.3f}{mark}")

    lines.append("")
    if result["agree"]:
        lines.append("BOTH PARADIGMS AGREE.")
    else:
        lines.append("PARADIGMS DISAGREE — treat this attribution as low confidence.")
    lines.append(bar)
    return "\n".join(lines)


def explain_passage(passage: str, attributor: Attributor | None = None) -> str:
    attributor = attributor or Attributor.load()
    return render(attributor.predict(passage), attributor)


DEMO_PASSAGE = textwrap.dedent(
    """\
    বৃষ্টি থামিলে সে জানালার পাশে দাঁড়াইয়া দূরের আকাশের দিকে তাকাইয়া রহিল।
    তাহার মনে হইল, এই পৃথিবীতে সুখ বলিয়া কিছু নাই; যাহা আছে তাহা কেবল
    প্রতীক্ষা মাত্র।
    """
)
