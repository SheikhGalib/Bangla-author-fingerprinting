"""Stylometric feature extraction.

Four feature families, each a scikit-learn transformer so that ablations are a
matter of swapping blocks in and out rather than rewriting code:

``FunctionWordFeatures``
    Relative frequency of each item in a *curated closed-class lexicon*.  Using
    a fixed lexicon rather than the corpus's own top-N word list is a
    deliberate choice: a corpus-derived list would quietly admit topic words
    (character names, place names) and inflate accuracy for the wrong reason.

``StructuralFeatures``
    Sentence-length distribution (mean, sd, skew, median, IQR), word length,
    lexical richness, punctuation rates — and a *sadhu/chalit register ratio*.
    That last one is specific to Bangla and worth spelling out: Bangla prose of
    this period is written in either the literary ``সাধু`` register (হইল,
    করিয়া, তাহার) or the colloquial ``চলিত`` one (হল, করে, তার).  Bankim and
    Vidyasagar are almost pure sadhu, Pramatha Chaudhuri campaigned for chalit,
    and Sarat Chandra sits between.  It is the single most interpretable axis in
    the whole feature set.

``CharNgramFeatures``
    TF-IDF over character 2–4-grams.  Cheap, language-independent, and the
    strongest single family in most authorship-attribution literature.

``PosNgramFeatures``
    TF-IDF over POS 1–3-grams from :mod:`banglastylo.postag`.  Captures
    syntactic habit rather than vocabulary.

Every transformer exposes ``get_feature_names_out`` so that a linear model's
coefficients can be read back as named, human-meaningful features — which is
what the interpretability section of the report needs.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config, postag
from .normalize import (
    DANDA,
    PUNCT_CHARS,
    sentence_split,
    word_tokenize,
)


# ---------------------------------------------------------------------------
# Function-word lexicon
# ---------------------------------------------------------------------------
def build_function_word_list(top_k: int = config.TOP_FUNCTION_WORDS) -> list[str]:
    """The closed-class lexicon used as the function-word feature space."""
    words = (
        postag.PRONOUNS
        | postag.POSTPOSITIONS
        | postag.CONJUNCTIONS
        | postag.PARTICLES
        | postag.DETERMINERS
        | postag.ADVERBS
        | postag.QUANTIFIERS_NUM
    )
    return sorted(words)[:top_k] if top_k else sorted(words)


FUNCTION_WORDS = build_function_word_list(top_k=0)

# ---------------------------------------------------------------------------
# Sadhu / chalit register markers
# ---------------------------------------------------------------------------
#: Verb and pronoun forms that only occur in the literary ``সাধু`` register.
SADHU_MARKERS = {
    "হইল", "হইলেন", "হইবে", "হইয়া", "হইতে", "হইয়াছে", "হইয়াছিল", "হইত",
    "করিল", "করিলেন", "করিবে", "করিয়া", "করিতে", "করিয়াছে", "করিতেছে",
    "করিতেছিল", "কহিল", "কহিলেন", "বলিল", "বলিলেন", "বলিয়া", "বলিতে",
    "দিল", "দিলেন", "দিয়া", "দিতে", "লইল", "লইলেন", "লইয়া", "লইতে",
    "গেল", "গেলেন", "গিয়া", "যাইতে", "যাইবে", "আসিল", "আসিলেন", "আসিয়া",
    "দেখিল", "দেখিলেন", "দেখিয়া", "দেখিতে", "উঠিল", "উঠিলেন", "উঠিয়া",
    "বসিল", "বসিলেন", "বসিয়া", "রহিল", "রহিলেন", "রহিয়া", "থাকিল",
    "পাইল", "পাইলেন", "পাইয়া", "পাইতে", "ছিলেন", "তাহার", "তাহাকে",
    "তাহারা", "তাহাদের", "তাঁহার", "তাঁহাকে", "তাঁহারা", "ইহার", "ইহাকে",
    "উহার", "যাহার", "যাহাকে", "কাহার", "কাহাকে", "ইহা", "উহা", "তাহা",
    "যাহা", "নাই", "নহে", "হয়েন", "অপেক্ষা", "কিন্তু", "পরন্তু",
}

#: The matching colloquial ``চলিত`` forms.
CHALIT_MARKERS = {
    "হল", "হলেন", "হবে", "হয়ে", "হতে", "হয়েছে", "হয়েছিল", "হত",
    "করল", "করলেন", "করবে", "করে", "করতে", "করেছে", "করছে", "করছিল",
    "বলল", "বললেন", "বলে", "বলতে", "দিল", "দিলেন", "দিয়ে", "দিতে",
    "নিল", "নিলেন", "নিয়ে", "নিতে", "গেল", "গেলেন", "গিয়ে", "যেতে",
    "যাবে", "এল", "এলেন", "এসে", "দেখল", "দেখলেন", "দেখে", "দেখতে",
    "উঠল", "উঠলেন", "উঠে", "বসল", "বসলেন", "বসে", "রইল", "রইলেন",
    "থাকল", "পেল", "পেলেন", "পেয়ে", "পেতে", "ছিলেন", "তার", "তাকে",
    "তারা", "তাদের", "তাঁর", "তাঁকে", "তাঁরা", "এর", "একে", "ওর",
    "যার", "যাকে", "কার", "কাকে", "এটা", "ওটা", "সেটা", "যেটা",
    "নেই", "নয়", "চেয়ে",
}


# ---------------------------------------------------------------------------
# Shared per-passage analysis
# ---------------------------------------------------------------------------
class PassageAnalysis:
    """Tokenisation, sentence split and POS tagging, computed once per passage.

    The transformers below all need the same three views of a passage, so it is
    worth paying for them exactly once and caching by text.
    """

    __slots__ = ("sentences", "tags", "text", "tokens", "tokens_punct")

    def __init__(self, text: str):
        self.text = text
        self.sentences = sentence_split(text)
        self.tokens_punct = word_tokenize(text, keep_punct=True)
        self.tokens = [t for t in self.tokens_punct if t not in PUNCT_CHARS]
        self.tags = postag.tag(self.tokens_punct)


_ANALYSIS_CACHE: dict[str, PassageAnalysis] = {}


def analyse(text: str) -> PassageAnalysis:
    a = _ANALYSIS_CACHE.get(text)
    if a is None:
        a = PassageAnalysis(text)
        if len(_ANALYSIS_CACHE) < 60_000:
            _ANALYSIS_CACHE[text] = a
    return a


# ---------------------------------------------------------------------------
# 1. Function words
# ---------------------------------------------------------------------------
class FunctionWordFeatures(BaseEstimator, TransformerMixin):
    """Relative frequency of each closed-class item, per 1 000 tokens."""

    def __init__(self, vocabulary: list[str] | None = None):
        self.vocabulary = vocabulary

    def fit(self, X, y=None):
        self.vocabulary_ = list(self.vocabulary or FUNCTION_WORDS)
        self.index_ = {w: i for i, w in enumerate(self.vocabulary_)}
        return self

    def transform(self, X):
        out = np.zeros((len(X), len(self.vocabulary_)), dtype=np.float32)
        for r, text in enumerate(X):
            toks = analyse(text).tokens
            if not toks:
                continue
            scale = 1000.0 / len(toks)
            for t in toks:
                j = self.index_.get(t)
                if j is not None:
                    out[r, j] += scale
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"fw:{w}" for w in self.vocabulary_], dtype=object)


# ---------------------------------------------------------------------------
# 2. Structural / distributional statistics
# ---------------------------------------------------------------------------
STRUCTURAL_NAMES = [
    "sent_len_mean", "sent_len_sd", "sent_len_skew", "sent_len_median",
    "sent_len_iqr", "sent_len_min", "sent_len_max", "n_sentences",
    "word_len_mean", "word_len_sd", "long_word_rate",
    "type_token_ratio", "hapax_rate", "dis_legomena_rate", "yule_k",
    "danda_rate", "comma_rate", "semicolon_rate", "question_rate",
    "exclaim_rate", "quote_rate", "dash_rate", "punct_rate",
    "sadhu_rate", "chalit_rate", "sadhu_chalit_ratio",
    "conjunct_rate", "hasanta_rate", "candrabindu_rate",
]

_HASANTA = "্"
_CANDRABINDU = "ঁ"


def _skew(a: np.ndarray) -> float:
    if a.size < 3:
        return 0.0
    sd = a.std()
    if sd < 1e-9:
        return 0.0
    return float(((a - a.mean()) ** 3).mean() / sd**3)


def _yule_k(counts: np.ndarray, n: int) -> float:
    """Yule's characteristic K — vocabulary-richness measure, length-robust."""
    if n == 0:
        return 0.0
    m2 = float((counts**2).sum())
    return 1e4 * (m2 - n) / (n * n)


class StructuralFeatures(BaseEstimator, TransformerMixin):
    """Sentence-length distribution, lexical richness, punctuation, register."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = np.zeros((len(X), len(STRUCTURAL_NAMES)), dtype=np.float32)
        for r, text in enumerate(X):
            a = analyse(text)
            toks, tp, sents = a.tokens, a.tokens_punct, a.sentences
            n = max(len(toks), 1)

            lens = np.array(
                [len(word_tokenize(s)) for s in sents] or [0], dtype=np.float64
            )
            wlens = np.array([len(t) for t in toks] or [0], dtype=np.float64)

            from collections import Counter

            c = Counter(toks)
            counts = np.array(list(c.values()) or [0], dtype=np.float64)
            n_types = len(c)

            def rate(ch: str, _t: str = text) -> float:
                """Occurrences per 1 000 characters of this passage."""
                return 1000.0 * _t.count(ch) / max(len(_t), 1)

            sadhu = sum(1 for t in toks if t in SADHU_MARKERS)
            chalit = sum(1 for t in toks if t in CHALIT_MARKERS)

            out[r] = [
                lens.mean(), lens.std(), _skew(lens), np.median(lens),
                float(np.percentile(lens, 75) - np.percentile(lens, 25)),
                lens.min(), lens.max(), len(sents),
                wlens.mean(), wlens.std(),
                float((wlens > 8).mean()),
                n_types / n,
                sum(1 for v in c.values() if v == 1) / n,
                sum(1 for v in c.values() if v == 2) / n,
                _yule_k(counts, n),
                rate(DANDA), rate(","), rate(";"), rate("?"), rate("!"),
                rate('"') + rate("'"), rate("-"),
                1000.0 * sum(1 for t in tp if t in PUNCT_CHARS) / max(len(tp), 1),
                1000.0 * sadhu / n,
                1000.0 * chalit / n,
                (sadhu + 1.0) / (chalit + 1.0),
                1000.0 * sum(t.count(_HASANTA) for t in toks) / n,
                rate(_HASANTA),
                rate(_CANDRABINDU),
            ]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"st:{n}" for n in STRUCTURAL_NAMES], dtype=object)


# ---------------------------------------------------------------------------
# 3. Character n-grams
# ---------------------------------------------------------------------------
class CharNgramFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        ngram_range: tuple[int, int] = config.CHAR_NGRAM_RANGE,
        max_features: int = config.CHAR_NGRAM_MAX_FEATURES,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features

    def fit(self, X, y=None):
        self.vec_ = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=3,
        )
        self.vec_.fit(X)
        return self

    def transform(self, X):
        return self.vec_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [f"ch:{g}" for g in self.vec_.get_feature_names_out()], dtype=object
        )


# ---------------------------------------------------------------------------
# 4. POS n-grams
# ---------------------------------------------------------------------------
class PosNgramFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        ngram_range: tuple[int, int] = config.POS_NGRAM_RANGE,
        max_features: int = config.POS_NGRAM_MAX_FEATURES,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features

    @staticmethod
    def _tag_string(text: str) -> str:
        return " ".join(analyse(text).tags)

    def fit(self, X, y=None):
        self.vec_ = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[A-Z]+",
            lowercase=False,       # tags are upper-case; lowercasing kills them
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=3,
        )
        self.vec_.fit([self._tag_string(t) for t in X])
        return self

    def transform(self, X):
        return self.vec_.transform([self._tag_string(t) for t in X])

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [f"pos:{g}" for g in self.vec_.get_feature_names_out()], dtype=object
        )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
FAMILIES = {
    "funcword": FunctionWordFeatures,
    "structural": StructuralFeatures,
    "charngram": CharNgramFeatures,
    "posngram": PosNgramFeatures,
}


class StylometricFeatures(BaseEstimator, TransformerMixin):
    """Horizontal concatenation of a chosen subset of the feature families."""

    def __init__(self, families: tuple[str, ...] = tuple(FAMILIES)):
        self.families = families

    def fit(self, X, y=None):
        self.blocks_ = {}
        for name in self.families:
            block = FAMILIES[name]()
            block.fit(X, y)
            self.blocks_[name] = block
        return self

    def transform(self, X):
        mats = [self.blocks_[n].transform(X) for n in self.families]
        if any(sparse.issparse(m) for m in mats):
            return sparse.hstack(
                [m if sparse.issparse(m) else sparse.csr_matrix(m) for m in mats]
            ).tocsr()
        return np.hstack(mats)

    def get_feature_names_out(self, input_features=None):
        names: list[str] = []
        for n in self.families:
            names += list(self.blocks_[n].get_feature_names_out())
        return np.array(names, dtype=object)
