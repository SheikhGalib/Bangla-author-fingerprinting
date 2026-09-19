"""Per-book analytics, computed from the corpus rather than asserted about it.

Everything this module reports is derived from the text on disk: word counts,
vocabulary richness, the 29 structural measurements the stylometric SVM uses,
and the words that most distinguish one book from the other eight.  Nothing is
looked up and nothing is remembered, which matters for a page whose purpose is
to be explained to an examiner -- every number on it can be recomputed in front
of them.

The distinctive-word calculation is the interesting one.  Ranking a book's
words by raw frequency returns the same Bangla function words for all nine
books, which says nothing.  Ranking by TF--IDF *across the nine books* instead
asks "which words does this book use that the others do not", and that is the
question a reader actually has.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

from . import config
from .normalize import PUNCT_CHARS, sentence_split, word_tokenize
from .stylometry import STRUCTURAL_NAMES, StructuralFeatures

#: Words too common in Bangla to distinguish anything; excluded from the
#: "distinctive words" list so it does not fill up with particles.  This is the
#: one place in the project where function words are *unwanted* -- everywhere
#: else they are the signal.
_STOPWORDS = {
    "এবং", "আর", "কিন্তু", "তবে", "যে", "যা", "যাহা", "তার", "তাহার", "তাকে",
    "সে", "তিনি", "আমি", "আমার", "আমাকে", "তুমি", "তোমার", "আপনি", "এই", "ওই",
    "সেই", "একটা", "একটি", "এক", "না", "নাই", "নয়", "হয়", "হয়ে", "হইয়া",
    "করে", "করিয়া", "করা", "বলে", "বলিয়া", "থেকে", "হইতে", "দিয়ে", "জন্য",
    "সঙ্গে", "মধ্যে", "উপর", "পর", "সব", "কিছু", "কোন", "কোনো", "যদি", "তাহলে",
    "আছে", "ছিল", "ছিলেন", "হল", "হলো", "গেল", "নিয়ে", "মত", "মতো", "খুব",
    "আবার", "শুধু", "তো", "ই", "ও", "এ", "তা", "কি", "কেন", "যায়", "যাবে",
    "<num>",
}

#: A token must occur at least this often in a book before it can be called
#: distinctive; without it the list fills with one-off typos and OCR debris.
_MIN_COUNT = 4


@dataclass
class BookProfile:
    author: str
    book: str
    role: str
    chapters: list[str] = field(default_factory=list)
    n_chapters: int = 0
    n_chars: int = 0
    n_tokens: int = 0
    n_sentences: int = 0
    n_unique: int = 0
    type_token_ratio: float = 0.0
    hapax_rate: float = 0.0
    mean_sentence_len: float = 0.0
    mean_word_len: float = 0.0
    dialogue_rate: float = 0.0
    sadhu_chalit_ratio: float = 0.0
    distinctive: list[dict] = field(default_factory=list)
    frequent: list[dict] = field(default_factory=list)
    recurring_names: list[dict] = field(default_factory=list)
    structural: dict[str, float] = field(default_factory=dict)


def _load_raw() -> dict[str, dict[str, dict[str, str]]]:
    path = config.RAW / "corpus_raw.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run scripts/01_build_corpus.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def _roles() -> dict[str, str]:
    from .ebangla import BOOKS
    return {b.title_bn: b.role for bs in BOOKS.values() for b in bs}


def _content_tokens(text: str) -> list[str]:
    return [t for t in word_tokenize(text)
            if t not in PUNCT_CHARS and t not in _STOPWORDS and len(t) > 1]


def _structural_row(text: str) -> dict[str, float]:
    """The 29 features the stylometric SVM sees, for one book."""
    values = StructuralFeatures().transform([text])[0]
    return {name: round(float(v), 4) for name, v in zip(STRUCTURAL_NAMES, values)}


def book_profiles(top_k: int = 15) -> list[BookProfile]:
    """Build a profile for every book in the corpus."""
    raw = _load_raw()
    roles = _roles()

    # Flatten to one document per book, and count document frequency across
    # the nine books so TF-IDF has something to divide by.
    docs: dict[tuple[str, str], str] = {}
    chapters: dict[tuple[str, str], list[str]] = {}
    chapter_df: dict[tuple[str, str], Counter] = {}
    for author, books in raw.items():
        for book, chs in books.items():
            key = (author, book)
            docs[key] = "\n".join(chs.values())
            chapters[key] = list(chs)
            # How many chapters of this book each word appears in: a recurring
            # character spreads across the narrative, a topic word does not.
            df: Counter = Counter()
            for ch_text in chs.values():
                df.update(set(_content_tokens(ch_text)))
            chapter_df[key] = df

    counts = {k: Counter(_content_tokens(v)) for k, v in docs.items()}
    n_docs = len(docs)
    doc_freq: Counter = Counter()
    for c in counts.values():
        doc_freq.update(set(c))

    profiles: list[BookProfile] = []
    for key, text in docs.items():
        author, book = key
        toks = word_tokenize(text)
        words = [t for t in toks if t not in PUNCT_CHARS]
        freq = Counter(words)
        sents = sentence_split(text)
        c = counts[key]
        total = sum(c.values()) or 1

        # TF-IDF against the other eight books.
        scored = []
        for term, n in c.items():
            if n < _MIN_COUNT:
                continue
            tf = n / total
            idf = math.log(n_docs / doc_freq[term])
            if idf <= 0:          # appears in every book: says nothing
                continue
            scored.append((tf * idf, term, n))
        scored.sort(reverse=True)

        struct = _structural_row(text)
        hapax = sum(1 for w, n in freq.items() if n == 1)

        profiles.append(BookProfile(
            author=author,
            book=book,
            role=roles.get(book, "train"),
            chapters=chapters[key],
            n_chapters=len(chapters[key]),
            n_chars=len(text),
            n_tokens=len(words),
            n_sentences=len(sents),
            n_unique=len(freq),
            type_token_ratio=round(len(freq) / max(len(words), 1), 4),
            hapax_rate=round(hapax / max(len(freq), 1), 4),
            mean_sentence_len=round(len(words) / max(len(sents), 1), 2),
            mean_word_len=round(
                sum(len(w) for w in words) / max(len(words), 1), 2),
            dialogue_rate=struct.get("quote_rate", 0.0),
            sadhu_chalit_ratio=struct.get("sadhu_chalit_ratio", 0.0),
            distinctive=[{"word": t, "count": n, "score": round(s, 5)}
                         for s, t, n in scored[:top_k]],
            frequent=[{"word": w, "count": n}
                      for w, n in c.most_common(top_k)],
            recurring_names=_likely_names(c, doc_freq, n_docs, chapter_df[key]),
            structural=struct,
        ))
    return profiles


#: Bangla inflectional endings.  A token carrying one of these is a verb or a
#: participle, not a name.  The list exists because the first version of
#: :func:`_likely_names` returned nothing but sadhu verb forms -- আসিয়া, বসিয়া,
#: কহিলেন -- which are a real and interesting finding about register, but are
#: not what a reader scanning for characters is looking for.
_VERBAL_SUFFIX = re.compile(
    r"(িয়া|ইয়া|য়া|িলেন|িলাম|িলে|িল|িতেছে|িতেছিল|িতে|িয়াছে|িয়াছি|িয়াছিল"
    r"|েছেন|েছিলেন|েছিল|েছে|লেন|লাম|লুম|চ্ছে|চ্ছি|বেন|বার|তেন|ইতে|ইল"
    # Sadhu imperfect: করিত, হইত, যাইত.  Costs the occasional real noun in
    # -িত (সঙ্গীত), which is an acceptable trade in a list of names.
    r"|িত|ইত)$"
)

#: Archaic third-person pronouns.  They are frequent, book-specific enough to
#: survive the TF-IDF filter, and obviously not characters.
_ARCHAIC_PRONOUNS = {
    "তাহারা", "তাহাদের", "তাহাকে", "তাহার", "তাহা", "তাঁহারা", "তাঁহার",
    "তাঁহাকে", "তাঁহাদের", "ইহার", "ইহা", "ইহাকে", "ইহাদের", "উহার", "উহা",
    "যাহারা", "যাহাদের", "যাহাকে", "সেইরূপ", "তথায়", "তদ্রূপ",
}


def _likely_names(counts: Counter, doc_freq: Counter, n_docs: int,
                  chapter_df: Counter | None = None,
                  top_k: int = 10) -> list[dict]:
    """Tokens that behave like recurring characters or places.

    Bangla has no capitalisation, so proper nouns cannot be spotted
    orthographically.  Three behavioural signals are used instead, and all
    three are needed:

    * **frequent in this book** -- a name is repeated, a passing noun is not;
    * **rare across the other books** -- shared vocabulary is language, not
      cast;
    * **spread across chapters** -- a character recurs through the narrative,
      whereas a topic word clusters in the one chapter that is about it.

    Inflected verb forms are filtered out explicitly; without that the list
    fills with sadhu participles, which dominate exactly the frequent-and-
    distinctive corner this heuristic searches.

    It is a heuristic and it is labelled as one in the interface: it recovers
    most recurring names and will occasionally admit a thematic noun.
    """
    out = []
    for term, n in counts.most_common(600):
        if n < 8 or doc_freq[term] > max(2, n_docs // 3):
            continue
        if len(term) < 3 or re.search(r"\d", term):
            continue
        if _VERBAL_SUFFIX.search(term) or term in _ARCHAIC_PRONOUNS:
            continue
        spread = chapter_df.get(term, 0) if chapter_df else 0
        if chapter_df is not None and spread < 3:
            continue
        out.append({"word": term, "count": n, "chapters": spread,
                    "books": doc_freq[term]})
        if len(out) >= top_k:
            break
    return out


# ---------------------------------------------------------------------------
# Corpus-level views for the methods page
# ---------------------------------------------------------------------------
def feature_families() -> list[dict]:
    """What the stylometric SVM actually looks at, family by family."""
    return [
        {
            "key": "funcword",
            "name": "Function words",
            "count": config.TOP_FUNCTION_WORDS,
            "what": "Relative frequency of the most common Bangla function "
                    "words (particles, postpositions, pronouns).",
            "why": "Topic classification deletes these because they carry no "
                   "topic. Authorship keeps them for exactly that reason: a "
                   "word that says nothing about what a text is about is free "
                   "to say something about who wrote it, and it is the hardest "
                   "thing for a writer to vary on purpose.",
        },
        {
            "key": "structural",
            "name": "Structural statistics",
            "count": len(STRUCTURAL_NAMES),
            "what": "Sentence length (mean, spread, skew), word length, "
                    "vocabulary richness (type-token ratio, hapax rate, "
                    "Yule's K), punctuation and dialogue rates, and the "
                    "sadhu/chalit register ratio.",
            "why": "These are the measurements a human reader would make by "
                   "eye. They are the reason the model's evidence can be "
                   "checked against the page.",
            "names": list(STRUCTURAL_NAMES),
        },
        {
            "key": "charngram",
            "name": "Character n-grams",
            "count": config.CHAR_NGRAM_MAX_FEATURES,
            "what": f"TF-IDF over character sequences of length "
                    f"{config.CHAR_NGRAM_RANGE[0]}-{config.CHAR_NGRAM_RANGE[1]}, "
                    f"padded at word boundaries.",
            "why": "Bangla marks case and tense with suffixes, so a character "
                   "model sees inside the word. It recognises two inflected "
                   "forms of one verb as related without either having been "
                   "seen whole. This family alone carries most of the signal.",
        },
        {
            "key": "posngram",
            "name": "Part-of-speech n-grams",
            "count": config.POS_NGRAM_MAX_FEATURES,
            "what": f"Sequences of {config.POS_NGRAM_RANGE[0]}-"
                    f"{config.POS_NGRAM_RANGE[1]} coarse POS tags from a "
                    f"hand-written Bangla tagger.",
            "why": "Captures grammatical habit rather than vocabulary: how an "
                   "author orders noun, verb and postposition. The weakest "
                   "family on its own, and the most nearly topic-free.",
        },
    ]


def representation_demo(text: str, top_k: int = 12) -> dict:
    """Show one passage as Bag-of-Words, TF-IDF and n-grams side by side.

    The methods page runs this on whatever the visitor types, so the three
    representations stop being definitions and become three concrete lists
    built from their own sentence.
    """
    toks = [t for t in word_tokenize(text) if t not in PUNCT_CHARS]
    bow = Counter(toks)

    bigrams = Counter(pairwise(toks))
    trigrams = Counter(zip(toks, toks[1:], toks[2:]))

    stripped = re.sub(r"\s+", " ", text)
    char3 = Counter(stripped[i:i + 3] for i in range(max(0, len(stripped) - 2)))

    return {
        "n_tokens": len(toks),
        "n_unique": len(bow),
        "bag_of_words": [{"item": w, "count": n} for w, n in bow.most_common(top_k)],
        "word_bigrams": [{"item": " ".join(g), "count": n}
                         for g, n in bigrams.most_common(top_k)],
        "word_trigrams": [{"item": " ".join(g), "count": n}
                          for g, n in trigrams.most_common(top_k)],
        "char_trigrams": [{"item": g.replace(" ", "␣"), "count": n}
                          for g, n in char3.most_common(top_k)],
        "structural": _structural_row(text),
    }
