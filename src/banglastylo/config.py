"""Central configuration for the Bangla authorship-attribution project.

Every path and hyper-parameter that more than one module needs lives here so
that the notebooks and the command-line entry points stay in agreement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"

ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
FIGURES = ARTIFACTS / "figures"
TABLES = ARTIFACTS / "tables"

REPORTS = ROOT / "reports"
REPORT_FIGURES = REPORTS / "figures"

#: Human-readable plain-text corpus.  This is the copy a reader opens: one
#: ``.txt`` per book, split into the two folders the methodology turns on.
#: ``train/`` is everything the models may learn from; ``unseen/`` holds one
#: book per author that no fitting step is ever allowed to touch.
CORPUS_TXT = ROOT / "corpus"
CORPUS_TRAIN = CORPUS_TXT / "train"
CORPUS_UNSEEN = CORPUS_TXT / "unseen"

# Creating the output tree on import is a convenience for local runs, but it
# must never be a condition of importing the package.  On Kaggle the library is
# read off a read-only dataset mount, where every one of these raises
# OSError(30) — which surfaced as the whole GPU job failing at `import config`
# with no obvious connection to directory creation.  Remote jobs write to their
# own working directory and do not need these at all.
for _p in (RAW, PROCESSED, SPLITS, MODELS, FIGURES, TABLES, REPORT_FIGURES,
           CORPUS_TRAIN, CORPUS_UNSEEN):
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# ----------------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------------
WIKISOURCE_API = "https://bn.wikisource.org/w/api.php"

# Wikimedia asks for a descriptive User-Agent naming the project.
USER_AGENT = (
    "BanglaStylometryResearch/1.0 "
    "(KUET CSE 4122 NLP Laboratory course project; academic, non-commercial)"
)

#: Seconds to wait between successive Wikisource API calls.  The API returns
#: HTTP 429 quickly if this is dropped, so we stay deliberately polite.
REQUEST_DELAY = 1.5

#: The three authors under study.  ``key`` is what filenames, label vectors
#: and plot legends use; ``bn`` is what a Bengali reader sees in the UI.
#:
#: The roster is deliberately famous.  An examiner who does not read the
#: source code still knows these three names and can judge whether a
#: prediction is plausible, which a roster of nineteenth-century essayists
#: does not allow.  They also span a century and three very different
#: registers -- Tagore's literary prose, Nazrul's charged romanticism,
#: Humayun Ahmed's spare modern dialogue -- so a confusion between them is
#: informative rather than inevitable.
AUTHORS: dict[str, dict[str, str]] = {
    "tagore": {
        "bn": "রবীন্দ্রনাথ ঠাকুর",
        "en": "Rabindranath Tagore",
        "short": "Tagore",
        "years": "1861-1941",
    },
    "nazrul": {
        "bn": "কাজী নজরুল ইসলাম",
        "en": "Kazi Nazrul Islam",
        "short": "Nazrul",
        "years": "1899-1976",
    },
    "humayun": {
        "bn": "হুমায়ূন আহমেদ",
        "en": "Humayun Ahmed",
        "short": "Humayun",
        "years": "1948-2012",
    },
}

#: The earlier Wikisource roster, kept because :mod:`wikisource` and the
#: crawler in :mod:`corpus` still build against it, and because the report's
#: leakage experiment was run on it.  It is not the corpus under study.
WIKISOURCE_AUTHORS: dict[str, dict[str, str]] = {

    "tagore": {
        "wikisource": "লেখক:রবীন্দ্রনাথ ঠাকুর",
        "bn": "রবীন্দ্রনাথ ঠাকুর",
        "en": "Rabindranath Tagore",
        "short": "R. Tagore",
        "years": "1861-1941",
    },
    "bankim": {
        "wikisource": "লেখক:বঙ্কিমচন্দ্র চট্টোপাধ্যায়",
        "bn": "বঙ্কিমচন্দ্র চট্টোপাধ্যায়",
        "en": "Bankim Chandra Chattopadhyay",
        "short": "Bankim",
        "years": "1838-1894",
    },
    "sarat": {
        "wikisource": "লেখক:শরৎচন্দ্র চট্টোপাধ্যায়",
        "bn": "শরৎচন্দ্র চট্টোপাধ্যায়",
        "en": "Sarat Chandra Chattopadhyay",
        "short": "Sarat Ch.",
        "years": "1876-1938",
    },
    "vidyasagar": {
        "wikisource": "লেখক:ঈশ্বরচন্দ্র বিদ্যাসাগর",
        "bn": "ঈশ্বরচন্দ্র বিদ্যাসাগর",
        "en": "Ishwar Chandra Vidyasagar",
        "short": "Vidyasagar",
        "years": "1820-1891",
    },
    "haraprasad": {
        "wikisource": "লেখক:হরপ্রসাদ শাস্ত্রী",
        "bn": "হরপ্রসাদ শাস্ত্রী",
        "en": "Haraprasad Shastri",
        "short": "Haraprasad",
        "years": "1853-1931",
    },
    "abanindranath": {
        "wikisource": "লেখক:অবনীন্দ্রনাথ ঠাকুর",
        "bn": "অবনীন্দ্রনাথ ঠাকুর",
        "en": "Abanindranath Tagore",
        "short": "A. Tagore",
        "years": "1871-1951",
    },
    "vivekananda": {
        "wikisource": "লেখক:স্বামী বিবেকানন্দ",
        "bn": "স্বামী বিবেকানন্দ",
        "en": "Swami Vivekananda",
        "short": "Vivekananda",
        "years": "1863-1902",
    },
    "pramatha": {
        "wikisource": "লেখক:প্রমথ চৌধুরী",
        "bn": "প্রমথ চৌধুরী",
        "en": "Pramatha Chaudhuri",
        "short": "Pramatha",
        "years": "1868-1946",
    },
}

#: Works to exclude even though Wikisource links them from an author page.
#:
#: A Wikisource ``লেখক:`` page links everything a person is associated with,
#: including volumes they *edited* rather than wrote.  Those are poison for
#: stylometry: Haraprasad Shastri's ``বৌদ্ধগান ও দোহা`` is his celebrated
#: edition of the Charyapada — 10th-century Old Bengali, not his prose.  Each
#: entry below is a substring matched against the work label.
WORK_BLOCKLIST: tuple[str, ...] = (
    "বৌদ্ধগান ও দোহা",       # Haraprasad Shastri's edition of the Charyapada
    "চর্যাচর্যবিনিশ্চয়",     # the same text under its other name
    "হাজার বছরের পুরাণ",     # Old/Middle Bengali anthologies
)

# ----------------------------------------------------------------------------
# Passage sampling
# ----------------------------------------------------------------------------
#: Target size of one attribution unit ("passage"), in whitespace tokens.
#:
#: Two things set this.  The smallest author's training material is ~30k
#: tokens, so a large passage would leave too few units to train on; and the
#: intended demonstration is a reader pasting a paragraph in, which is
#: typically 50-150 words rather than 220.  120 satisfies both.
PASSAGE_TOKENS = 120

#: A passage shorter than this after cleaning is discarded.
MIN_PASSAGE_TOKENS = 60

#: Authors with fewer than this many passages are dropped from the study.
MIN_PASSAGES_PER_AUTHOR = 200

#: …and an author needs at least this many distinct works.  A one-book author
#: cannot participate in a work-disjoint split at all: every passage of theirs
#: would land in a single split.  Each author here contributes two training
#: books plus one held-out book, so two is the binding figure for the training
#: side.
MIN_WORKS_PER_AUTHOR = 2

#: Passages per author after balancing (``None`` -> use the smallest author).
BALANCE_TO: int | None = None

# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------
#: Splits are grouped by *work*, never by passage: all passages from one book
#: land in exactly one split.  This is what stops the classifiers from winning
#: on topic memorisation instead of style.
TEST_FRACTION = 0.25
VAL_FRACTION = 0.15
SEED = 20242025          # KUET session 2024-25

# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
TOP_FUNCTION_WORDS = 150
CHAR_NGRAM_RANGE = (2, 4)
CHAR_NGRAM_MAX_FEATURES = 3000
POS_NGRAM_RANGE = (1, 3)
POS_NGRAM_MAX_FEATURES = 800

# ----------------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------------
W2V_DIM = 200
W2V_WINDOW = 5
W2V_NEGATIVE = 8
W2V_EPOCHS = 6
W2V_MIN_COUNT = 5
W2V_BATCH = 4096
W2V_LR = 2.5e-3

PRETRAINED_MODEL = "csebuetnlp/banglabert"
BERT_MAX_LEN = 256

# ----------------------------------------------------------------------------
# Generative language models
# ----------------------------------------------------------------------------
CHAR_LM_ORDER = 5
WORD_LM_ORDER = 3

# ----------------------------------------------------------------------------
# Fine-tuning
# ----------------------------------------------------------------------------
#: Fine-tuning runs on CPU here (no CUDA device on the lab machine), which costs
#: roughly 0.42 s per sample at 192 sub-word tokens and 0.60 s at 256.  The
#: budget below is ~45 minutes; the report states this constraint explicitly
#: rather than presenting the fine-tuned score as compute-unconstrained.
FT_EPOCHS = 2
FT_BATCH = 8
FT_LR = 2e-5
FT_MAX_LEN = 192


@dataclass(frozen=True)
class RunConfig:
    """Snapshot of the knobs that change between experiment runs."""

    seed: int = SEED
    passage_tokens: int = PASSAGE_TOKENS
    authors: tuple[str, ...] = field(default_factory=lambda: tuple(AUTHORS))
