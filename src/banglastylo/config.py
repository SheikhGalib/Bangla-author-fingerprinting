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

for _p in (RAW, PROCESSED, SPLITS, MODELS, FIGURES, TABLES, REPORT_FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

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

#: Candidate authors.  ``AUTHORS`` maps a short ASCII key (used in filenames and
#: plot labels) to the Bengali Wikisource ``লেখক:`` page title.  Every one of
#: these is out of copyright: all died more than 60 years before this project.
AUTHORS: dict[str, dict[str, str]] = {
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
PASSAGE_TOKENS = 220

#: A passage shorter than this after cleaning is discarded.
MIN_PASSAGE_TOKENS = 120

#: Authors with fewer than this many passages are dropped from the study.
MIN_PASSAGES_PER_AUTHOR = 300

#: …and an author needs at least this many distinct works, which is the harder
#: constraint.  A one-book author cannot participate in a work-disjoint split at
#: all: every passage of theirs would land in a single split.  Dropping them is
#: the honest move, and the report names who was dropped and why.
MIN_WORKS_PER_AUTHOR = 3

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
