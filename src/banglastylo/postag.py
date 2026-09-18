"""A rule-based coarse POS tagger for Bangla.

Why this exists
---------------
The proposal calls for POS n-gram features "from a Bangla tagger".  There is no
usable off-the-shelf option in this environment:

* ``stanza`` ships only character language models for Bengali (``bn``), no POS
  model;
* ``bnlp-toolkit`` depends on ``gensim``, which has no Python 3.14 wheel and
  needs a C toolchain this machine does not have;
* the sole Universal Dependencies Bengali treebank (``UD_Bengali-BRU``) is 56
  sentences and test-only — far too small to train a tagger on.

So the tagger is written here.  It is a two-stage lexicon-plus-morphology
tagger over the UD coarse tagset, which is the right granularity for stylometry
anyway: authorship signal lives in the *rate* of pronouns, postpositions and
particles, not in fine-grained morphological features.

How it works
------------
1. **Closed classes by lexicon.** Bangla pronouns, postpositions, conjunctions,
   particles and quantifiers are a closed, enumerable set.  They are listed
   below in both the *chalit* (colloquial) and *sadhu* (literary) registers,
   because a corpus spanning Vidyasagar to Sarat Chandra contains both.
2. **Open classes by morphology.** Verbs are recognised from Bangla's very
   regular finite-verb inflection (``-িলেন``, ``-েছে``, ``-বেন``, ``-িতেছিল`` …)
   combined with a root list; adjectives and adverbs from derivational suffixes
   (``-ীয়``, ``-ময়``, ``-ভাবে``, ``-রূপে`` …).  Nominal case clitics
   (``-এর``, ``-কে``, ``-টি``, ``-গুলো`` …) are stripped before lookup.
3. **Default to NOUN**, the majority open class.

``evaluate_on_ud`` scores the tagger against ``UD_Bengali-BRU``; the number it
returns is reported in the write-up so the reader can discount the POS features
appropriately rather than take them on faith.
"""
from __future__ import annotations

import re
from pathlib import Path

from .normalize import PUNCT_CHARS

# ---------------------------------------------------------------------------
# Closed-class lexicons
# ---------------------------------------------------------------------------
PRONOUNS = {
    # 1st / 2nd person, both registers
    "আমি", "আমরা", "আমাকে", "আমার", "আমাদের", "আমাদিগকে", "মোরা", "মোর",
    "তুমি", "তোমরা", "তোমাকে", "তোমার", "তোমাদের", "তোমাদিগকে",
    "তুই", "তোরা", "তোকে", "তোর", "তোদের",
    "আপনি", "আপনারা", "আপনাকে", "আপনার", "আপনাদের",
    # 3rd person
    "সে", "তারা", "তাকে", "তার", "তাদের", "তাহারা", "তাহাকে", "তাহার",
    "তাহাদের", "তাহাদিগকে", "তিনি", "তাঁরা", "তাঁহারা", "তাঁকে", "তাঁহাকে",
    "তাঁর", "তাঁহার", "তাঁদের", "তাঁহাদের", "উনি", "ইনি", "এরা", "ইহারা",
    "ইহাকে", "ইহার", "ইহাদের", "এঁরা", "এঁকে", "এঁর", "ও", "ওরা", "ওকে",
    "ওর", "ওদের", "ইহা", "উহা", "তাহা", "তা", "এ", "এই", "সেই",
    # interrogative / relative / indefinite
    "কে", "কারা", "কাকে", "কার", "কাদের", "কি", "কী", "কিসে", "কিছু",
    "যে", "যাঁরা", "যারা", "যাকে", "যার", "যাদের", "যাহা", "যাহারা",
    "যাহাকে", "যাহার", "যিনি", "যাঁহারা", "যাঁহাকে", "যাঁহার",
    "কেউ", "কেহ", "কোনো", "কোন", "কেউই", "সকলে", "সকলকে", "সবাই", "সব",
    "নিজে", "নিজের", "নিজেকে", "স্বয়ং", "পরস্পর", "উভয়ে", "উভয়",
    "ইহাদিগকে", "তদ্রূপ", "এতদ্", "যাহাদিগকে",
}

#: Bangla postpositions.  These carry a lot of the stylistic load in the
#: proposal's worked example, so the list is deliberately generous.
POSTPOSITIONS = {
    "থেকে", "হইতে", "হতে", "চেয়ে", "অপেক্ষা", "দিয়ে", "দিয়া", "দ্বারা",
    "কর্তৃক", "সঙ্গে", "সহিত", "সাথে", "সহ", "বিনা", "ব্যতীত", "ছাড়া",
    "ছাড়িয়া", "ব্যতিরেকে", "জন্য", "জন্যে", "নিমিত্ত", "তরে", "উদ্দেশে",
    "পাশে", "পার্শ্বে", "কাছে", "নিকট", "নিকটে", "সন্নিকটে", "দিকে",
    "প্রতি", "অভিমুখে", "উপরে", "উপর", "নিচে", "নীচে", "অধীনে", "মধ্যে",
    "ভিতরে", "অভ্যন্তরে", "বাহিরে", "বাইরে", "সামনে", "সম্মুখে", "পিছনে",
    "পশ্চাতে", "আগে", "পূর্বে", "পরে", "পরেই", "পশ্চাৎ", "মত", "মতো",
    "ন্যায়", "সদৃশ", "অনুসারে", "অনুযায়ী", "মধ্য", "বিষয়ে", "সম্বন্ধে",
    "সম্পর্কে", "বিরুদ্ধে", "পক্ষে", "বদলে", "পরিবর্তে", "মারফত", "মাঝে",
    "ধরে", "ধরিয়া", "লাগিয়া", "করিয়া", "বৈ", "অবধি", "পর্যন্ত", "যাবৎ",
    "ভিতর", "তলে", "ওপরে", "ওপর", "কাছেই", "নিকটেই", "সহকারে",
}

CONJUNCTIONS = {
    "এবং", "ও", "আর", "কিন্তু", "তবে", "অথবা", "কিংবা", "বা", "নতুবা",
    "নয়তো", "অথচ", "তথাপি", "যদিও", "যেহেতু", "কারণ", "কেননা", "সুতরাং",
    "অতএব", "তাই", "ফলে", "তথা", "বরং", "নচেৎ", "পরন্তু", "কিন্ত",
    "যদি", "তবু", "তবুও", "যেন", "যাতে", "বলিয়া", "বলে",
    "তারপর", "তাহলে", "অতঃপর", "তদুপরি", "নইলে",
}

PARTICLES = {
    "না", "নাই", "নেই", "নি", "নয়", "নহে", "নহি", "নও",
    "ই", "ও", "তো", "তোহ", "কি", "কিনা", "যে", "গো", "রে", "হে",
    "যদি", "বটে", "মাত্র", "শুধু", "কেবল", "অবশ্য", "নিশ্চয়", "হয়তো",
    "বুঝি", "নাকি", "যেন", "খানি", "খানা", "টুকু", "টুকুন",
}

ADVERBS = {
    "এখন", "তখন", "কখন", "যখন", "তবে", "সবে", "সদ্য", "আজ", "কাল",
    "গতকাল", "পরশু", "এখনই", "তৎক্ষণাৎ", "অবিলম্বে", "সহসা", "হঠাৎ",
    "ক্রমে", "ক্রমশ", "পুনরায়", "আবার", "পুনঃ", "বারবার", "প্রায়",
    "কদাচিৎ", "কখনও", "কখনো", "সর্বদা", "সততই", "নিত্য", "এখানে",
    "সেখানে", "কোথায়", "যেখানে", "এইখানে", "সেইখানে", "কোথাও", "সর্বত্র",
    "উপরন্তু", "অতিশয়", "অত্যন্ত", "অতি", "খুব", "বেশ", "যথেষ্ট", "একটু",
    "অল্প", "বহু", "অনেক", "কম", "বেশি", "বেশী", "ধীরে", "দ্রুত", "শীঘ্র",
    "সত্বর", "একেবারে", "নিতান্ত", "মোটেই", "আদৌ", "তথায়", "ইতিমধ্যে",
    "পুনর্বার", "যথা", "তথা", "এইরূপ", "সেইরূপ", "কিরূপে", "কেমনে", "কেন",
    "কিরূপ", "এমন", "তেমন", "যেমন", "কেমন", "যথার্থ",
}

QUANTIFIERS_NUM = {
    "এক", "দুই", "দুটি", "দুটো", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট",
    "নয়", "দশ", "শত", "সহস্র", "হাজার", "লক্ষ", "কোটি", "প্রথম", "দ্বিতীয়",
    "তৃতীয়", "চতুর্থ", "পঞ্চম", "ষষ্ঠ", "সপ্তম", "অষ্টম", "নবম", "দশম",
    "কয়েক", "কয়েকটি", "একজন", "দুজন", "তিনজন", "কতজন", "বহুজন",
}

#: Determiners.  UD separates these from pronouns; keeping them apart matters
#: for stylometry too, since demonstrative density is an authorial habit.
DETERMINERS = {
    "এই", "সেই", "ঐ", "ওই", "এইসব", "সেইসব", "এসব", "ওসব", "সব", "সকল",
    "সমস্ত", "সমুদয়", "প্রতি", "প্রত্যেক", "প্রতিটি", "অন্য", "অপর",
    "একটি", "একটা", "একখানি", "একখানা", "কোনো", "কোন", "যেই", "কতগুলি",
}

INTERJECTIONS = {
    "হ্যাঁ", "হাঁ", "হুম", "আচ্ছা", "ওহ", "ওঃ", "আহা", "আহ", "বাঃ", "বাহ",
    "আরে", "ছি", "উঃ", "ইস", "ওরে", "হায়", "অহো", "এস", "বেশ",
}

#: Frequent adjectives that no suffix rule would catch.
ADJECTIVES = {
    "ভাল", "ভালো", "খারাপ", "মন্দ", "সুন্দর", "কুৎসিত", "বড়", "বড়ো",
    "ছোট", "ছোটো", "নতুন", "নূতন", "পুরাতন", "পুরানো", "পুরনো", "প্রাচীন",
    "নব", "লাল", "নীল", "সবুজ", "শাদা", "সাদা", "কালো", "কৃষ্ণ", "শুভ্র",
    "উচ্চ", "নিম্ন", "দীর্ঘ", "ক্ষুদ্র", "বৃহৎ", "মহৎ", "মহা", "শ্রেষ্ঠ",
    "উত্তম", "অধম", "প্রধান", "মুখ্য", "গৌণ", "প্রিয়", "কঠিন", "সহজ",
    "সরল", "জটিল", "মিষ্টি", "তিক্ত", "গরম", "ঠাণ্ডা", "শীতল", "উষ্ণ",
    "ধনী", "দরিদ্র", "গরিব", "সুখী", "দুঃখী", "ক্ষুধার্ত", "তৃষ্ণার্ত",
    "মজার", "সোনার", "রূপার", "যুবক", "বৃদ্ধ", "তরুণ", "সত্য", "মিথ্যা",
    "পূর্ণ", "শূন্য", "একা", "একাকী", "সমগ্র", "বিশাল", "অসীম", "সীমিত",
}

#: Very common verb roots.  Recognising the root lets the suffix rules fire on
#: forms whose ending alone would be ambiguous with a noun.  Bangla stems mutate
#: their vowel under inflection (খা -> খে, দে -> দি, যা -> গে), so the mutated
#: stems are listed too rather than modelled.
VERB_ROOTS = {
    "কর", "করি", "বল", "কহ", "যা", "আস", "দে", "দি", "নে", "নি", "হ", "হই",
    "দেখ", "থাক", "পার", "চা", "চাহ", "খা", "খে", "শোন", "শুন", "জান",
    "বুঝ", "ভাব", "চল", "উঠ", "বস", "পড়", "রাখ", "রেখ", "লাগ", "দাঁড়া",
    "ফির", "ফের", "মর", "বাঁচ", "লিখ", "ডাক", "হাস", "কাঁদ", "ধর", "ছাড়",
    "মিল", "পাঠা", "আন", "তুল", "নাম", "ঘুম", "জাগ", "গা", "খুঁজ", "পাই",
    "পা", "ভুল", "শিখ", "শেখ", "বাড়", "কম", "রহ", "রয়", "গি", "গে",
    "যাই", "এল", "গেল", "আঁক", "কিন", "ঘুর", "পর", "পরি", "ধো", "ধোব",
    "ভালবাস", "ভালোবাস", "নাও", "নিয়", "দিয়", "করিয়", "হইয়", "আছ",
    "ছিল", "থাকি", "চাই", "চায়", "হয়", "হব", "হত", "হচ্ছ", "যাচ্ছ",
    "আসছ", "বলছ", "করছ", "দেখছ", "শুনছ", "বসছ", "উঠছ", "লেখ", "বেড়া",
}

#: Frequent finite forms whose morphology is irregular enough that the suffix
#: rules miss them.  Small, closed, and worth listing explicitly.
VERB_FORMS = {
    "আছে", "আছেন", "আছি", "আছিস", "আছ", "নাই", "নেই",
    "ছিল", "ছিলেন", "ছিলাম", "ছিলে", "ছিলি",
    "হয়", "হন", "হও", "হই", "হবে", "হবেন", "হব", "হবো", "হল", "হলো",
    "হলেন", "হলাম", "হলে", "হইল", "হইলেন", "হইবে", "হইয়াছে", "হইয়াছিল",
    "যায়", "যান", "যাও", "যাই", "যাবে", "যাবেন", "যাব", "গেল", "গেলেন",
    "গেলাম", "গিয়াছে", "গিয়াছেন", "গিয়েছে", "গেছে", "গেছেন",
    "দেয়", "দেন", "দাও", "দিই", "দেবে", "দেবেন", "দিব", "দিল", "দিলেন",
    "নেয়", "নেন", "নাও", "নিই", "নেবে", "নিবে", "নিল", "নিলেন",
    "খায়", "খান", "খাও", "খাই", "খাবে", "খাবেন", "খেয়েছ", "খেয়েছে",
    "করে", "করেন", "কর", "করি", "করব", "করবে", "করবেন", "করল", "করলেন",
    "পারে", "পারেন", "পার", "পারি", "পারো", "পারব", "পারবে",
    "চায়", "চান", "চাই", "চাও", "ধোবো", "ধোব", "ভালবাসি", "ভালোবাসি",
}

# ---------------------------------------------------------------------------
# Morphological patterns
# ---------------------------------------------------------------------------
#: Finite / participial verb endings, sadhu and chalit.  Ordered longest-first
#: so that ``-িতেছিলেন`` wins over ``-েন``.
VERB_SUFFIXES = [
    "িতেছিলেন", "িতেছিলাম", "িতেছিলে", "িতেছিল", "িয়াছিলেন", "িয়াছিলাম",
    "িয়াছিলে", "িয়াছিল", "েছিলেন", "েছিলাম", "েছিলে", "েছিল", "চ্ছিলেন",
    "চ্ছিলাম", "চ্ছিলে", "চ্ছিল", "িতেছেন", "িতেছি", "িতেছে", "িয়াছেন",
    "িয়াছি", "িয়াছে", "েছেন", "েছিস", "েছি", "েছ", "েছে", "চ্ছেন",
    "চ্ছি", "চ্ছে", "িলেন", "িলাম", "িলে", "িল", "লেন", "লাম", "লে",
    "েন", "িব", "িবে", "িবেন", "বেন", "বে", "বি", "ব", "িতাম", "িতেন",
    "িত", "তাম", "তেন", "তে", "ত", "িয়া", "য়ে", "িতে", "ইতে", "ইয়া",
    "ায়", "ান", "াও", "িস", "িয়", "ুন", "ো", "ে", "ি",
]

ADJ_SUFFIXES = [
    "ীয়", "ময়", "বান", "বতী", "মান", "িক", "তর", "তম", "শীল",
    "যুক্ত", "হীন", "পূর্ণ", "জনক", "ময়ী", "বিশিষ্ট", "ার্ত", "িষ্ঠ",
]

ADV_SUFFIXES = ["ভাবে", "রূপে", "তঃ", "মতে", "স্বরূপ", "পূর্বক", "সহকারে"]

#: Nominal case clitics stripped before lexicon lookup.
NOUN_CLITICS = [
    "গুলিকে", "গুলোকে", "দিগকে", "গুলির", "গুলোর", "খানার", "খানির",
    "টিকে", "টাকে", "গুলি", "গুলো", "দের", "টির", "টার", "খানা", "খানি",
    "কে", "রা", "এর", "ের", "রে", "য়ে", "তে", "েতে", "র", "টি", "টা",
]

BANGLA_DIGIT_RE = re.compile(r"^[০-৯0-9]+$")
NUM_PLACEHOLDER = "<num>"

TAGSET = ("NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CCONJ",
          "PART", "NUM", "INTJ", "PUNCT", "X")

#: The tagger targets a *coarse* tagset.  Three UD distinctions are deliberately
#: not attempted, and gold tags are folded accordingly before scoring:
#:
#: ``AUX``   -- Bangla auxiliaries are ordinary verbs in ordinary verb forms;
#:              telling them apart needs syntax this tagger does not have.
#: ``PROPN`` -- would need a Bangla name gazetteer, which does not exist for
#:              19th-century literary names.
#: ``SCONJ`` -- merged with ``CCONJ``; the stylometric feature of interest is
#:              conjunction density, not conjunction subtype.
#:
#: All three folds *lose* information rather than invent it, and each one is
#: reported alongside the accuracy figure so the reader can judge the trade.
UD_FOLD = {"AUX": "VERB", "PROPN": "NOUN", "SCONJ": "CCONJ"}


def _strip_clitic(token: str) -> str:
    for c in NOUN_CLITICS:
        if token.endswith(c) and len(token) - len(c) >= 2:
            return token[: -len(c)]
    return token


def _looks_verbal(token: str) -> bool:
    if token in VERB_FORMS or token in VERB_ROOTS:
        return True
    for suf in VERB_SUFFIXES:
        if not token.endswith(suf):
            continue
        stem = token[: -len(suf)]
        if len(stem) < 1:
            continue
        # Long, unambiguous inflections are enough on their own.
        if len(suf) >= 3:
            return True
        # Short endings only count when the stem is a known verb root.
        if stem in VERB_ROOTS or stem + "া" in VERB_ROOTS:
            return True
    return False


#: Particles that are never anything else, so they short-circuit the lexicons.
_HARD_PARTICLES = {"না", "নি", "তো", "ই", "কিনা", "বটে", "নাকি"}


def tag_token(token: str) -> str:
    """Return the coarse tag for a single token."""
    if not token:
        return "X"
    if token in PUNCT_CHARS or all(c in PUNCT_CHARS for c in token):
        return "PUNCT"
    if token == NUM_PLACEHOLDER or BANGLA_DIGIT_RE.match(token):
        return "NUM"
    if re.fullmatch(r"[A-Za-z]+", token):
        return "X"

    # Closed classes, most-specific first.  Order encodes the resolution of the
    # genuinely ambiguous forms: 'না' is always the negator, 'এই'/'সেই' are
    # determiners rather than pronouns, and 'ও' is far more often the
    # conjunction "and" than the third-person pronoun.
    if token in _HARD_PARTICLES:
        return "PART"
    if token in INTERJECTIONS:
        return "INTJ"
    if token in DETERMINERS:
        return "DET"
    if token in PRONOUNS:
        return "PRON"
    if token in POSTPOSITIONS:
        return "ADP"
    if token in CONJUNCTIONS:
        return "CCONJ"
    if token in PARTICLES:
        return "PART"
    if token in ADVERBS:
        return "ADV"
    if token in ADJECTIVES:
        return "ADJ"
    if token in QUANTIFIERS_NUM:
        return "NUM"
    if token in VERB_FORMS:
        return "VERB"

    for suf in ADV_SUFFIXES:
        if token.endswith(suf) and len(token) > len(suf) + 1:
            return "ADV"

    if _looks_verbal(token):
        return "VERB"

    for suf in ADJ_SUFFIXES:
        if token.endswith(suf) and len(token) > len(suf) + 1:
            return "ADJ"

    # Retry the closed classes on the stem, e.g. 'তাহাদিগকে' -> 'তাহাদিগ'.
    stem = _strip_clitic(token)
    if stem != token:
        if stem in PRONOUNS:
            return "PRON"
        if stem in POSTPOSITIONS:
            return "ADP"
        if stem in ADJECTIVES:
            return "ADJ"

    return "NOUN"


def tag(tokens: list[str]) -> list[str]:
    return [tag_token(t) for t in tokens]


# ---------------------------------------------------------------------------
# Validation against UD_Bengali-BRU
# ---------------------------------------------------------------------------
def read_conllu(path: str | Path) -> list[list[tuple[str, str]]]:
    """Return sentences as lists of ``(form, upos)`` pairs."""
    sents: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if cur:
                sents.append(cur)
                cur = []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 4 or "-" in cols[0]:
            continue
        cur.append((cols[1], cols[3]))
    if cur:
        sents.append(cur)
    return sents


def evaluate_on_ud(path: str | Path) -> dict:
    """Token accuracy plus a per-tag breakdown and a confusion list."""
    from collections import Counter

    sents = read_conllu(path)
    gold_counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    confusions: Counter[tuple[str, str]] = Counter()
    n = ok = 0
    for sent in sents:
        for form, gold in sent:
            gold_c = UD_FOLD.get(gold, gold)
            pred = tag_token(form)
            n += 1
            gold_counts[gold_c] += 1
            if pred == gold_c:
                ok += 1
                correct[gold_c] += 1
            else:
                confusions[(gold_c, pred)] += 1
    return {
        "n_sentences": len(sents),
        "n_tokens": n,
        "accuracy": ok / n if n else 0.0,
        "per_tag": {
            t: {"support": gold_counts[t], "recall": correct[t] / gold_counts[t]}
            for t in sorted(gold_counts)
        },
        "top_confusions": confusions.most_common(12),
    }
