"""Bangla text normalisation, sentence segmentation and word tokenisation.

Bangla orthography needs a few decisions that Latin-script pipelines never have
to make, so they are spelled out here rather than buried in a regex:

* the sentence terminator is the *danda* ``।`` (U+0964), not a full stop; a
  full stop still appears inside abbreviations and dates, so it is only treated
  as a terminator when followed by whitespace and a Bangla letter;
* Bangla has its own digits ``০-৯`` (U+09E6-U+09EF), which we fold to a single
  ``<num>`` placeholder so that page numbers and dates do not become authorship
  evidence;
* ZWJ/ZWNJ (U+200C/U+200D) are inconsistently typed by Wikisource contributors.
  They are stripped so that two spellings of the same conjunct do not become two
  different tokens.
"""
from __future__ import annotations

import re
import unicodedata

# --- Unicode ranges ---------------------------------------------------------
BANGLA_BLOCK = "\u0980-\u09FF"
BANGLA_DIGITS = "\u09E6-\u09EF"
DANDA = "\u0964"
DOUBLE_DANDA = "\u0965"

_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff\u00ad]")
_NON_BANGLA_PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2012": "-", "\u2015": "-",
    "\u2026": "...", "\u00a0": " ",
}

#: Wikisource editorial furniture that is not part of the author's prose.
_EDITORIAL = re.compile(
    r"(\[\[[^\]]*\]\])"            # leftover wiki links
    r"|(\{\{[^}]*\}\})"            # leftover templates
    r"|(\[[০-৯0-9ivxlcIVXLC]+\])"  # bracketed page / footnote numbers
    r"|(<[^>]+>)"                  # stray html
)

_MULTISPACE = re.compile(r"[ \t\u3000]+")
_MULTINEWLINE = re.compile(r"\n{3,}")

# Sentence terminators: danda, double danda, ? and !.  A Latin '.' only counts
# when it is *not* part of an abbreviation-looking run of single letters.
_SENT_SPLIT = re.compile(
    rf"(?<=[{DANDA}{DOUBLE_DANDA}?!])\s+"
    rf"|(?<=\.)\s+(?=[{BANGLA_BLOCK}A-Z])"
)

_NUM_TOKEN = re.compile(rf"^[{BANGLA_DIGITS}0-9]+([.,\-/][{BANGLA_DIGITS}0-9]+)*$")

# A token is a run of Bangla letters/marks (optionally with an internal hyphen),
# a run of digits, or a single punctuation mark.
_TOKEN = re.compile(
    rf"[{BANGLA_BLOCK}]+(?:[-\u2010][{BANGLA_BLOCK}]+)*"
    rf"|[A-Za-z]+"
    rf"|[{BANGLA_DIGITS}0-9]+(?:[.,][{BANGLA_DIGITS}0-9]+)*"
    rf"|[{DANDA}{DOUBLE_DANDA}?!,;:\"'()\[\]\-.]"
)

PUNCT_CHARS = set(f"{DANDA}{DOUBLE_DANDA}?!,;:\"'()[]-.")


def normalize_text(text: str) -> str:
    """Return `text` in a canonical form suitable for feature extraction."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    for src, dst in _NON_BANGLA_PUNCT_MAP.items():
        text = text.replace(src, dst)
    text = _EDITORIAL.sub(" ", text)
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()


def bangla_ratio(text: str) -> float:
    """Fraction of the non-space characters that live in the Bangla block.

    Used to throw away English prefaces, transliteration tables and OCR noise.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    n_bn = sum(1 for c in chars if "\u0980" <= c <= "\u09FF")
    return n_bn / len(chars)


def sentence_split(text: str) -> list[str]:
    """Split normalised `text` into sentences on danda / ? / ! / guarded '.'."""
    text = text.replace("\n", " ")
    parts = _SENT_SPLIT.split(text)
    return [s.strip() for s in parts if s and s.strip()]


def word_tokenize(text: str, keep_punct: bool = False) -> list[str]:
    """Tokenise normalised `text`.

    Digits collapse to ``<num>`` so that page numbers, dates and chapter counts
    cannot leak into the stylistic signal.
    """
    toks = _TOKEN.findall(text)
    out: list[str] = []
    for t in toks:
        if t in PUNCT_CHARS:
            if keep_punct:
                out.append(t)
            continue
        if _NUM_TOKEN.match(t):
            out.append("<num>")
        else:
            out.append(t)
    return out


def strip_punctuation(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in PUNCT_CHARS]
