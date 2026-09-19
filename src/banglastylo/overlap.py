"""Detect and remove text shared between the training and held-out books.

This module exists because the corpus build turned up a real instance of the
problem, not as a precaution.

Tagore's *রহস্য সমগ্র* is an **anthology**: a later compilation that re-prints
stories which also appear in *গল্পগুচ্ছ*.  Both were selected -- গল্পগুচ্ছ to
train on, রহস্য সমগ্র to hold out -- and several stories (খোকাবাবুর
প্রত্যাবর্তন, সম্পত্তি সমর্পণ, ক্ষুধিত পাষাণ and others) sit on both sides.
Left alone, the "unseen" book would contain pages the model had memorised
word for word, and the headline number would be meaningless.  Bengali
literature is full of such compilations, so a title-level eyeball check is not
enough; this is detected from the text itself.

The rule is: **the held-out book is never modified.**  Overlap is always
resolved by dropping the offending chapter from the *training* side.  That
keeps the promise the corpus makes to a reader -- open any page of
``corpus/unseen/`` and the model provably has not seen it -- at the cost of
some training material, which is the cheaper of the two.

Detection uses token shingles rather than exact string equality, because the
two printings differ in punctuation and spelling normalisation even where the
prose is identical.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .normalize import word_tokenize

#: Tokens per shingle.  Long enough that ordinary shared phrasing between two
#: works by the same author does not register; short enough to survive the
#: typographic differences between two printings of one story.
SHINGLE_N = 10

#: Step between shingle starts.  This must be 1, and the reason is worth
#: recording: with any stride > 1 the two printings fall out of phase.  A
#: single extra token early in one edition shifts every later window, so two
#: copies of the *same story* scored only 0.10-0.32 containment at stride 5 and
#: slipped under the threshold.  At stride 1 every position is a window start,
#: so a shared passage matches regardless of alignment.
SHINGLE_STRIDE = 1

#: Containment above this fraction means the training chapter is substantially
#: reprinted in the held-out book.  Two different stories by one author sit far
#: below this; two printings of one story sit near 1.0, so the threshold is not
#: delicately placed.
CONTAINMENT_THRESHOLD = 0.35


@dataclass
class OverlapHit:
    author: str
    train_work: str
    train_chapter: str
    unseen_work: str
    unseen_chapter: str
    containment: float


def shingles(text: str, n: int = SHINGLE_N, stride: int = SHINGLE_STRIDE) -> set[str]:
    """Hashed token n-grams of ``text``.

    Hashing keeps the set small: a book is tens of thousands of shingles, and
    storing 8-byte digests instead of the joined token strings is what lets the
    all-pairs comparison below stay in memory.
    """
    toks = word_tokenize(text)
    return {
        hashlib.blake2b(" ".join(toks[i : i + n]).encode("utf-8"),
                        digest_size=8).hexdigest()
        for i in range(0, max(1, len(toks) - n), stride)
    }


def containment(a: set[str], b: set[str]) -> float:
    """Fraction of ``a`` that also occurs in ``b``.

    Containment, not Jaccard.  A short story reprinted inside a large anthology
    has a small Jaccard score simply because the anthology is bigger, while its
    containment in that anthology is close to 1 -- which is the question being
    asked.
    """
    return len(a & b) / len(a) if a else 0.0


def find_overlaps(
    train: dict[str, dict[str, str]],
    unseen: dict[str, dict[str, str]],
    threshold: float = CONTAINMENT_THRESHOLD,
) -> list[OverlapHit]:
    """Report training chapters substantially reprinted in a held-out book.

    Both arguments are ``{author: {chapter_label: text}}``.  Comparison is
    within an author only: two authors sharing wording is a fact about the
    language, not a leak.
    """
    hits: list[OverlapHit] = []
    for author, train_chapters in train.items():
        if author not in unseen:
            continue
        unseen_sh = {k: shingles(v) for k, v in unseen[author].items() if v.strip()}
        for t_label, t_text in train_chapters.items():
            if not t_text.strip():
                continue
            t_sh = shingles(t_text)
            for u_label, u_sh in unseen_sh.items():
                c = containment(t_sh, u_sh)
                if c >= threshold:
                    t_work, _, t_chap = t_label.partition("::")
                    u_work, _, u_chap = u_label.partition("::")
                    hits.append(OverlapHit(
                        author=author,
                        train_work=t_work.strip(), train_chapter=t_chap.strip(),
                        unseen_work=u_work.strip(), unseen_chapter=u_chap.strip(),
                        containment=round(c, 3),
                    ))
                    break          # one match is enough to drop the chapter
    return hits


def drop_overlapping(
    train: dict[str, dict[str, str]], hits: list[OverlapHit]
) -> dict[str, dict[str, str]]:
    """Return ``train`` without the chapters named in ``hits``."""
    doomed = {(h.author, f"{h.train_work} :: {h.train_chapter}") for h in hits}
    out: dict[str, dict[str, str]] = {}
    for author, chapters in train.items():
        kept = {}
        for label, text in chapters.items():
            work, _, chap = label.partition("::")
            key = (author, f"{work.strip()} :: {chap.strip()}")
            if key not in doomed:
                kept[label] = text
        out[author] = kept
    return out


def format_report(hits: list[OverlapHit]) -> str:
    """Human-readable summary, for the notebook and the report to quote."""
    if not hits:
        return "no train/unseen overlap detected"
    lines = [f"{len(hits)} training chapter(s) reprinted in a held-out book:"]
    for h in hits:
        lines.append(
            f"  [{h.author}] {h.train_work} / {h.train_chapter}"
            f"  ->  {h.unseen_work} / {h.unseen_chapter}"
            f"   (containment {h.containment:.2f})"
        )
    return "\n".join(lines)
