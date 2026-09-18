"""Corpus construction: crawl, clean, segment and balance.

The unit of attribution is a **passage** of roughly ``config.PASSAGE_TOKENS``
tokens, cut on sentence boundaries.  Every passage remembers which *work* it
came from, because train/test splitting is done at the work level: a book is
never split across the train and test sides.  Without that discipline a
classifier can score very highly by memorising the proper nouns of a novel and
we would be reporting topic identification while calling it style.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass

from . import config
from .normalize import bangla_ratio, normalize_text, sentence_split, word_tokenize
from .wikisource import WikisourceClient

# Wikisource titles that are indexes, portals or translations rather than prose.
_TITLE_BLOCKLIST = re.compile(
    r"(রচনাবলী)|(সমগ্র)|(তালিকা)|(নির্ঘণ্ট)|(অনুবাদ)|(সূচী)|(সূচি)|(পাতা:)"
)

#: Stop crawling an author once this many characters of clean text are held.
CHAR_BUDGET_PER_AUTHOR = 900_000
#: Never fetch more than this many scanned pages from one book.
MAX_PAGES_PER_INDEX = 400
#: A book contributing fewer clean characters than this is not worth a work slot.
MIN_CHARS_PER_WORK = 3_000


@dataclass
class Passage:
    author: str
    work: str
    passage_id: str
    text: str
    n_tokens: int
    n_sentences: int


# ---------------------------------------------------------------------------
# Crawling
# ---------------------------------------------------------------------------
def clean_body(raw: str) -> str:
    """Keep only lines that look like Bangla prose.

    Wikisource scan pages carry running headers, page numbers, English
    epigraphs, and — on badly OCR'd scans — runs of Devanagari look-alike
    characters.  Requiring a high Bangla-block ratio removes all four, because
    Devanagari lives in a different Unicode block from Bangla.
    """
    text = normalize_text(raw)
    kept: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 30:
            continue
        if bangla_ratio(line) < 0.88:
            continue
        kept.append(line)
    return "\n".join(kept)


def crawl_author(
    key: str, client: WikisourceClient, verbose: bool = True
) -> tuple[dict[str, str], list[dict]]:
    """Return ``({work_title: clean_text}, provenance)`` for one author.

    Text is taken from the ``পাতা:`` (Page:) namespace, keeping only pages the
    Wikisource community has marked proofread.  Works that hold their prose
    inline instead of transcluding a scan fall back to rendered HTML.
    """
    meta = config.AUTHORS[key]
    works = client.author_works(meta["wikisource"])
    works = [w for w in works if not _TITLE_BLOCKLIST.search(w)]
    works.sort()
    if verbose:
        print(f"[{key}] {len(works)} candidate works", flush=True)

    indexes, inline = client.work_indexes(works)

    # One scan index often backs many chapter pages; crawl each index once and
    # attribute it to the shortest work title that transcludes it (that title is
    # the book, the longer ones are its chapters).
    index_owner: dict[str, str] = {}
    for work, names in indexes.items():
        for name in names:
            cur = index_owner.get(name)
            if cur is None or len(work) < len(cur):
                index_owner[name] = work

    out: dict[str, str] = {}
    provenance: list[dict] = []
    n_chars = 0

    for name, work in sorted(index_owner.items(), key=lambda kv: kv[1]):
        if n_chars >= CHAR_BUDGET_PER_AUTHOR:
            break
        body, stats = client.index_body(name, max_pages=MAX_PAGES_PER_INDEX)
        text = clean_body(body)
        stats.update(work=work, n_chars=len(text))
        provenance.append(stats)
        if len(text) < MIN_CHARS_PER_WORK:
            if verbose:
                print(f"  [{key}] skip {name[:52]:54s} "
                      f"{stats['n_proofread']}/{stats['n_pages']} proofread, "
                      f"{len(text):,} chars", flush=True)
            continue
        label = f"{work} :: {name}"
        out[label] = text
        n_chars += len(text)
        if verbose:
            print(f"  [{key}] {name[:52]:54s} "
                  f"{stats['n_proofread']:>3}/{stats['n_pages']:>3} proofread  "
                  f"{len(text):>8,} chars  (total {n_chars:,})", flush=True)

    # Fallback: works with prose inline in the wikitext, no scan behind them.
    if n_chars < CHAR_BUDGET_PER_AUTHOR:
        for work in works:
            if n_chars >= CHAR_BUDGET_PER_AUTHOR:
                break
            if work in indexes:
                continue
            wt = inline.get(work, "")
            if len(wt) < 1500:
                continue
            from .wikisource import wikitext_to_text

            text = clean_body(wikitext_to_text(wt))
            if len(text) < MIN_CHARS_PER_WORK:
                continue
            out[f"{work} :: inline"] = text
            n_chars += len(text)
            provenance.append(
                {"index": "(inline wikitext)", "work": work,
                 "n_pages": 1, "n_proofread": 1, "n_chars": len(text)}
            )
            if verbose:
                print(f"  [{key}] {work[:52]:54s} inline        "
                      f"{len(text):>8,} chars  (total {n_chars:,})", flush=True)

    return out, provenance


def build_raw_corpus(
    author_keys: list[str] | None = None, verbose: bool = True
) -> dict[str, dict[str, str]]:
    """Crawl every requested author and persist to ``data/raw/corpus_raw.json``."""
    keys = author_keys or list(config.AUTHORS)
    client = WikisourceClient(verbose=verbose)
    corpus: dict[str, dict[str, str]] = {}
    prov: dict[str, list[dict]] = {}
    for k in keys:
        corpus[k], prov[k] = crawl_author(k, client, verbose=verbose)
    path = config.RAW / "corpus_raw.json"
    path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    (config.RAW / "provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verbose:
        print(f"\nwrote {path} ({client.n_requests} API requests)", flush=True)
    return corpus


# ---------------------------------------------------------------------------
# Segmentation into passages
# ---------------------------------------------------------------------------
def _shingles(tokens: list[str], n: int = 8) -> list[str]:
    return [
        hashlib.blake2b(" ".join(tokens[i : i + n]).encode("utf-8"), digest_size=8).hexdigest()
        for i in range(0, max(0, len(tokens) - n), 4)
    ]


def segment_author(
    author: str,
    works: dict[str, str],
    target: int = config.PASSAGE_TOKENS,
    minimum: int = config.MIN_PASSAGE_TOKENS,
    seen_shingles: set[str] | None = None,
) -> list[Passage]:
    """Cut an author's works into sentence-aligned passages, de-duplicating."""
    seen_shingles = seen_shingles if seen_shingles is not None else set()
    seen_exact: set[str] = set()
    passages: list[Passage] = []

    for work, text in sorted(works.items()):
        sentences = sentence_split(text.replace("\n", " "))
        buf: list[str] = []
        buf_tokens = 0
        for sent in sentences:
            n = len(word_tokenize(sent))
            if n == 0:
                continue
            buf.append(sent)
            buf_tokens += n
            if buf_tokens < target:
                continue

            body = " ".join(buf)
            buf, buf_tokens = [], 0
            toks = word_tokenize(body)
            if len(toks) < minimum:
                continue

            digest = hashlib.sha1(body.encode("utf-8")).hexdigest()
            if digest in seen_exact:
                continue
            sh = _shingles(toks)
            if sh and sum(1 for s in sh if s in seen_shingles) / len(sh) > 0.5:
                continue  # near-duplicate of an already-kept passage
            seen_exact.add(digest)
            seen_shingles.update(sh)

            passages.append(
                Passage(
                    author=author,
                    work=work,
                    passage_id=f"{author}:{digest[:12]}",
                    text=body,
                    n_tokens=len(toks),
                    n_sentences=len(sentence_split(body)),
                )
            )
    return passages


def build_passages(
    raw: dict[str, dict[str, str]] | None = None, verbose: bool = True
) -> list[Passage]:
    if raw is None:
        raw = json.loads((config.RAW / "corpus_raw.json").read_text(encoding="utf-8"))
    seen: set[str] = set()
    all_passages: list[Passage] = []
    for author, works in raw.items():
        works = {
            w: t for w, t in works.items()
            if not any(b in w for b in config.WORK_BLOCKLIST)
        }
        ps = segment_author(author, works, seen_shingles=seen)
        if verbose:
            print(f"[{author}] {len(ps):>5} passages from {len(works)} works")
        all_passages += ps
    return all_passages


def balance(
    passages: list[Passage],
    min_per_author: int = config.MIN_PASSAGES_PER_AUTHOR,
    min_works: int = config.MIN_WORKS_PER_AUTHOR,
    cap: int | None = config.BALANCE_TO,
    seed: int = config.SEED,
    verbose: bool = True,
) -> tuple[list[Passage], list[dict]]:
    """Drop thin authors, then down-sample everyone to the smallest survivor.

    An author is dropped for too few passages *or* too few distinct works; the
    second criterion is the binding one, since a one-book author cannot take
    part in a work-disjoint split at all.

    Down-sampling walks each author's works round-robin so that a balanced
    author keeps material from *all* of their works rather than from whichever
    book happened to be crawled first.

    Returns ``(passages, dropped)`` where ``dropped`` records who was excluded
    and on which criterion, for the report to state rather than gloss over.
    """
    import random

    by_author: dict[str, list[Passage]] = defaultdict(list)
    for p in passages:
        by_author[p.author].append(p)

    dropped: list[dict] = []
    for a in sorted(by_author):
        ps = by_author[a]
        n_works = len({p.work for p in ps})
        reasons = []
        if len(ps) < min_per_author:
            reasons.append(f"{len(ps)} passages < {min_per_author}")
        if n_works < min_works:
            reasons.append(f"{n_works} works < {min_works}")
        if reasons:
            dropped.append({"author": a, "passages": len(ps),
                            "works": n_works, "reason": "; ".join(reasons)})
            if verbose:
                print(f"  dropping {a}: {'; '.join(reasons)}")
            del by_author[a]

    if not by_author:
        raise RuntimeError("no author cleared the minimum passage threshold")

    n = cap or min(len(ps) for ps in by_author.values())
    rng = random.Random(seed)
    out: list[Passage] = []
    for author, ps in sorted(by_author.items()):
        by_work: dict[str, list[Passage]] = defaultdict(list)
        for p in ps:
            by_work[p.work].append(p)
        for w in by_work.values():
            rng.shuffle(w)
        works = sorted(by_work)
        picked: list[Passage] = []
        i = 0
        while len(picked) < min(n, len(ps)):
            w = works[i % len(works)]
            if by_work[w]:
                picked.append(by_work[w].pop())
            i += 1
            if all(not by_work[w] for w in works):
                break
        out += picked
        if verbose:
            print(f"  {author}: {len(picked)} passages from {len(by_work)} works")
    return out, dropped


def save_passages(passages: list[Passage], name: str = "passages.jsonl") -> None:
    path = config.PROCESSED / name
    with path.open("w", encoding="utf-8") as fh:
        for p in passages:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(passages)} passages)")


def load_passages(name: str = "passages.jsonl") -> list[Passage]:
    path = config.PROCESSED / name
    return [
        Passage(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
