"""Corpus source: chapter-serialised Bangla books from ebanglalibrary.com.

Why a second source at all.  The Wikisource crawler in :mod:`wikisource` gives
excellent provenance but only reaches authors who died before ~1950, and the
prose it recovers is scanned-and-proofread nineteenth-century Bengali.  A
reader asked to *verify* the corpus has to navigate the ``পাতা:`` namespace to
do it.  This module targets the opposite trade-off: three authors a Bengali
reader already knows by name, served as ordinary chapter pages, written to
plain ``.txt`` files that anyone can open and read.

The site serialises a book as one ``books/<slug>/`` index page that links to
``lessons/<slug>/`` chapter pages in reading order.  Chapter slugs are *not*
predictable — some books number them ``-1``/``-2``, others do not — so the
index page is always enumerated rather than guessed.

Everything fetched is cached under ``data/raw/ebangla_cache/``.  Re-running the
build after the first pass costs no network at all, which matters because the
polite delay between requests dominates the runtime.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html import unescape

from . import config
from .normalize import bangla_ratio, normalize_text

#: A desktop User-Agent.  The site serves a JS-only shell to unknown agents.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

#: Seconds between successive requests.  The site is a small community archive;
#: there is no reason to hit it faster than a human reader would.
REQUEST_DELAY = 1.5

#: Retries for a failed fetch, with exponential backoff between them.
MAX_RETRIES = 4

CACHE = config.RAW / "ebangla_cache"
CACHE.mkdir(parents=True, exist_ok=True)

#: A paragraph shorter than this is furniture — a caption, a "next chapter"
#: link, a stray byline — not prose.
MIN_PARAGRAPH_CHARS = 30

#: …and one below this Bangla-character ratio is English navigation chrome.
MIN_PARAGRAPH_BANGLA = 0.80

# Boilerplate that appears inside the content div on every page.
_BOILERPLATE = re.compile(
    r"(ebanglalibrary)"
    r"|(পূর্ববর্তী)"          # "previous"
    r"|(পরবর্তী)"            # "next"
    r"|(সূচিপত্র)"           # "table of contents"
    r"|(Pages:)"
    r"|(Post navigation)"
)


# ---------------------------------------------------------------------------
# The book roster
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Book:
    """One serialised book.

    ``role`` is the whole methodology in one field.  ``train`` books build the
    models; the single ``unseen`` book per author is never touched by any
    fitting step and becomes the reported test set.  Because the split is drawn
    at the level of a whole book, work-disjointness is true by construction
    rather than by a shuffling argument.
    """

    key: str
    title_bn: str
    url: str
    role: str  # "train" | "unseen"


@dataclass
class Chapter:
    title: str
    url: str
    text: str


@dataclass
class BookText:
    book: Book
    author: str
    chapters: list[Chapter] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(c.text for c in self.chapters if c.text)

    @property
    def n_chars(self) -> int:
        return len(self.text)


#: Authors, in the order the report presents them.  ``bn`` is what the UI and
#: the plots show; ``key`` is what filenames and label vectors use.
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

_B = "https://www.ebanglalibrary.com/books/"

#: Three books per author: two to learn from, one held out.  The held-out book
#: is chosen to match the *genre* of at least one training book, so that the
#: test measures a change of book and not a change of form — holding out essays
#: after training on fiction would conflate the two.
BOOKS: dict[str, list[Book]] = {
    "tagore": [
        Book(
            "galpaguchchha",
            "গল্পগুচ্ছ",
            _B + "%e0%a6%97%e0%a6%b2%e0%a7%8d%e0%a6%aa%e0%a6%97%e0%a7%81%e0%a6%9a"
            "%e0%a7%8d%e0%a6%9b-%e0%a6%b0%e0%a6%ac%e0%a7%80%e0%a6%a8%e0%a7%8d"
            "%e0%a6%a6%e0%a7%8d%e0%a6%b0%e0%a6%a8%e0%a6%be%e0%a6%a5-%e0%a6%a0/",
            "train",
        ),
        Book(
            "sadhana",
            "সাধনা: জীবনের উপলব্ধি",
            _B + "%e0%a6%b8%e0%a6%be%e0%a6%a7%e0%a6%a8%e0%a6%be-%e0%a6%9c%e0%a7%80"
            "%e0%a6%ac%e0%a6%a8%e0%a7%87%e0%a6%b0-%e0%a6%89%e0%a6%aa%e0%a6%b2"
            "%e0%a6%ac%e0%a7%8d%e0%a6%a7%e0%a6%bf-%e0%a6%b0%e0%a6%ac%e0%a7%80/",
            "train",
        ),
        Book(
            "rahasya-samagra",
            "রহস্য সমগ্র",
            _B + "%e0%a6%b0%e0%a6%b9%e0%a6%b8%e0%a7%8d%e0%a6%af-%e0%a6%b8%e0%a6%ae"
            "%e0%a6%97%e0%a7%8d%e0%a6%b0-%e0%a6%b0%e0%a6%ac%e0%a7%80%e0%a6%a8"
            "%e0%a7%8d%e0%a6%a6%e0%a7%8d%e0%a6%b0%e0%a6%a8%e0%a6%be%e0%a6%a5/",
            "unseen",
        ),
    ],
    "nazrul": [
        Book(
            "kuhelika",
            "কুহেলিকা",
            _B + "%e0%a6%95%e0%a7%81%e0%a6%b9%e0%a7%87%e0%a6%b2%e0%a6%bf%e0%a6%95"
            "%e0%a6%be-%e0%a6%95%e0%a6%be%e0%a6%9c%e0%a7%80-%e0%a6%a8%e0%a6%9c"
            "%e0%a6%b0%e0%a7%81%e0%a6%b2-%e0%a6%87%e0%a6%b8%e0%a6%b2%e0%a6%be/",
            "train",
        ),
        Book(
            "rikter-bedan",
            "রিক্তের বেদন",
            _B + "%e0%a6%b0%e0%a6%bf%e0%a6%95%e0%a7%8d%e0%a6%a4%e0%a7%87%e0%a6%b0-"
            "%e0%a6%ac%e0%a7%87%e0%a6%a6%e0%a6%a8-%e0%a6%95%e0%a6%be%e0%a6%9c"
            "%e0%a7%80-%e0%a6%a8%e0%a6%9c%e0%a6%b0%e0%a7%81%e0%a6%b2-%e0%a6%87/",
            "train",
        ),
        Book(
            "byathar-dan",
            "ব্যথার দান",
            _B + "%e0%a6%ac%e0%a7%8d%e0%a6%af%e0%a6%a5%e0%a6%be%e0%a6%b0-%e0%a6%a6"
            "%e0%a6%be%e0%a6%a8-%e0%a6%95%e0%a6%be%e0%a6%9c%e0%a7%80-%e0%a6%a8"
            "%e0%a6%9c%e0%a6%b0%e0%a7%81%e0%a6%b2-%e0%a6%87%e0%a6%b8%e0%a6%b2/",
            "unseen",
        ),
    ],
    "humayun": [
        Book(
            "himu",
            "হিমু",
            _B + "%e0%a6%b9%e0%a6%bf%e0%a6%ae%e0%a7%81-%e0%a6%b9%e0%a7%81%e0%a6%ae"
            "%e0%a6%be%e0%a6%af%e0%a6%bc%e0%a7%82%e0%a6%a8-%e0%a6%86%e0%a6%b9"
            "%e0%a6%ae%e0%a7%87%e0%a6%a6/",
            "train",
        ),
        Book(
            "bipod",
            "বিপদ",
            _B + "%e0%a6%ac%e0%a6%bf%e0%a6%aa%e0%a6%a6-%e0%a6%b9%e0%a7%81%e0%a6%ae"
            "%e0%a6%be%e0%a6%af%e0%a6%bc%e0%a7%82%e0%a6%a8-%e0%a6%86%e0%a6%b9"
            "%e0%a6%ae%e0%a7%87%e0%a6%a6/",
            "train",
        ),
        Book(
            "himur-hate-nilpadma",
            "হিমুর হাতে কয়েকটি নীলপদ্ম",
            # NB: this slug spells য় *decomposed* (য + nukta, %af%bc) where the
            # others use the precomposed U+09DF (%9f).  The two are visually
            # identical and the site 404s on the wrong one, so leave it alone.
            _B + "%e0%a6%b9%e0%a6%bf%e0%a6%ae%e0%a7%81%e0%a6%b0-%e0%a6%b9%e0%a6%be"
            "%e0%a6%a4%e0%a7%87-%e0%a6%95%e0%a6%af%e0%a6%bc%e0%a7%87%e0%a6%95"
            "%e0%a6%9f%e0%a6%bf-%e0%a6%a8%e0%a7%80%e0%a6%b2%e0%a6%aa%e0%a6%a6/",
            "unseen",
        ),
    ],
}


def books(role: str | None = None) -> list[tuple[str, Book]]:
    """Return ``(author_key, Book)`` pairs, optionally filtered by role."""
    out = []
    for author, bs in BOOKS.items():
        for b in bs:
            if role is None or b.role == role:
                out.append((author, b))
    return out


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
class EBanglaClient:
    """Cached, rate-limited HTML fetcher.

    The cache is keyed by a hash of the URL and holds the decoded HTML, so a
    second build run does no network I/O.  That is what makes the corpus step
    cheap to re-run while developing the cleaner below it.
    """

    def __init__(self, verbose: bool = True, delay: float = REQUEST_DELAY) -> None:
        self.verbose = verbose
        self.delay = delay
        self.n_requests = 0
        self._last = 0.0

    def _cache_path(self, url: str):
        return CACHE / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".json")

    def get(self, url: str) -> str:
        path = self._cache_path(url)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["html"]

        elapsed = time.monotonic() - self._last
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    html = resp.read().decode("utf-8", "replace")
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = exc
                wait = self.delay * (2**attempt)
                if self.verbose:
                    print(f"    retry {attempt + 1}/{MAX_RETRIES} after {wait:.1f}s "
                          f"({type(exc).__name__})", flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError(f"failed to fetch {url}") from last_err

        self._last = time.monotonic()
        self.n_requests += 1
        path.write_text(json.dumps({"url": url, "html": html}, ensure_ascii=False),
                        encoding="utf-8")
        return html


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def chapter_links(html: str) -> list[tuple[str, str]]:
    """Return ``(anchor_text, url)`` for each chapter, in reading order.

    Order matters and is taken from the document: the index page lists chapters
    in sequence, whereas sorting the slugs alphabetically would interleave
    Bengali numerals unpredictably.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    pattern = re.compile(
        r'<a[^>]+href="(https://www\.ebanglalibrary\.com/lessons/[^"]+)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    for url, label in pattern.findall(html):
        if url in seen:
            continue
        seen.add(url)
        text = unescape(re.sub(r"<[^>]+>", "", label)).strip()
        text = re.sub(r"\s+", " ", text)
        out.append((text, url))
    return out


def _content_block(html: str) -> str:
    """Isolate the chapter body.

    The prose sits in ``div.entry-content`` (the theme also tags it
    ``ld-tab-content``).  Falling back to the whole document would drag in the
    sidebar's book list, so a miss is reported by returning an empty string and
    letting the caller decide.

    Most chapters hold their paragraphs as direct children of that div, but a
    minority nest them one or two levels deeper inside wrapper divs -- the same
    theme, rendered identically, different markup.  Taking only direct children
    silently returned nothing for those, which cost two chapters of ``ব্যথার
    দান`` before it was noticed, so descendants are used when there are no
    direct children.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    best, best_len = "", 0
    for div in soup.find_all(["div", "article"]):
        classes = div.get("class") or []
        if "entry-content" not in classes:
            continue
        paras = div.find_all("p", recursive=False) or div.find_all("p")
        length = sum(len(p.get_text(" ", strip=True)) for p in paras)
        if length > best_len:
            best, best_len = "".join(str(p) for p in paras), length
    return best


def chapter_text(html: str) -> str:
    """Extract clean Bangla prose paragraphs from a chapter page."""
    block = _content_block(html)
    if not block:
        return ""
    kept: list[str] = []
    for raw in re.findall(r"<p[^>]*>(.*?)</p>", block, re.DOTALL):
        text = re.sub(r"<[^>]+>", " ", raw)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < MIN_PARAGRAPH_CHARS:
            continue
        if _BOILERPLATE.search(text):
            continue
        if bangla_ratio(text) < MIN_PARAGRAPH_BANGLA:
            continue
        kept.append(normalize_text(text))
    return "\n\n".join(kept)


def download_book(
    author: str, book: Book, client: EBanglaClient, verbose: bool = True
) -> BookText:
    """Fetch every chapter of ``book`` and return it as a :class:`BookText`."""
    if verbose:
        print(f"  [{author}/{book.key}] {book.title_bn}", flush=True)
    index = client.get(book.url)
    links = chapter_links(index)
    if not links:
        raise RuntimeError(f"no chapter links found on {book.url}")

    out = BookText(book=book, author=author)
    for i, (label, url) in enumerate(links, 1):
        text = chapter_text(client.get(url))
        title = label or f"অধ্যায় {i}"
        out.chapters.append(Chapter(title=title, url=url, text=text))
        if verbose:
            flag = "" if text else "   <- EMPTY"
            print(f"      {i:>3}/{len(links)}  {len(text):>6} chars  {title[:40]}{flag}",
                  flush=True)
    if verbose:
        print(f"      = {out.n_chars:,} chars over {len(out.chapters)} chapters",
              flush=True)
    return out
