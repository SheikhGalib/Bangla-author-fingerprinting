"""A polite, cached client for the Bengali Wikisource API.

Three facts about Bengali Wikisource shape this module:

1. **Main-namespace chapter pages contain no text.**  They are transclusions of
   the form ``<pages index="Book.pdf" from=149 to=156/>``.  The prose lives in
   the ``পাতা:`` (Page:) namespace, one wiki page per scanned page.

2. **Page: pages carry a proofreading level.**  ``<pagequality level="N">`` is
   0 (no text) … 4 (validated).  Level 1 means *raw, uncorrected OCR*, and for
   Bangla that OCR is frequently garbled into Devanagari look-alikes.  Only
   level ≥ 3 is usable, and filtering on it is the single most important
   quality control in the whole corpus pipeline.

3. **Page: pages can be fetched 50 at a time** via ``prop=revisions``.  That is
   roughly fifty times cheaper than ``action=parse`` per chapter, which matters
   because the anonymous API rate limit is strict enough to return HTTP 429
   within a handful of requests otherwise.

Every response is cached under ``data/raw/cache``, so an interrupted crawl
resumes for free and the exact corpus behind the reported numbers can be
rebuilt.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from . import config

CACHE = config.RAW / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

PAGE_NS = "পাতা"          # Page: namespace (id 104)
PAGE_NS_ID = "104"

#: Minimum proofreading level accepted.  3 = "Proofread", 4 = "Validated".
MIN_PAGE_QUALITY = 3

_BN_DIGITS = "০১২৩৪৫৬৭৮৯"
_BN_TO_ASCII = {ord(d): str(i) for i, d in enumerate(_BN_DIGITS)}


def bn_int(s: str) -> int | None:
    """Parse a page-number string that may use Bangla or ASCII digits."""
    t = s.translate(_BN_TO_ASCII)
    return int(t) if t.isdigit() else None


# ---------------------------------------------------------------------------
# HTML fallback (used only for the few works stored as plain wikitext)
# ---------------------------------------------------------------------------
_SKIP_TAGS = {"style", "script", "sup", "table", "figure", "figcaption"}
_SKIP_CLASS_MARKERS = (
    "headertemplate",      # book/author header -> would leak the label
    "header_notes", "ws-noexport", "noprint", "navigation", "prp-page-image",
    "mw-editsection", "reference", "catlinks", "printfooter", "mw-references",
    "sisterproject", "licence",
)
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4",
               "blockquote", "dd"}


class _WikiTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_stack: list[int] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        self._depth += 1
        cls = (dict(attrs).get("class") or "").lower()
        if (tag in _SKIP_TAGS or any(m in cls for m in _SKIP_CLASS_MARKERS)) \
                and tag != "br":
            self._skip_stack.append(self._depth)
        if tag in _BLOCK_TAGS and not self._skip_stack:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self._skip_stack and self._skip_stack[-1] == self._depth:
            self._skip_stack.pop()
        if tag in _BLOCK_TAGS and not self._skip_stack:
            self.parts.append("\n")
        self._depth = max(0, self._depth - 1)

    def handle_data(self, data):
        if not self._skip_stack:
            self.parts.append(data)

    @property
    def text(self) -> str:
        return html.unescape("".join(self.parts))


def html_to_text(raw_html: str) -> str:
    p = _WikiTextExtractor()
    p.feed(raw_html)
    return p.text


# ---------------------------------------------------------------------------
# Wikitext -> plain text
# ---------------------------------------------------------------------------
_RE_NOINCLUDE = re.compile(r"<noinclude>.*?</noinclude>", re.DOTALL)
_RE_OPEN_NOINCLUDE = re.compile(r"<noinclude>.*", re.DOTALL)
_RE_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL)
_RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_RE_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_RE_LINK_PIPED = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")
_RE_LINK_PLAIN = re.compile(r"\[\[([^\]]*)\]\]")
_RE_EXTLINK = re.compile(r"\[https?://\S+\s+([^\]]*)\]")
_RE_QUOTES = re.compile(r"'{2,5}")
_RE_HEADING = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$", re.MULTILINE)
_RE_LISTMARK = re.compile(r"^[*#:;]+", re.MULTILINE)
_RE_PAGEQUALITY = re.compile(r'<pagequality\s+level="(\d)"')
_RE_INDEX = re.compile(r'index\s*=\s*"([^"]+)"')


def page_quality(wikitext: str) -> int:
    m = _RE_PAGEQUALITY.search(wikitext)
    return int(m.group(1)) if m else -1


def wikitext_to_text(wikitext: str) -> str:
    """Strip wiki markup from a ``পাতা:`` page, keeping only the body prose."""
    t = _RE_COMMENT.sub(" ", wikitext)
    t = _RE_NOINCLUDE.sub(" ", t)      # running headers / footers / page numbers
    t = _RE_OPEN_NOINCLUDE.sub(" ", t)  # unbalanced trailing noinclude
    t = _RE_REF.sub(" ", t)
    t = _RE_TABLE.sub(" ", t)
    for _ in range(4):                  # nested templates
        t, n = _RE_TEMPLATE.subn(" ", t)
        if not n:
            break
    t = _RE_LINK_PIPED.sub(r"\1", t)
    t = _RE_LINK_PLAIN.sub(r"\1", t)
    t = _RE_EXTLINK.sub(r"\1", t)
    t = _RE_HEADING.sub(r"\1", t)
    t = _RE_LISTMARK.sub(" ", t)
    t = _RE_TAG.sub(" ", t)
    t = _RE_QUOTES.sub("", t)
    t = html.unescape(t)
    return t


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class WikisourceClient:
    """Rate-limited, disk-cached wrapper around ``api.php``."""

    def __init__(self, delay: float = config.REQUEST_DELAY, verbose: bool = True):
        self.delay = delay
        self.verbose = verbose
        self._last_call = 0.0
        self.n_requests = 0

    # -- low level ---------------------------------------------------------
    def _cache_path(self, params: dict[str, str]) -> Path:
        key = json.dumps(params, sort_keys=True, ensure_ascii=False)
        return CACHE / (hashlib.sha1(key.encode("utf-8")).hexdigest() + ".json")

    def call(self, **params: str) -> dict:
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")
        cache_file = self._cache_path(params)
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cache_file.unlink(missing_ok=True)

        # POST keeps very long title lists off the URL line.
        data = urllib.parse.urlencode(params).encode("utf-8")
        req = urllib.request.Request(
            config.WIKISOURCE_API,
            data=data,
            headers={
                "User-Agent": config.USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept-Encoding": "gzip",
            },
        )

        backoff = 5.0
        for attempt in range(7):
            wait = self.delay - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        import gzip
                        raw = gzip.decompress(raw)
                    payload = json.loads(raw.decode("utf-8"))
                self._last_call = time.time()
                self.n_requests += 1
                cache_file.write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                return payload
            except urllib.error.HTTPError as exc:
                self._last_call = time.time()
                if exc.code in (429, 500, 502, 503, 504) and attempt < 6:
                    retry_after = exc.headers.get("Retry-After")
                    pause = float(retry_after) if (retry_after or "").isdigit() else backoff
                    if self.verbose:
                        print(f"    HTTP {exc.code}; sleeping {pause:.0f}s", flush=True)
                    time.sleep(pause)
                    backoff = min(backoff * 2, 120)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                self._last_call = time.time()
                if attempt < 6:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)
                    continue
                raise
        raise RuntimeError("unreachable")

    # -- discovery ---------------------------------------------------------
    def author_works(self, author_page: str) -> list[str]:
        """Main-namespace pages linked from an ``লেখক:`` (Author:) page."""
        titles: list[str] = []
        cont: dict[str, str] = {}
        while True:
            d = self.call(action="query", titles=author_page, prop="links",
                          plnamespace="0", pllimit="500", **cont)
            pages = d.get("query", {}).get("pages", [])
            if not pages or pages[0].get("missing"):
                return []
            titles += [l["title"] for l in pages[0].get("links", [])]
            if "continue" in d:
                cont = d["continue"]
            else:
                return titles

    def wikitext(self, titles: list[str]) -> dict[str, str]:
        """Batch-fetch raw wikitext for up to 50 titles per request."""
        out: dict[str, str] = {}
        for i in range(0, len(titles), 50):
            batch = titles[i : i + 50]
            d = self.call(action="query", titles="|".join(batch),
                          prop="revisions", rvprop="content", rvslots="main")
            for pg in d.get("query", {}).get("pages", []):
                if "revisions" not in pg:
                    continue
                out[pg["title"]] = pg["revisions"][0]["slots"]["main"]["content"]
        return out

    def work_indexes(
        self, work_titles: list[str]
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Map each main-namespace work to the scan indexes it transcludes.

        Returns ``(indexes, raw_wikitext)``.  Works that transclude nothing keep
        their prose inline, and the caller falls back to the rendered HTML for
        those.
        """
        texts = self.wikitext(work_titles)
        out: dict[str, list[str]] = {}
        for title, wt in texts.items():
            names = list(dict.fromkeys(_RE_INDEX.findall(wt)))
            if names:
                out[title] = [n.replace("_", " ") for n in names]
        return out, texts

    def index_page_titles(self, index_name: str) -> list[str]:
        """Every ``পাতা:<index>/<n>`` page, sorted by page number."""
        prefix = index_name.replace("_", " ") + "/"
        titles: list[str] = []
        cont: dict[str, str] = {}
        while True:
            d = self.call(action="query", list="allpages", apprefix=prefix,
                          apnamespace=PAGE_NS_ID, aplimit="500", **cont)
            titles += [p["title"] for p in d.get("query", {}).get("allpages", [])]
            if "continue" in d:
                cont = d["continue"]
            else:
                break

        def key(t: str) -> int:
            n = bn_int(t.rsplit("/", 1)[-1])
            return n if n is not None else 10**9

        return sorted(titles, key=key)

    def index_body(
        self, index_name: str, max_pages: int | None = None
    ) -> tuple[str, dict]:
        """Concatenated proofread text of one scan index.

        Returns ``(text, stats)`` where ``stats`` records how many scanned pages
        were available and how many survived the proofreading-quality filter.
        """
        titles = self.index_page_titles(index_name)
        if max_pages is not None:
            titles = titles[:max_pages]
        texts = self.wikitext(titles)

        kept: list[str] = []
        n_ok = 0
        for t in titles:
            wt = texts.get(t)
            if wt is None:
                continue
            if page_quality(wt) < MIN_PAGE_QUALITY:
                continue
            n_ok += 1
            kept.append(wikitext_to_text(wt))
        stats = {"index": index_name, "n_pages": len(titles), "n_proofread": n_ok}
        return "\n".join(kept), stats

    def rendered_text(self, title: str) -> str:
        """Fallback for works that hold their prose directly in wikitext."""
        try:
            d = self.call(action="parse", page=title, prop="text",
                          disabletoc="1", disableeditsection="1",
                          disablelimitreport="1")
        except urllib.error.HTTPError:
            return ""
        return html_to_text(d["parse"]["text"]) if "parse" in d else ""
