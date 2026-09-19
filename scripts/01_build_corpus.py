"""Stage 1: download the books, de-overlap them, write the plain-text corpus.

    .venv/Scripts/python.exe scripts/01_build_corpus.py

Produces three things.

1. ``corpus/train/`` and ``corpus/unseen/`` -- one readable ``.txt`` per book,
   grouped by author.  These are the copy a human opens.  They are written
   *after* de-overlapping, so what is in ``train/`` is exactly what the models
   are allowed to learn from, and a passage copied out of ``unseen/`` is
   guaranteed to be text no model has seen.

2. ``data/raw/corpus_raw.json`` -- the same text as
   ``{author: {book: {chapter: text}}}``, which the segmentation stage
   consumes.  Chapter granularity is kept because overlap is resolved one
   chapter at a time.

3. ``data/processed/overlap_report.json`` -- what was dropped and why.

Every page is cached under ``data/raw/ebangla_cache``, so re-running costs no
network.  The first run takes roughly five minutes, almost all of it the polite
delay between requests.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from banglastylo import config, overlap
from banglastylo.ebangla import AUTHORS, BOOKS, EBanglaClient, download_book

#: Windows rejects these in a filename; some titles carry a colon.
_UNSAFE = str.maketrans({c: "-" for c in '<>:"/\\|?*'})

#: Anthology chapter titles repeat the author's name ("কঙ্কাল - রবীন্দ্রনাথ
#: ঠাকুর").  Stripping it keeps the chapter headings in the .txt readable.
_AUTHOR_SUFFIX = re.compile(
    r"\s*[-–—]\s*(?:" + "|".join(re.escape(a["bn"]) for a in AUTHORS.values()) + r")\s*$"
)


def _safe(name: str) -> str:
    return name.translate(_UNSAFE).strip()


def _clean_title(title: str) -> str:
    return _AUTHOR_SUFFIX.sub("", title).strip()


def write_plain_text(author: str, title_bn: str, role: str, url: str,
                     chapters: list[tuple[str, str]]) -> Path:
    """Write one book as a readable ``.txt`` under ``corpus/<role>/<author>/``."""
    root = config.CORPUS_TRAIN if role == "train" else config.CORPUS_UNSEEN
    folder = root / _safe(AUTHORS[author]["en"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (_safe(title_bn) + ".txt")

    n_chars = sum(len(t) for _, t in chapters)
    role_bn = "প্রশিক্ষণ (training)" if role == "train" else "অদেখা (held-out / unseen)"
    lines = [
        "=" * 70,
        f"বই      : {title_bn}",
        f"লেখক    : {AUTHORS[author]['bn']}  ({AUTHORS[author]['en']})",
        f"অধ্যায়  : {len(chapters)}",
        f"অক্ষর   : {n_chars:,}",
        f"উৎস     : {url}",
        f"ভূমিকা  : {role_bn}",
        "=" * 70,
    ]
    for i, (title, text) in enumerate(chapters, 1):
        lines += ["", "", f"--- {i:02d}. {title} ---", "", text]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


README = """\
বাংলা লেখক-শনাক্তকরণ কর্পাস  /  Bangla Authorship Attribution Corpus
====================================================================

train/   - the books the models learn from.  Two books per author.
unseen/  - one book per author that NO training step ever reads.

Why the split is by whole book, not by paragraph
------------------------------------------------
If paragraphs from the same novel appear on both sides of a train/test split,
a classifier can score highly just by recognising character names and places.
That is topic identification wearing style's clothes.  Here an entire book is
held back instead, so every sentence in unseen/ comes from a work the model has
never seen -- the accuracy reported is accuracy on a *new book*.

To test it yourself: open any file in unseen/, copy a paragraph, and paste it
into the app.  The model was never shown that book.

A note on anthologies
---------------------
Bengali publishers reissue short stories in later compilations, so two books by
one author can contain the same story under the same name.  Any training
chapter found to be reprinted in a held-out book was removed from the training
side before these files were written -- see overlap_report.txt.  The held-out
books are never edited.  What you see in train/ is exactly what was trained on.

Files are UTF-8 plain text and open in any editor.
"""


def main() -> int:
    client = EBanglaClient(verbose=True)

    # --- download -----------------------------------------------------------
    downloaded: dict[str, dict] = {}
    for author in AUTHORS:
        print(f"\n=== {AUTHORS[author]['bn']} ===", flush=True)
        for book in BOOKS[author]:
            bt = download_book(author, book, client)
            chapters = [(_clean_title(c.title), c.text) for c in bt.chapters if c.text]
            if not chapters:
                print(f"  FAILED: no text for {author}/{book.key}", file=sys.stderr)
                return 1
            downloaded[f"{author}/{book.key}"] = {
                "author": author, "book": book, "chapters": chapters,
            }

    # --- overlap between the training and held-out sides --------------------
    # Flatten to {author: {"<book> :: <chapter>": text}} for the detector.
    def flatten(role: str) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for d in downloaded.values():
            if d["book"].role != role:
                continue
            for title, text in d["chapters"]:
                out.setdefault(d["author"], {})[f"{d['book'].title_bn} :: {title}"] = text
        return out

    hits = overlap.find_overlaps(flatten("train"), flatten("unseen"))
    print(f"\n{'=' * 78}\noverlap check\n{'=' * 78}")
    print(overlap.format_report(hits))
    dropped = {(h.author, h.train_work, h.train_chapter) for h in hits}

    (config.PROCESSED / "overlap_report.json").write_text(
        json.dumps([asdict(h) for h in hits], ensure_ascii=False, indent=2),
        encoding="utf-8")
    (config.CORPUS_TXT / "overlap_report.txt").write_text(
        overlap.format_report(hits) + "\n", encoding="utf-8")

    # --- write the corpus, minus the overlapping training chapters ----------
    raw: dict[str, dict[str, dict[str, str]]] = {}
    manifest: list[dict] = []
    for d in downloaded.values():
        author, book = d["author"], d["book"]
        kept, removed = [], []
        for title, text in d["chapters"]:
            if book.role == "train" and (author, book.title_bn, title) in dropped:
                removed.append(title)
            else:
                kept.append((title, text))
        path = write_plain_text(author, book.title_bn, book.role, book.url, kept)
        raw.setdefault(author, {})[book.title_bn] = dict(kept)
        manifest.append({
            "author": author,
            "author_bn": AUTHORS[author]["bn"],
            "book": book.key,
            "title_bn": book.title_bn,
            "role": book.role,
            "url": book.url,
            "chapters": len(kept),
            "chapters_removed": removed,
            "chars": sum(len(t) for _, t in kept),
            "txt": str(path.relative_to(config.ROOT)),
        })

    (config.CORPUS_TXT / "README.txt").write_text(README, encoding="utf-8")
    (config.RAW / "corpus_raw.json").write_text(
        json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    (config.RAW / "provenance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- summary ------------------------------------------------------------
    print(f"\n{'=' * 78}\ncorpus summary  ({client.n_requests} pages fetched)\n{'=' * 78}")
    print(f"{'author':<10}{'role':<8}{'book':<28}{'chaps':>6}{'drop':>5}{'chars':>10}")
    for m in manifest:
        print(f"{m['author']:<10}{m['role']:<8}{m['title_bn'][:26]:<28}"
              f"{m['chapters']:>6}{len(m['chapters_removed']):>5}{m['chars']:>10,}")
    for role in ("train", "unseen"):
        rows = [m for m in manifest if m["role"] == role]
        print(f"  {role:<7} {sum(m['chars'] for m in rows):>10,} chars "
              f"over {sum(m['chapters'] for m in rows)} chapters")
    print(f"\nwrote {config.CORPUS_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
