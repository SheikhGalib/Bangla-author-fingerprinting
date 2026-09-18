"""Stage 1: crawl Bengali Wikisource and write data/raw/corpus_raw.json.

Run from the project root:

    .venv/Scripts/python.exe scripts/01_build_corpus.py

Every API response is cached under ``data/raw/cache``, so interrupting and
re-running this script costs nothing and resumes where it stopped.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from banglastylo import config
from banglastylo.corpus import build_raw_corpus

if __name__ == "__main__":
    raw = build_raw_corpus(list(config.AUTHORS))
    print("\n=== crawl summary ===")
    for author, works in raw.items():
        chars = sum(len(t) for t in works.values())
        print(f"{author:16s} {len(works):>3} works  {chars:>9,} chars")
