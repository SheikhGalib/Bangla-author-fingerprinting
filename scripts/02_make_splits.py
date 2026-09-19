"""Stage 2: segment the corpus into passages and fix the train/val/test split.

    .venv/Scripts/python.exe scripts/02_make_splits.py

Writes ``data/processed/passages.jsonl`` and ``data/splits/split.json``.  Both
are small and deterministic, and together they are the only input the GPU
training job needs -- which is why this stage is separate from the notebooks.

Two decisions are made here and are worth stating plainly.

*Training is not balanced; the test set is.*  Down-sampling Tagore to match the
smallest author would throw away most of his training passages to buy a tidier
table, so the classifiers are class-weighted instead and see everything.  The
*test* set is balanced, because that is where balance actually buys something:
accuracy is then directly comparable against a 1/3 chance rate and the
confusion matrix is not distorted by unequal row totals.

*The test set is three whole books*, one per author, chosen before any text was
downloaded.  Nothing here selects it; this stage only verifies it.
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from banglastylo import config, corpus, splits


def balance_test(part, seed: int = config.SEED):
    """Down-sample each author to the smallest author's count, evenly by work."""
    by_author = defaultdict(list)
    for p in part:
        by_author[p.author].append(p)
    n = min(len(v) for v in by_author.values())
    rng = random.Random(seed)
    out = []
    for author in sorted(by_author):
        ps = sorted(by_author[author], key=lambda p: p.passage_id)
        rng.shuffle(ps)
        out += ps[:n]
    return out


def main() -> int:
    print("=== segmenting ===")
    passages = corpus.segment_books()

    print("\n=== splitting ===")
    split = splits.split_by_role(passages)
    split.test = balance_test(split.test)

    rep = splits.leakage_report(split)
    print("\n=== leakage check ===")
    for k, v in rep["overlaps"].items():
        print(f"  {k:32s} {v}")
    if not rep["clean"]:
        print("\nLEAKAGE DETECTED — refusing to write the split", file=sys.stderr)
        return 1
    print("  clean: every test book is unseen, no chapter spans train and val")

    print("\n=== split sizes ===")
    print(f"{'author':<10}{'train':>8}{'val':>7}{'test':>7}   test book")
    test_book = {p.author: p.work for p in split.test}
    for a in config.AUTHORS:
        c = {n: sum(1 for p in part if p.author == a)
             for n, part in (("train", split.train), ("val", split.val),
                             ("test", split.test))}
        print(f"{a:<10}{c['train']:>8}{c['val']:>7}{c['test']:>7}   {test_book.get(a, '-')}")
    print(f"{'TOTAL':<10}{len(split.train):>8}{len(split.val):>7}{len(split.test):>7}")

    n_classes = len({p.author for p in split.test})
    print(f"\nchance accuracy on the balanced test set: {1 / n_classes:.3f}")

    corpus.save_passages(split.train + split.val + split.test)
    splits.save(split)

    print("\ntrain work counts:", dict(Counter(p.work for p in split.train)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
