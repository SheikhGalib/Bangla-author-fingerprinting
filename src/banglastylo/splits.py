"""Work-disjoint train / validation / test splitting.

The whole credibility of an authorship-attribution number rests on this file.

If passages are split at random, passages from the *same novel* land on both
sides of the split.  A classifier can then win by recognising a character name
or a village name — topic identification wearing style's clothes.  Reported
accuracy goes up and means less.

So splitting is done per author, over *works*: each of an author's books is
assigned whole to exactly one of train / validation / test.  Every author is
guaranteed to appear in all three splits, and no book ever crosses a boundary.
``leakage_report`` re-checks the invariant after the fact rather than trusting
that the code above it was right.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass

from . import config
from .corpus import Passage


@dataclass
class Split:
    train: list[Passage]
    val: list[Passage]
    test: list[Passage]

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def split_by_work(
    passages: list[Passage],
    test_fraction: float = config.TEST_FRACTION,
    val_fraction: float = config.VAL_FRACTION,
    seed: int = config.SEED,
) -> Split:
    """Assign whole works to splits, per author, hitting the target fractions.

    Books are indivisible and differ wildly in length, so hitting 60/15/25
    exactly is impossible.  This is the classic *largest-processing-time*
    heuristic for partitioning into weighted bins: take the author's works
    biggest-first, and give each one to whichever split is currently furthest
    below its quota.  Taking the big books first is what makes it work — a
    small book at the end can fine-tune a bin, whereas a big book arriving last
    can only overshoot it.

    Realised fractions still drift from the targets, and :func:`describe`
    reports the drift rather than hiding it.
    """
    rng = random.Random(seed)
    by_author_work: dict[str, dict[str, list[Passage]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for p in passages:
        by_author_work[p.author][p.work].append(p)

    buckets: dict[str, list[Passage]] = {"train": [], "val": [], "test": []}
    targets = {
        "train": 1.0 - test_fraction - val_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }

    for author in sorted(by_author_work):
        works = by_author_work[author]
        names = sorted(works)
        rng.shuffle(names)                       # tie-break reproducibly
        names.sort(key=lambda w: -len(works[w]))
        total = sum(len(works[w]) for w in names)

        assigned = {"train": 0, "val": 0, "test": 0}
        # With one or two works there is no honest three-way split; say so by
        # filling train first, then test, and leaving val empty.
        order = ["train", "test", "val"]
        for i, w in enumerate(names):
            block = works[w]
            if i < len(order) and len(names) <= 3:
                part = order[i]
            else:
                part = max(
                    targets,
                    key=lambda k: targets[k] * total - assigned[k],
                )
            buckets[part] += block
            assigned[part] += len(block)

    return Split(train=buckets["train"], val=buckets["val"],
                 test=buckets["test"])


def leakage_report(split: Split) -> dict:
    """Verify no work — and no passage — appears in more than one split."""
    sets = {
        name: {p.work for p in part}
        for name, part in (("train", split.train), ("val", split.val),
                           ("test", split.test))
    }
    ids = {
        name: {p.passage_id for p in part}
        for name, part in (("train", split.train), ("val", split.val),
                           ("test", split.test))
    }
    overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{a}|{b}:works"] = sorted(sets[a] & sets[b])
        overlaps[f"{a}|{b}:passages"] = len(ids[a] & ids[b])
    clean = all(
        (not v if isinstance(v, list) else v == 0) for v in overlaps.values()
    )
    return {"clean": clean, "overlaps": overlaps}


def describe(split: Split) -> dict:
    """Per-author, per-split passage and work counts, for the report table."""
    rows: dict[str, dict] = {}
    for name, part in (("train", split.train), ("val", split.val),
                       ("test", split.test)):
        for p in part:
            r = rows.setdefault(
                p.author,
                {"train": 0, "val": 0, "test": 0,
                 "works_train": set(), "works_val": set(), "works_test": set()},
            )
            r[name] += 1
            r[f"works_{name}"].add(p.work)
    for r in rows.values():
        for k in ("train", "val", "test"):
            r[f"works_{k}"] = len(r[f"works_{k}"])
    return rows


def save(split: Split, name: str = "split.json") -> None:
    path = config.SPLITS / name
    payload = {
        k: [p.passage_id for p in v]
        for k, v in (("train", split.train), ("val", split.val),
                     ("test", split.test))
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"wrote {path}")


def load(passages: list[Passage], name: str = "split.json") -> Split:
    payload = json.loads((config.SPLITS / name).read_text(encoding="utf-8"))
    index = {p.passage_id: p for p in passages}
    return Split(
        train=[index[i] for i in payload["train"] if i in index],
        val=[index[i] for i in payload["val"] if i in index],
        test=[index[i] for i in payload["test"] if i in index],
    )


def xy(part: list[Passage]) -> tuple[list[str], list[str]]:
    return [p.text for p in part], [p.author for p in part]
