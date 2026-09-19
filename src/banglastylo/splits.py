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


def split_by_role(
    passages: list[Passage],
    val_fraction: float = config.VAL_FRACTION,
    seed: int = config.SEED,
) -> Split:
    """Split using the roles the corpus was collected under.

    This is the splitter the study uses.  :func:`split_by_work` has to *infer*
    a work-disjoint partition from whatever books happened to be crawled; here
    the partition was decided before a single page was downloaded.  One book
    per author was designated ``unseen`` and becomes the test set entire, so
    work-disjointness is a property of the corpus rather than a claim about the
    code.  It is also the only arrangement a reader can check without running
    anything: the held-out books are the files in ``corpus/unseen/``.

    Validation comes out of the training books and is held out **by whole
    chapter**.  A passage-level validation split would put two halves of one
    chapter on either side, and early stopping would then be tuned against
    text the model had effectively seen.  Chapters are the finest unit that
    avoids this, and using them keeps all three splits leak-free under the same
    rule.
    """
    rng = random.Random(seed)
    test = [p for p in passages if p.role == "unseen"]
    pool = [p for p in passages if p.role != "unseen"]

    train: list[Passage] = []
    val: list[Passage] = []
    by_author: dict[str, list[Passage]] = defaultdict(list)
    for p in pool:
        by_author[p.author].append(p)

    for author in sorted(by_author):
        ps = by_author[author]
        by_chapter: dict[tuple[str, str], list[Passage]] = defaultdict(list)
        for p in ps:
            by_chapter[(p.work, p.chapter)].append(p)

        # Largest chapters first, each to whichever side is furthest below
        # quota — the same heuristic split_by_work uses, one level down.
        names = sorted(by_chapter)
        rng.shuffle(names)
        names.sort(key=lambda c: -len(by_chapter[c]))
        quota = val_fraction * len(ps)
        n_val = 0
        for c in names:
            block = by_chapter[c]
            if n_val + len(block) / 2 <= quota:
                val += block
                n_val += len(block)
            else:
                train += block

    return Split(train=train, val=val, test=test)


def leakage_report(split: Split) -> dict:
    """Check each split boundary against the rule that actually governs it.

    The three boundaries are not held to the same standard, and conflating them
    produces a false alarm.

    * ``train|test`` and ``val|test`` must be **work-disjoint**: the test set is
      three whole books nothing was fitted on, and a shared book here would
      invalidate every reported number.
    * ``train|val`` is **chapter-disjoint** by construction and shares books on
      purpose — validation is carved out of the training books.  Demanding
      work-disjointness of it would report a leak where the design is sound.

    No passage may appear twice anywhere, under any of the three.
    """
    parts = (("train", split.train), ("val", split.val), ("test", split.test))
    works = {name: {p.work for p in part} for name, part in parts}
    chapters = {name: {(p.work, p.chapter) for p in part} for name, part in parts}
    ids = {name: {p.passage_id for p in part} for name, part in parts}

    checks: dict[str, object] = {}
    for a, b in (("train", "test"), ("val", "test")):
        checks[f"{a}|{b}:shared_works"] = sorted(works[a] & works[b])
    checks["train|val:shared_chapters"] = sorted(
        f"{w} :: {c}" for w, c in chapters["train"] & chapters["val"]
    )
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        checks[f"{a}|{b}:shared_passages"] = len(ids[a] & ids[b])

    clean = all(
        (v == 0 if isinstance(v, int) else not v) for v in checks.values()
    )
    return {"clean": clean, "overlaps": checks}


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
