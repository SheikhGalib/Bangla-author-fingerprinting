"""How stable is the BiLSTM result, really?

Two runs of "the same" BiLSTM on the same split disagreed by ten accuracy
points: 0.844 from the training job, 0.940 from the presentation notebook.
That is too large to report either number on its own, and the report leans on
this model to argue that adding word order did not help.  So before the claim
stands, the spread has to be measured rather than assumed.

Two things differed between those runs, and this job separates them:

* the **random seed**, and
* the **embedding initialisation scale** -- ``neural.py`` fills the table from
  N(0, 0.1) while PyTorch's ``nn.Embedding`` default is N(0, 1), ten times
  wider.

So: five seeds at each of the two scales, everything else held fixed, on the
identical split the rest of the project uses.  Ten runs at roughly two minutes
each.  The output is a mean and a range per scale, which is what the report
should quote.

Nothing here touches the held-out books beyond scoring on them.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("/kaggle/working")
SEEDS = (20242025, 7, 1234, 99991, 314159)
SCALES = (0.1, 1.0)
EPOCHS = 25


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_input() -> Path:
    root = Path("/kaggle/input")
    for c in sorted(root.rglob("passages.jsonl")):
        return c.parent
    raise SystemExit(f"dataset not found; mounted: {[str(p) for p in root.glob('*')]}")


IN = find_input()
sys.path.insert(0, str(IN / "src"))


def load_split():
    rows = {}
    with (IN / "passages.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                p = json.loads(line)
                rows[p["passage_id"]] = p
    assign = json.loads((IN / "split.json").read_text(encoding="utf-8"))
    out = {}
    for name in ("train", "val", "test"):
        part = [rows[i] for i in assign[name] if i in rows]
        out[name] = ([r["text"] for r in part], [r["author"] for r in part])
    return out


def main() -> int:
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    from banglastylo import neural

    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    data = load_split()
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    xte, yte = data["test"]
    log(f"train={len(xtr)} val={len(xva)} test={len(xte)}")

    results = []
    for scale in SCALES:
        # Patch the initialisation scale for this block of runs.  The function
        # is module-level, so swapping it is the least invasive way to vary the
        # one thing under test without forking the class.
        original = neural._embedding_matrix

        def make_matrix(vocab, dim, w2v=None, seed=0, _scale=scale):
            rng = np.random.default_rng(seed)
            mat = rng.normal(0.0, _scale, size=(len(vocab), dim)).astype(np.float32)
            mat[neural.PAD] = 0.0
            return mat

        neural._embedding_matrix = make_matrix
        try:
            for seed in SEEDS:
                t0 = time.time()
                clf = neural.BiLSTMClassifier(epochs=EPOCHS, lr=1.5e-3, seed=seed)
                clf.fit(xtr, ytr, val=(xva, yva), verbose=False)
                pred = list(clf.predict(xte))
                acc = float(accuracy_score(yte, pred))
                f1 = float(f1_score(yte, pred, average="macro"))
                best_val = max((h.get("val_acc", 0) for h in clf.history_), default=0)
                results.append({"init_scale": scale, "seed": seed,
                                "test_accuracy": acc, "test_macro_f1": f1,
                                "best_val_accuracy": best_val})
                log(f"  scale={scale}  seed={seed:<9} test={acc:.4f}  "
                    f"val={best_val:.4f}  ({time.time() - t0:.0f}s)")
        finally:
            neural._embedding_matrix = original

    (OUT / "bilstm_seeds.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    log("=" * 64)
    summary = {}
    for scale in SCALES:
        accs = [r["test_accuracy"] for r in results if r["init_scale"] == scale]
        summary[str(scale)] = {
            "mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "min": float(np.min(accs)), "max": float(np.max(accs)),
            "n": len(accs),
        }
        log(f"init N(0,{scale}):  mean {np.mean(accs):.4f}  sd {np.std(accs):.4f}  "
            f"range {np.min(accs):.4f}-{np.max(accs):.4f}")
    allacc = [r["test_accuracy"] for r in results]
    summary["overall"] = {"mean": float(np.mean(allacc)),
                          "min": float(np.min(allacc)),
                          "max": float(np.max(allacc)), "n": len(allacc)}
    log(f"overall:        mean {np.mean(allacc):.4f}  "
        f"range {np.min(allacc):.4f}-{np.max(allacc):.4f}  ({len(allacc)} runs)")

    (OUT / "bilstm_seeds_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
