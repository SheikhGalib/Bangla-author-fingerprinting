"""GPU training job, executed as a Kaggle kernel.

Only the three models that actually need a GPU run here: the fine-tuned
BanglaBERT, the BiLSTM and the from-scratch Transformer encoder.  Everything
else in the project -- the Kneser-Ney language models, Naive Bayes, softmax
regression, the SVM over stylometric features -- trains in seconds on a CPU
and stays in the local notebooks, where it is easier to inspect.

The split is *not* recomputed here.  It arrives in the attached dataset exactly
as ``scripts/02_make_splits.py`` wrote it, so the numbers this job reports are
comparable with the locally-trained models line for line.  Recomputing it would
reintroduce the one thing the project is careful about.

Outputs land in ``/kaggle/working``: one metrics JSON, per-model predictions on
the test set, and the fine-tuned checkpoint.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("/kaggle/working")
SEED = 20242025


def find_input() -> Path:
    """Locate the attached dataset instead of assuming its mount point.

    Kaggle mounts a dataset at ``/kaggle/input/<slug>``, but the slug can differ
    from what was pushed, and a version still being processed when the kernel
    starts mounts as nothing at all.  Both failures look identical from inside
    the job -- a bare ``FileNotFoundError`` -- so the marker file is searched
    for and every candidate directory is listed when it is missing.
    """
    root = Path("/kaggle/input")
    # Recursive: the mount point is not reliably the dataset slug.  This one
    # actually landed under /kaggle/input/datasets/<owner>/<slug>/, which a
    # single-level glob missed.
    for candidate in sorted(root.rglob("passages.jsonl")):
        return candidate.parent
    listing = [str(p) for p in sorted(root.rglob("*"))][:40] if root.exists() else []
    raise SystemExit(
        f"passages.jsonl not found under {root}; mounted inputs: {listing}"
    )


IN = find_input()
sys.path.insert(0, str(IN / "src"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_split():
    passages = {}
    with (IN / "passages.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                p = json.loads(line)
                passages[p["passage_id"]] = p
    assign = json.loads((IN / "split.json").read_text(encoding="utf-8"))
    out = {}
    for name in ("train", "val", "test"):
        rows = [passages[i] for i in assign[name] if i in passages]
        out[name] = ([r["text"] for r in rows], [r["author"] for r in rows])
    return out


def accuracy(y_true, y_pred) -> float:
    return float(np.mean([a == b for a, b in zip(y_true, y_pred)]))


def macro_f1(y_true, y_pred, classes) -> float:
    fs = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(fs))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def run_bert(data, results):
    """Fine-tune BanglaBERT as a genuine multi-class classifier."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "csebuetnlp/banglabert"
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    xte, yte = data["test"]
    classes = sorted(set(ytr))
    label = {a: i for i, a in enumerate(classes)}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"BanglaBERT fine-tune on {dev}, {len(classes)} classes, {len(xtr)} train")

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(classes)).to(dev)

    # Class weights: the training side is intentionally unbalanced (Tagore has
    # the most material), so the loss is reweighted rather than the data cut.
    counts = np.array([ytr.count(c) for c in classes], dtype=np.float64)
    weights = torch.tensor(counts.sum() / (len(classes) * counts),
                           dtype=torch.float32, device=dev)
    log(f"  class weights: {dict(zip(classes, weights.tolist()))}")

    max_len, epochs, batch = 256, 4, 16
    enc = tok(xtr, padding="max_length", truncation=True, max_length=max_len,
              return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                       torch.tensor([label[a] for a in ytr]))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-5, total_steps=epochs * len(dl), pct_start=0.1)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def predict(texts):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                e = tok(texts[i : i + 64], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(dev)
                out.append(model(**e).logits.cpu().numpy())
        z = np.vstack(out)
        return [classes[i] for i in z.argmax(1)], z

    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for ids, mask, yy in dl:
            ids, mask, yy = ids.to(dev), mask.to(dev), yy.to(dev)
            loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, yy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            tot += float(loss.detach()) * yy.numel()
        va, _ = predict(xva)
        acc = accuracy(yva, va)
        log(f"  epoch {ep + 1}/{epochs} loss={tot / len(ytr):.4f} val_acc={acc:.4f}")
        if acc > best:
            best = acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    pred, logits = predict(xte)
    results["banglabert_finetuned"] = {
        "family": "discriminative / pretrained transformer",
        "test_accuracy": accuracy(yte, pred),
        "test_macro_f1": macro_f1(yte, pred, classes),
        "val_accuracy": best,
        "classes": classes,
        "predictions": pred,
        "epochs": epochs, "max_len": max_len, "device": dev,
    }
    log(f"  BanglaBERT test acc = {results['banglabert_finetuned']['test_accuracy']:.4f}")

    d = OUT / "banglabert_finetuned"
    d.mkdir(exist_ok=True)
    model.save_pretrained(d)
    tok.save_pretrained(d)
    (d / "classes.txt").write_text("\n".join(classes), encoding="utf-8")


def run_scratch(data, results):
    """BiLSTM and from-scratch Transformer, using the project's own module."""
    from banglastylo import neural

    xtr, ytr = data["train"]
    xva, yva = data["val"]
    xte, yte = data["test"]
    classes = sorted(set(ytr))

    for key, cls, kw, family in (
        ("bilstm", neural.BiLSTMClassifier, {"epochs": 25, "lr": 1.5e-3},
         "discriminative / recurrent"),
        ("transformer_scratch", neural.TransformerClassifier,
         {"epochs": 30, "lr": 8e-4}, "discriminative / transformer from scratch"),
    ):
        log(f"{key}: training")
        clf = cls(seed=SEED, **kw)
        clf.fit(xtr, ytr, val=(xva, yva), verbose=True)
        pred = list(clf.predict(xte))
        results[key] = {
            "family": family,
            "test_accuracy": accuracy(yte, pred),
            "test_macro_f1": macro_f1(yte, pred, classes),
            "val_accuracy": max((h.get("val_acc", 0) for h in clf.history_),
                                default=None),
            "classes": classes,
            "predictions": pred,
            "history": clf.history_,
            **kw,
        }
        log(f"  {key} test acc = {results[key]['test_accuracy']:.4f}")
        clf.save(OUT / key)


def main() -> int:
    log(f"python {sys.version.split()[0]}")
    try:
        import torch
        log(f"torch {torch.__version__}  cuda={torch.cuda.is_available()} "
            f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    except Exception as exc:
        log(f"torch unavailable: {exc}")
        return 1

    data = load_split()
    for k, (x, y) in data.items():
        log(f"{k}: {len(x)} passages, {sorted(set(y))}")
    (OUT / "test_labels.json").write_text(
        json.dumps(data["test"][1], ensure_ascii=False), encoding="utf-8")

    results: dict = {}
    for fn in (run_bert, run_scratch):
        try:
            fn(data, results)
        except Exception as exc:
            import traceback
            log(f"FAILED {fn.__name__}: {exc}")
            traceback.print_exc()
            results[fn.__name__ + "_error"] = str(exc)
        (OUT / "gpu_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    log("=== summary ===")
    for k, v in results.items():
        if isinstance(v, dict) and "test_accuracy" in v:
            log(f"  {k:26s} acc={v['test_accuracy']:.4f}  f1={v['test_macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
