"""Stage 4: merge the local and Kaggle-GPU results into one table.

    .venv/Scripts/python.exe scripts/04_collect_results.py

The two halves of the experiment run in different places -- classical models on
this machine, the three GPU-bound models as a Kaggle kernel -- but they are one
experiment, because both read the same ``split.json``.  This stage joins them
and is the only thing the report reads.

It also recomputes accuracy for the GPU models from their stored predictions
rather than trusting the accuracy the remote job printed.  That is cheap, and
it catches the failure that matters most here: a remote job that silently
trained against a different split would otherwise produce a plausible number
nobody could distinguish from a real one.

Writes ``artifacts/tables/results_all.csv``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from banglastylo import config, corpus, splits

KAGGLE_DIR = config.ARTIFACTS / "kaggle"

#: Display names, so the merged table does not mix code identifiers with prose.
PRETTY = {
    "banglabert_finetuned": "BanglaBERT (fine-tuned)",
    "bilstm": "BiLSTM (from scratch)",
    "transformer_scratch": "Transformer (from scratch)",
}


def main() -> int:
    local_path = config.TABLES / "results_local.csv"
    if not local_path.exists():
        print("run scripts/03_train_local.py first", file=sys.stderr)
        return 1
    rows = pd.read_csv(local_path).to_dict("records")

    gpu_path = KAGGLE_DIR / "gpu_results.json"
    if not gpu_path.exists():
        print(f"no GPU results at {gpu_path} — "
              "run scripts/kaggle_run.py pull once the kernel finishes")
    else:
        passages = corpus.load_passages()
        split = splits.load(passages)
        _, y_true = splits.xy(split.test)

        gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
        for key, res in gpu.items():
            if not isinstance(res, dict) or "predictions" not in res:
                continue
            pred = res["predictions"]
            if len(pred) != len(y_true):
                print(f"  SKIP {key}: {len(pred)} predictions but "
                      f"{len(y_true)} test passages — split mismatch",
                      file=sys.stderr)
                continue
            acc = accuracy_score(y_true, pred)
            f1 = f1_score(y_true, pred, average="macro")
            drift = abs(acc - res.get("test_accuracy", acc))
            if drift > 1e-6:
                print(f"  NOTE {key}: recomputed acc {acc:.4f} differs from "
                      f"reported {res['test_accuracy']:.4f}")
            rows.append({
                "model": PRETTY.get(key, key),
                "family": res.get("family", "neural"),
                "accuracy": acc,
                "macro_f1": f1,
                "notes": f"Kaggle T4 GPU; val_acc={res.get('val_accuracy')}",
            })

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    df.to_csv(config.TABLES / "results_all.csv", index=False)

    print(f"\n{'=' * 82}")
    print("all models — test set is three held-out books, one per author")
    print("=" * 82)
    print(df.to_string(index=False,
                       formatters={"accuracy": "{:.4f}".format,
                                   "macro_f1": "{:.4f}".format}))
    print(f"\nchance = {1 / len(config.AUTHORS):.4f}")
    print(f"wrote {config.TABLES / 'results_all.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
