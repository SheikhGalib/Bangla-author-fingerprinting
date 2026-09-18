"""Execute the notebooks in order, in place, so their stored outputs are real.

    .venv/Scripts/python.exe scripts/run_notebooks.py            # all of them
    .venv/Scripts/python.exe scripts/run_notebooks.py 04 05      # just these

Each notebook is executed with the ``banglastylo`` kernel and its outputs are
written back into the file, so the committed notebooks show the numbers the
report quotes.  Execution stops at the first failure rather than carrying a
broken pipeline forward.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

ORDER = [
    "01_corpus_construction.ipynb",
    "02_splits_and_pos_tagger.ipynb",
    "03_embeddings.ipynb",
    "04_models_and_evaluation.ipynb",
    "05_interpretability.ipynb",
    "06_interface.ipynb",
]

# Fine-tuning BanglaBERT on CPU dominates notebook 04's runtime.
TIMEOUT = {"04_models_and_evaluation.ipynb": 14400}
DEFAULT_TIMEOUT = 5400


def run(name: str) -> bool:
    path = NB / name
    print(f"\n{'=' * 78}\nexecuting {name}\n{'=' * 78}", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [
            str(PY), "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.kernel_name=banglastylo",
            f"--ExecutePreprocessor.timeout={TIMEOUT.get(name, DEFAULT_TIMEOUT)}",
            str(path),
        ],
        cwd=str(NB),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,          # the return code is inspected below
    )
    dt = time.time() - t0
    tail = (proc.stderr or "").strip().splitlines()[-40:]
    print("\n".join(tail))
    print(f"\n{name}: {'OK' if proc.returncode == 0 else 'FAILED'} "
          f"in {dt / 60:.1f} min", flush=True)
    return proc.returncode == 0


if __name__ == "__main__":
    wanted = sys.argv[1:]
    todo = [n for n in ORDER if not wanted or any(w in n for w in wanted)]
    for name in todo:
        if not run(name):
            print(f"\nstopping: {name} failed", file=sys.stderr)
            raise SystemExit(1)
    print("\nall notebooks executed")
