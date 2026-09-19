"""Push the GPU training job to Kaggle, watch it, and pull the results back.

    .venv/Scripts/python.exe scripts/kaggle_run.py push     # upload + start
    .venv/Scripts/python.exe scripts/kaggle_run.py status   # poll
    .venv/Scripts/python.exe scripts/kaggle_run.py pull     # download output

Only the GPU-bound models run remotely (see ``kaggle/train_gpu.py``).  What
travels is the *already-split* corpus, never the raw books and never the
splitting logic, so the remote numbers line up with the local ones instead of
being a second experiment that happens to use the same data.

Both the dataset and the kernel are created **private**.  The corpus is
assembled for coursework and has no business being a public Kaggle dataset.

Credentials come from ``.env`` (``KAGGLE_API_TOKEN``) and are written to
``~/.kaggle/access_token``, which is where the Kaggle client looks.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

PY = ROOT / ".venv" / "Scripts" / "python.exe"
STAGE = ROOT / "data" / "kaggle_stage"
DATASET_DIR = STAGE / "dataset"
KERNEL_DIR = STAGE / "kernel"

USER = "galibmahim"
DATASET_SLUG = "bangla-authorship"
KERNEL_SLUG = "bangla-authorship-gpu"
DATASET_ID = f"{USER}/{DATASET_SLUG}"
KERNEL_ID = f"{USER}/{KERNEL_SLUG}"

#: The presentation notebook is a *separate* kernel from the training job.
#: The training job is a headless script whose output is a results file; this
#: one is a notebook whose output is the executed notebook itself, complete
#: with charts, and is the artefact shown to an examiner.
NOTEBOOK_SLUG = "bangla-authorship-notebook"
NOTEBOOK_ID = f"{USER}/{NOTEBOOK_SLUG}"
NOTEBOOK_DIR = STAGE / "notebook"

RESULTS = ROOT / "artifacts" / "kaggle"


def ensure_credentials() -> None:
    """Copy the token out of ``.env`` into the location the client reads."""
    home = Path.home() / ".kaggle"
    home.mkdir(exist_ok=True)
    target = home / "access_token"
    if target.exists() and target.read_text(encoding="utf-8").strip():
        return
    env = ROOT / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("KAGGLE_API_TOKEN="):
            target.write_text(line.split("=", 1)[1].strip(), encoding="utf-8")
            print(f"wrote {target}")
            return
    raise SystemExit("KAGGLE_API_TOKEN not found in .env")


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run([str(PY), "-m", "kaggle", *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          check=False)   # return code inspected below
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out.rstrip())
    if check and proc.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed")
    return proc


def stage() -> None:
    """Assemble the dataset payload and the kernel payload."""
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    (DATASET_DIR / "src" / "banglastylo").mkdir(parents=True)
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("passages.jsonl",):
        shutil.copy(ROOT / "data" / "processed" / name, DATASET_DIR / name)
    shutil.copy(ROOT / "data" / "splits" / "split.json", DATASET_DIR / "split.json")

    # Ship the library so the remote job imports the same code, not a copy.
    for py in (ROOT / "src" / "banglastylo").glob("*.py"):
        shutil.copy(py, DATASET_DIR / "src" / "banglastylo" / py.name)

    (DATASET_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "Bangla Authorship Passages",
        "id": DATASET_ID,
        "licenses": [{"name": "unknown"}],
    }, indent=2), encoding="utf-8")

    shutil.copy(ROOT / "kaggle" / "train_gpu.py", KERNEL_DIR / "train_gpu.py")
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_ID,
        "title": KERNEL_SLUG,
        "code_file": "train_gpu.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,          # needed to pull csebuetnlp/banglabert
        "dataset_sources": [DATASET_ID],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2), encoding="utf-8")

    n = sum(1 for _ in (DATASET_DIR / "passages.jsonl").open(encoding="utf-8"))
    print(f"staged {n} passages + {len(list((DATASET_DIR / 'src' / 'banglastylo').glob('*.py')))} modules")


def push() -> None:
    stage()
    existing = kaggle("datasets", "list", "-m", "-s", DATASET_SLUG, check=False)
    if DATASET_ID in (existing.stdout or ""):
        print(f"\n--- updating dataset {DATASET_ID} ---")
        kaggle("datasets", "version", "-p", str(DATASET_DIR),
               "-m", "refresh", "--dir-mode", "zip")
    else:
        print(f"\n--- creating dataset {DATASET_ID} ---")
        kaggle("datasets", "create", "-p", str(DATASET_DIR), "--dir-mode", "zip")

    # The kernel fails to attach a dataset whose first version is still being
    # processed, so wait for it to become queryable before pushing.
    for attempt in range(30):
        probe = kaggle("datasets", "status", DATASET_ID, check=False)
        if "ready" in (probe.stdout or "").lower():
            break
        print(f"  dataset not ready yet ({attempt + 1}/30)")
        time.sleep(10)

    print(f"\n--- pushing kernel {KERNEL_ID} ---")
    kaggle("kernels", "push", "-p", str(KERNEL_DIR))
    print(f"\nrunning: https://www.kaggle.com/code/{KERNEL_ID}")


def push_notebook() -> None:
    """Upload and run the presentation notebook against the same dataset."""
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    src = ROOT / "kaggle" / "bangla_authorship_complete.ipynb"
    if not src.exists():
        raise SystemExit("run scripts/make_kaggle_notebook.py first")
    shutil.copy(src, NOTEBOOK_DIR / src.name)
    (NOTEBOOK_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": NOTEBOOK_ID,
        "title": NOTEBOOK_SLUG,
        "code_file": src.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,      # needed to pull csebuetnlp/banglabert
        "dataset_sources": [DATASET_ID],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2), encoding="utf-8")

    print(f"--- pushing notebook {NOTEBOOK_ID} ---")
    kaggle("kernels", "push", "-p", str(NOTEBOOK_DIR))
    print(f"\nrunning: https://www.kaggle.com/code/{NOTEBOOK_ID}")


def pull_notebook() -> None:
    """Fetch the executed notebook, outputs and all."""
    dest = ROOT / "notebooks"
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "pull", NOTEBOOK_ID, "-p", str(dest), "-m")
    print(f"\npulled into {dest}")


def status(kernel_id: str = KERNEL_ID) -> str:
    proc = kaggle("kernels", "status", kernel_id, check=False)
    return (proc.stdout or "") + (proc.stderr or "")


def pull() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", KERNEL_ID, "-p", str(RESULTS))
    path = RESULTS / "gpu_results.json"
    if not path.exists():
        print("no gpu_results.json in the output — check the kernel log",
              file=sys.stderr)
        return
    res = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n{'model':<28}{'test acc':>10}{'macro F1':>10}")
    for k, v in res.items():
        if isinstance(v, dict) and "test_accuracy" in v:
            print(f"{k:<28}{v['test_accuracy']:>10.4f}{v['test_macro_f1']:>10.4f}")


COMMANDS = {
    "push": push,
    "status": lambda: print(status()),
    "pull": pull,
    "push-notebook": push_notebook,
    "status-notebook": lambda: print(status(NOTEBOOK_ID)),
    "pull-notebook": pull_notebook,
}

if __name__ == "__main__":
    ensure_credentials()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd not in COMMANDS:
        raise SystemExit(f"usage: kaggle_run.py [{' | '.join(COMMANDS)}]")
    COMMANDS[cmd]()
