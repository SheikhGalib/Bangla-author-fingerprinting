"""Command-line interface: paste a Bangla passage, get an audited attribution.

    .venv/Scripts/python.exe app.py --demo
    .venv/Scripts/python.exe app.py --text "…বাংলা অনুচ্ছেদ…"
    .venv/Scripts/python.exe app.py --file passage.txt
    .venv/Scripts/python.exe app.py            # reads stdin until EOF

Models are loaded from ``artifacts/models``; run notebook 04 to create them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from banglastylo import config
from banglastylo.interface import (
    DEMO_PASSAGE,
    Attributor,
    render,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Authorship attribution and style fingerprinting for Bangla."
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--text", help="passage given inline")
    src.add_argument("--file", help="path to a UTF-8 file holding the passage")
    src.add_argument("--demo", action="store_true",
                     help="run on a built-in sample passage")
    ap.add_argument("--models", default=str(config.MODELS),
                    help="directory holding the trained models")
    ap.add_argument("--json", action="store_true",
                    help="emit the raw result as JSON instead of a report")
    ap.add_argument("--top-k", type=int, default=8,
                    help="how many contributing features to show")
    args = ap.parse_args()

    if args.demo:
        passage = DEMO_PASSAGE
    elif args.text:
        passage = args.text
    elif args.file:
        passage = Path(args.file).read_text(encoding="utf-8")
    else:
        print("Paste a Bangla passage, then press Ctrl-Z (Windows) / Ctrl-D "
              "and Enter:\n", file=sys.stderr)
        passage = sys.stdin.read()

    if not passage.strip():
        print("empty passage", file=sys.stderr)
        return 2

    try:
        attributor = Attributor.load(args.models)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = attributor.predict(passage, top_k=args.top_k)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render(result, attributor))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
