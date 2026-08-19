"""Build clean phase snapshots from one isolated optimizer history."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


NAMES = ("Ab", "kb", "eField", "lifetime", "tran_diff", "long_diff")
FIELDS = ("phase", *NAMES, "native_LLHD")


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in FIELDS} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("bo1", "botr"), required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    raw = root / "raw"
    continuation = raw / "continuation"
    continuation.mkdir(parents=True, exist_ok=True)

    main_rows = list(csv.DictReader((raw / "traces/bo_6d_history.csv").open()))
    if len(main_rows) != 172:
        raise RuntimeError(f"expected 172 initial/global rows, got {len(main_rows)}")
    clean = []
    for row in main_rows:
        phase = {"initial": "INITIAL", "BO": "BO_1"}.get(row["phase"])
        if phase is None:
            raise RuntimeError(f"unexpected main-history phase {row['phase']!r}")
        clean.append({**row, "phase": phase})

    if args.stage == "bo1":
        write_rows(continuation / "clean_172_training.csv", clean)
        return

    tr_rows = list(csv.DictReader((continuation / "bo_tr_history.csv").open()))
    if len(tr_rows) != 100 or any(row["phase"] != "BO_TR" for row in tr_rows):
        raise RuntimeError(f"expected 100 BO_TR rows, got {len(tr_rows)}")
    write_rows(continuation / "clean_bo_tr_training.csv", clean + tr_rows)


if __name__ == "__main__":
    main()
