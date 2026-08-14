"""Compatibility gate: prove that the reconstructed workflows/objective.py
produces the same LLHD as the retained history for three previously-simulated
6D coordinates, plus repeatability at one of them.

Writes:
  .local/six_d/current/raw/direct_validation/objective_compatibility_check.csv
"""
from __future__ import annotations
import csv, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_6d import ROOT, RAW, NAMES, make_objective

DV = RAW / "direct_validation"
OUT = DV / "objective_compatibility_check.csv"

SNAPSHOT = RAW / "continuation/final_training_snapshot.csv"


def _pick_reference_rows():
    """Return the three canonical historical points: BO_1 best, pre-BO_2 best, BO_2 best."""
    rows = list(csv.DictReader(SNAPSHOT.open()))
    L = np.array([float(r["native_LLHD"]) for r in rows])
    phase = np.array([r["phase"] for r in rows])
    picks = {}
    # BO_1 best
    bo1_idx = np.where(phase == "BO_1")[0]
    picks["BO_1_best"] = rows[int(bo1_idx[np.argmin(L[bo1_idx])])]
    # Pre-BO_2 best (min over INITIAL/BO_1/DIRECT/INTERACTION)
    pre_mask = np.isin(phase, ("INITIAL", "BO_1", "DIRECT", "INTERACTION"))
    pre_idx = np.where(pre_mask)[0]
    picks["pre_BO_2_best"] = rows[int(pre_idx[np.argmin(L[pre_idx])])]
    # BO_2 best
    bo2_idx = np.where(phase == "BO_2")[0]
    picks["BO_2_best"] = rows[int(bo2_idx[np.argmin(L[bo2_idx])])]
    return picks


def main():
    DV.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise FileExistsError(OUT)
    picks = _pick_reference_rows()
    obj, tgt_obj = make_objective()
    fields = ["label", "phase", "evaluation_index", "historical_LLHD",
              "regenerated_LLHD", "abs_diff", "rel_diff",
              "repeat_index", "elapsed_seconds", "timestamp"]
    with OUT.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for label, row in picks.items():
            x = np.array([float(row[n]) for n in NAMES], dtype=float)
            for rep in (1, 2) if label == "BO_2_best" else (1,):
                t0 = time.perf_counter()
                llhd = obj.evaluate(dict(zip(NAMES, [float(v) for v in x])), tgt_obj)
                el = time.perf_counter() - t0
                hist = float(row["native_LLHD"])
                out_row = {
                    "label": label,
                    "phase": row["phase"],
                    "evaluation_index": row["evaluation_index"],
                    "historical_LLHD": hist,
                    "regenerated_LLHD": float(llhd),
                    "abs_diff": float(llhd) - hist,
                    "rel_diff": (float(llhd) - hist) / hist if hist != 0 else float("nan"),
                    "repeat_index": rep,
                    "elapsed_seconds": el,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                w.writerow(out_row); f.flush(); os.fsync(f.fileno())
                print(json.dumps(out_row), flush=True)
    print(json.dumps({"status": "complete", "csv": str(OUT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
