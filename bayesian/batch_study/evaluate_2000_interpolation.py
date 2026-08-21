"""Evaluate seven held-out points from 2000 cm repeat-1 best to nominal.

This is a diagnostic validation only. It reads frozen optimizer provenance and
writes one CSV under batch_study/2000cm/provenance; it never touches training
histories or checkpoints.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

BAYESIAN = Path("/sdf/home/i/iatif/REAL_BEACON/bayesian")
UPSTREAM = Path("/sdf/home/i/iatif/larnd-sim-jax")
SUMMARY = BAYESIAN / "batch_study/comparison/optimizer_repeats.csv"
TARGET = BAYESIAN / "batch_study/2000cm/target_2000cm.npz"
OUTPUT = BAYESIAN / "batch_study/2000cm/provenance/nominal_best_interpolation.csv"
EXPECTED_TARGET_SHA = "c2b7c0a7c2c370efe6af34c02acc248a9ce04bb44d4176f09b1752ecb4d8503d"
NAMES = ("Ab", "kb", "eField", "lifetime", "tran_diff", "long_diff")
NOMINAL = np.array([0.8, 0.0486, 0.5, 2200.0, 8.8e-6, 4.0e-6])

sys.path[:0] = [
    str(BAYESIAN / "workflows"),
    str(BAYESIAN / "workflows/six_d"),
    str(UPSTREAM),
    str(UPSTREAM / "src"),
]

os.environ.update({
    "BAYESIAN_PHYSICAL_LENGTH_CM": "2000",
    "BAYESIAN_N_EVENTS": "-1",
    "BAYESIAN_MAX_NBATCH": "1",
    "BAYESIAN_SIMULATOR_SEED": "0",
    "BAYESIAN_DATA_SEED": "0",
    "BAYESIAN_TARGET_NPZ": str(TARGET),
})

from build_6d import eval_point, make_objective  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic: {OUTPUT}")
    actual_sha = sha256(TARGET)
    if actual_sha != EXPECTED_TARGET_SHA:
        raise RuntimeError(f"target SHA mismatch: {actual_sha}")

    with SUMMARY.open(newline="") as stream:
        matches = [row for row in csv.DictReader(stream)
                   if row["data_size_label"] == "2000 cm" and row["repeat"] == "1"]
    if len(matches) != 1:
        raise RuntimeError(f"expected one authoritative repeat-1 row, got {len(matches)}")
    source = matches[0]
    best = np.array([float(source[name]) for name in NAMES])

    obj, target = make_objective()
    if len(obj.dataset) != 1 or len(target) != 1:
        raise RuntimeError("interpolation requires exactly one logical batch")

    fields = ["fraction_best_to_nominal", *NAMES, "native_LLHD", "score_log",
              "elapsed_seconds", "target_sha256", "slurm_job_id"]
    rows = []
    for step in range(1, 8):
        fraction = step / 8.0
        point = best + fraction * (NOMINAL - best)
        started = time.perf_counter()
        value = float(eval_point(obj, target, point))
        elapsed = time.perf_counter() - started
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite LLHD at interpolation fraction {fraction}")
        row = {
            "fraction_best_to_nominal": fraction,
            **{name: float(point[index]) for index, name in enumerate(NAMES)},
            "native_LLHD": value,
            "score_log": -math.log(value),
            "elapsed_seconds": elapsed,
            "target_sha256": actual_sha,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    with OUTPUT.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "complete", "output": str(OUTPUT),
                      "n_new_validation_observations": len(rows),
                      "entered_training": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
