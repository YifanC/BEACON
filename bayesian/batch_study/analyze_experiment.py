"""Validate and summarize one isolated six-dimensional optimizer run."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import spearmanr


NAMES = ("Ab", "kb", "eField", "lifetime", "tran_diff", "long_diff")
LO = np.array([0.75, 0.03, 0.49, 400.0, 3.0e-6, 1.0e-6])
HI = np.array([0.90, 0.08, 0.51, 6000.0, 15.0e-6, 10.0e-6])


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_training(root: Path) -> list[dict[str, str]]:
    raw = root / "raw"
    main = rows(raw / "traces/bo_6d_history.csv")
    tr = rows(raw / "continuation/bo_tr_history.csv")
    tr2 = rows(raw / "continuation/bo_tr2_history.csv")
    for row in main:
        row["phase"] = {"initial": "INITIAL", "BO": "BO_1"}[row["phase"]]
    combined = main + tr + tr2
    counts = Counter(row["phase"] for row in combined)
    expected = {"INITIAL": 72, "BO_1": 100, "BO_TR": 100, "BO_TR2": 60}
    if counts != expected or len(combined) != 332:
        raise RuntimeError(f"training lineage mismatch: {dict(counts)}")
    return combined


def cross_validation(x: np.ndarray, loss: np.ndarray, workflow: Path) -> dict:
    sys.path.insert(0, str(workflow))
    from build_6d import fit_gp, unit
    rng = np.random.default_rng(20260813)
    shuffled = np.arange(len(loss))
    rng.shuffle(shuffled)
    predicted = np.full(len(loss), np.nan)
    predicted_sd = np.full(len(loss), np.nan)
    for fold in np.array_split(shuffled, 5):
        train = np.setdiff1d(shuffled, fold)
        model = fit_gp(x[train], loss[train], nu=1.5)
        with torch.no_grad():
            posterior = model.posterior(torch.tensor(unit(x[fold]), dtype=torch.double))
            predicted[fold] = posterior.mean.squeeze(-1).cpu().numpy()
            predicted_sd[fold] = posterior.variance.sqrt().squeeze(-1).cpu().numpy()
    actual = -np.log(loss)
    residual = predicted - actual
    standardized = residual / predicted_sd
    return {
        "actual": actual,
        "predicted": predicted,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "baseline_rmse": float(np.sqrt(np.mean((actual - actual.mean()) ** 2))),
        "spearman": float(spearmanr(actual, predicted)[0]),
        "coverage_95": float(np.mean(np.abs(standardized) <= 1.96)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
    run_root = experiment / "history"
    result_dir = experiment / "results"
    result_dir.mkdir(exist_ok=True)
    training = load_training(run_root)
    x = np.array([[float(row[name]) for name in NAMES] for row in training])
    loss = np.array([float(row["native_LLHD"]) for row in training])
    if not np.isfinite(x).all() or not np.isfinite(loss).all() or np.any(loss <= 0):
        raise RuntimeError("non-finite or non-positive training data")
    if np.any(x < LO) or np.any(x > HI):
        raise RuntimeError("training coordinate outside configured bounds")
    unique_count = len(np.unique(x, axis=0))
    duplicate_count = len(x) - unique_count
    best_index = int(np.argmin(loss))
    best = x[best_index]
    elapsed = sum(float(row["simulator_elapsed_seconds"]) for row in training)

    cv = cross_validation(x, loss, experiment.parents[1] / "workflows/six_d")

    config = json.loads((run_root / "raw/config/run_config.json").read_text())
    summary = {
        "experiment": experiment.name,
        "completed": True,
        "n_evaluations": len(training),
        "phase_counts": dict(Counter(row["phase"] for row in training)),
        "best_index": best_index + 1,
        "best_native_LLHD": float(loss[best_index]),
        "best_point": dict(zip(NAMES, map(float, best))),
        "physical_length_cm": config["physical_dataset"]["physical_track_length_cm"],
        "requested_length_cm": float(config["physical_dataset"]["requested_track_length_cm"]),
        "simulator_seed": config["physical_dataset"]["candidate_simulator_seed"],
        "data_seed": config["physical_dataset"]["data_seed"],
        "target_npz": config["physical_dataset"]["target_npz"],
        "simulator_elapsed_seconds_sum": elapsed,
        "unique_coordinates": unique_count,
        "duplicate_coordinates": duplicate_count,
        "gp_cv": {key: value for key, value in cv.items()
                  if key not in ("actual", "predicted")},
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (result_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        for key in ("completed", "n_evaluations", "best_native_LLHD",
                    "physical_length_cm", "requested_length_cm", "simulator_seed",
                    "data_seed", "simulator_elapsed_seconds_sum",
                    "unique_coordinates", "duplicate_coordinates"):
            writer.writerow([key, summary[key]])
        for name, value in summary["best_point"].items():
            writer.writerow([name, value])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
