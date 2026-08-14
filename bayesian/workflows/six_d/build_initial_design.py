"""Generate + simulate 72 Sobol 6D initial-design points.

72 = 12 × 6 ≈ 12 per dim, comparable to a 2D×4 density and enough to
train a stable Matern-5/2 ARD(6) GP before BO. Appends rows to the shared
history CSV with the same schema as later BO iterations.

Writes:
  raw/initial_design/initial_design_6d.csv
  raw/initial_design/initial_design_6d.npz
  raw/traces/bo_6d_history.csv   (88 rows written when done — 88=72 is our N here)

Wait — this run uses N=72 (not 88). We document that explicitly in the config.
"""
from __future__ import annotations
import csv, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from torch.quasirandom import SobolEngine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_6d import (
    ROOT, RAW, NAMES, LO, HI, NOM, unit, physical, guard,
    make_objective, eval_point,
)

N_INITIAL = 72
SOBOL_SEED = 20260812
BO_ITERS = 100

HIST = RAW / "traces/bo_6d_history.csv"
NPZ = RAW / "initial_design/initial_design_6d.npz"
CSV_INIT = RAW / "initial_design/initial_design_6d.csv"
CFG = RAW / "config/run_config.json"

FIELDS = [
    "run", "evaluation_index", "simulator_iteration", "phase",
    *NAMES,
    *[f"normalized_{n}" for n in NAMES],
    "native_LLHD", "score_log", "Yvar_log",
    "acquisition_value",
    *[f"{n}_lengthscale" for n in NAMES],
    "outputscale", "simulator_elapsed_seconds",
    "timestamp", "simulator_status", "checkpoint_path",
]


def _append(row):
    new = not HIST.exists()
    with HIST.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _no_overwrite(p: Path):
    if p.exists():
        raise FileExistsError(f"immutable output already exists: {p}")
    return p


def main():
    guard(RAW)
    if HIST.exists():
        rows = list(csv.DictReader(HIST.open()))
        if len(rows) >= N_INITIAL:
            print(json.dumps({"status": "initial_design_present",
                               "rows": len(rows)}))
            return
        raise RuntimeError(f"partial history exists with {len(rows)} rows; "
                           "delete or resume manually")
    # Sobol 6D, scrambled, deterministic seed
    engine = SobolEngine(dimension=6, scramble=True, seed=SOBOL_SEED)
    U = engine.draw(N_INITIAL).cpu().numpy()  # shape (N, 6), values in [0,1)
    X = np.array([physical(u) for u in U], dtype=float)

    obj, tgt = make_objective()
    stamp0 = datetime.now(timezone.utc).isoformat()
    L = np.zeros(N_INITIAL, dtype=float)
    for i, (x, u) in enumerate(zip(X, U), start=1):
        t0 = time.perf_counter()
        loss = eval_point(obj, tgt, x)
        elapsed = time.perf_counter() - t0
        L[i - 1] = loss
        stamp = datetime.now(timezone.utc).isoformat()
        _append({
            "run": "6D-1000cm",
            "evaluation_index": i,
            "simulator_iteration": 0,
            "phase": "initial",
            **dict(zip(NAMES, x)),
            **{f"normalized_{n}": float(u[j]) for j, n in enumerate(NAMES)},
            "native_LLHD": float(loss),
            "score_log": -float(np.log(loss)),
            "Yvar_log": 1.0 / float(loss) ** 2,
            "acquisition_value": "NA",
            **{f"{n}_lengthscale": "NA" for n in NAMES},
            "outputscale": "NA",
            "simulator_elapsed_seconds": elapsed,
            "timestamp": stamp,
            "simulator_status": "success",
            "checkpoint_path": "",
        })
        print(json.dumps({"phase": "initial", "i": i, "native_LLHD": float(loss),
                          "elapsed_s": float(elapsed),
                          **dict(zip(NAMES, map(float, x)))}), flush=True)

    # Also drop a compact NPZ + a self-contained CSV under initial_design/
    with _no_overwrite(NPZ).open("wb") as f:
        np.savez_compressed(f, X=X, U=U, L=L, names=np.asarray(NAMES),
                             LO=LO, HI=HI, NOM=NOM,
                             sobol_seed=SOBOL_SEED, n_initial=N_INITIAL)
        f.flush(); os.fsync(f.fileno())
    with _no_overwrite(CSV_INIT).open("x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design_index", *NAMES, "native_LLHD"])
        for i, (x, l) in enumerate(zip(X, L), start=1):
            w.writerow([i, *[float(v) for v in x], float(l)])
        f.flush(); os.fsync(f.fileno())

    # Config snapshot
    with _no_overwrite(CFG).open("x") as f:
        json.dump({
            "run_label": "6D-1000cm",
            "created_at": stamp0,
            "physical_dataset": {
                "label": "999.93-cm safe dataset",
                "input_hdf5": ("/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/"
                               "production_884072/job_23771825_0000/"
                               "output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_"
                               "costheta0.966_range_0.05cm.h5"),
                "n_events": 176,
                "hdf5_rows": 100010,
                "physical_track_length_cm": 999.926641702652,
                "target_hits": 9368,
                "target_seed": 0,
                "target_npz": str((ROOT.parent /
                                    ".local/two_d/target.npz").resolve()),
            },
            "parameters": {n: {"lower": float(LO[i]), "upper": float(HI[i]),
                                "nominal": float(NOM[i]), "unit": "" if i not in (1,2,3,4,5)
                                else ["kV*g/(MeV*cm^3)", "kV/cm", "us", "cm^2/us", "cm^2/us"][i-1]}
                            for i, n in enumerate(NAMES)},
            "initial_design": {"scheme": "Sobol6D (scrambled)",
                                "seed": SOBOL_SEED, "n": N_INITIAL,
                                "reused_prior_observations": False,
                                "note": ("Prior 2D observations are NOT reused. "
                                          "Their tran_diff and long_diff coordinates were held "
                                          "fixed at 8.8e-6 / 4.0e-6 respectively, so they do not "
                                          "sample the two new 6D axes.")},
            "surrogate": {"score": "-ln(native_LLHD)",
                           "train_Yvar": "1/native_LLHD^2",
                           "kernel": "Matern-5/2 ARD(6)",
                           "outcome_transform": "Standardize(m=1)",
                           "acquisition": "qLogExpectedImprovement",
                           "q": 1,
                           "num_restarts": 24,
                           "raw_samples": 2048},
            "bo_iterations_planned": BO_ITERS,
            "expected_history_rows": N_INITIAL + BO_ITERS,
        }, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    print(json.dumps({"status": "initial_design_complete",
                       "n": N_INITIAL,
                       "best_initial_LLHD": float(L.min()),
                       "history_csv": str(HIST),
                       "npz": str(NPZ),
                       "config": str(CFG)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
