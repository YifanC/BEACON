"""Diagnostic simulator calls for the BO_TR postmortem.

7 BRIDGE + 12 SHELL simulator evaluations, single job. Reads BO_TR best from
the frozen clean 272-row snapshot; reads nominal from constants. The
all-nominal endpoint is already cached in
`.local/six_d/current/raw/direct_validation/bo_tr_reference_points.csv`.

DIAG_BRIDGE / DIAG_SHELL rows go to a canonical file each; they must NEVER be
merged into the GP training set. This script does not touch continuation
history or checkpoints.

Writes:
  .local/six_d/current/raw/direct_validation/surrogate_bridge.csv
  .local/six_d/current/raw/direct_validation/surrogate_shell.csv
"""
from __future__ import annotations
import csv, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.quasirandom import SobolEngine

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_6d import RAW, NAMES, LO, HI, NOM, unit, physical, fit_gp
from objective import LLHDObjective, LossSettings, SimSettings, load_target

BASE = Path("/sdf/home/i/iatif/larnd-sim-jax/optimize/bayesian").resolve()
SNAPSHOT = RAW / "continuation/clean_bo_tr_training.csv"
DV = RAW / "direct_validation"
BRIDGE_CSV = DV / "surrogate_bridge.csv"
SHELL_CSV = DV / "surrogate_shell.csv"
REF_CSV = DV / "bo_tr_reference_points.csv"

TARGET = BASE / ".local/two_d/target.npz"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def _load_best_from_snapshot():
    rows = list(csv.DictReader(SNAPSHOT.open()))
    ll = [float(r["native_LLHD"]) for r in rows]
    bi = int(np.argmin(ll))
    return {n: float(rows[bi][n]) for n in NAMES}, float(ll[bi])


def _build_gp():
    rows = list(csv.DictReader(SNAPSHOT.open()))
    X = np.array([[float(r[n]) for n in NAMES] for r in rows], dtype=float)
    L = np.array([float(r["native_LLHD"]) for r in rows], dtype=float)
    torch.set_default_dtype(torch.double)
    return fit_gp(X, L)


def _gp_predict(model, X_phys):
    with torch.no_grad():
        post = model.posterior(torch.tensor(unit(X_phys), dtype=torch.double))
        m = post.mean.squeeze(-1).cpu().numpy()
        sd = post.variance.sqrt().squeeze(-1).cpu().numpy()
    return m, sd


def _build_objective():
    input_hdf5 = ("/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/"
                  "production_884072/job_23771825_0000/"
                  "output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_"
                  "costheta0.966_range_0.05cm.h5")
    sim = SimSettings(input_file=input_hdf5, n_events=176, seed=0,
                       sim_seed_strategy="same")
    obj = LLHDObjective(sim=sim, loss=LossSettings(sigma_charge=500.0, eps=1e-10),
                        larndsim_repo="/sdf/home/i/iatif/larnd-sim-jax",
                        tunable_params=tuple(NAMES))
    tgt = load_target(str(TARGET))
    return obj, tgt


def _load_all_nominal():
    """Return the cached deterministic all-nominal LLHD from the reference file."""
    rows = list(csv.DictReader(REF_CSV.open()))
    for r in rows:
        if r.get("label") == "all_nominal":
            return float(r["native_LLHD"])
    return None


def _no_overwrite(p: Path):
    if p.exists():
        raise FileExistsError(p)
    return p


def main():
    DV.mkdir(parents=True, exist_ok=True)
    _no_overwrite(BRIDGE_CSV); _no_overwrite(SHELL_CSV)

    BEST, BEST_LLHD = _load_best_from_snapshot()
    print(f"[diag] BO_TR best LLHD = {BEST_LLHD}  point={BEST}", flush=True)
    NOM_pt = np.array(NOM, dtype=float)
    BEST_pt = np.array([BEST[n] for n in NAMES], dtype=float)
    best_u = unit(BEST_pt)
    nom_u = unit(NOM_pt)
    normalized_radius = float(np.linalg.norm(nom_u - best_u))
    print(f"[diag] normalized BO_TR-best-to-nominal distance = {normalized_radius:.4f}", flush=True)

    model = _build_gp()
    obj, tgt = _build_objective()

    all_nom_llhd = _load_all_nominal()
    if all_nom_llhd is None:
        raise RuntimeError("could not read cached all-nominal LLHD from reference file")

    # ---------------- BRIDGE ----------------
    ts = [0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000]
    bridge_fields = ["phase", "t", *NAMES, *[f"normalized_{n}" for n in NAMES],
                     "actual_native_LLHD", "actual_score",
                     "gp_score_mean", "gp_score_sd",
                     "gp_llhd_like", "score_residual", "native_like_residual",
                     "simulator_elapsed_seconds", "was_cached", "timestamp",
                     "slurm_job_id"]

    with BRIDGE_CSV.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bridge_fields); w.writeheader()
        for t in ts:
            x = BEST_pt + t * (NOM_pt - BEST_pt)
            was_cached = False; elapsed = 0.0
            if abs(t) < 1e-12:
                llhd = BEST_LLHD; was_cached = True
            elif abs(t - 1.0) < 1e-12:
                llhd = all_nom_llhd; was_cached = True
            else:
                t0 = time.perf_counter()
                llhd = obj.evaluate(dict(zip(NAMES, [float(v) for v in x])), tgt)
                elapsed = time.perf_counter() - t0
            score = -float(np.log(llhd))
            m, sd = _gp_predict(model, x.reshape(1, -1))
            gp_llhd = float(np.exp(-m[0]))
            row = {"phase": "DIAG_BRIDGE", "t": float(t),
                   **{n: float(x[i]) for i, n in enumerate(NAMES)},
                   **{f"normalized_{n}": float(unit(x)[i]) for i, n in enumerate(NAMES)},
                   "actual_native_LLHD": float(llhd),
                   "actual_score": score,
                   "gp_score_mean": float(m[0]),
                   "gp_score_sd": float(sd[0]),
                   "gp_llhd_like": gp_llhd,
                   "score_residual": float(m[0] - score),
                   "native_like_residual": float(gp_llhd - llhd),
                   "simulator_elapsed_seconds": elapsed,
                   "was_cached": was_cached,
                   "timestamp": datetime.now(timezone.utc).isoformat(),
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")}
            w.writerow(row); f.flush(); os.fsync(f.fileno())
            print(json.dumps({"BRIDGE": t, "llhd": float(llhd),
                              "gp": gp_llhd, "cached": was_cached}), flush=True)

    # ---------------- SHELL ----------------
    eng = SobolEngine(dimension=6, scramble=True, seed=20260826)
    n_want = 12
    U_raw = eng.draw(64).cpu().numpy() * 2.0 - 1.0    # in [-1, 1]^6 before normalization
    # Normalize each row to unit length then scale to `normalized_radius`
    picked_u = []
    seen_train = set()
    for r in csv.DictReader(SNAPSHOT.open()):
        seen_train.add(tuple(float(r[n]) for n in NAMES))
    seen_train.add(tuple(BEST_pt.tolist()))
    seen_train.add(tuple(NOM_pt.tolist()))
    for row in U_raw:
        v = row / np.linalg.norm(row)
        cand_u = np.clip(best_u + v * normalized_radius, 0.0, 1.0)
        cand_phys = physical(cand_u)
        key = tuple(float(x) for x in cand_phys)
        if key in seen_train:
            continue
        # Also drop points that landed on any global bound after clip (they alter the radius)
        if np.any(np.abs(cand_u - 0.0) < 1e-12) or np.any(np.abs(cand_u - 1.0) < 1e-12):
            continue
        picked_u.append(cand_u)
        seen_train.add(key)
        if len(picked_u) >= n_want:
            break
    if len(picked_u) < n_want:
        raise RuntimeError(f"shell design underfilled: got {len(picked_u)}/{n_want}")

    shell_fields = ["phase", "index", *NAMES, *[f"normalized_{n}" for n in NAMES],
                    "distance_from_best_unit",
                    "actual_native_LLHD", "actual_score",
                    "gp_score_mean", "gp_score_sd",
                    "gp_llhd_like", "score_residual", "native_like_residual",
                    "simulator_elapsed_seconds", "timestamp",
                    "slurm_job_id"]
    llhds = []
    with SHELL_CSV.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=shell_fields); w.writeheader()
        for i, cu in enumerate(picked_u, start=1):
            cphys = physical(cu)
            t0 = time.perf_counter()
            llhd = obj.evaluate(dict(zip(NAMES, [float(v) for v in cphys])), tgt)
            elapsed = time.perf_counter() - t0
            m, sd = _gp_predict(model, cphys.reshape(1, -1))
            gp_llhd = float(np.exp(-m[0]))
            score = -float(np.log(llhd))
            d_from_best = float(np.linalg.norm(cu - best_u))
            row = {"phase": "DIAG_SHELL", "index": i,
                   **{n: float(cphys[j]) for j, n in enumerate(NAMES)},
                   **{f"normalized_{n}": float(cu[j]) for j, n in enumerate(NAMES)},
                   "distance_from_best_unit": d_from_best,
                   "actual_native_LLHD": float(llhd),
                   "actual_score": score,
                   "gp_score_mean": float(m[0]),
                   "gp_score_sd": float(sd[0]),
                   "gp_llhd_like": gp_llhd,
                   "score_residual": float(m[0] - score),
                   "native_like_residual": float(gp_llhd - llhd),
                   "simulator_elapsed_seconds": elapsed,
                   "timestamp": datetime.now(timezone.utc).isoformat(),
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")}
            w.writerow(row); f.flush(); os.fsync(f.fileno())
            llhds.append(float(llhd))
            print(json.dumps({"SHELL": i, "d": d_from_best, "llhd": float(llhd),
                              "gp": gp_llhd}), flush=True)
    print(json.dumps({"status": "complete",
                       "n_bridge": len(ts),
                       "n_shell": len(picked_u),
                       "bridge_csv": str(BRIDGE_CSV),
                       "shell_csv": str(SHELL_CSV)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
