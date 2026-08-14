"""Final held-out 6D validation for the 6D BO continuation.

Reads the frozen training snapshot at
    .local/six_d/current/raw/continuation/final_training_snapshot.csv
and creates VALIDATION observations that were never used for GP fitting or BO
selection:

  - a global scrambled Sobol design of 12 points over the full 6D bounds;
  - a local scrambled Sobol design of 12 points around the final best (10%
    half-width per axis, clipped to bounds);
  - one all-nominal reference point;
  - three independent evaluations each of the exact all-nominal point and the
    exact final best point (fixed-seed repeatability check);
  - six conditional direct simulator profiles, one per parameter, at the final
    best coordinates, using the union of coarse normalized levels
    {0, 0.25, 0.5, 0.75, 1}, the nominal coordinate, the exact best coordinate,
    and fine local offsets {-0.05, -0.025, -0.01, 0, +0.01, +0.025, +0.05} of
    the full allowed range on the varied axis.

All output goes under `.local/six_d/current/raw/direct_validation/`. Never
modifies raw/traces/ or raw/continuation/.
"""
from __future__ import annotations
import csv, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.quasirandom import SobolEngine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_6d import ROOT, RAW, NAMES, LO, HI, NOM, unit, physical, guard, make_objective, eval_point

DV = RAW / "direct_validation"
DV_SIM = DV / "simulator_heldout_bo_tr2"
CONT = RAW / "continuation"
# Point validate.py at the CLEAN 332-row BO_TR2 snapshot for the new validation.
SNAPSHOT = CONT / "final_training_snapshot_bo_tr2.csv"

# BO_TR2 validation uses a fresh Sobol seed (20260827) and 32 held-out points.
VALIDATION_SEED = 20260827
GLOBAL_N = 16
LOCAL_N = 16
LOCAL_HALFWIDTH_FRACTION = 0.10
NOMINAL_REPEATS = 1  # nominal is deterministic; one call is enough
BEST_REPEATS = 1
# Output filename prefix for BO_TR2-era files
_PREFIX = "bo_tr2_"

# Fine local offsets, expressed as fractions of the full allowed range on that axis
FINE_OFFSETS = np.array([-0.05, -0.025, -0.01, 0.0, +0.01, +0.025, +0.05])
COARSE_NORM_LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _no_overwrite(p: Path):
    if p.exists():
        raise FileExistsError(f"immutable output already exists: {p}")
    return p


def _load_training():
    if not SNAPSHOT.exists():
        raise RuntimeError(f"final training snapshot missing: {SNAPSHOT}")
    rows = list(csv.DictReader(SNAPSHOT.open()))
    X = np.array([[float(r[n]) for n in NAMES] for r in rows], dtype=float)
    L = np.array([float(r["native_LLHD"]) for r in rows], dtype=float)
    phase = np.array([r["phase"] for r in rows])
    return rows, X, L, phase


def _cache_key(pt: dict):
    return tuple(float(pt[n]) for n in NAMES)


def main():
    DV.mkdir(parents=True, exist_ok=True); guard(DV)
    DV_SIM.mkdir(parents=True, exist_ok=True); guard(DV_SIM)

    rows, X, L, phase = _load_training()
    bi = int(np.argmin(L))
    BEST = {n: float(rows[bi][n]) for n in NAMES}
    BEST_LLHD = float(L[bi])
    print(f"[validate] final best row {bi+1}/{len(rows)}  phase={phase[bi]}  LLHD={BEST_LLHD:.6f}", flush=True)
    print(f"[validate] BEST point: {BEST}", flush=True)

    # Snapshot metadata + evaluation cache built from training observations
    snap_sha = _sha256(SNAPSHOT)
    cache: dict[tuple, float] = {}
    for i in range(len(X)):
        cache[tuple(float(v) for v in X[i])] = float(L[i])

    # Provenance JSON
    prov_path = _no_overwrite(DV / f"{_PREFIX}heldout_provenance.json")
    tgt = Path("/sdf/home/i/iatif/larnd-sim-jax/optimize/bayesian/.local/two_d/target.npz")
    prov = {
        "final_training_snapshot": str(SNAPSHOT),
        "final_training_snapshot_sha256": snap_sha,
        "n_training_rows": len(rows),
        "final_best": {"point": BEST, "native_LLHD": BEST_LLHD,
                        "row_index": bi + 1, "phase": str(phase[bi])},
        "target_npz": str(tgt),
        "target_sha256": _sha256(tgt) if tgt.exists() else None,
        "validation_seed": VALIDATION_SEED,
        "global_n": GLOBAL_N, "local_n": LOCAL_N,
        "local_halfwidth_fraction": LOCAL_HALFWIDTH_FRACTION,
        "nominal_repeats": NOMINAL_REPEATS,
        "best_repeats": BEST_REPEATS,
        "fine_offsets": FINE_OFFSETS.tolist(),
        "coarse_norm_levels": COARSE_NORM_LEVELS.tolist(),
        "candidate_sim_deterministic_note":
            "The LUT probabilistic candidate simulator is deterministic in "
            "(params, tracks, response). See optimize/bayesian/bo_2d/src/objective.py:16-26.",
    }
    with prov_path.open("x") as f:
        json.dump(prov, f, indent=2); f.flush(); os.fsync(f.fileno())

    obj, tgt_obj = make_objective()

    # ---------------- Held-out 6D design ----------------
    # 12 global + 12 local Sobol points, then reference points.
    # Filter out any coordinates that exactly match a training row.
    eng_g = SobolEngine(dimension=6, scramble=True, seed=VALIDATION_SEED)
    U_g = eng_g.draw(GLOBAL_N * 4).cpu().numpy()  # over-sample to allow filtering
    X_g_all = np.array([physical(u) for u in U_g], dtype=float)

    eng_l = SobolEngine(dimension=6, scramble=True, seed=VALIDATION_SEED + 1)
    U_l = eng_l.draw(LOCAL_N * 4).cpu().numpy()
    hw = LOCAL_HALFWIDTH_FRACTION * (HI - LO)
    ctr = np.array([BEST[n] for n in NAMES], dtype=float)
    lo_l = np.maximum(ctr - hw, LO)
    hi_l = np.minimum(ctr + hw, HI)
    X_l_all = lo_l + U_l * (hi_l - lo_l)

    def pick_unique(candidates, n):
        picked = []
        for cand in candidates:
            key = tuple(float(v) for v in cand)
            if key in cache:
                continue
            picked.append(cand)
            cache[key] = None  # placeholder so we don't pick same twice
            if len(picked) == n:
                break
        return picked

    global_pts = pick_unique(X_g_all, GLOBAL_N)
    local_pts = pick_unique(X_l_all, LOCAL_N)

    heldout_csv = _no_overwrite(DV / f"{_PREFIX}heldout_6d_results.csv")
    heldout_design_csv = _no_overwrite(DV / f"{_PREFIX}heldout_6d_design.csv")
    ref_csv = _no_overwrite(DV / f"{_PREFIX}reference_points.csv")

    fields_heldout = ["set", "index", *NAMES, "native_LLHD", "score_log",
                       "Yvar_log", "elapsed_seconds", "timestamp", "slurm_job_id"]
    fields_design = ["set", "index", *NAMES]

    with heldout_design_csv.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_design); w.writeheader()
        for i, pt in enumerate(global_pts, start=1):
            w.writerow({"set": "global", "index": i, **{n: float(pt[j]) for j, n in enumerate(NAMES)}})
        for i, pt in enumerate(local_pts, start=1):
            w.writerow({"set": "local", "index": i, **{n: float(pt[j]) for j, n in enumerate(NAMES)}})
        f.flush(); os.fsync(f.fileno())

    with heldout_csv.open("x", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fields_heldout); w.writeheader()
        for label, pts in [("global", global_pts), ("local", local_pts)]:
            for i, pt in enumerate(pts, start=1):
                t0 = time.perf_counter()
                llhd = eval_point(obj, tgt_obj, pt)
                elapsed = time.perf_counter() - t0
                stamp = datetime.now(timezone.utc).isoformat()
                row = {"set": label, "index": i,
                       **{n: float(pt[j]) for j, n in enumerate(NAMES)},
                       "native_LLHD": float(llhd),
                       "score_log": -float(np.log(llhd)),
                       "Yvar_log": 1.0 / float(llhd) ** 2,
                       "elapsed_seconds": float(elapsed),
                       "timestamp": stamp,
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")}
                w.writerow(row); fout.flush(); os.fsync(fout.fileno())
                cache[tuple(float(v) for v in pt)] = float(llhd)
                sim_p = _no_overwrite(DV_SIM / f"heldout_{label}_{i:03d}.json")
                with sim_p.open("x") as sf:
                    json.dump(row, sf, indent=2); sf.flush(); os.fsync(sf.fileno())
                print(json.dumps({"set": label, "i": i, "llhd": float(llhd),
                                   "elapsed_s": float(elapsed)}), flush=True)

    # Reference: all-nominal + repeated best (3x each) + repeated nominal (3x)
    fields_ref = ["label", "repeat", *NAMES, "native_LLHD", "score_log",
                   "elapsed_seconds", "timestamp"]
    with ref_csv.open("x", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fields_ref); w.writeheader()
        nom_pt = np.array(NOM, dtype=float)
        for k in range(1, NOMINAL_REPEATS + 1):
            t0 = time.perf_counter(); llhd = eval_point(obj, tgt_obj, nom_pt); el = time.perf_counter() - t0
            stamp = datetime.now(timezone.utc).isoformat()
            row = {"label": "all_nominal", "repeat": k,
                   **{n: float(nom_pt[j]) for j, n in enumerate(NAMES)},
                   "native_LLHD": float(llhd),
                   "score_log": -float(np.log(llhd)),
                   "elapsed_seconds": float(el), "timestamp": stamp}
            w.writerow(row); fout.flush(); os.fsync(fout.fileno())
            print(json.dumps(row), flush=True)
        best_pt = np.array([BEST[n] for n in NAMES], dtype=float)
        for k in range(1, BEST_REPEATS + 1):
            t0 = time.perf_counter(); llhd = eval_point(obj, tgt_obj, best_pt); el = time.perf_counter() - t0
            stamp = datetime.now(timezone.utc).isoformat()
            row = {"label": "final_best", "repeat": k,
                   **{n: float(best_pt[j]) for j, n in enumerate(NAMES)},
                   "native_LLHD": float(llhd),
                   "score_log": -float(np.log(llhd)),
                   "elapsed_seconds": float(el), "timestamp": stamp}
            w.writerow(row); fout.flush(); os.fsync(fout.fileno())
            print(json.dumps(row), flush=True)

    # ---------------- Six conditional profiles ----------------
    profiles_meta = []
    for j, p in enumerate(NAMES):
        pcsv = _no_overwrite(DV / f"{_PREFIX}final_profile_{p}.csv")
        pmeta = _no_overwrite(DV / f"{_PREFIX}final_profile_{p}_meta.json")
        lo, hi = float(LO[j]), float(HI[j])
        rng = hi - lo
        coarse = lo + COARSE_NORM_LEVELS * rng
        fine = float(BEST[p]) + FINE_OFFSETS * rng
        vals = np.unique(np.concatenate([coarse, [float(NOM[j])], [float(BEST[p])], fine]))
        vals = vals[(vals >= lo) & (vals <= hi)]
        vals = np.sort(vals)

        fields_prof = [p, "native_LLHD", "score_log", "elapsed_seconds",
                        "was_cached", "timestamp"]
        out_rows = []
        with pcsv.open("x", newline="") as fout:
            w = csv.DictWriter(fout, fieldnames=fields_prof); w.writeheader()
            for i, v in enumerate(vals, start=1):
                point = dict(BEST); point[p] = float(v)
                x = np.array([point[n] for n in NAMES], dtype=float)
                key = tuple(float(vv) for vv in x)
                if key in cache and cache[key] is not None:
                    llhd = cache[key]; elapsed = 0.0; was_cached = True
                else:
                    t0 = time.perf_counter(); llhd = eval_point(obj, tgt_obj, x); elapsed = time.perf_counter() - t0
                    cache[key] = float(llhd); was_cached = False
                stamp = datetime.now(timezone.utc).isoformat()
                row = {p: float(v), "native_LLHD": float(llhd),
                       "score_log": -float(np.log(llhd)),
                       "elapsed_seconds": float(elapsed),
                       "was_cached": was_cached,
                       "timestamp": stamp}
                w.writerow(row); fout.flush(); os.fsync(fout.fileno())
                out_rows.append(row)
                print(json.dumps({"profile": p, "i": i, "value": float(v),
                                   "llhd": float(llhd), "cached": was_cached}), flush=True)
        vs = np.array([r[p] for r in out_rows]); ls = np.array([r["native_LLHD"] for r in out_rows])
        imin = int(np.argmin(ls))
        with pmeta.open("x") as f:
            json.dump({
                "parameter": p,
                "fixed_point": BEST,
                "final_best_native_LLHD": BEST_LLHD,
                "n_points": int(len(out_rows)),
                "grid_min": float(vs.min()), "grid_max": float(vs.max()),
                "direct_min_coord": float(vs[imin]),
                "direct_min_llhd": float(ls[imin]),
                "delta_llhd_direct_min_minus_best": float(ls[imin] - BEST_LLHD),
                "exact_best_included": bool(np.any(np.abs(vs - BEST[p]) < 1e-14)),
                "exact_nominal_included": bool(np.any(np.abs(vs - NOM[j]) < 1e-14)),
                "at_lower_bound": bool(abs(vs[imin] - lo) < 1e-14),
                "at_upper_bound": bool(abs(vs[imin] - hi) < 1e-14),
            }, f, indent=2); f.flush(); os.fsync(f.fileno())
        profiles_meta.append({"parameter": p, "csv": str(pcsv), "n_points": int(len(out_rows))})

    summary_path = _no_overwrite(DV / f"{_PREFIX}final_validation_summary.json")
    with summary_path.open("x") as f:
        json.dump({
            "final_training_snapshot": str(SNAPSHOT),
            "final_training_snapshot_sha256": snap_sha,
            "target_sha256": _sha256(tgt) if tgt.exists() else None,
            "final_best": {"point": BEST, "native_LLHD": BEST_LLHD},
            "profiles": profiles_meta,
            "heldout_results_csv": str(heldout_csv),
            "reference_csv": str(ref_csv),
        }, f, indent=2); f.flush(); os.fsync(f.fileno())
    print(json.dumps({"status": "complete",
                       "heldout": str(heldout_csv),
                       "reference": str(ref_csv),
                       "profiles": [m["csv"] for m in profiles_meta]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
