"""Stage 3 + 4 — merge interaction results into augmented design, refit GP,
run 50 additional BO iterations (labelled BO_2).

Reads augmented_training_design.csv + interaction_results.csv → deduplicated
merged design. Writes:
  raw/continuation/traces/bo_6d_continuation_history.csv (append-only)
  raw/continuation/traces/bo_6d_continuation_trace_latest.npz
  raw/continuation/traces/bo_6d_continuation_summary.json
  raw/continuation/checkpoints/checkpoint_iter_XXX.pt + checkpoint_latest.pt

Never modifies raw/traces/bo_6d_history.csv or raw/direct_validation/*.
Deduplicates candidates against the merged history before simulating: if a
proposed point matches an existing observation to 1e-14 in every axis, the
cached deterministic LLHD is reused and the simulator is NOT called.
"""
from __future__ import annotations
import csv, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch

BASE = Path("/sdf/home/i/iatif/larnd-sim-jax/optimize/bayesian/.local/six_d/current").resolve()
sys.path.insert(0, str(BASE / "scripts"))
from build_6d import (NAMES, LO, HI, unit, physical, fit_gp,
                       make_objective, eval_point, guard)
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf

CONT = BASE / "raw/continuation"
CONT.mkdir(parents=True, exist_ok=True)
TRACES = CONT / "traces"; TRACES.mkdir(parents=True, exist_ok=True)
CPS = CONT / "checkpoints"; CPS.mkdir(parents=True, exist_ok=True)
SIM = CONT / "simulator"; SIM.mkdir(parents=True, exist_ok=True)
LOGS = CONT / "logs"; LOGS.mkdir(parents=True, exist_ok=True)

AUG_CSV = CONT / "augmented_training_design.csv"
INTER_CSV = CONT / "interaction_results.csv"
HIST_OUT = TRACES / "bo_6d_continuation_history.csv"
TRACE_OUT = TRACES / "bo_6d_continuation_trace_latest.npz"
SUMMARY_OUT = TRACES / "bo_6d_continuation_summary.json"
CP_LATEST = CPS / "checkpoint_latest.pt"

N_BO_2 = 50
SEED = 20260815   # different from the initial 6D BO (20260812) so acquisition randomness is fresh

FIELDS = [
    "run", "evaluation_index", "simulator_iteration", "phase",
    "source", "source_file", "original_row",
    *NAMES,
    *[f"normalized_{n}" for n in NAMES],
    "native_LLHD", "score_log", "Yvar_log",
    "acquisition_value",
    *[f"{n}_lengthscale" for n in NAMES],
    "outputscale", "simulator_elapsed_seconds",
    "timestamp", "simulator_status", "checkpoint_path",
]


def _atomic_torch(path, state):
    q = guard(path.with_suffix(path.suffix + ".part"))
    torch.save(state, q); os.replace(q, path)


def _atomic_npz(path, **arrays):
    q = guard(path.with_suffix(path.suffix + ".part"))
    with q.open("wb") as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(q, path)


def _append_row(row):
    new = not HIST_OUT.exists()
    with HIST_OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush(); os.fsync(f.fileno())


def _load_merged():
    """Return (X, L, meta_rows_for_seed_rows) from augmented + interaction results,
    deduplicated by exact 6D coordinate."""
    rows_aug = list(csv.DictReader(AUG_CSV.open()))
    rows_int = list(csv.DictReader(INTER_CSV.open())) if INTER_CSV.exists() else []
    seen = {}
    merged = []
    for r in rows_aug:
        pt = tuple(float(r[n]) for n in NAMES)
        if pt in seen:
            continue
        seen[pt] = len(merged)
        merged.append({"pt": pt, "L": float(r["native_LLHD"]),
                        "source": r["source"], "source_file": r["source_file"],
                        "original_row": r["original_row"], "phase": r["phase"]})
    for r in rows_int:
        pt = tuple(float(r[n]) for n in NAMES)
        if pt in seen:
            continue
        seen[pt] = len(merged)
        merged.append({"pt": pt, "L": float(r["native_LLHD"]),
                        "source": "interaction",
                        "source_file": str(INTER_CSV),
                        "original_row": r["combo_index"],
                        "phase": "INTERACTION"})
    X = np.array([m["pt"] for m in merged], dtype=float)
    L = np.array([m["L"] for m in merged], dtype=float)
    return X, L, merged, seen


def _lookup_cached(seen_map, X, L, x):
    key = tuple(float(v) for v in x)
    if key in seen_map:
        return float(L[seen_map[key]])
    return None


def main():
    X, L, meta, seen = _load_merged()
    print(f"merged rows: {len(X)}  best_LLHD: {L.min():.4f}", flush=True)

    # Emit seed rows into continuation history CSV once (so the continuation history
    # is self-contained and future readers can find each row's provenance)
    if not HIST_OUT.exists():
        for i, m in enumerate(meta, start=1):
            x = np.array(m["pt"], dtype=float)
            u = unit(x)
            _append_row({
                "run": "6D-1000cm-continuation",
                "evaluation_index": i,
                "simulator_iteration": 0,
                "phase": m["phase"],
                "source": m["source"], "source_file": m["source_file"],
                "original_row": m["original_row"],
                **{n: float(x[j]) for j, n in enumerate(NAMES)},
                **{f"normalized_{n}": float(u[j]) for j, n in enumerate(NAMES)},
                "native_LLHD": float(m["L"]),
                "score_log": -float(np.log(m["L"])),
                "Yvar_log": 1.0 / float(m["L"]) ** 2,
                "acquisition_value": "NA",
                **{f"{n}_lengthscale": "NA" for n in NAMES},
                "outputscale": "NA",
                "simulator_elapsed_seconds": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "simulator_status": "reused_from_history_or_dv_or_interaction",
                "checkpoint_path": "",
            })
    seed_row_count = len(meta)

    # Resume support
    if CP_LATEST.exists():
        st = torch.load(CP_LATEST, map_location="cpu", weights_only=False)
        if st.get("complete"):
            print(json.dumps({"status": "already_complete"}), flush=True); return
        X = st["X"]; L = st["L"]; phase_list = st["phase"]
        A = st["A"]; LS = st["LS"]; OS = st["OS"]
        start = st["next_iteration"]
        torch.random.set_rng_state(st["torch_rng"])
        np.random.set_state(st["numpy_rng"])
        # Reload seen map from current arrays
        seen = {tuple(float(v) for v in X[i]): i for i in range(len(X))}
        # Re-sync any BO_2 rows already appended after the last checkpoint
        rows_now = list(csv.DictReader(HIST_OUT.open()))
        appended_bo2 = [r for r in rows_now if r["phase"] == "BO_2"][start:]
        for r in appended_bo2:
            xr = np.array([float(r[n]) for n in NAMES])
            key = tuple(float(v) for v in xr)
            if key in seen:
                continue
            X = np.vstack([X, xr]); L = np.r_[L, float(r["native_LLHD"])]
            phase_list.append("BO_2")
            A.append(float(r["acquisition_value"]))
            LS.append([float(r[f"{n}_lengthscale"]) for n in NAMES])
            OS.append(float(r["outputscale"]))
            seen[key] = len(X) - 1
        start = st["next_iteration"] + len(appended_bo2)
    else:
        torch.manual_seed(SEED); np.random.seed(SEED)
        phase_list = [m["phase"] for m in meta]
        A = [float("nan")] * len(meta)
        LS = [[float("nan")] * 6 for _ in meta]
        OS = [float("nan")] * len(meta)
        start = 0

    obj = None; tgt = None
    bounds = torch.tensor([[0.0] * 6, [1.0] * 6], dtype=torch.double)
    boundary_props = 0
    cache_hits = 0

    for it in range(start, N_BO_2):
        model = fit_gp(X, L, seed=SEED + it)
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
        outs = float(model.covar_module.outputscale.detach().cpu())
        Y = -torch.log(torch.tensor(L, dtype=torch.double)).view(-1, 1)
        acq = qLogExpectedImprovement(model, best_f=Y.max())
        cand, val = optimize_acqf(acq, bounds, q=1, num_restarts=24, raw_samples=2048)
        u = cand.detach().cpu().numpy()[0]
        x = physical(u)
        av = float(val.detach().cpu().reshape(-1)[0])
        boundary_flag = int(np.any((u < 1e-3) | (u > 1 - 1e-3)))
        boundary_props += boundary_flag

        # Duplicate check (candidate objective is deterministic)
        key = tuple(float(v) for v in x)
        cached = _lookup_cached(seen, X, L, x)
        if cached is not None:
            loss = cached
            elapsed = 0.0
            sim_status = "cached_duplicate"
            cache_hits += 1
        else:
            if obj is None:
                obj, tgt = make_objective()
            t0 = time.perf_counter()
            loss = eval_point(obj, tgt, x)
            elapsed = time.perf_counter() - t0
            sim_status = "success"

        stamp = datetime.now(timezone.utc).isoformat()
        X = np.vstack([X, x]); L = np.r_[L, loss]
        phase_list.append("BO_2"); A.append(av); LS.append(ls.tolist()); OS.append(outs)
        if sim_status == "success":
            seen[key] = len(X) - 1

        _append_row({
            "run": "6D-1000cm-continuation",
            "evaluation_index": seed_row_count + it + 1,
            "simulator_iteration": it + 1,
            "phase": "BO_2",
            "source": "continuation",
            "source_file": str(HIST_OUT),
            "original_row": "",
            **dict(zip(NAMES, [float(v) for v in x])),
            **{f"normalized_{n}": float(u[j]) for j, n in enumerate(NAMES)},
            "native_LLHD": float(loss),
            "score_log": -float(np.log(loss)),
            "Yvar_log": 1.0 / float(loss) ** 2,
            "acquisition_value": av,
            **{f"{n}_lengthscale": float(ls[j]) for j, n in enumerate(NAMES)},
            "outputscale": outs,
            "simulator_elapsed_seconds": elapsed,
            "timestamp": stamp,
            "simulator_status": sim_status,
            "checkpoint_path": str(CP_LATEST),
        })

        # Per-iter metadata JSON
        sim_p = SIM / f"bo2_iter_{it + 1:03d}.json"
        if not sim_p.exists():
            with sim_p.open("x") as sf:
                json.dump({"iteration": it + 1,
                            "requested": dict(zip(NAMES, map(float, x))),
                            "native_LLHD": float(loss),
                            "elapsed_seconds": float(elapsed),
                            "acquisition_value": av,
                            "boundary_flag": bool(boundary_flag),
                            "sim_status": sim_status,
                            "timestamp": stamp,
                            "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
                            sf, indent=2); sf.flush(); os.fsync(sf.fileno())

        _atomic_npz(TRACE_OUT,
                    X_physical=X, X_unit=unit(X),
                    native_LLHD=L, score_log=-np.log(L), Yvar_log=1.0 / L ** 2,
                    phase=np.asarray(phase_list), acquisition_value=np.asarray(A),
                    lengthscales=np.asarray(LS), outputscale=np.asarray(OS),
                    seed=SEED)
        state = {"X": X, "L": L, "phase": phase_list, "A": A, "LS": LS, "OS": OS,
                 "next_iteration": it + 1,
                 "torch_rng": torch.random.get_rng_state(),
                 "numpy_rng": np.random.get_state(),
                 "seed": SEED, "run": "6D-1000cm-continuation",
                 "complete": (it + 1) == N_BO_2}
        _atomic_torch(CP_LATEST, state)
        if (it + 1) % 10 == 0:
            numbered = CPS / f"checkpoint_iter_{it + 1:03d}.pt"
            if not numbered.exists():
                torch.save(state, numbered)

        print(json.dumps({"iteration": it + 1,
                          **dict(zip(NAMES, map(float, x))),
                          "native_LLHD": float(loss),
                          "running_best": float(L.min()),
                          "acquisition_value": av,
                          "boundary_flag": bool(boundary_flag),
                          "sim_status": sim_status,
                          "elapsed_seconds": float(elapsed)}), flush=True)

    bi = int(np.argmin(L))
    summary = {
        "status": "complete", "run": "6D-1000cm-continuation", "seed": SEED,
        "seed_rows": seed_row_count, "bo2_iterations": N_BO_2,
        "total_rows": len(L),
        "best_index_in_continuation_history": bi + 1,
        "best_point": dict(zip(NAMES, map(float, X[bi]))),
        "best_native_LLHD": float(L[bi]),
        "boundary_proposals": boundary_props,
        "cache_hits": cache_hits,
        "final_lengthscales": {n: float(v) for n, v in zip(NAMES, LS[-1])},
        "final_outputscale": float(OS[-1]),
    }
    if not SUMMARY_OUT.exists():
        with SUMMARY_OUT.open("x") as f:
            json.dump(summary, f, indent=2); f.flush(); os.fsync(f.fileno())
    print(json.dumps(summary, indent=2), flush=True)


# =====================================================================
# Trust-region mode (BO_TR)
# =====================================================================
CLEAN_CSV = CONT / "clean_172_training.csv"
BO_TR_HIST = CONT / "bo_tr_history.csv"
BO_TR_CP_LATEST = CPS / "bo_tr_checkpoint_latest.pt"
BO_TR_TRACE = TRACES / "bo_tr_trace_latest.npz"
BO_TR_SUMMARY = TRACES / "bo_tr_summary.json"

N_BO_TR = 100
BO_TR_SEED = 20260830

# Trust-region hyperparameters (frozen before the run)
TR_LEN_INIT = 0.40
TR_LEN_MAX = 0.80
TR_LEN_MIN = 0.025
TR_SUCCESS_TOL = 3
TR_FAILURE_TOL = 6
TR_EXPAND = 2.0
TR_SHRINK = 0.5
TR_STAG_LIMIT = 12  # extra failures at min length before stagnation reset
TR_ARD_CLIP = (0.1, 10.0)  # per-dim weight clip (documented)
IMP_ABS = 1.0
IMP_REL = 1e-4

BO_TR_FIELDS = [
    "iteration", "phase",
    *NAMES,
    *[f"normalized_{n}" for n in NAMES],
    "native_LLHD", "score_log", "Yvar_log",
    "acquisition_value",
    *[f"{n}_lengthscale" for n in NAMES],
    "outputscale",
    "tr_length",
    *[f"tr_center_{n}" for n in NAMES],
    *[f"tr_lo_{n}" for n in NAMES],
    *[f"tr_hi_{n}" for n in NAMES],
    "success_counter", "failure_counter",
    "tr_event",
    "boundary_flag",
    "gp_post_mean_at_candidate",
    "gp_post_sd_at_candidate",
    "simulator_elapsed_seconds",
    "timestamp",
]


def _bo_tr_append(row):
    new = not BO_TR_HIST.exists()
    with BO_TR_HIST.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BO_TR_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row); f.flush(); os.fsync(f.fileno())


def _load_clean_172():
    if not CLEAN_CSV.exists():
        raise RuntimeError(f"clean baseline missing: {CLEAN_CSV}")
    rows = list(csv.DictReader(CLEAN_CSV.open()))
    X = np.array([[float(r[n]) for n in NAMES] for r in rows], dtype=float)
    L = np.array([float(r["native_LLHD"]) for r in rows], dtype=float)
    phase = np.array([r["phase"] for r in rows])
    n_init = int((phase == "INITIAL").sum()); n_bo = int((phase == "BO_1").sum())
    if n_init != 72 or n_bo != 100 or len(rows) != 172:
        raise RuntimeError(f"clean baseline shape wrong: INITIAL={n_init} BO_1={n_bo} total={len(rows)}")
    return X, L, phase


def _tr_weighted_bounds(center_u, length, ls_unit):
    """Construct per-dimension trust-region bounds in unit space.

    ARD lengthscales are converted to relative weights (geometric-mean=1),
    clipped to TR_ARD_CLIP, and each dimension's half-width scales as
    length * weight_i / 2. Clipped to global [0,1] bounds.
    """
    ls_unit = np.asarray(ls_unit, dtype=float)
    ls_unit = np.clip(ls_unit, 1e-6, None)
    w = ls_unit / np.exp(np.mean(np.log(ls_unit)))  # geometric mean 1
    w = np.clip(w, *TR_ARD_CLIP)
    half = 0.5 * length * w
    lo = np.clip(center_u - half, 0.0, 1.0)
    hi = np.clip(center_u + half, 0.0, 1.0)
    return lo, hi, w


def _run_trust_region():
    """Adaptive trust-region qLogEI 6D BO on the clean 72+100 seed."""
    X0, L0, phase0 = _load_clean_172()

    if BO_TR_CP_LATEST.exists():
        st = torch.load(BO_TR_CP_LATEST, map_location="cpu", weights_only=False)
        if st.get("complete"):
            print(json.dumps({"status": "already_complete", "run": "BO_TR"}), flush=True)
            return
        X = st["X"]; L = st["L"]; phase = st["phase"]
        tr_length = float(st["tr_length"])
        success = int(st["success_counter"]); failure = int(st["failure_counter"])
        stag = int(st.get("stag_since_min", 0))
        start = int(st["next_iteration"])
        expansions = int(st.get("expansions", 0)); shrinks = int(st.get("shrinks", 0))
        stag_resets = int(st.get("stag_resets", 0))
        boundary_props = int(st.get("boundary_props", 0))
        torch.random.set_rng_state(st["torch_rng"])
        np.random.set_state(st["numpy_rng"])
        seen = {tuple(float(v) for v in X[i]): i for i in range(len(X))}
    else:
        torch.manual_seed(BO_TR_SEED); np.random.seed(BO_TR_SEED)
        X = X0.copy(); L = L0.copy(); phase = list(phase0)
        tr_length = TR_LEN_INIT
        success = 0; failure = 0; stag = 0
        expansions = 0; shrinks = 0; stag_resets = 0
        boundary_props = 0
        start = 0
        seen = {tuple(float(v) for v in X[i]): i for i in range(len(X))}

    obj = None; tgt = None
    unit_bounds = torch.tensor([[0.0] * 6, [1.0] * 6], dtype=torch.double)

    for it in range(start, N_BO_TR):
        model = fit_gp(X, L, seed=BO_TR_SEED + it)
        ls_unit = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
        outs = float(model.covar_module.outputscale.detach().cpu())

        # Trust region: center = current best actual observation among {X}
        best_i = int(np.argmin(L))
        best_llhd_pre = float(L[best_i])
        center_u = unit(X[best_i])
        lo_u, hi_u, weights = _tr_weighted_bounds(center_u, tr_length, ls_unit)
        tr_bounds = torch.tensor([lo_u, hi_u], dtype=torch.double)

        Y = -torch.log(torch.tensor(L, dtype=torch.double)).view(-1, 1)
        acq = qLogExpectedImprovement(model, best_f=Y.max())
        # Robust acquisition optimization inside the trust region
        cand, val = optimize_acqf(acq, tr_bounds, q=1, num_restarts=64, raw_samples=8192)
        u = cand.detach().cpu().numpy()[0]
        x = physical(u)
        av = float(val.detach().cpu().reshape(-1)[0])
        boundary_flag = int(np.any((u < 1e-3) | (u > 1 - 1e-3)))
        boundary_props += boundary_flag

        # Duplicate check
        key = tuple(float(v) for v in x)
        if key in seen:
            loss = float(L[seen[key]]); elapsed = 0.0; sim_status = "cached_duplicate"
        else:
            if obj is None:
                obj, tgt = make_objective()
            t0 = time.perf_counter()
            loss = eval_point(obj, tgt, x)
            elapsed = time.perf_counter() - t0
            sim_status = "success"

        # GP posterior at the candidate (before appending)
        with torch.no_grad():
            post_cand = model.posterior(torch.tensor(u.reshape(1, -1), dtype=torch.double))
            gp_m = float(post_cand.mean.squeeze().cpu())
            gp_s = float(post_cand.variance.sqrt().squeeze().cpu())

        # Update training set
        X = np.vstack([X, x]); L = np.r_[L, loss]; phase.append("BO_TR")
        if sim_status == "success":
            seen[key] = len(X) - 1

        # Success / failure logic
        improvement = best_llhd_pre - loss
        improved = improvement > max(IMP_ABS, IMP_REL * best_llhd_pre)
        tr_event = ""
        if improved:
            success += 1; failure = 0
            if success >= TR_SUCCESS_TOL:
                new_len = min(tr_length * TR_EXPAND, TR_LEN_MAX)
                if new_len > tr_length:
                    tr_event = "expand"; expansions += 1
                tr_length = new_len
                success = 0
        else:
            failure += 1; success = 0
            if failure >= TR_FAILURE_TOL:
                new_len = max(tr_length * TR_SHRINK, TR_LEN_MIN)
                if new_len < tr_length:
                    tr_event = "shrink"; shrinks += 1
                tr_length = new_len
                failure = 0

        # Stagnation counter — increments only once we've bottomed to min length
        if abs(tr_length - TR_LEN_MIN) < 1e-12 and not improved:
            stag += 1
            if stag >= TR_STAG_LIMIT:
                tr_length = TR_LEN_INIT
                stag = 0
                tr_event = "stagnation_reset"
                stag_resets += 1
        elif improved:
            stag = 0

        stamp = datetime.now(timezone.utc).isoformat()
        _bo_tr_append({
            "iteration": it + 1, "phase": "BO_TR",
            **{n: float(x[j]) for j, n in enumerate(NAMES)},
            **{f"normalized_{n}": float(u[j]) for j, n in enumerate(NAMES)},
            "native_LLHD": float(loss),
            "score_log": -float(np.log(loss)),
            "Yvar_log": 1.0 / float(loss) ** 2,
            "acquisition_value": av,
            **{f"{n}_lengthscale": float(ls_unit[j]) for j, n in enumerate(NAMES)},
            "outputscale": outs,
            "tr_length": float(tr_length),
            **{f"tr_center_{NAMES[j]}": float(center_u[j] * (HI[j] - LO[j]) + LO[j]) for j in range(6)},
            **{f"tr_lo_{NAMES[j]}": float(lo_u[j] * (HI[j] - LO[j]) + LO[j]) for j in range(6)},
            **{f"tr_hi_{NAMES[j]}": float(hi_u[j] * (HI[j] - LO[j]) + LO[j]) for j in range(6)},
            "success_counter": int(success), "failure_counter": int(failure),
            "tr_event": tr_event,
            "boundary_flag": int(boundary_flag),
            "gp_post_mean_at_candidate": gp_m,
            "gp_post_sd_at_candidate": gp_s,
            "simulator_elapsed_seconds": float(elapsed),
            "timestamp": stamp,
        })

        # NPZ trace
        _atomic_npz(BO_TR_TRACE,
                    X_physical=X, X_unit=unit(X),
                    native_LLHD=L, phase=np.asarray(phase),
                    seed=BO_TR_SEED)
        state = {"X": X, "L": L, "phase": phase,
                 "next_iteration": it + 1,
                 "tr_length": tr_length,
                 "success_counter": success, "failure_counter": failure,
                 "stag_since_min": stag,
                 "expansions": expansions, "shrinks": shrinks,
                 "stag_resets": stag_resets,
                 "boundary_props": boundary_props,
                 "torch_rng": torch.random.get_rng_state(),
                 "numpy_rng": np.random.get_state(),
                 "seed": BO_TR_SEED, "run": "BO_TR",
                 "complete": (it + 1) == N_BO_TR}
        _atomic_torch(BO_TR_CP_LATEST, state)
        if (it + 1) % 10 == 0:
            numbered = CPS / f"bo_tr_checkpoint_iter_{it + 1:03d}.pt"
            if not numbered.exists():
                torch.save(state, numbered)

        print(json.dumps({"iter": it + 1, "phase": "BO_TR",
                           **{n: float(x[j]) for j, n in enumerate(NAMES)},
                           "llhd": float(loss),
                           "running_best": float(L.min()),
                           "tr_length": float(tr_length),
                           "improved": bool(improved),
                           "tr_event": tr_event}), flush=True)

    bi = int(np.argmin(L))
    summary = {"status": "complete", "run": "BO_TR", "seed": BO_TR_SEED,
                "seed_rows": 172, "bo_tr_iterations": N_BO_TR,
                "total_rows": len(L),
                "best_index": bi + 1,
                "best_point": dict(zip(NAMES, map(float, X[bi]))),
                "best_native_LLHD": float(L[bi]),
                "boundary_proposals": int(boundary_props),
                "expansions": int(expansions),
                "shrinks": int(shrinks),
                "stagnation_resets": int(stag_resets),
                "final_tr_length": float(tr_length)}
    if not BO_TR_SUMMARY.exists():
        with BO_TR_SUMMARY.open("x") as f:
            json.dump(summary, f, indent=2); f.flush(); os.fsync(f.fileno())
    print(json.dumps(summary, indent=2), flush=True)


# =====================================================================
# BO_TR2 — Matérn-3/2 continuation on the clean 272-row snapshot
# =====================================================================
CLEAN_272_CSV = CONT / "clean_bo_tr_training.csv"          # frozen 272 rows
BO_TR2_HIST = CONT / "bo_tr2_history.csv"
BO_TR2_CP_LATEST = CPS / "bo_tr2_checkpoint_latest.pt"
BO_TR2_TRACE = TRACES / "bo_tr2_trace_latest.npz"
BO_TR2_SUMMARY = TRACES / "bo_tr2_summary.json"

N_BO_TR2 = 60
BO_TR2_SEED = 20260840
BO_TR2_KERNEL_NU = 1.5  # Matérn-3/2 (Model C)

# Fresh trust-region state (audit-authorised policy)
TR2_LEN_INIT = 0.20
TR2_LEN_MAX = 0.20
TR2_LEN_MIN = 0.005
TR2_SUCCESS_TOL = 3
TR2_FAILURE_TOL = 4
TR2_EXPAND = 1.5
TR2_SHRINK = 0.5
TR2_STAG_LIMIT = 8
TR2_ARD_CLIP = (0.1, 10.0)
IMP2_ABS = 1.0
IMP2_REL = 1e-4

# Same schema as BO_TR history (fits under the same fields for downstream tools)
BO_TR2_FIELDS = list(BO_TR_FIELDS)


def _bo_tr2_append(row):
    new = not BO_TR2_HIST.exists()
    with BO_TR2_HIST.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BO_TR2_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row); f.flush(); os.fsync(f.fileno())


def _load_clean_272():
    if not CLEAN_272_CSV.exists():
        raise RuntimeError(f"clean 272-row snapshot missing: {CLEAN_272_CSV}")
    rows = list(csv.DictReader(CLEAN_272_CSV.open()))
    X = np.array([[float(r[n]) for n in NAMES] for r in rows], dtype=float)
    L = np.array([float(r["native_LLHD"]) for r in rows], dtype=float)
    phase = np.array([r["phase"] for r in rows])
    n_init = int((phase == "INITIAL").sum())
    n_bo1 = int((phase == "BO_1").sum())
    n_botr = int((phase == "BO_TR").sum())
    if n_init != 72 or n_bo1 != 100 or n_botr != 100 or len(rows) != 272:
        raise RuntimeError(f"clean 272 snapshot phase counts wrong: "
                            f"INITIAL={n_init} BO_1={n_bo1} BO_TR={n_botr} total={len(rows)}")
    # Guarantee no leakage
    forbidden = {"DIRECT", "INTERACTION", "BO_2", "DIAG_BRIDGE", "DIAG_SHELL",
                  "VALIDATION"}
    if any(ph in forbidden for ph in np.unique(phase)):
        raise RuntimeError("forbidden phase leaked into clean 272-row snapshot")
    return X, L, phase


def _run_trust_region_2():
    """Adaptive trust-region qLogEI 6D BO on the clean 272-row snapshot,
    using Matérn-3/2 (Model C, selected by the audit shootout).
    """
    X0, L0, phase0 = _load_clean_272()

    if BO_TR2_CP_LATEST.exists():
        st = torch.load(BO_TR2_CP_LATEST, map_location="cpu", weights_only=False)
        if st.get("complete"):
            print(json.dumps({"status": "already_complete", "run": "BO_TR2"}), flush=True)
            return
        X = st["X"]; L = st["L"]; phase = st["phase"]
        tr_length = float(st["tr_length"])
        success = int(st["success_counter"]); failure = int(st["failure_counter"])
        stag = int(st.get("stag_since_min", 0))
        start = int(st["next_iteration"])
        expansions = int(st.get("expansions", 0)); shrinks = int(st.get("shrinks", 0))
        stag_resets = int(st.get("stag_resets", 0))
        boundary_props = int(st.get("boundary_props", 0))
        torch.random.set_rng_state(st["torch_rng"])
        np.random.set_state(st["numpy_rng"])
        seen = {tuple(float(v) for v in X[i]): i for i in range(len(X))}
    else:
        torch.manual_seed(BO_TR2_SEED); np.random.seed(BO_TR2_SEED)
        X = X0.copy(); L = L0.copy(); phase = list(phase0)
        tr_length = TR2_LEN_INIT
        success = 0; failure = 0; stag = 0
        expansions = 0; shrinks = 0; stag_resets = 0
        boundary_props = 0
        start = 0
        seen = {tuple(float(v) for v in X[i]): i for i in range(len(X))}

    obj = None; tgt = None
    unit_bounds = torch.tensor([[0.0] * 6, [1.0] * 6], dtype=torch.double)

    for it in range(start, N_BO_TR2):
        # Selected surrogate: Matérn-3/2 (Model C) via nu=1.5
        model = fit_gp(X, L, seed=BO_TR2_SEED + it, nu=BO_TR2_KERNEL_NU)
        ls_unit = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
        outs = float(model.covar_module.outputscale.detach().cpu())

        best_i = int(np.argmin(L))
        best_llhd_pre = float(L[best_i])
        center_u = unit(X[best_i])
        # ARD-aware per-axis weighting (identical rule to BO_TR)
        ls_clip = np.clip(ls_unit, 1e-6, None)
        w = ls_clip / np.exp(np.mean(np.log(ls_clip)))
        w = np.clip(w, *TR2_ARD_CLIP)
        half = 0.5 * tr_length * w
        lo_u = np.clip(center_u - half, 0.0, 1.0)
        hi_u = np.clip(center_u + half, 0.0, 1.0)
        tr_bounds = torch.tensor([lo_u, hi_u], dtype=torch.double)

        Y = -torch.log(torch.tensor(L, dtype=torch.double)).view(-1, 1)
        acq = qLogExpectedImprovement(model, best_f=Y.max())
        cand, val = optimize_acqf(acq, tr_bounds, q=1, num_restarts=64, raw_samples=8192)
        u = cand.detach().cpu().numpy()[0]
        x = physical(u)
        av = float(val.detach().cpu().reshape(-1)[0])
        boundary_flag = int(np.any((u < 1e-3) | (u > 1 - 1e-3)))
        boundary_props += boundary_flag

        key = tuple(float(v) for v in x)
        if key in seen:
            loss = float(L[seen[key]]); elapsed = 0.0; sim_status = "cached_duplicate"
        else:
            if obj is None:
                obj, tgt = make_objective()
            t0 = time.perf_counter()
            loss = eval_point(obj, tgt, x)
            elapsed = time.perf_counter() - t0
            sim_status = "success"

        with torch.no_grad():
            post_cand = model.posterior(torch.tensor(u.reshape(1, -1), dtype=torch.double))
            gp_m = float(post_cand.mean.squeeze().cpu())
            gp_s = float(post_cand.variance.sqrt().squeeze().cpu())

        X = np.vstack([X, x]); L = np.r_[L, loss]; phase.append("BO_TR2")
        if sim_status == "success":
            seen[key] = len(X) - 1

        improvement = best_llhd_pre - loss
        improved = improvement > max(IMP2_ABS, IMP2_REL * best_llhd_pre)
        tr_event = ""
        if improved:
            success += 1; failure = 0
            if success >= TR2_SUCCESS_TOL:
                new_len = min(tr_length * TR2_EXPAND, TR2_LEN_MAX)
                if new_len > tr_length:
                    tr_event = "expand"; expansions += 1
                tr_length = new_len
                success = 0
        else:
            failure += 1; success = 0
            if failure >= TR2_FAILURE_TOL:
                new_len = max(tr_length * TR2_SHRINK, TR2_LEN_MIN)
                if new_len < tr_length:
                    tr_event = "shrink"; shrinks += 1
                tr_length = new_len
                failure = 0

        if abs(tr_length - TR2_LEN_MIN) < 1e-12 and not improved:
            stag += 1
            if stag >= TR2_STAG_LIMIT:
                tr_length = TR2_LEN_INIT
                stag = 0
                tr_event = "stagnation_reset"
                stag_resets += 1
        elif improved:
            stag = 0

        stamp = datetime.now(timezone.utc).isoformat()
        _bo_tr2_append({
            "iteration": it + 1, "phase": "BO_TR2",
            **{n: float(x[j]) for j, n in enumerate(NAMES)},
            **{f"normalized_{n}": float(u[j]) for j, n in enumerate(NAMES)},
            "native_LLHD": float(loss),
            "score_log": -float(np.log(loss)),
            "Yvar_log": 1.0 / float(loss) ** 2,
            "acquisition_value": av,
            **{f"{n}_lengthscale": float(ls_unit[j]) for j, n in enumerate(NAMES)},
            "outputscale": outs,
            "tr_length": float(tr_length),
            **{f"tr_center_{NAMES[j]}": float(center_u[j] * (HI[j] - LO[j]) + LO[j]) for j in range(6)},
            **{f"tr_lo_{NAMES[j]}": float(lo_u[j] * (HI[j] - LO[j]) + LO[j]) for j in range(6)},
            **{f"tr_hi_{NAMES[j]}": float(hi_u[j] * (HI[j] - LO[j]) + LO[j]) for j in range(6)},
            "success_counter": int(success), "failure_counter": int(failure),
            "tr_event": tr_event,
            "boundary_flag": int(boundary_flag),
            "gp_post_mean_at_candidate": gp_m,
            "gp_post_sd_at_candidate": gp_s,
            "simulator_elapsed_seconds": float(elapsed),
            "timestamp": stamp,
        })

        _atomic_npz(BO_TR2_TRACE,
                    X_physical=X, X_unit=unit(X),
                    native_LLHD=L, phase=np.asarray(phase),
                    seed=BO_TR2_SEED)
        state = {"X": X, "L": L, "phase": phase,
                 "next_iteration": it + 1,
                 "tr_length": tr_length,
                 "success_counter": success, "failure_counter": failure,
                 "stag_since_min": stag,
                 "expansions": expansions, "shrinks": shrinks,
                 "stag_resets": stag_resets,
                 "boundary_props": boundary_props,
                 "torch_rng": torch.random.get_rng_state(),
                 "numpy_rng": np.random.get_state(),
                 "seed": BO_TR2_SEED, "run": "BO_TR2",
                 "complete": (it + 1) == N_BO_TR2}
        _atomic_torch(BO_TR2_CP_LATEST, state)
        # Numbered checkpoints only at 20/40/60
        if (it + 1) in (20, 40, 60):
            numbered = CPS / f"bo_tr2_checkpoint_{it + 1:03d}.pt"
            if not numbered.exists():
                torch.save(state, numbered)

        print(json.dumps({"iter": it + 1, "phase": "BO_TR2",
                           **{n: float(x[j]) for j, n in enumerate(NAMES)},
                           "llhd": float(loss),
                           "running_best": float(L.min()),
                           "tr_length": float(tr_length),
                           "improved": bool(improved),
                           "tr_event": tr_event}), flush=True)

    bi = int(np.argmin(L))
    summary = {"status": "complete", "run": "BO_TR2", "seed": BO_TR2_SEED,
                "seed_rows": 272, "bo_tr2_iterations": N_BO_TR2,
                "total_rows": len(L),
                "best_index": bi + 1,
                "best_point": dict(zip(NAMES, map(float, X[bi]))),
                "best_native_LLHD": float(L[bi]),
                "boundary_proposals": int(boundary_props),
                "expansions": int(expansions),
                "shrinks": int(shrinks),
                "stagnation_resets": int(stag_resets),
                "final_tr_length": float(tr_length),
                "kernel_nu": float(BO_TR2_KERNEL_NU)}
    if not BO_TR2_SUMMARY.exists():
        with BO_TR2_SUMMARY.open("x") as f:
            json.dump(summary, f, indent=2); f.flush(); os.fsync(f.fileno())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("bo_2", "trust_region", "trust_region_2"),
                    default="bo_2")
    args = ap.parse_args()
    if args.mode == "trust_region":
        _run_trust_region()
    elif args.mode == "trust_region_2":
        _run_trust_region_2()
    else:
        main()
