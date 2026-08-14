"""6D 100-iteration BO run.

Same qLogEI+SingleTaskGP setup as run_4d_bo.py. Resumable via
`raw/checkpoints/checkpoint_latest.pt`. Refuses to run before the initial
design is complete (72 rows). Appends BO rows to the shared history CSV;
never rewrites the initial 72 rows.
"""
from __future__ import annotations
import csv, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_6d import (
    ROOT, RAW, NAMES, LO, HI, NOM, unit, physical, fit_gp,
    make_objective, eval_point, guard,
)
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf


RUN = "6D-1000cm"
SEED = 20260812
BO_ITERS = 100
N_INITIAL = 72

HIST = RAW / "traces/bo_6d_history.csv"
TRACE = RAW / "traces/bo_6d_trace_latest.npz"
SUMMARY = RAW / "traces/bo_6d_summary.json"
CP_DIR = RAW / "checkpoints"
CP_LATEST = CP_DIR / "checkpoint_latest.pt"
SIM_DIR = RAW / "simulator"

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


def _atomic_torch(path: Path, state):
    q = guard(path.with_suffix(path.suffix + ".part"))
    torch.save(state, q); os.replace(q, path)


def _atomic_npz(path: Path, **arrays):
    q = guard(path.with_suffix(path.suffix + ".part"))
    with q.open("wb") as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(q, path)


def _append(row):
    with HIST.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writerow(row); f.flush(); os.fsync(f.fileno())


def _load_history():
    rows = list(csv.DictReader(HIST.open()))
    X = np.array([[float(r[n]) for n in NAMES] for r in rows], dtype=float)
    L = np.array([float(r["native_LLHD"]) for r in rows], dtype=float)
    phase = [r["phase"] for r in rows]
    A = []
    LS = []
    OS = []
    for r in rows:
        if r["phase"] == "BO":
            A.append(float(r["acquisition_value"]))
            LS.append([float(r[f"{n}_lengthscale"]) for n in NAMES])
            OS.append(float(r["outputscale"]))
        else:
            A.append(np.nan)
            LS.append([np.nan] * 6)
            OS.append(np.nan)
    return rows, X, L, phase, A, LS, OS


def _consolidate(X, L, digits):
    k = torch.round(torch.tensor(unit(X), dtype=torch.double) * 10 ** digits).long()
    _, inv = torch.unique(k, dim=0, return_inverse=True)
    xx = []; ll = []
    for i in range(int(inv.max()) + 1):
        m = (inv == i)
        xx.append(np.asarray(X)[m.numpy()].mean(0))
        ll.append(np.asarray(L)[m.numpy()].mean())
    return np.array(xx), np.array(ll)


def _robust_fit(X, L, seed):
    last = None
    for d in [None, 6, 5, 4]:
        xx, ll = (X, L) if d is None else _consolidate(X, L, d)
        if d is not None and len(xx) == len(X):
            continue
        try:
            return fit_gp(xx, ll, seed=seed + (0 if d is None else d * 1009)), \
                   ("full" if d is None else f"round{d}")
        except Exception as e:
            last = e
            print(f"GP fit failure {d}: {e}", flush=True)
    raise last


def main():
    CP_DIR.mkdir(parents=True, exist_ok=True)
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    if not HIST.exists():
        raise RuntimeError("initial design missing; run build_initial_design.py first")
    rows, X, L, phase, A, LS, OS = _load_history()
    n_init = sum(1 for p in phase if p == "initial")
    if n_init != N_INITIAL:
        raise RuntimeError(f"expected {N_INITIAL} initial rows, got {n_init}")

    # Resume from checkpoint if present
    if CP_LATEST.exists():
        st = torch.load(CP_LATEST, map_location="cpu", weights_only=False)
        if st.get("complete"):
            print(json.dumps({"status": "already_complete"}), flush=True)
            return
        X = st["X"]; L = st["L"]; phase = st["phase"]; A = st["A"]; LS = st["LS"]; OS = st["OS"]
        start = st["next_iteration"]
        torch.random.set_rng_state(st["torch_rng"])
        np.random.set_state(st["numpy_rng"])
        # Align with any BO rows appended after the checkpoint
        rows_now = list(csv.DictReader(HIST.open()))
        bo_rows_after = [r for r in rows_now if r["phase"] == "BO"][start:]
        for r in bo_rows_after:
            xr = np.array([float(r[n]) for n in NAMES])
            X = np.vstack([X, xr]); L = np.r_[L, float(r["native_LLHD"])]
            phase.append("BO"); A.append(float(r["acquisition_value"]))
            LS.append([float(r[f"{n}_lengthscale"]) for n in NAMES])
            OS.append(float(r["outputscale"]))
        start = len(bo_rows_after) + st["next_iteration"]
    else:
        torch.manual_seed(SEED); np.random.seed(SEED)
        start = 0

    obj, tgt = make_objective()
    bounds = torch.tensor([[0.0] * 6, [1.0] * 6], dtype=torch.double)

    for it in range(start, BO_ITERS):
        model, fit_label = _robust_fit(X, L, SEED + it)
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
        outs = float(model.covar_module.outputscale.detach().cpu())
        Y = -torch.log(torch.tensor(L, dtype=torch.double)).view(-1, 1)
        acq = qLogExpectedImprovement(model, best_f=Y.max())
        cand, val = optimize_acqf(acq, bounds, q=1, num_restarts=24, raw_samples=2048)
        u = cand.detach().cpu().numpy()[0]
        x = physical(u)
        av = float(val.detach().cpu().reshape(-1)[0])

        t0 = time.perf_counter()
        loss = eval_point(obj, tgt, x)
        elapsed = time.perf_counter() - t0
        stamp = datetime.now(timezone.utc).isoformat()

        X = np.vstack([X, x]); L = np.r_[L, loss]
        phase.append("BO"); A.append(av); LS.append(ls.tolist()); OS.append(outs)

        _append({
            "run": RUN,
            "evaluation_index": len(L),
            "simulator_iteration": it + 1,
            "phase": "BO",
            **dict(zip(NAMES, x)),
            **{f"normalized_{n}": u[j] for j, n in enumerate(NAMES)},
            "native_LLHD": float(loss),
            "score_log": -float(np.log(loss)),
            "Yvar_log": 1.0 / float(loss) ** 2,
            "acquisition_value": av,
            **{f"{n}_lengthscale": float(ls[j]) for j, n in enumerate(NAMES)},
            "outputscale": outs,
            "simulator_elapsed_seconds": elapsed,
            "timestamp": stamp,
            "simulator_status": "success",
            "checkpoint_path": str(CP_LATEST),
        })

        # per-iter simulator metadata
        sim_meta = SIM_DIR / f"bo_iter_{it + 1:03d}.json"
        if not sim_meta.exists():
            with sim_meta.open("x") as f:
                json.dump({"iteration": it + 1,
                            "requested": dict(zip(NAMES, map(float, x))),
                            "native_LLHD": float(loss),
                            "elapsed_seconds": float(elapsed),
                            "timestamp": stamp,
                            "fit_source": fit_label,
                            "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
                           f, indent=2); f.flush(); os.fsync(f.fileno())

        _atomic_npz(TRACE,
                    X_physical=X, X_unit=unit(X),
                    native_LLHD=L, score_log=-np.log(L), Yvar_log=1.0 / L ** 2,
                    phase=np.asarray(phase), acquisition_value=np.asarray(A),
                    lengthscales=np.asarray(LS), outputscale=np.asarray(OS),
                    run=RUN, seed=SEED)
        state = {"X": X, "L": L, "phase": phase, "A": A, "LS": LS, "OS": OS,
                 "next_iteration": it + 1,
                 "torch_rng": torch.random.get_rng_state(),
                 "numpy_rng": np.random.get_state(),
                 "seed": SEED, "run": RUN,
                 "complete": (it + 1) == BO_ITERS}
        _atomic_torch(CP_LATEST, state)
        if (it + 1) % 10 == 0:
            numbered = CP_DIR / f"checkpoint_iter_{it + 1:03d}.pt"
            if not numbered.exists():
                torch.save(state, numbered)

        print(json.dumps({"iteration": it + 1,
                          **dict(zip(NAMES, map(float, x))),
                          "native_LLHD": float(loss),
                          "running_best_LLHD": float(L.min()),
                          "acquisition_value": av, "fit_source": fit_label,
                          "elapsed_seconds": float(elapsed)}), flush=True)

    bi = int(np.argmin(L))
    summary = {
        "status": "complete", "run": RUN, "seed": SEED,
        "initial_N": N_INITIAL, "bo_iterations": BO_ITERS, "total_N": len(L),
        "best_index": bi + 1,
        "best_point": dict(zip(NAMES, map(float, X[bi]))),
        "best_native_LLHD": float(L[bi]),
        "boundary_proposals": int(sum(
            np.any((unit(x) < 1e-3) | (unit(x) > 1 - 1e-3)) for x in X[N_INITIAL:])),
        "final_lengthscales": dict(zip(NAMES, [float(v) for v in LS[-1]])),
        "final_outputscale": float(OS[-1]),
    }
    if not SUMMARY.exists():
        with SUMMARY.open("x") as f:
            json.dump(summary, f, indent=2); f.flush(); os.fsync(f.fileno())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
