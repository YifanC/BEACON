"""Two-parameter Bayesian optimization on the 999.93-cm safe dataset.

One reusable runner used by all three pairs (Ab+kb, eField+lifetime, diffusion).
The pair is selected by `--config workflows/two_d/configs/<name>.yaml`.

Writes into `.local/two_d/<pair>/`:
  history.csv, trace_latest.npz, checkpoint_latest.pt,
  checkpoint_iter_NNN.pt (every 10 iters + final),
  simulator/{init_NNN,bo_iter_NNN}.json,
  summary.json

Reuses the shared target at `.local/two_d/target.npz` when present; otherwise
the caller must supply it separately (target generation is not part of this
runner).
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    REPO, BAY, DTYPE, fit_gp, propose_next,
    make_llhd_objective, unit_of, physical_of,
)


def _atomic_write(path: Path, blob):
    q = path.with_suffix(path.suffix + ".part")
    torch.save(blob, q); os.replace(q, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    LO = np.array(cfg["lower"], dtype=float)
    HI = np.array(cfg["upper"], dtype=float)
    NAMES = list(cfg["tunable_params"])
    if len(NAMES) != 2:
        raise SystemExit(f"two_d requires 2 tunable params, got {NAMES}")

    outdir = BAY / f".local/two_d/{cfg['pair']}"
    (outdir / "simulator").mkdir(parents=True, exist_ok=True)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    target_npz = BAY / ".local/two_d/target.npz"
    if not target_npz.exists():
        raise SystemExit(f"expected shared 2D target at {target_npz}")

    obj, load_target = make_llhd_objective(
        n_events=cfg["n_events"], tunable_params=NAMES,
    )
    targets = load_target(str(target_npz))

    # Sobol init
    eng = torch.quasirandom.SobolEngine(dimension=2, scramble=True,
                                        seed=cfg["sobol_seed"])
    U = eng.draw(cfg["sobol_n"]).double().numpy()
    Xp = np.column_stack([LO[0] + U[:, 0] * (HI[0] - LO[0]),
                          LO[1] + U[:, 1] * (HI[1] - LO[1])])

    hist_path = outdir / "history.csv"
    fields = ["phase", "iteration", *NAMES,
              *[f"u_{n}" for n in NAMES],
              "native_LLHD", "acquisition_value", "elapsed_seconds", "timestamp"]

    def append(row):
        new = not hist_path.exists()
        with hist_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new: w.writeheader()
            w.writerow(row); f.flush(); os.fsync(f.fileno())

    X, L = [], []
    for i, (u, xp) in enumerate(zip(U, Xp), 1):
        t = time.perf_counter()
        llhd = float(obj.evaluate(dict(zip(NAMES, [float(xp[0]), float(xp[1])])), targets))
        elapsed = time.perf_counter() - t
        stamp = datetime.now(timezone.utc).isoformat()
        X.append(xp.tolist()); L.append(llhd)
        row = {"phase": "INITIAL", "iteration": i,
               NAMES[0]: float(xp[0]), NAMES[1]: float(xp[1]),
               f"u_{NAMES[0]}": float(u[0]), f"u_{NAMES[1]}": float(u[1]),
               "native_LLHD": llhd, "acquisition_value": "NA",
               "elapsed_seconds": elapsed, "timestamp": stamp}
        append(row)
        with (outdir / "simulator" / f"init_{i:03d}.json").open("x") as f:
            json.dump({"i": i, **{NAMES[j]: float(xp[j]) for j in (0, 1)},
                       "native_LLHD": llhd, "elapsed_seconds": elapsed,
                       "timestamp": stamp,
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, f, indent=2)
        print(json.dumps({"phase": "INITIAL", "i": i, "native_LLHD": llhd}), flush=True)

    X = np.asarray(X); L = np.asarray(L)
    torch.manual_seed(cfg["bo_seed"]); np.random.seed(cfg["bo_seed"])
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DTYPE)

    for it in range(1, cfg["bo_iters"] + 1):
        Xu = unit_of(X, LO, HI)
        model = fit_gp(Xu, L, seed=cfg["bo_seed"] + it, ard_num_dims=2)
        u_next, av = propose_next(model, L, bounds)
        xp = physical_of(u_next, LO, HI)
        t = time.perf_counter()
        llhd = float(obj.evaluate(dict(zip(NAMES, [float(xp[0]), float(xp[1])])), targets))
        elapsed = time.perf_counter() - t
        stamp = datetime.now(timezone.utc).isoformat()
        X = np.vstack([X, xp]); L = np.r_[L, llhd]
        row = {"phase": "BO", "iteration": cfg["sobol_n"] + it,
               NAMES[0]: float(xp[0]), NAMES[1]: float(xp[1]),
               f"u_{NAMES[0]}": float(u_next[0]), f"u_{NAMES[1]}": float(u_next[1]),
               "native_LLHD": llhd, "acquisition_value": av,
               "elapsed_seconds": elapsed, "timestamp": stamp}
        append(row)
        with (outdir / "simulator" / f"bo_iter_{it:03d}.json").open("x") as f:
            json.dump({"iteration": it, "requested": {NAMES[j]: float(xp[j]) for j in (0, 1)},
                       "native_LLHD": llhd, "acquisition_value": av,
                       "elapsed_seconds": elapsed, "timestamp": stamp,
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, f, indent=2)

        state = {"X": X, "L": L, "next_iteration": it, "seed": cfg["bo_seed"],
                 "torch_rng": torch.random.get_rng_state(),
                 "numpy_rng": np.random.get_state()}
        _atomic_write(outdir / "checkpoints/checkpoint_latest.pt", state)
        if it % 10 == 0 or it == cfg["bo_iters"]:
            numbered = outdir / f"checkpoints/checkpoint_iter_{it:03d}.pt"
            if not numbered.exists():
                torch.save(state, numbered)
        # trace
        np.savez_compressed(outdir / "trace_latest.npz",
                            X_physical=X, native_LLHD=L)
        print(json.dumps({"phase": "BO", "iter": it, "native_LLHD": llhd,
                          "running_best": float(L.min())}), flush=True)

    bi = int(np.argmin(L))
    summary = {"pair": cfg["pair"],
               "sobol_n": cfg["sobol_n"], "bo_iters": cfg["bo_iters"],
               "best_index": bi + 1,
               "best_point": {NAMES[j]: float(X[bi, j]) for j in (0, 1)},
               "best_native_LLHD": float(L[bi])}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
