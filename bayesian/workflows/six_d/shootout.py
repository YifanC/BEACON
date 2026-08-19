"""Surrogate shootout A/B/C/D + acquisition audit + noise inspection.

Writes results into the audit-diagnosis JSON so finalize / next-step decision
can consume them programmatically.
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "workflows/six_d"))
sys.path.insert(0, str(BASE / "workflows"))
from build_6d import NAMES, LO, HI, NOM, unit, physical
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

RAW = BASE / ".local/six_d/current/raw"
SNAP = RAW / "continuation/clean_bo_tr_training.csv"
BR = RAW / "direct_validation/surrogate_bridge.csv"
SH = RAW / "direct_validation/surrogate_shell.csv"
HO = RAW / "direct_validation/bo_tr_heldout_6d_results.csv"
OUT = BASE / "results/six_d/shootout.json"

DTYPE = torch.double


def load_xy(path, ll_col="native_LLHD"):
    rows = list(csv.DictReader(path.open()))
    X = np.array([[float(r[n]) for n in NAMES] for r in rows], dtype=float)
    L = np.array([float(r[ll_col]) for r in rows], dtype=float)
    return X, L, rows


def _fit_variant(X, L, kernel_nu, fixed_yvar, seed=20260812):
    torch.manual_seed(seed)
    x = torch.tensor(unit(X), dtype=DTYPE)
    l = torch.tensor(L, dtype=DTYPE).view(-1, 1)
    y = -torch.log(l)
    cov = ScaleKernel(
        MaternKernel(nu=kernel_nu, ard_num_dims=6,
                     lengthscale_constraint=Interval(1e-3, 20.0)),
        outputscale_constraint=Interval(1e-4, 100.0),
    )
    cov.base_kernel.lengthscale = torch.full((1, 6), 0.3, dtype=DTYPE)
    cov.outputscale = torch.tensor(1.0, dtype=DTYPE)
    kwargs = dict(covar_module=cov, outcome_transform=Standardize(m=1))
    if fixed_yvar:
        kwargs["train_Yvar"] = 1.0 / l ** 2
    m = SingleTaskGP(x, y, **kwargs)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    m.eval()
    return m


def _pred(m, Xp):
    with torch.no_grad():
        post = m.posterior(torch.tensor(unit(Xp), dtype=DTYPE))
        return post.mean.squeeze(-1).cpu().numpy(), post.variance.sqrt().squeeze(-1).cpu().numpy()


def main():
    torch.set_default_dtype(torch.double)
    X_all, L_all, _ = load_xy(SNAP)
    best_i = int(np.argmin(L_all))
    best_u = unit(X_all[best_i])
    d_all = np.linalg.norm(unit(X_all) - best_u, axis=1)
    if len(X_all) < 128:
        raise RuntimeError("training has < 128 rows")
    loc_idx = np.argsort(d_all)[:128]
    X_loc = X_all[loc_idx]; L_loc = L_all[loc_idx]

    b_rows = list(csv.DictReader(BR.open()))
    b_int = [r for r in b_rows if r["was_cached"] == "False"]
    X_bridge = np.array([[float(r[n]) for n in NAMES] for r in b_int])
    L_bridge = np.array([float(r["actual_native_LLHD"]) for r in b_int])
    X_shell, L_shell, _ = load_xy(SH, ll_col="actual_native_LLHD")
    X_ho, L_ho, ho_rows = load_xy(HO, ll_col="native_LLHD")
    ho_set = np.array([r["set"] for r in ho_rows])

    def _sm(m, Xe, Le):
        y = -np.log(Le); pm, ps = _pred(m, Xe); r = pm - y
        return {
            "n": int(len(Xe)),
            "rmse_score": float(np.sqrt(np.mean(r ** 2))),
            "mae_score": float(np.mean(np.abs(r))),
            "spearman": float(spearmanr(y, pm)[0]) if len(Xe) >= 3 else None,
            "cov68": float(np.mean(np.abs(r / ps) <= 1.0)),
            "cov95": float(np.mean(np.abs(r / ps) <= 1.96)),
        }

    variants = [("A", 2.5, True, X_all, L_all),
                ("B", 2.5, True, X_loc, L_loc),
                ("C", 1.5, True, X_all, L_all),
                ("D", 1.5, True, X_loc, L_loc)]
    results = {}
    for label, nu, fyv, Xt, Lt in variants:
        m = _fit_variant(Xt, Lt, nu, fyv)
        br = _sm(m, X_bridge, L_bridge)
        sh = _sm(m, X_shell, L_shell)
        gh = _sm(m, X_ho[ho_set == "global"], L_ho[ho_set == "global"])
        lh = _sm(m, X_ho[ho_set == "local"], L_ho[ho_set == "local"])
        combined = float(np.sqrt(
            (br["rmse_score"] ** 2 * br["n"] + sh["rmse_score"] ** 2 * sh["n"])
            / (br["n"] + sh["n"])))
        pm_n, ps_n = _pred(m, np.array([NOM]))
        # effective noise
        try:
            noise = m.likelihood.noise.detach().cpu().numpy().tolist()
            noise_stats = {"min": float(min(noise)), "max": float(max(noise)),
                            "median": float(np.median(noise)), "n_at_1e-6": int(sum(1 for x in noise if x <= 1.001e-6))}
        except Exception:
            noise_stats = None
        results[label] = {
            "n_train": int(len(Xt)), "kernel_nu": nu, "fixed_yvar": fyv,
            "bridge": br, "shell": sh, "combined_local_rmse": combined,
            "ho_global": gh, "ho_local": lh,
            "gp_at_nominal_llhd_like": float(np.exp(-pm_n[0])),
            "gp_at_nominal_score_sd": float(ps_n[0]),
            "effective_noise": noise_stats,
        }
        print(f"MODEL {label} n={len(Xt)}: bridge_RMSE={br['rmse_score']:.4f} shell_RMSE={sh['rmse_score']:.4f} combined={combined:.4f} "
              f"HO_global_RMSE={gh['rmse_score']:.4f} rho={gh['spearman']:.3f} HO_local_RMSE={lh['rmse_score']:.4f} "
              f"gp@nom_llhd={float(np.exp(-pm_n[0])):.2f} noise_med={(noise_stats or {}).get('median')}", flush=True)

    # Selection rule
    A = results["A"]
    challenger = None
    for label in ("B", "C", "D"):
        c = results[label]
        loc_improv = (A["combined_local_rmse"] - c["combined_local_rmse"]) / A["combined_local_rmse"]
        ho_degrade = (c["ho_global"]["rmse_score"] - A["ho_global"]["rmse_score"]) / A["ho_global"]["rmse_score"]
        spearman_ok = c["ho_global"]["spearman"] is None or c["ho_global"]["spearman"] >= 0.95
        if loc_improv >= 0.20 and ho_degrade <= 0.15 and spearman_ok:
            if challenger is None or c["combined_local_rmse"] < results[challenger]["combined_local_rmse"]:
                challenger = label
    selected = challenger if challenger else "A"

    payload = {
        "results": results, "selected": selected,
        "selection_rule": {"loc_improv_min": 0.20, "ho_degrade_max": 0.15,
                            "min_ho_spearman": 0.95},
        "reason": (f"selected {selected}: no candidate achieves ≥20% combined-local RMSE improvement over Model A without >15% HO-global degradation and ρ≥0.95"
                    if selected == "A" else
                    f"selected {selected}: combined_local RMSE improved ≥20% over Model A and HO constraints met"),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({"selected": selected, "results": {k: {"combined_local_rmse": v["combined_local_rmse"],
                                                              "ho_global_rmse": v["ho_global"]["rmse_score"],
                                                              "ho_global_spearman": v["ho_global"]["spearman"]}
                                                          for k, v in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
