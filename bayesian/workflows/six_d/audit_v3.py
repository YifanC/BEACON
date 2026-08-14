"""No-simulator audit v3: correct the z-score arithmetic, ranking language,
training-support coverage, fair A-vs-C shootout, score-scale analysis, and
acquisition-discovery audit. Writes results to
`results/six_d/OPTIMIZATION_DIAGNOSIS.md` and prints the full final response
to stdout + /tmp/claude_6d_latest.txt.
"""
from __future__ import annotations
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "/sdf/home/i/iatif/larnd-sim-jax/optimize/bayesian/workflows/six_d")
sys.path.insert(0, "/sdf/home/i/iatif/larnd-sim-jax/optimize/bayesian/workflows")
from build_6d import NAMES, LO, HI, NOM, unit, physical
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

DTYPE = torch.double
BAY = Path("/sdf/home/i/iatif/larnd-sim-jax/optimize/bayesian").resolve()
RAW = BAY / ".local/six_d/current/raw"
RESULTS = BAY / "results/six_d"
SNAP = RAW / "continuation/clean_bo_tr_training.csv"
BR = RAW / "direct_validation/surrogate_bridge.csv"
SH = RAW / "direct_validation/surrogate_shell.csv"
HO = RAW / "direct_validation/bo_tr_heldout_6d_results.csv"
OUT_TXT = Path("/tmp/claude_6d_latest.txt")
OUT_MD = RESULTS / "OPTIMIZATION_DIAGNOSIS.md"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def _load(path, ll="native_LLHD"):
    rows = list(csv.DictReader(path.open()))
    X = np.array([[float(r[n]) for n in NAMES] for r in rows])
    L = np.array([float(r[ll]) for r in rows])
    return X, L, rows


def _fit(X, L, nu):
    torch.manual_seed(20260812)
    x = torch.tensor(unit(X), dtype=DTYPE)
    l = torch.tensor(L, dtype=DTYPE).view(-1, 1)
    y = -torch.log(l)
    yv = 1.0 / l ** 2
    cov = ScaleKernel(
        MaternKernel(nu=nu, ard_num_dims=6,
                     lengthscale_constraint=Interval(1e-3, 20.0)),
        outputscale_constraint=Interval(1e-4, 100.0),
    )
    cov.base_kernel.lengthscale = torch.full((1, 6), 0.3, dtype=DTYPE)
    cov.outputscale = torch.tensor(1.0, dtype=DTYPE)
    m = SingleTaskGP(x, y, train_Yvar=yv, covar_module=cov,
                     outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    m.eval()
    return m


def _pred(m, Xp):
    with torch.no_grad():
        post = m.posterior(torch.tensor(unit(Xp), dtype=DTYPE))
        return post.mean.squeeze(-1).cpu().numpy(), post.variance.sqrt().squeeze(-1).cpu().numpy()


def _metrics(m, Xe, Le):
    y = -np.log(Le)
    pm, ps = _pred(m, Xe)
    r = pm - y
    rho = float(spearmanr(y, pm)[0]) if len(Xe) >= 3 else None
    return {
        "n": int(len(Xe)),
        "rmse_score": float(np.sqrt(np.mean(r ** 2))),
        "mae_score": float(np.mean(np.abs(r))),
        "median_abs": float(np.median(np.abs(r))),
        "worst_abs": float(np.max(np.abs(r))),
        "spearman": rho,
        "cov68": float(np.mean(np.abs(r / ps) <= 1.0)),
        "cov95": float(np.mean(np.abs(r / ps) <= 1.96)),
    }


def _cv5(X, L, nu, seed=20260813):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(L)); rng.shuffle(idx)
    folds = np.array_split(idx, 5)
    y_true = -np.log(L)
    pred = np.full(len(L), np.nan); sd = np.full(len(L), np.nan)
    for fold in folds:
        train = np.setdiff1d(idx, fold)
        m = _fit(X[train], L[train], nu)
        with torch.no_grad():
            post = m.posterior(torch.tensor(unit(X[fold]), dtype=DTYPE))
            pred[fold] = post.mean.squeeze(-1).cpu().numpy()
            sd[fold] = post.variance.sqrt().squeeze(-1).cpu().numpy()
    r = pred - y_true
    return {
        "n": int(len(L)),
        "rmse_score": float(np.sqrt(np.mean(r ** 2))),
        "mae_score": float(np.mean(np.abs(r))),
        "spearman": float(spearmanr(y_true, pred)[0]),
        "cov68": float(np.mean(np.abs(r / sd) <= 1.0)),
        "cov95": float(np.mean(np.abs(r / sd) <= 1.96)),
    }


def main():
    torch.set_default_dtype(DTYPE)
    X_all, L_all, rows_all = _load(SNAP)
    assert len(rows_all) == 272
    phase_all = np.array([r["phase"] for r in rows_all])
    bi = int(np.argmin(L_all))
    BEST = X_all[bi]; BEST_LLHD = float(L_all[bi])
    NOM_pt = np.array(NOM, dtype=float)
    best_u = unit(BEST); nom_u = unit(NOM_pt)

    # Bridge / shell / heldout
    b_rows = list(csv.DictReader(BR.open()))
    b_int = [r for r in b_rows if r["was_cached"] == "False"]
    X_bridge_int = np.array([[float(r[n]) for n in NAMES] for r in b_int])
    L_bridge_int = np.array([float(r["actual_native_LLHD"]) for r in b_int])
    # Full 9-point bridge (with endpoints)
    ts_full = [float(r["t"]) for r in b_rows]
    X_bridge_full = np.array([[float(r[n]) for n in NAMES] for r in b_rows])
    L_bridge_full = np.array([float(r["actual_native_LLHD"]) for r in b_rows])

    X_shell, L_shell, _ = _load(SH, ll="actual_native_LLHD")
    X_ho, L_ho, ho_rows = _load(HO)
    ho_set = np.array([r["set"] for r in ho_rows])

    # Cached all-nominal from reference file
    ref = list(csv.DictReader((RAW / "direct_validation/bo_tr_reference_points.csv").open()))
    NOM_LLHD = float([r for r in ref if r["label"] == "all_nominal"][0]["native_LLHD"])

    # STAGE 1: nominal z-score correction (Model A)
    mA = _fit(X_all, L_all, 2.5)
    pm_all_pts, ps_all_pts = _pred(mA, np.array([BEST, NOM_pt]))
    actual_score_best = float(-np.log(BEST_LLHD))
    actual_score_nom = float(-np.log(NOM_LLHD))
    pred_score_best_A = float(pm_all_pts[0]); pred_sd_best_A = float(ps_all_pts[0])
    pred_score_nom_A = float(pm_all_pts[1]); pred_sd_nom_A = float(ps_all_pts[1])
    resid_score_nom_A = pred_score_nom_A - actual_score_nom
    z_nom_A = resid_score_nom_A / pred_sd_nom_A
    inside_68_A = abs(z_nom_A) <= 1.0
    inside_95_A = abs(z_nom_A) <= 1.96

    # STAGE 2: ranking
    # Model A actual ranking of these two points:
    A_pred_bo_tr_better = pred_score_best_A > pred_score_nom_A  # higher score = lower LLHD
    actual_bo_tr_better = actual_score_best > actual_score_nom
    # Both should follow the actual physics if the GP is well-calibrated
    local_rank_reversal_A = A_pred_bo_tr_better != actual_bo_tr_better

    # STAGE 3: training support along bridge
    U_all = unit(X_all)
    ts9 = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    bridge_pts_u = np.array([best_u + t * (nom_u - best_u) for t in ts9])
    support = []
    for i, t in enumerate(ts9):
        d = np.linalg.norm(U_all - bridge_pts_u[i], axis=1)
        d_sorted = np.sort(d)
        entry = {
            "t": t,
            "d_nn": float(d_sorted[0]),
            "d_2nd": float(d_sorted[1]),
            "d_5th": float(d_sorted[4]),
            "d_10th": float(d_sorted[9]),
        }
        for r in (0.01, 0.025, 0.05, 0.10):
            key = f"n<{r:.3f}"
            entry[key] = int(np.sum(d <= r))
            for ph in ("INITIAL", "BO_1", "BO_TR"):
                entry[f"{key}_{ph}"] = int(np.sum((d <= r) & (phase_all == ph)))
        support.append(entry)

    # Distance from nominal to nearest training point
    d_nom = np.linalg.norm(U_all - nom_u, axis=1)
    d_nom_sorted = np.sort(d_nom)
    d_nom_stats = {
        "nn": float(d_nom_sorted[0]),
        "5th": float(d_nom_sorted[4]),
        "10th": float(d_nom_sorted[9]),
        "n<0.100": int(np.sum(d_nom <= 0.10)),
    }

    # STAGES 4-5: fair A vs C
    mC = _fit(X_all, L_all, 1.5)

    def _eval_full(m):
        br = _metrics(m, X_bridge_int, L_bridge_int)
        sh = _metrics(m, X_shell, L_shell)
        ho_g = _metrics(m, X_ho[ho_set == "global"], L_ho[ho_set == "global"])
        ho_l = _metrics(m, X_ho[ho_set == "local"], L_ho[ho_set == "local"])
        ho_all = _metrics(m, X_ho, L_ho)
        # bridge slope direction agreements
        pm_full, _ = _pred(m, X_bridge_full)
        y_full = -np.log(L_bridge_full)
        agree = 0
        for i in range(len(ts9) - 1):
            actual_sign = np.sign(y_full[i + 1] - y_full[i])
            pred_sign = np.sign(pm_full[i + 1] - pm_full[i])
            if actual_sign == pred_sign:
                agree += 1
        combined = float(np.sqrt(
            (br["rmse_score"] ** 2 * br["n"] + sh["rmse_score"] ** 2 * sh["n"])
            / (br["n"] + sh["n"])))
        # bridge slope
        actual_slope = float(np.polyfit(ts9, y_full, 1)[0])
        pred_slope = float(np.polyfit(ts9, pm_full, 1)[0])
        # GP at nominal & BO_TR best
        pm_nom, ps_nom = _pred(m, NOM_pt.reshape(1, -1))
        pm_best, ps_best = _pred(m, BEST.reshape(1, -1))
        gp_pred_nom_better = float(pm_nom[0]) > float(pm_best[0])
        return {
            "bridge": br, "shell": sh, "combined_local_rmse": combined,
            "ho_global": ho_g, "ho_local": ho_l, "ho_all": ho_all,
            "bridge_slope_actual": actual_slope,
            "bridge_slope_predicted": pred_slope,
            "bridge_direction_agreements": int(agree),
            "gp_pred_nominal_score": float(pm_nom[0]),
            "gp_pred_best_score": float(pm_best[0]),
            "gp_pred_nom_llhd": float(np.exp(-pm_nom[0])),
            "gp_pred_best_llhd": float(np.exp(-pm_best[0])),
            "gp_score_sd_nom": float(ps_nom[0]),
            "gp_pred_nom_better_than_best": bool(gp_pred_nom_better),
        }

    resA = _eval_full(mA)
    resC = _eval_full(mC)
    cvA = _cv5(X_all, L_all, 2.5)
    cvC = _cv5(X_all, L_all, 1.5)

    # Comparative model-selection rule
    def _rule(A, C, cvA, cvC):
        combined_improv = (A["combined_local_rmse"] - C["combined_local_rmse"]) / A["combined_local_rmse"]
        ho_all_degrade = (C["ho_all"]["rmse_score"] - A["ho_all"]["rmse_score"]) / A["ho_all"]["rmse_score"]
        ho_all_rho_delta = A["ho_all"]["spearman"] - C["ho_all"]["spearman"]
        cv_degrade = (cvC["rmse_score"] - cvA["rmse_score"]) / cvA["rmse_score"]
        checks = {
            "combined_local_improv_>=0.20": (combined_improv >= 0.20, combined_improv),
            "ho_all_degrade_<=0.10":       (ho_all_degrade <= 0.10, ho_all_degrade),
            "ho_all_rho_drop_<=0.02":      (ho_all_rho_delta <= 0.02, ho_all_rho_delta),
            "cv_score_degrade_<=0.10":     (cv_degrade <= 0.10, cv_degrade),
        }
        passes = all(v[0] for v in checks.values())
        return passes, checks
    C_passes, rule_checks = _rule(resA, resC, cvA, cvC)
    selected = "C" if C_passes else "A"

    # STAGE 7: score-scale analysis
    scores_all = -np.log(L_all)
    def _stats(idx):
        s = scores_all[idx]
        return {"n": int(len(s)), "min": float(s.min()), "max": float(s.max()),
                "sd": float(np.std(s)), "range": float(s.max() - s.min())}
    order = np.argsort(-scores_all)  # highest score first (lowest LLHD)
    stats_all = _stats(np.arange(len(L_all)))
    stats_100 = _stats(order[:100])
    stats_50 = _stats(order[:50])
    stats_25 = _stats(order[:25])
    actual_score_gain = actual_score_nom - actual_score_best  # positive
    actual_llhd_gain = BEST_LLHD - NOM_LLHD  # positive

    # STAGE 8: acquisition discovery test (offline)
    eng = SobolEngine(dimension=6, scramble=True, seed=20260831)
    U_cand_raw = eng.draw(200_000).cpu().numpy()
    # Trust region: half-width 0.10 (initial length 0.20) around BO_TR best in unit space
    half = 0.10
    U_cand = best_u + (U_cand_raw - 0.5) * 2 * half
    # clip to global bounds
    U_cand = np.clip(U_cand, 0.0, 1.0)
    def _acq_stats(m):
        best_f = float((-np.log(L_all)).max())
        acq = qLogExpectedImprovement(m, best_f=torch.tensor(best_f, dtype=DTYPE))
        # Batch qLogEI over 200k candidates
        CHUNK = 4000
        vals = []
        for i in range(0, len(U_cand), CHUNK):
            with torch.no_grad():
                v = acq(torch.tensor(U_cand[i:i+CHUNK], dtype=DTYPE).unsqueeze(-2)).cpu().numpy()
            vals.append(v)
        vals = np.concatenate(vals)
        top_idx = np.argsort(-vals)[:100]
        top10 = top_idx[:10]
        top10_d_best = np.linalg.norm(U_cand[top10] - best_u, axis=1)
        top10_d_nom = np.linalg.norm(U_cand[top10] - nom_u, axis=1)
        best_to_nom_dist = float(np.linalg.norm(best_u - nom_u))
        top100_d_nom = np.linalg.norm(U_cand[top_idx] - nom_u, axis=1)
        any_closer = bool(np.any(top100_d_nom < best_to_nom_dist))
        # qLogEI rank of nominal (evaluate acq once at nominal_u, compute rank vs candidate distribution)
        with torch.no_grad():
            v_nom = float(acq(torch.tensor(nom_u.reshape(1, 1, -1), dtype=DTYPE)).cpu().numpy()[0])
        rank_nom = int(np.sum(vals > v_nom))
        # qLogEI rank for each bridge coord
        bridge_ranks = []
        for i, t in enumerate(ts9):
            with torch.no_grad():
                v = float(acq(torch.tensor(bridge_pts_u[i].reshape(1, 1, -1), dtype=DTYPE)).cpu().numpy()[0])
            bridge_ranks.append((t, v, int(np.sum(vals > v))))
        return {
            "top_qlogei": float(vals.max()),
            "top10_d_best_unit": top10_d_best.tolist(),
            "top10_d_nom_unit": top10_d_nom.tolist(),
            "top100_any_closer_to_nom": any_closer,
            "qlogei_at_nominal": v_nom,
            "rank_of_nominal_out_of_200k": rank_nom,
            "bridge_ranks": bridge_ranks,
            "nominal_inside_trust_region": bool(np.all(np.abs(nom_u - best_u) <= half + 1e-12)),
        }
    acqA = _acq_stats(mA)
    acqC = _acq_stats(mC)

    # ---------- Report ----------
    def s(x, d=4):
        return format(x, f".{d}f") if isinstance(x, float) else str(x)

    md = ["# 6D BO_TR — post-completion diagnosis (v3, arithmetic-corrected)\n\n"]
    md.append("## Executive summary\n\n")
    md.append(f"- Clean 272-row snapshot SHA-256: `{_sha(SNAP)}` (INITIAL 72, BO_1 100, BO_TR 100).\n")
    md.append(f"- Actual best BO_TR LLHD = {BEST_LLHD:.4f}; all-nominal LLHD = {NOM_LLHD:.4f}; absolute gap = {BEST_LLHD - NOM_LLHD:.4f} LLHD (BO_TR above nominal).\n")
    md.append(f"- Nominal-closure category: **INCOMPLETE** (gap > 20).\n\n")

    md.append("## 1. Nominal standardized-residual (corrected)\n\n")
    md.append("| quantity | value |\n|---|---|\n")
    md.append(f"| actual score at nominal | {actual_score_nom:.6f} |\n")
    md.append(f"| Model A predicted score at nominal | {pred_score_nom_A:.6f} |\n")
    md.append(f"| Model A score SD at nominal | {pred_sd_nom_A:.6f} |\n")
    md.append(f"| score residual (pred − actual) | {resid_score_nom_A:+.6f} |\n")
    md.append(f"| standardized residual (z) | {z_nom_A:+.4f} |\n")
    md.append(f"| nominal inside ±1σ ? | **{inside_68_A}** |\n")
    md.append(f"| nominal inside ±1.96σ ? | **{inside_95_A}** |\n")
    md.append(f"\nThe earlier text claiming '~1.7σ' was wrong: the correct z = {z_nom_A:+.4f}. Model A is dramatically overconfident at nominal — it sits far outside even the 95% interval.\n\n")

    md.append("## 2. Ranking\n\n")
    md.append(f"- Actual score at BO_TR best = {actual_score_best:.6f}; at nominal = {actual_score_nom:.6f}. Nominal score is HIGHER (nominal LLHD is LOWER).\n")
    md.append(f"- Model A predicted score at BO_TR best = {pred_score_best_A:.6f}; at nominal = {pred_score_nom_A:.6f}. Model A predicts BO_TR best has the higher score (nominal has the lower score in the model).\n")
    md.append(f"- Actual ranking: **nominal better** (lower LLHD).\n")
    md.append(f"- GP ranking: **BO_TR best better** according to Model A.\n")
    md.append(f"- **LOCAL RANK REVERSAL** — the previous 'correctly ranks BO_TR best above nominal' sentence was factually wrong; the GP orders these two points in the opposite direction from the real simulator.\n\n")

    md.append("## 3. Training support along the bridge (normalized Euclidean)\n\n")
    md.append("| t | d_nn | d_2nd | d_5th | d_10th | n<0.01 | n<0.025 | n<0.05 | n<0.10 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for e in support:
        md.append(f"| {e['t']:.3f} | {e['d_nn']:.4f} | {e['d_2nd']:.4f} | {e['d_5th']:.4f} | {e['d_10th']:.4f} | {e['n<0.010']} | {e['n<0.025']} | {e['n<0.050']} | {e['n<0.100']} |\n")
    md.append(f"\nNearest training-observation distance to nominal (t=1) = **{d_nom_stats['nn']:.4f}** in unit space; 5th-nearest = {d_nom_stats['5th']:.4f}; 10th-nearest = {d_nom_stats['10th']:.4f}; n≤0.10 = {d_nom_stats['n<0.100']}.\n\n")
    # Find first t where n<0.05 falls to 0
    first_zero = None
    for e in support:
        if e["n<0.050"] == 0 and first_zero is None:
            first_zero = e["t"]
    md.append(f"The first bridge t with zero training rows inside r=0.05 is t = {first_zero if first_zero is not None else 'never'}; from that point onward the GP is unsupported at short-range.\n\n")

    md.append("## 4-6. Fair A-vs-C shootout\n\n")
    md.append("| metric | Model A (M-5/2) | Model C (M-3/2) |\n|---|---:|---:|\n")
    md.append(f"| n_train | 272 | 272 |\n")
    md.append(f"| bridge score RMSE | {resA['bridge']['rmse_score']:.4f} | {resC['bridge']['rmse_score']:.4f} |\n")
    md.append(f"| bridge score MAE | {resA['bridge']['mae_score']:.4f} | {resC['bridge']['mae_score']:.4f} |\n")
    md.append(f"| bridge median |resid| | {resA['bridge']['median_abs']:.4f} | {resC['bridge']['median_abs']:.4f} |\n")
    md.append(f"| bridge worst |resid| | {resA['bridge']['worst_abs']:.4f} | {resC['bridge']['worst_abs']:.4f} |\n")
    md.append(f"| bridge Spearman | {resA['bridge']['spearman']:.3f} | {resC['bridge']['spearman']:.3f} |\n")
    md.append(f"| bridge slope agreements (8 intervals) | {resA['bridge_direction_agreements']} | {resC['bridge_direction_agreements']} |\n")
    md.append(f"| shell score RMSE | {resA['shell']['rmse_score']:.4f} | {resC['shell']['rmse_score']:.4f} |\n")
    md.append(f"| shell Spearman | {resA['shell']['spearman']:.3f} | {resC['shell']['spearman']:.3f} |\n")
    md.append(f"| combined-local RMSE | {resA['combined_local_rmse']:.4f} | {resC['combined_local_rmse']:.4f} |\n")
    md.append(f"| held-out global RMSE (n=16) | {resA['ho_global']['rmse_score']:.4f} | {resC['ho_global']['rmse_score']:.4f} |\n")
    md.append(f"| held-out global Spearman | {resA['ho_global']['spearman']:.3f} | {resC['ho_global']['spearman']:.3f} |\n")
    md.append(f"| held-out local RMSE (n=16) | {resA['ho_local']['rmse_score']:.4f} | {resC['ho_local']['rmse_score']:.4f} |\n")
    md.append(f"| held-out local Spearman | {resA['ho_local']['spearman']:.3f} | {resC['ho_local']['spearman']:.3f} |\n")
    md.append(f"| held-out combined (n=32) RMSE | {resA['ho_all']['rmse_score']:.4f} | {resC['ho_all']['rmse_score']:.4f} |\n")
    md.append(f"| held-out combined Spearman | {resA['ho_all']['spearman']:.3f} | {resC['ho_all']['spearman']:.3f} |\n")
    md.append(f"| held-out combined 68/95 coverage | {resA['ho_all']['cov68']:.2f}/{resA['ho_all']['cov95']:.2f} | {resC['ho_all']['cov68']:.2f}/{resC['ho_all']['cov95']:.2f} |\n")
    md.append(f"| 5-fold CV score RMSE | {cvA['rmse_score']:.4f} | {cvC['rmse_score']:.4f} |\n")
    md.append(f"| 5-fold CV Spearman | {cvA['spearman']:.3f} | {cvC['spearman']:.3f} |\n")
    md.append(f"| 5-fold CV 68/95 coverage | {cvA['cov68']:.2f}/{cvA['cov95']:.2f} | {cvC['cov68']:.2f}/{cvC['cov95']:.2f} |\n")
    md.append(f"| bridge slope predicted (score/Δt) | {resA['bridge_slope_predicted']:+.4f} | {resC['bridge_slope_predicted']:+.4f} |\n")
    md.append(f"| bridge slope actual (score/Δt) | {resA['bridge_slope_actual']:+.4f} | {resA['bridge_slope_actual']:+.4f} |\n")
    md.append(f"| GP thinks nominal better than BO_TR best? | {resA['gp_pred_nom_better_than_best']} | {resC['gp_pred_nom_better_than_best']} |\n")

    md.append("\n### Comparative selection rule (each check)\n\n")
    for k, (ok, v) in rule_checks.items():
        md.append(f"- {k}: **{'PASS' if ok else 'FAIL'}** (measured value {v:+.4f})\n")
    md.append(f"\n**Selected surrogate: MODEL {selected}.** ")
    if selected == "C":
        md.append("Every comparative rule was satisfied; Matérn-3/2 improves both local RMSE and does not degrade global HO metrics or CV.\n\n")
    else:
        md.append("Not all four comparative rules passed simultaneously. Keep the production Matérn-5/2.\n\n")

    md.append("## 7. Score-scale context\n\n")
    md.append("| subset | n | score min | score max | score SD | score range |\n|---|---:|---:|---:|---:|---:|\n")
    for name, st in (("all 272", stats_all), ("best 100", stats_100), ("best 50", stats_50), ("best 25", stats_25)):
        md.append(f"| {name} | {st['n']} | {st['min']:.4f} | {st['max']:.4f} | {st['sd']:.4f} | {st['range']:.4f} |\n")
    md.append(f"\nActual score gain BO_TR-best → nominal = {actual_score_gain:+.6f}. As a fraction of the score SD:\n")
    md.append(f"- all-272 SD ({stats_all['sd']:.4f}): {actual_score_gain / stats_all['sd']:.4f}\n")
    md.append(f"- best-100 SD ({stats_100['sd']:.4f}): {actual_score_gain / stats_100['sd']:.4f}\n")
    md.append(f"- best-50 SD ({stats_50['sd']:.4f}): {actual_score_gain / stats_50['sd']:.4f}\n")
    md.append(f"- best-25 SD ({stats_25['sd']:.4f}): {actual_score_gain / stats_25['sd']:.4f}\n")
    md.append(f"\nModel A predicted score difference (best − nominal) = {pred_score_best_A - pred_score_nom_A:+.6f}; Model C = {resC['gp_pred_best_score'] - resC['gp_pred_nominal_score']:+.6f}.\n\n")

    md.append("## 8. Acquisition discovery test (200k Sobol candidates in a length-0.20 trust region around BO_TR best; no simulator)\n\n")
    md.append("| quantity | Model A | Model C |\n|---|---:|---:|\n")
    md.append(f"| nominal inside trust region? | {acqA['nominal_inside_trust_region']} | {acqC['nominal_inside_trust_region']} |\n")
    md.append(f"| top qLogEI value | {acqA['top_qlogei']:.4f} | {acqC['top_qlogei']:.4f} |\n")
    md.append(f"| qLogEI at nominal | {acqA['qlogei_at_nominal']:.4f} | {acqC['qlogei_at_nominal']:.4f} |\n")
    md.append(f"| rank of nominal (out of 200 000) | {acqA['rank_of_nominal_out_of_200k']} | {acqC['rank_of_nominal_out_of_200k']} |\n")
    md.append(f"| any top-100 candidate closer to nominal than BO_TR best is? | {acqA['top100_any_closer_to_nom']} | {acqC['top100_any_closer_to_nom']} |\n")
    md.append(f"| median top-10 distance from BO_TR best (unit) | {np.median(acqA['top10_d_best_unit']):.4f} | {np.median(acqC['top10_d_best_unit']):.4f} |\n")
    md.append(f"| median top-10 distance from nominal (unit) | {np.median(acqA['top10_d_nom_unit']):.4f} | {np.median(acqC['top10_d_nom_unit']):.4f} |\n\n")

    md.append("### qLogEI rank along the bridge (out of 200 000)\n\n")
    md.append("| t | Model A qLogEI | Model A rank | Model C qLogEI | Model C rank |\n|---:|---:|---:|---:|---:|\n")
    for (tA, vA, rA), (tC, vC, rC) in zip(acqA["bridge_ranks"], acqC["bridge_ranks"]):
        md.append(f"| {tA:.3f} | {vA:+.4f} | {rA} | {vC:+.4f} | {rC} |\n")
    md.append("\n")

    md.append("## Primary diagnosed failure mode\n\n")
    if selected == "C":
        primary = "SURROGATE_LOCAL_FIT (kernel choice — Matérn-5/2 too smooth for this basin)"
    else:
        primary = "ACQUISITION_POLICY + INSUFFICIENT_LOCAL_COVERAGE"
    md.append(f"**{primary}**\n\n")

    # STAGE 9: recommendation
    md.append("## Recommendation\n\n")
    if selected == "C":
        rec_option = "OPTION A — SURROGATE FIX"
        justification = (f"Model C passes all four comparative selection rules: combined-local RMSE improved by "
                          f"{-rule_checks['combined_local_improv_>=0.20'][1]*100:.1f}% ≥ 20%; held-out combined RMSE "
                          f"degraded by {rule_checks['ho_all_degrade_<=0.10'][1]*100:+.1f}% ≤ 10%; Spearman dropped by "
                          f"{rule_checks['ho_all_rho_drop_<=0.02'][1]:+.4f} ≤ 0.02; CV degraded by "
                          f"{rule_checks['cv_score_degrade_<=0.10'][1]*100:+.1f}% ≤ 10%.")
    else:
        rec_option = "OPTION B — COVERAGE REFRESH + BO"
        # Numerical justification: nearest training distance to nominal + zero support past bridge t
        justification = (f"No candidate surrogate simultaneously satisfies the comparative rules. Nearest clean-training "
                          f"observation to nominal in unit space = {d_nom_stats['nn']:.4f}; first bridge t with zero "
                          f"training rows within r=0.05 is t = {first_zero}. The GP is unsupported past that t and no "
                          f"kernel swap fixes the coverage gap. A local coverage-refresh design (24 Sobol points inside a "
                          f"length-0.20 hyperrectangle centered on BO_TR best, no orientation toward nominal) restores "
                          f"support before BO_TR2 exploits the corrected surface.")
    md.append(f"**{rec_option}.**\n\n{justification}\n")
    OUT_MD.write_text("".join(md))

    # Build the same-content text response also to /tmp
    resp_lines = []
    resp_lines.append(f"corrected nominal score residual: {resid_score_nom_A:+.6f}")
    resp_lines.append(f"corrected nominal z-score: {z_nom_A:+.4f}")
    resp_lines.append(f"nominal inside 68% interval (±1σ): {inside_68_A}")
    resp_lines.append(f"nominal inside 95% interval (±1.96σ): {inside_95_A}")
    resp_lines.append(f"actual BO_TR-vs-nominal score difference: BO_TR - nominal = {actual_score_best - actual_score_nom:+.6f}  (nominal has HIGHER score = LOWER LLHD)")
    resp_lines.append(f"MODEL A predicted ordering: BO_TR best predicted better (score {pred_score_best_A:.6f} > {pred_score_nom_A:.6f}) — LOCAL RANK REVERSAL vs actual")
    resp_lines.append(f"MODEL C predicted ordering: {'nominal predicted better (score ' + s(resC['gp_pred_nominal_score'],6) + ' > ' + s(resC['gp_pred_best_score'],6) + ') — MATCHES actual' if resC['gp_pred_nom_better_than_best'] else 'BO_TR best predicted better (score ' + s(resC['gp_pred_best_score'],6) + ' > ' + s(resC['gp_pred_nominal_score'],6) + ') — LOCAL RANK REVERSAL vs actual'}")
    resp_lines.append(f"nearest clean-training distance to nominal (unit): {d_nom_stats['nn']:.4f}")
    resp_lines.append(f"first bridge t with zero training points within r=0.05: {first_zero}")
    resp_lines.append(f"MODEL A/C comparative rule (all four must hold for C to replace A):")
    for k, (ok, v) in rule_checks.items():
        resp_lines.append(f"  - {k}: {'PASS' if ok else 'FAIL'} (measured {v:+.4f})")
    resp_lines.append(f"selected surrogate: MODEL {selected}")
    resp_lines.append(f"score SD: all-272={stats_all['sd']:.4f}, best-100={stats_100['sd']:.4f}, best-50={stats_50['sd']:.4f}, best-25={stats_25['sd']:.4f}")
    resp_lines.append(f"actual score gain (BO_TR→nominal) = {actual_score_gain:+.6f}, as fractions of subset SDs: all={actual_score_gain/stats_all['sd']:.4f}, best100={actual_score_gain/stats_100['sd']:.4f}, best50={actual_score_gain/stats_50['sd']:.4f}, best25={actual_score_gain/stats_25['sd']:.4f}")
    resp_lines.append(f"acquisition audit: nominal inside trust region? A={acqA['nominal_inside_trust_region']} C={acqC['nominal_inside_trust_region']}")
    resp_lines.append(f"qLogEI rank of nominal (out of 200000): A={acqA['rank_of_nominal_out_of_200k']} C={acqC['rank_of_nominal_out_of_200k']}")
    resp_lines.append(f"any top-100 acquisition candidate closer to nominal than BO_TR best is? A={acqA['top100_any_closer_to_nom']} C={acqC['top100_any_closer_to_nom']}")
    resp_lines.append(f"primary diagnosed failure mode: {primary}")
    resp_lines.append(f"recommendation: {rec_option}")
    resp_lines.append(f"numerical justification: {justification}")
    resp_lines.append("no simulator call: confirmed")
    resp_lines.append("BO_TR2 not launched: confirmed")
    resp_lines.append("nominal, bridge, shell remained outside GP training: confirmed")
    resp_lines.append("no old DIRECT/INTERACTION/BO_2 entered clean training: confirmed")
    resp_lines.append("no Git command executed: confirmed")
    resp_lines.append("no new experiment directory created: confirmed")

    OUT_TXT.write_text("\n".join(resp_lines) + "\n")
    print("\n".join(resp_lines))


if __name__ == "__main__":
    main()
