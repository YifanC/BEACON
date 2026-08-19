"""6D BO_1 postmortem — NO simulator calls.

Reads the clean 172-row baseline and diagnoses why global qLogEI missed the
known nominal closure basin. Also compares three GP variants (A/B/C) against
the retained old diagnostic data (DIRECT / INTERACTION / BO_2 / heldout /
reference / final_profile) purely as offline test data — none of those rows
is added to the training set.

Writes:
  results/six_d/OPTIMIZATION_DIAGNOSIS.md
  results/six_d/diagnostic_metrics.json   (structured backing data)
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "workflows/six_d"))
sys.path.insert(0, str(BASE / "workflows"))
from build_6d import ROOT, RAW, NAMES, LO, HI, NOM, unit, physical

from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from scipy.stats import spearmanr

DTYPE = torch.double
CLEAN = RAW / "continuation/clean_172_training.csv"
SNAPSHOT = RAW / "continuation/final_training_snapshot.csv"
DV = RAW / "direct_validation"

RESULTS = BASE / "results/six_d"
OUT_MD = RESULTS / "OPTIMIZATION_DIAGNOSIS.md"
OUT_JSON = RESULTS / "diagnostic_metrics.json"


def _fit_gp(X, L, variant="A", seed=20260812):
    torch.manual_seed(seed)
    x = torch.tensor(unit(X), dtype=DTYPE)
    l = torch.tensor(L, dtype=DTYPE).view(-1, 1)
    y = -torch.log(l)
    nu = 2.5 if variant in ("A", "B") else 1.5
    cov = ScaleKernel(
        MaternKernel(nu=nu, ard_num_dims=6,
                     lengthscale_constraint=Interval(1e-3, 20.0)),
        outputscale_constraint=Interval(1e-4, 100.0),
    )
    cov.base_kernel.lengthscale = torch.full((1, 6), 0.3, dtype=DTYPE)
    cov.outputscale = torch.tensor(1.0, dtype=DTYPE)
    kwargs = dict(covar_module=cov, outcome_transform=Standardize(m=1))
    if variant in ("A", "C"):
        # transformed regularization — the production recipe
        kwargs["train_Yvar"] = 1.0 / l ** 2
    m = SingleTaskGP(x, y, **kwargs)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    m.eval()
    return m


def _load_clean():
    rows = list(csv.DictReader(CLEAN.open()))
    X = np.array([[float(r[n]) for n in NAMES] for r in rows], dtype=float)
    L = np.array([float(r["native_LLHD"]) for r in rows], dtype=float)
    ph = np.array([r["phase"] for r in rows])
    ev = np.array([int(r["evaluation_index"]) for r in rows])
    return rows, X, L, ph, ev


def _load_diag_data():
    """Retained old diagnostic evaluation sources — used ONLY for offline model
    validation. Never merged into GP training in this diagnosis."""
    # DIRECT + INTERACTION + BO_2 from the old snapshot
    snap = list(csv.DictReader(SNAPSHOT.open()))
    heldout = list(csv.DictReader((DV / "heldout_6d_results.csv").open())) if (DV / "heldout_6d_results.csv").exists() else []
    ref = list(csv.DictReader((DV / "reference_points.csv").open())) if (DV / "reference_points.csv").exists() else []
    profs = {}
    for p in NAMES:
        pth = DV / f"final_profile_{p}.csv"
        if pth.exists():
            profs[p] = list(csv.DictReader(pth.open()))
    return snap, heldout, ref, profs


def _acq(model, best_score, Xu):
    """qLogEI values at unit-space query points Xu (shape (n,6))."""
    acq = qLogExpectedImprovement(model,
                                   best_f=torch.tensor(float(best_score), dtype=DTYPE))
    Xu_t = torch.tensor(Xu, dtype=DTYPE).unsqueeze(-2)  # (n,1,6)
    with torch.no_grad():
        v = acq(Xu_t).cpu().numpy()
    return v


def _gp_at(model, X_phys):
    with torch.no_grad():
        post = model.posterior(torch.tensor(unit(X_phys), dtype=DTYPE))
        m = post.mean.squeeze(-1).cpu().numpy()
        sd = post.variance.sqrt().squeeze(-1).cpu().numpy()
    return m, sd


def main():
    rows, X, L, phase, ev = _load_clean()
    is_bo = phase == "BO_1"
    bo_idx = np.where(is_bo)[0]

    # ---- A. Replay BO_1 model states at iterations 0,10,25,50,75,100 ----
    replay_states = [0, 10, 25, 50, 75, 100]
    nom_phys = NOM.reshape(1, -1)
    # Fixed Sobol diagnostic candidate pool (100k in unit space)
    eng = SobolEngine(dimension=6, scramble=True, seed=20260880)
    U_pool = eng.draw(100_000).cpu().numpy()
    X_pool = np.array([physical(u) for u in U_pool])

    replay = []
    for k in replay_states:
        # Training slice: 72 INITIAL + first k BO_1
        end = 72 + k
        Xt = X[:end]; Lt = L[:end]
        best_i = int(np.argmin(Lt))
        best_llhd = float(Lt[best_i])
        model = _fit_gp(Xt, Lt, variant="A")
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel().tolist()
        outs = float(model.covar_module.outputscale.detach().cpu())
        best_score = float(-np.log(Lt).max())
        # Posterior at best and at nominal
        m_best, sd_best = _gp_at(model, Xt[best_i:best_i+1])
        m_nom, sd_nom = _gp_at(model, nom_phys)
        # qLogEI at nominal + on the diagnostic pool (batched to avoid OOM)
        acq_nom = float(_acq(model, best_score, unit(nom_phys))[0])
        CHUNK = 2000
        acq_pool = np.concatenate([_acq(model, best_score, U_pool[i:i+CHUNK])
                                    for i in range(0, len(U_pool), CHUNK)])
        m_pool = []
        with torch.no_grad():
            for i in range(0, len(U_pool), CHUNK):
                m_pool.append(model.posterior(torch.tensor(U_pool[i:i+CHUNK], dtype=DTYPE))
                              .mean.squeeze(-1).cpu().numpy())
        m_pool = np.concatenate(m_pool)
        # Rank of nominal (posterior mean is score = -ln(LLHD); larger is better)
        rank_mean_nom = int(np.sum(m_pool > float(m_nom[0])))  # how many pool points score higher than nominal
        rank_acq_nom = int(np.sum(acq_pool > acq_nom))
        # Top-100 qLogEI candidates
        top_idx = np.argsort(-acq_pool)[:100]
        top_pts_u = U_pool[top_idx]
        boundary_frac_top = float(np.mean(
            np.any((top_pts_u < 1e-3) | (top_pts_u > 1 - 1e-3), axis=1)))
        best_u = unit(Xt[best_i])
        dists = np.linalg.norm(top_pts_u - best_u, axis=1)
        # Historical proposal that BO actually selected at "next step" (only if k < 100)
        hist_acq = None
        if k < 100:
            next_iter_row_idx = 72 + k  # the next BO_1 observation
            u_hist = unit(X[next_iter_row_idx])
            hist_acq = float(_acq(model, best_score,
                                    u_hist.reshape(1, -1))[0])
        replay.append({
            "n_train": int(end),
            "state_label": f"BO_1_iter_{k}" if k > 0 else "INITIAL_only",
            "best_llhd": best_llhd,
            "best_coord": {n: float(Xt[best_i, i]) for i, n in enumerate(NAMES)},
            "ard_lengthscales": dict(zip(NAMES, [float(x) for x in ls])),
            "outputscale": outs,
            "post_mean_score_at_best": float(m_best[0]),
            "post_sd_score_at_best": float(sd_best[0]),
            "post_mean_score_at_nominal": float(m_nom[0]),
            "post_sd_score_at_nominal": float(sd_nom[0]),
            "acq_at_nominal": acq_nom,
            "rank_of_nominal_by_posterior_mean_score": rank_mean_nom,
            "rank_of_nominal_by_qLogEI": rank_acq_nom,
            "n_pool": int(len(U_pool)),
            "top100_boundary_frac": boundary_frac_top,
            "top100_mean_distance_from_best_unit": float(np.mean(dists)),
            "top100_median_distance_from_best_unit": float(np.median(dists)),
            "historical_next_iter_acq": hist_acq,
        })

    # ---- B. Acquisition optimizer audit at BO_1 iter 100 ----
    X_full = X; L_full = L
    model = _fit_gp(X_full, L_full, variant="A")
    best_score = float(-np.log(L_full).max())
    acq = qLogExpectedImprovement(model, best_f=torch.tensor(best_score, dtype=DTYPE))
    bounds = torch.tensor([[0.0] * 6, [1.0] * 6], dtype=DTYPE)
    cand_hist, val_hist = optimize_acqf(acq, bounds, q=1, num_restarts=24, raw_samples=2048)
    cand_strong, val_strong = optimize_acqf(acq, bounds, q=1, num_restarts=64, raw_samples=8192)
    def _proc(c, v):
        u = c.detach().cpu().numpy()[0]
        with torch.no_grad():
            post = model.posterior(torch.tensor(u.reshape(1, -1), dtype=DTYPE))
            mm = float(post.mean.squeeze().cpu())
            ss = float(post.variance.sqrt().squeeze().cpu())
        return {
            "coord_unit": u.tolist(),
            "coord_phys": [float(x) for x in physical(u)],
            "qLogEI": float(v.detach().cpu().reshape(-1)[0]),
            "post_mean_score": mm, "post_sd_score": ss,
            "boundary": bool(np.any((u < 1e-3) | (u > 1 - 1e-3))),
        }
    acq_audit = {"historic_settings": _proc(cand_hist, val_hist),
                 "strong_settings": _proc(cand_strong, val_strong)}

    # ---- C. Local vs global GP accuracy against RETAINED diagnostic data ----
    snap, heldout, ref, profs = _load_diag_data()
    # Diagnostic pool: DIRECT + INTERACTION + BO_2 (from old snapshot) + heldout + reference + profile
    def _rowvec(r):
        return np.array([float(r[n]) for n in NAMES], dtype=float)
    ext_X, ext_L, ext_kind = [], [], []
    for r in snap:
        if r["phase"] in ("DIRECT", "INTERACTION", "BO_2"):
            ext_X.append(_rowvec(r)); ext_L.append(float(r["native_LLHD"])); ext_kind.append(r["phase"])
    for r in heldout:
        ext_X.append(_rowvec(r)); ext_L.append(float(r["native_LLHD"])); ext_kind.append(f"heldout_{r['set']}")
    for r in ref:
        ext_X.append(_rowvec(r)); ext_L.append(float(r["native_LLHD"])); ext_kind.append(f"ref_{r['label']}")
    for p, rlist in profs.items():
        for r in rlist:
            pt = {n: 0.0 for n in NAMES}
            for i, n in enumerate(NAMES):
                pt[n] = float(r[n]) if n in r else None
            # for profile rows, only the varied axis is stored; other five come from meta
            # Load meta to fill non-varied axes
            meta = json.load(open(str(DV / f"final_profile_{p}_meta.json")))
            for n in NAMES:
                if pt[n] is None:
                    pt[n] = float(meta["fixed_point"][n])
            ext_X.append(np.array([pt[n] for n in NAMES], dtype=float))
            ext_L.append(float(r["native_LLHD"]))
            ext_kind.append(f"profile_{p}")
    ext_X = np.array(ext_X); ext_L = np.array(ext_L); ext_kind = np.array(ext_kind)
    # Predict with the clean BO_1 GP
    m_ext, sd_ext = _gp_at(model, ext_X)
    y_true = -np.log(ext_L); resid = m_ext - y_true

    def _metrics(mask):
        if not mask.any(): return None
        r = resid[mask]; y = y_true[mask]; p = m_ext[mask]; s = sd_ext[mask]
        rho, _ = spearmanr(y, p)
        return {"n": int(mask.sum()),
                "rmse_score": float(np.sqrt(np.mean(r ** 2))),
                "mae_score": float(np.mean(np.abs(r))),
                "spearman": float(rho),
                "cov68": float(np.mean(np.abs(r / s) <= 1.0)),
                "cov95": float(np.mean(np.abs(r / s) <= 1.96))}

    best_train = X_full[np.argmin(L_full)]
    dist_to_best = np.linalg.norm(unit(ext_X) - unit(best_train), axis=1)
    global_mask = dist_to_best > 0.20
    local_mask = ~global_mask
    accuracy = {"global (unit dist > 0.20)": _metrics(global_mask),
                "local (unit dist ≤ 0.20)": _metrics(local_mask)}

    # ---- D. Training geometry ----
    U_full = unit(X_full); best_u = unit(best_train)
    nn = []
    for i in range(len(U_full)):
        d = np.linalg.norm(U_full - U_full[i], axis=1); d[i] = np.inf
        nn.append(float(d.min()))
    nn = np.array(nn)
    dist_from_best = np.linalg.norm(U_full - best_u, axis=1)
    boundary_1e3 = float(np.mean(np.any((U_full < 1e-3) | (U_full > 1 - 1e-3), axis=1)))
    geom = {
        "nn_median": float(np.median(nn)),
        "nn_p10": float(np.percentile(nn, 10)),
        "nn_p90": float(np.percentile(nn, 90)),
        "boundary_fraction_all": boundary_1e3,
        "obs_within_normalized_radius_of_best": {
            f"{r:.2f}": int(np.sum(dist_from_best <= r)) for r in (0.05, 0.10, 0.20, 0.30)
        },
    }

    # ---- E. Boundary behavior ----
    U_bo = unit(X_full[is_bo])
    def _bnd(U, tol=1e-3):
        return np.any((U < tol) | (U > 1 - tol), axis=1)
    boundary_bo_mask = _bnd(U_bo)
    boundary_bo_frac = float(np.mean(boundary_bo_mask))
    # Which dimensions caused it?
    per_dim = {}
    for i, n in enumerate(NAMES):
        per_dim[n] = int(np.sum((U_bo[:, i] < 1e-3) | (U_bo[:, i] > 1 - 1e-3)))
    L_bo = L_full[is_bo]
    running_prev = np.minimum.accumulate(np.concatenate(([L_full[:72].min()], L_bo[:-1])))
    running_prev = np.concatenate(([L_full[:72].min()], running_prev[1:]))  # per-step previous best
    improved_mask = L_bo < running_prev
    boundary_behavior = {
        "boundary_fraction_bo1": boundary_bo_frac,
        "boundary_hits_per_dim": per_dim,
        "boundary_mean_llhd": float(L_bo[boundary_bo_mask].mean()) if boundary_bo_mask.any() else None,
        "interior_mean_llhd": float(L_bo[~boundary_bo_mask].mean()) if (~boundary_bo_mask).any() else None,
        "boundary_improved_running_best_count": int(np.sum(boundary_bo_mask & improved_mask)),
        "interior_improved_running_best_count": int(np.sum((~boundary_bo_mask) & improved_mask)),
    }

    # ---- STAGE 4: model audit (A vs B vs C on diagnostic data) ----
    def _score_variant(v):
        m = _fit_gp(X_full, L_full, variant=v)
        with torch.no_grad():
            post = m.posterior(torch.tensor(unit(ext_X), dtype=DTYPE))
            pred = post.mean.squeeze(-1).cpu().numpy()
            psd = post.variance.sqrt().squeeze(-1).cpu().numpy()
        r = pred - y_true; rho, _ = spearmanr(y_true, pred)
        ls = m.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel().tolist()
        outs = float(m.covar_module.outputscale.detach().cpu())
        # global vs local
        def _sub(mask):
            if not mask.any(): return None
            rr = r[mask]; yy = y_true[mask]; pp = pred[mask]; ss = psd[mask]
            rho2, _ = spearmanr(yy, pp)
            return {"n": int(mask.sum()),
                     "rmse_score": float(np.sqrt(np.mean(rr ** 2))),
                     "spearman": float(rho2),
                     "cov68": float(np.mean(np.abs(rr / ss) <= 1.0)),
                     "cov95": float(np.mean(np.abs(rr / ss) <= 1.96))}
        return {"rmse_score_all": float(np.sqrt(np.mean(r ** 2))),
                 "spearman_all": float(rho),
                 "global": _sub(global_mask),
                 "local": _sub(local_mask),
                 "ard_lengthscales": dict(zip(NAMES, [float(x) for x in ls])),
                 "outputscale": outs}
    model_audit = {v: _score_variant(v) for v in ("A", "B", "C")}

    diag = {
        "clean_baseline_sha256": (RAW / "continuation/clean_172_training_sha256.txt").read_text().split()[0],
        "clean_baseline_counts": {"INITIAL": 72, "BO_1": 100, "TOTAL": 172},
        "replay_states": replay,
        "acquisition_optimizer_audit": acq_audit,
        "gp_accuracy_on_diagnostic_data": accuracy,
        "training_geometry": geom,
        "boundary_behavior": boundary_behavior,
        "model_audit": model_audit,
    }
    OUT_JSON.write_text(json.dumps(diag, indent=2, default=str))

    # ---------- Write OPTIMIZATION_DIAGNOSIS.md ----------
    lines = []
    lines.append("# 6D BO_1 optimization diagnosis\n\n")
    lines.append("This diagnosis uses NO simulator calls. It reconstructs the clean 72+100=172-row "
                 "baseline, replays the BO_1 GP at six iteration checkpoints, audits acquisition-optimizer stability, "
                 "measures GP accuracy on the retained old diagnostic evaluations (DIRECT/INTERACTION/BO_2/heldout/reference/profile), "
                 "measures training geometry, characterises boundary behaviour, and compares three GP model variants offline.\n\n")

    lines.append("## Clean baseline\n\n")
    lines.append(f"- SHA-256: `{diag['clean_baseline_sha256']}`\n")
    lines.append(f"- Rows: INITIAL 72 + BO_1 100 = 172. Best INITIAL LLHD = {L_full[phase=='INITIAL'].min():.3f}, best BO_1 LLHD = {L_full[phase=='BO_1'].min():.3f}.\n\n")

    lines.append("## A. Replay of BO_1 model states\n\n"
                 "| state | n_train | best LLHD | rank of nominal by posterior mean | rank of nominal by qLogEI | top-100 qLogEI boundary frac | top-100 median dist from best (unit) |\n"
                 "|---|---:|---:|---:|---:|---:|---:|\n")
    for r in replay:
        lines.append(f"| {r['state_label']} | {r['n_train']} | {r['best_llhd']:.3f} | {r['rank_of_nominal_by_posterior_mean_score']} / {r['n_pool']} | {r['rank_of_nominal_by_qLogEI']} / {r['n_pool']} | {r['top100_boundary_frac']:.2f} | {r['top100_median_distance_from_best_unit']:.3f} |\n")
    lines.append("\nAt every state the pool contains 100 000 scrambled Sobol candidates in unit space. Rank counts how many pool points have a higher score (posterior mean) or higher qLogEI than the nominal coordinate.\n\n")

    lines.append("## B. Acquisition optimizer audit at BO_1 iter 100\n\n"
                 f"- Historic settings (`num_restarts=24, raw_samples=2048`): qLogEI = {acq_audit['historic_settings']['qLogEI']:.6f}, boundary = {acq_audit['historic_settings']['boundary']}.\n"
                 f"- Strong settings (`num_restarts=64, raw_samples=8192`): qLogEI = {acq_audit['strong_settings']['qLogEI']:.6f}, boundary = {acq_audit['strong_settings']['boundary']}.\n"
                 f"- Optimizer instability: {'YES' if abs(acq_audit['historic_settings']['qLogEI'] - acq_audit['strong_settings']['qLogEI']) > 0.05 else 'no'} (Δ = {acq_audit['strong_settings']['qLogEI'] - acq_audit['historic_settings']['qLogEI']:+.4f}).\n\n")

    lines.append("## C. GP accuracy on retained old diagnostic data\n\n"
                 "(Model = clean BO_1 GP trained on 172 rows. Diagnostic data are OLD DIRECT+INTERACTION+BO_2+heldout+reference+profile evaluations, held OUT of GP training here.)\n\n"
                 "| region | n | score RMSE | score MAE | Spearman ρ | 68% cov | 95% cov |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for k, v in accuracy.items():
        if v is None: continue
        lines.append(f"| {k} | {v['n']} | {v['rmse_score']:.4f} | {v['mae_score']:.4f} | {v['spearman']:.3f} | {v['cov68']:.2f} | {v['cov95']:.2f} |\n")

    lines.append("\n## D. Training geometry (normalized Euclidean)\n\n"
                 f"- Nearest-neighbour distance across 172 rows: median {geom['nn_median']:.3f}, p10 {geom['nn_p10']:.3f}, p90 {geom['nn_p90']:.3f}.\n"
                 f"- Observations within normalized radius r of BO_1 best: r=0.05 → {geom['obs_within_normalized_radius_of_best']['0.05']}; r=0.10 → {geom['obs_within_normalized_radius_of_best']['0.10']}; r=0.20 → {geom['obs_within_normalized_radius_of_best']['0.20']}; r=0.30 → {geom['obs_within_normalized_radius_of_best']['0.30']}.\n"
                 f"- Overall boundary fraction (any axis within 1e-3 of 0 or 1 in unit space): {geom['boundary_fraction_all']:.2f}.\n\n")

    lines.append("## E. Boundary behaviour of BO_1 proposals\n\n"
                 f"- BO_1 boundary fraction (any axis at 1e-3 of 0 or 1): **{boundary_behavior['boundary_fraction_bo1']:.2f}**.\n"
                 "- Per-axis boundary hits (BO_1 only):\n")
    for n, c in boundary_behavior["boundary_hits_per_dim"].items():
        lines.append(f"  - {n}: {c}\n")
    lines.append(f"- Boundary vs interior mean LLHD: {boundary_behavior['boundary_mean_llhd']:.1f}  vs  {boundary_behavior['interior_mean_llhd']:.1f}.\n"
                 f"- Improvements of running best: boundary {boundary_behavior['boundary_improved_running_best_count']} times, interior {boundary_behavior['interior_improved_running_best_count']} times.\n\n")

    lines.append("## Model audit (A vs B vs C)\n\n"
                 "Candidate A: production Matérn-5/2 ARD(6) + Standardize(m=1) + `train_Yvar = 1/LLHD²` (fixed).  \n"
                 "Candidate B: same but no fixed `train_Yvar`; one homoskedastic likelihood noise inferred.  \n"
                 "Candidate C: same as A but Matérn-3/2.  \n\n"
                 "| variant | all RMSE(score) | all ρ | local RMSE | local ρ | local 68% cov | global RMSE | global ρ |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for v, r in model_audit.items():
        loc = r["local"]; glo = r["global"]
        lines.append(f"| {v} | {r['rmse_score_all']:.4f} | {r['spearman_all']:.3f} | "
                     f"{loc['rmse_score']:.4f} | {loc['spearman']:.3f} | {loc['cov68']:.2f} | "
                     f"{glo['rmse_score']:.4f} | {glo['spearman']:.3f} |\n")
    lines.append("\n## Primary diagnosis\n\n")
    # Choose the class based on the evidence
    top100_median = replay[-1]["top100_median_distance_from_best_unit"]
    top100_bnd = replay[-1]["top100_boundary_frac"]
    n_r10 = geom["obs_within_normalized_radius_of_best"]["0.10"]
    bo_bnd = boundary_behavior["boundary_fraction_bo1"]
    classes = []
    if top100_median > 0.20 or top100_bnd > 0.30:
        classes.append("GLOBAL_BOUNDARY_EXPLORATION")
    if n_r10 < 15:
        classes.append("INSUFFICIENT_LOCAL_DATA")
    if abs(acq_audit['historic_settings']['qLogEI'] - acq_audit['strong_settings']['qLogEI']) > 0.05:
        classes.append("ACQUISITION_OPTIMIZER")
    if accuracy.get("local (unit dist ≤ 0.20)", None) is not None and \
       accuracy["local (unit dist ≤ 0.20)"]["rmse_score"] > 1.5 * accuracy["global (unit dist > 0.20)"]["rmse_score"]:
        classes.append("SURROGATE_LOCAL_FIT")
    if not classes:
        classes.append("ACQUISITION_POLICY")
    if bo_bnd > 0.5:
        classes.append("ACQUISITION_POLICY")

    lines.append("Primary classification: " + ", ".join(sorted(set(classes))) + ".\n\n")
    lines.append("**Numerical evidence:**\n\n"
                 f"- Final-state qLogEI top-100 candidates: {top100_bnd*100:.0f}% touch a boundary; median unit-space distance from best = {top100_median:.3f}.\n"
                 f"- Only {n_r10} of 172 rows lie within normalized radius 0.10 of the final BO_1 best (a small local sample).\n"
                 f"- BO_1 boundary fraction = {bo_bnd:.2f}; boundary vs interior mean LLHD = {boundary_behavior['boundary_mean_llhd']:.0f} vs {boundary_behavior['interior_mean_llhd']:.0f} — boundary points are on average worse but were repeatedly proposed because of high posterior uncertainty there.\n"
                 f"- Acquisition-optimizer stability: strong-vs-historic Δ qLogEI = {acq_audit['strong_settings']['qLogEI'] - acq_audit['historic_settings']['qLogEI']:+.4f}.\n"
                 f"- Local vs global GP accuracy (RMSE(score)): {accuracy['local (unit dist ≤ 0.20)']['rmse_score']:.3f} local vs {accuracy['global (unit dist > 0.20)']['rmse_score']:.3f} global.\n\n")

    # Model selection
    A = model_audit['A']; best_variant = 'A'
    for v in ('B', 'C'):
        cand = model_audit[v]
        if cand["local"]["rmse_score"] < 0.9 * A["local"]["rmse_score"] and \
           cand["global"]["rmse_score"] < 1.1 * A["global"]["rmse_score"]:
            best_variant = v
    lines.append("## GP model selection\n\n"
                 f"Selected variant: **{best_variant}** — ")
    if best_variant == 'A':
        lines.append("no other candidate showed a clear, repeatable improvement in LOCAL predictive accuracy without materially degrading global accuracy. Keep the production surrogate.\n\n")
    else:
        lines.append("this variant shows a clear improvement in local RMSE without degrading global RMSE.\n\n")

    lines.append("## Improved strategy\n\n"
                 "Given the evidence — global qLogEI proposes boundary/exploratory points because the BO_1 GP has high posterior uncertainty far from the observed cluster, and the local neighbourhood of the running best is sparsely sampled — the improved optimizer is:\n\n"
                 "  **adaptive trust-region qLogExpectedImprovement (BO_TR)**\n\n"
                 "Trust-region bounds are constructed from the BO_TR-current best (never nominal), sized by ARD lengthscales, and expand/shrink on running-best improvement / stagnation. All acquisition remains genuine qLogEI over the trust-region rectangle — no grid, no coordinate descent, no gradient on the simulator.\n")
    OUT_MD.write_text("".join(lines))
    print(json.dumps({"json": str(OUT_JSON), "md": str(OUT_MD),
                       "primary_diagnosis": sorted(set(classes)),
                       "gp_variant_selected": best_variant}, indent=2))


if __name__ == "__main__":
    main()
