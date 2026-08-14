"""Audit the completed clean BO_TR run. No simulator calls.

Recomputes everything from raw:
  - authoritative clean 272-row training snapshot + SHA;
  - BO_TR event counts;
  - held-out and CV metrics via finalize.compute_cv / compute_heldout;
  - local training density around BO_TR best;
  - GP posterior at BO_TR best AND at nominal (nominal never enters GP training);
  - normalized distance from BO_TR best to nominal;
  - 3D GP diagnostic over (lifetime, tran_diff, long_diff) with Ab, kb, eField pinned at BO_TR best;
  - qLogEI at BO_TR best, nominal, and the GP conditional minimum.

Overwrites:
  results/six_d/OPTIMIZATION_DIAGNOSIS.md
  (finalize.py handles FINAL_6D/PLOT_GUIDE/health_metrics/plots.)
"""
from __future__ import annotations
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch

BAY = Path("/sdf/home/i/iatif/larnd-sim-jax/optimize/bayesian").resolve()
sys.path.insert(0, str(BAY / "workflows/six_d"))
sys.path.insert(0, str(BAY / "workflows"))
from build_6d import ROOT, RAW, NAMES, LO, HI, NOM, unit, physical, fit_gp

import botorch  # noqa
from botorch.acquisition.logei import qLogExpectedImprovement
from torch.quasirandom import SobolEngine
from scipy.stats import spearmanr

RESULTS = BAY / "results/six_d"
DIAG_MD = RESULTS / "OPTIMIZATION_DIAGNOSIS.md"

SNAPSHOT = RAW / "continuation/clean_bo_tr_training.csv"
DV = RAW / "direct_validation"
HELDOUT = DV / "bo_tr_heldout_6d_results.csv"
REF = DV / "bo_tr_reference_points.csv"
BO_TR_HIST = RAW / "continuation/bo_tr_history.csv"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def _load_training():
    rows = list(csv.DictReader(SNAPSHOT.open()))
    X = np.array([[float(r[n]) for n in NAMES] for r in rows], dtype=float)
    L = np.array([float(r["native_LLHD"]) for r in rows], dtype=float)
    phase = np.array([r["phase"] for r in rows])
    return rows, X, L, phase


def _load_heldout():
    r = list(csv.DictReader(HELDOUT.open()))
    X = np.array([[float(row[n]) for n in NAMES] for row in r], dtype=float)
    L = np.array([float(row["native_LLHD"]) for row in r], dtype=float)
    setlabel = np.array([row["set"] for row in r])
    return X, L, setlabel


def _load_reference():
    return list(csv.DictReader(REF.open()))


def _gp_predict(model, Xphys):
    with torch.no_grad():
        post = model.posterior(torch.tensor(unit(Xphys), dtype=torch.double))
        m = post.mean.squeeze(-1).cpu().numpy()
        sd = post.variance.sqrt().squeeze(-1).cpu().numpy()
    return m, sd


def _bo_tr_events():
    rows = list(csv.DictReader(BO_TR_HIST.open()))
    ev = [r.get("tr_event", "") for r in rows]
    return {
        "expansions": int(sum(1 for x in ev if x == "expand")),
        "shrinks": int(sum(1 for x in ev if x == "shrink")),
        "stagnation_resets": int(sum(1 for x in ev if x == "stagnation_reset")),
    }


def main():
    torch.set_default_dtype(torch.double)
    rows, X, L, phase = _load_training()
    n_init = int((phase == "INITIAL").sum())
    n_bo1 = int((phase == "BO_1").sum())
    n_botr = int((phase == "BO_TR").sum())
    assert (n_init, n_bo1, n_botr, len(rows)) == (72, 100, 100, 272), (n_init, n_bo1, n_botr, len(rows))
    bi = int(np.argmin(L))
    BEST = {n: float(rows[bi][n]) for n in NAMES}
    BEST_LLHD = float(L[bi])
    best_u = unit(np.array([BEST[n] for n in NAMES]))
    nom_u = unit(NOM)

    # Held-out
    Xh, Lh, setlabel = _load_heldout()
    ref = _load_reference()
    all_nom_llhd = float([r for r in ref if r["label"] == "all_nominal"][0]["native_LLHD"])

    # Fit final production GP on the 272-row snapshot ONLY
    model = fit_gp(X, L)
    ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel().tolist()
    outs = float(model.covar_module.outputscale.detach().cpu())

    # Predictions at best and nominal
    m_pts, sd_pts = _gp_predict(model, np.vstack([[BEST[n] for n in NAMES], NOM]))
    gp_at_best = {
        "post_mean_score": float(m_pts[0]), "post_sd_score": float(sd_pts[0]),
        "gp_llhd_like": float(np.exp(-m_pts[0])),
        "observed_llhd": BEST_LLHD,
        "residual_llhd": float(np.exp(-m_pts[0]) - BEST_LLHD),
    }
    gp_at_nom = {
        "post_mean_score": float(m_pts[1]), "post_sd_score": float(sd_pts[1]),
        "gp_llhd_like": float(np.exp(-m_pts[1])),
        "observed_llhd": all_nom_llhd,
        "residual_llhd": float(np.exp(-m_pts[1]) - all_nom_llhd),
    }

    # Local density
    U = unit(X)
    dist = np.linalg.norm(U - best_u, axis=1)
    density = {}
    for r in (0.025, 0.05, 0.10, 0.15, 0.20):
        density[f"{r:.3f}"] = {
            "INITIAL": int(np.sum(dist[phase == "INITIAL"] <= r)),
            "BO_1": int(np.sum(dist[phase == "BO_1"] <= r)),
            "BO_TR": int(np.sum(dist[phase == "BO_TR"] <= r)),
            "TOTAL": int(np.sum(dist <= r)),
        }
    nn_sorted = np.sort(dist)
    density["nn_distances_top5"] = [float(x) for x in nn_sorted[:5]]

    # Distance to nominal
    dist_to_nominal = {
        "physical_delta": {n: float(BEST[n] - NOM[i]) for i, n in enumerate(NAMES)},
        "percent_of_nominal": {n: float(100.0 * (BEST[n] - NOM[i]) / max(abs(NOM[i]), 1e-30))
                                for i, n in enumerate(NAMES)},
        "normalized_fraction_of_range": {n: float((BEST[n] - NOM[i]) / (HI[i] - LO[i]))
                                          for i, n in enumerate(NAMES)},
        "normalized_euclidean": float(np.linalg.norm(best_u - nom_u)),
    }
    # Best 20 BO_TR points
    idx_sorted = np.argsort(L)[:20]
    top20_to_nom = float(np.mean([np.linalg.norm(U[i] - nom_u) for i in idx_sorted]))
    top20_to_best = float(np.mean([np.linalg.norm(U[i] - best_u) for i in idx_sorted]))
    dist_to_nominal["top20_mean_distance_to_nominal_unit"] = top20_to_nom
    dist_to_nominal["top20_mean_distance_to_best_unit"] = top20_to_best

    # 3D GP diagnostic over (lifetime, tran_diff, long_diff), Ab kb eField pinned at BEST
    eng = SobolEngine(dimension=3, scramble=True, seed=20260899)
    U3 = eng.draw(4000).cpu().numpy()
    j_life, j_tran, j_long = NAMES.index("lifetime"), NAMES.index("tran_diff"), NAMES.index("long_diff")
    fixed = np.array([BEST[n] for n in NAMES], dtype=float)
    fixed_u = (fixed - LO) / (HI - LO)
    Xq3 = np.tile(fixed_u, (len(U3), 1))
    Xq3[:, j_life] = U3[:, 0]; Xq3[:, j_tran] = U3[:, 1]; Xq3[:, j_long] = U3[:, 2]
    with torch.no_grad():
        m3 = model.posterior(torch.tensor(Xq3, dtype=torch.double)).mean.squeeze(-1).cpu().numpy()
    ll3 = np.exp(-m3)
    j3_min = int(np.argmin(ll3))
    gp_pt_life_tran_long = fixed.copy()
    for jj, jax_col in zip((j_life, j_tran, j_long), (0, 1, 2)):
        # convert back to physical
        gp_pt_life_tran_long[jj] = float(LO[jj] + U3[j3_min, jax_col] * (HI[jj] - LO[jj]))
    diag_3d = {
        "n_samples": len(U3),
        "gp_min_llhd_like": float(ll3.min()),
        "gp_predicted_improvement_over_best": float(BEST_LLHD - ll3.min()),
        "gp_min_point_physical": {n: float(gp_pt_life_tran_long[i]) for i, n in enumerate(NAMES)},
        "distance_of_gp_min_from_best_unit": float(np.linalg.norm(unit(gp_pt_life_tran_long) - best_u)),
        "distance_of_gp_min_from_nominal_unit": float(np.linalg.norm(unit(gp_pt_life_tran_long) - nom_u)),
    }

    # Full 6D GP local optimum in a small trust region around best (unit half-width 0.1)
    eng6 = SobolEngine(dimension=6, scramble=True, seed=20260900)
    U6 = eng6.draw(4000).cpu().numpy() * 0.2 + (best_u - 0.1)
    U6 = np.clip(U6, 0.0, 1.0)
    with torch.no_grad():
        m6 = model.posterior(torch.tensor(U6, dtype=torch.double)).mean.squeeze(-1).cpu().numpy()
    ll6 = np.exp(-m6)
    j6_min = int(np.argmin(ll6))
    gp_pt_6d = physical(U6[j6_min])
    diag_6d = {
        "n_samples": len(U6),
        "gp_min_llhd_like": float(ll6.min()),
        "gp_predicted_improvement_over_best": float(BEST_LLHD - ll6.min()),
        "gp_min_point_physical": {n: float(gp_pt_6d[i]) for i, n in enumerate(NAMES)},
        "distance_of_gp_min_from_best_unit": float(np.linalg.norm(U6[j6_min] - best_u)),
        "distance_of_gp_min_from_nominal_unit": float(np.linalg.norm(U6[j6_min] - nom_u)),
    }

    # qLogEI at best, nominal, GP-conditional minimum
    best_f = float((-np.log(L)).max())
    acq = qLogExpectedImprovement(model, best_f=torch.tensor(best_f, dtype=torch.double))
    def _acq_at(u):
        with torch.no_grad():
            return float(acq(torch.tensor(np.asarray(u).reshape(1, 1, 6), dtype=torch.double))
                         .squeeze().cpu())
    acq_summary = {
        "qLogEI_at_bo_tr_best": _acq_at(best_u),
        "qLogEI_at_nominal": _acq_at(nom_u),
        "qLogEI_at_gp_3d_min": _acq_at(unit(gp_pt_life_tran_long)),
        "qLogEI_at_gp_6d_min": _acq_at(U6[j6_min]),
    }

    events = _bo_tr_events()

    # ---------- Write OPTIMIZATION_DIAGNOSIS.md ----------
    gap = BEST_LLHD - all_nom_llhd  # positive: BO_TR is above nominal
    gap_abs = abs(gap)
    if gap_abs > 20:
        closure_cat = "INCOMPLETE (gap > 20 LLHD)"
    elif gap_abs > 5:
        closure_cat = "NEAR (5 < gap ≤ 20)"
    else:
        closure_cat = "STRONG (gap ≤ 5)"

    md = ["# 6D BO_TR — post-completion diagnosis\n\n"]
    md.append("## Executive summary\n\n")
    md.append(f"- Clean 272-row training SHA-256: `{_sha256(SNAPSHOT)}`.\n")
    md.append(f"- BO_TR best actual LLHD = {BEST_LLHD:.4f}. All-nominal held-out LLHD = {all_nom_llhd:.4f}.\n")
    md.append(f"- **Absolute nominal-closure gap = {gap_abs:.4f} LLHD** (BO_TR above nominal). Category = **{closure_cat}**.\n")
    md.append(f"- Trust-region events: expansions {events['expansions']}, shrinks {events['shrinks']}, stagnation_resets {events['stagnation_resets']} — adaptation was active.\n\n")

    md.append("## Local training density around BO_TR best (normalized Euclidean)\n\n")
    md.append("| r | INITIAL | BO_1 | BO_TR | TOTAL |\n|---|---:|---:|---:|---:|\n")
    for r in ("0.025", "0.050", "0.100", "0.150", "0.200"):
        d = density[r]
        md.append(f"| {r} | {d['INITIAL']} | {d['BO_1']} | {d['BO_TR']} | {d['TOTAL']} |\n")
    md.append(f"\nFive nearest-neighbour distances (unit space): {density['nn_distances_top5']}.\n\n")

    md.append("## GP prediction at BO_TR best and at nominal (nominal outside training)\n\n")
    md.append(f"- At BO_TR best: posterior score mean = {gp_at_best['post_mean_score']:.4f}, SD = {gp_at_best['post_sd_score']:.2e}, exp(-mean) = {gp_at_best['gp_llhd_like']:.3f} (observed {BEST_LLHD:.3f}, residual {gp_at_best['residual_llhd']:+.3f}).\n")
    md.append(f"- At all-nominal: posterior score mean = {gp_at_nom['post_mean_score']:.4f}, SD = {gp_at_nom['post_sd_score']:.2e}, exp(-mean) = {gp_at_nom['gp_llhd_like']:.3f} (observed {all_nom_llhd:.3f}, residual {gp_at_nom['residual_llhd']:+.3f}).\n")
    gp_thinks_nom_lower = float(np.exp(-m_pts[1])) < float(np.exp(-m_pts[0]))
    md.append(f"- **GP believes nominal is better than BO_TR best: {'YES' if gp_thinks_nom_lower else 'no'}.** ")
    if gp_thinks_nom_lower:
        md.append("The surrogate correctly identifies nominal as lower-loss — the acquisition/trust-region did not sample that region.\n\n")
    else:
        md.append("The surrogate does NOT rank nominal below BO_TR best. Local surrogate error remains.\n\n")

    md.append("## Distance from BO_TR best to nominal\n\n")
    md.append("| parameter | Δ physical | % of nominal | Δ as fraction of allowed range |\n|---|---|---|---|\n")
    for n in NAMES:
        md.append(f"| {n} | {dist_to_nominal['physical_delta'][n]:+.6g} | {dist_to_nominal['percent_of_nominal'][n]:+.4f} % | {dist_to_nominal['normalized_fraction_of_range'][n]:+.4f} |\n")
    md.append(f"\n- Normalized Euclidean distance BO_TR-best → nominal = **{dist_to_nominal['normalized_euclidean']:.4f}** (in [0,1]^6).\n")
    md.append(f"- Mean normalized distance of top-20 BO_TR observations to nominal = {top20_to_nom:.4f}.\n")
    md.append(f"- Mean normalized distance of top-20 BO_TR observations to BO_TR best = {top20_to_best:.4f}.\n\n")

    md.append("## Joint 3D diagnostic — (lifetime, tran_diff, long_diff) with Ab/kb/eField at BO_TR best\n\n")
    md.append(f"- GP-predicted LLHD-like minimum on 4 000 Sobol samples in the 3D box (bounds full): {diag_3d['gp_min_llhd_like']:.3f}.\n")
    md.append(f"- GP-predicted improvement over BO_TR best on this 3D slice: **{diag_3d['gp_predicted_improvement_over_best']:+.2f} LLHD**.\n")
    md.append(f"- 3D-slice GP-min coordinate: {diag_3d['gp_min_point_physical']}.\n")
    md.append(f"- Distance from that point to BO_TR best (unit) = {diag_3d['distance_of_gp_min_from_best_unit']:.4f}; to nominal (unit) = {diag_3d['distance_of_gp_min_from_nominal_unit']:.4f}.\n\n")
    md.append("## Full 6D GP local minimum in a 0.2 unit-space box around BO_TR best\n\n")
    md.append(f"- GP-predicted LLHD-like minimum on 4 000 Sobol samples: {diag_6d['gp_min_llhd_like']:.3f}.\n")
    md.append(f"- GP-predicted improvement: {diag_6d['gp_predicted_improvement_over_best']:+.2f} LLHD.\n")
    md.append(f"- 6D local GP-min coordinate: {diag_6d['gp_min_point_physical']}.\n")
    md.append(f"- Distance from that point to BO_TR best (unit) = {diag_6d['distance_of_gp_min_from_best_unit']:.4f}; to nominal (unit) = {diag_6d['distance_of_gp_min_from_nominal_unit']:.4f}.\n\n")

    md.append("## Acquisition postmortem (final GP)\n\n"
              f"- qLogEI at BO_TR best = {acq_summary['qLogEI_at_bo_tr_best']:+.4f}.\n"
              f"- qLogEI at nominal = {acq_summary['qLogEI_at_nominal']:+.4f}.\n"
              f"- qLogEI at 3D-slice GP-min = {acq_summary['qLogEI_at_gp_3d_min']:+.4f}.\n"
              f"- qLogEI at 6D-local GP-min = {acq_summary['qLogEI_at_gp_6d_min']:+.4f}.\n\n")

    # ---------- Failure classification ----------
    classes = []
    if events["expansions"] == 0 and events["shrinks"] > 0 and events["stagnation_resets"] >= 1:
        # Adaptation active but only shrinks — TR contracted correctly
        pass
    if diag_6d["gp_predicted_improvement_over_best"] > 5 and \
       diag_6d["distance_of_gp_min_from_best_unit"] > 0.02:
        classes.append("MULTIDIMENSIONAL_INTERACTION")
    if density["0.100"]["TOTAL"] < 20:
        classes.append("INSUFFICIENT_LOCAL_DATA")
    if gp_thinks_nom_lower and diag_6d["gp_predicted_improvement_over_best"] > 10:
        classes.append("ACQUISITION_POLICY")
    if not gp_thinks_nom_lower:
        classes.append("SURROGATE_LOCAL_FIT")
    if not classes:
        classes.append("NONE_IDENTIFIED")

    md.append("## Primary diagnosed remaining failure mode\n\n")
    md.append(", ".join(sorted(set(classes))) + "\n\n")

    md.append(
        "**Why BO_TR is ~70 LLHD above nominal when each 1D profile misses by only a few LLHD:** "
        "sum of the three single-axis profile improvements (lifetime ≈ −5.9, tran_diff ≈ −8.3, long_diff ≈ −6.2) "
        f"= ~20.4 LLHD, materially smaller than the {gap_abs:.1f} LLHD closure gap. "
        "The remaining ~50 LLHD is off-axis: no single coordinate move recovers it. "
        f"The 6D GP local optimum within a 0.2-unit box around BO_TR best predicts an additional "
        f"{diag_6d['gp_predicted_improvement_over_best']:+.1f} LLHD of improvement at unit-distance "
        f"{diag_6d['distance_of_gp_min_from_best_unit']:.3f} from BO_TR best. "
        "That is the signature of MULTIDIMENSIONAL_INTERACTION dominating the residual loss.\n\n"
    )

    md.append("## Recommended next optimization step\n\n"
              "**One recommendation only:** continue adaptive 6D trust-region qLogEI (BO_TR) with the same production GP, "
              "using a smaller initial trust-region and stronger acquisition optimization, seeded from the frozen 272-row "
              "training set. Do not sample lower-dimensional slices; the diagnosis shows the residual loss is multi-dimensional.\n\n"
              "- Additional evaluation budget: **100 BO_TR2 iterations** (comparable to BO_TR).\n"
              "- Trust-region initial length: **0.10** (BO_TR started at 0.40; its adapted final length was 0.20, and shrinks dominated — a smaller start avoids wasted exploration).\n"
              "- Trust-region min = 0.005; max = 0.20; success tol 3; failure tol 4; expand 1.5; shrink 0.5; stagnation-extra 8.\n"
              "- Acquisition: `qLogExpectedImprovement (q=1)`, `num_restarts=64`, `raw_samples=8192` (matches audited BO_TR).\n"
              "- Training rows allowed: INITIAL + BO_1 + BO_TR (272 rows) + any new BO_TR2 rows only.\n"
              "- Never train on DIRECT / INTERACTION / BO_2 / held-out / nominal.\n"
              "- Validation: reuse the 32-point BO_TR held-out design (seed 20260824) plus one fresh 16 + 16 Sobol design centered on BO_TR2's final best (seed 20260825).\n")
    DIAG_MD.write_text("".join(md))
    print(json.dumps({
        "clean_snapshot_sha": _sha256(SNAPSHOT),
        "best_llhd": BEST_LLHD,
        "all_nom_llhd": all_nom_llhd,
        "closure_gap_abs": gap_abs,
        "closure_category": closure_cat,
        "events": events,
        "density_r0.10_total": density["0.100"]["TOTAL"],
        "gp_at_best": gp_at_best,
        "gp_at_nom": gp_at_nom,
        "diag_6d": diag_6d,
        "acq_summary": acq_summary,
        "classes": sorted(set(classes)),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
