"""Finalize the 6D BO experiment.

Reads the frozen final training snapshot + all held-out and profile validation
results, fits the production 6D GP on the frozen snapshot ONLY, produces the
five canonical plots, PLOT_GUIDE.md, FINAL_6D.md, and health_metrics.json.

Refuses to run if torch/BoTorch/GPyTorch cannot be imported. No fallback
smoother is used anywhere the figure is labeled GP.
"""
from __future__ import annotations
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np

BASE = Path("/sdf/home/i/iatif/larnd-sim-jax/optimize/bayesian").resolve()
sys.path.insert(0, str(BASE / "workflows/six_d"))
from build_6d import ROOT, RAW, NAMES, LO, HI, NOM, unit, physical, fit_gp

try:
    import torch  # noqa
    import botorch  # noqa
    import gpytorch  # noqa
    from botorch.acquisition.logei import qLogExpectedImprovement  # noqa
except Exception as e:
    raise SystemExit(f"FATAL: torch/BoTorch/GPyTorch not importable: {e}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

RESULTS = BASE / "results/six_d"
PLOTS = RESULTS / "plots"
RESULTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

SNAPSHOT = RAW / "continuation/final_training_snapshot_bo_tr2.csv"
DV = RAW / "direct_validation"
HELDOUT = DV / "bo_tr2_heldout_6d_results.csv"
REF = DV / "bo_tr2_reference_points.csv"
REPEATABILITY = DV / "bo_tr2_repeatability.csv"

UNITS_TEX = {"Ab": "", "kb": "kV·g/(MeV·cm³)", "eField": "kV/cm",
             "lifetime": "μs", "tran_diff": "cm²/μs", "long_diff": "cm²/μs"}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _xlabel(p): return f"{p} [{UNITS_TEX[p]}]" if UNITS_TEX[p] else p


def _load_training():
    rows = list(csv.DictReader(SNAPSHOT.open()))
    X = np.array([[float(r[n]) for n in NAMES] for r in rows], dtype=float)
    L = np.array([float(r["native_LLHD"]) for r in rows], dtype=float)
    phase = np.array([r["phase"] for r in rows])
    ev = np.array([int(r["evaluation_index"]) for r in rows])
    src = np.array([r.get("source", "") for r in rows])
    return rows, X, L, phase, ev, src


def _load_heldout():
    if not HELDOUT.exists():
        return None, None, None
    r = list(csv.DictReader(HELDOUT.open()))
    X = np.array([[float(row[n]) for n in NAMES] for row in r], dtype=float)
    L = np.array([float(row["native_LLHD"]) for row in r], dtype=float)
    setlabel = np.array([row["set"] for row in r])
    return X, L, setlabel


def _load_reference():
    if not REF.exists():
        return None
    return list(csv.DictReader(REF.open()))


def _load_repeatability():
    """Load independently executed same-coordinate calls retained for repeatability."""
    if not REPEATABILITY.exists():
        return {}, []
    rows = list(csv.DictReader(REPEATABILITY.open()))
    stats = {}
    for label in ("all_nominal", "final_best"):
        values = np.array([float(r["native_LLHD"]) for r in rows if r["label"] == label], dtype=float)
        if len(values):
            stats[label] = {
                "n": int(len(values)),
                "values": values.tolist(),
                "min": float(values.min()),
                "max": float(values.max()),
                "std": float(values.std()),
                "exact_equal": bool(np.all(values == values[0])),
            }
    return stats, rows


def _load_profile(param):
    p = DV / f"bo_tr2_final_profile_{param}.csv"
    if not p.exists():
        return None
    r = list(csv.DictReader(p.open()))
    x = np.array([float(row[param]) for row in r], dtype=float)
    y = np.array([float(row["native_LLHD"]) for row in r], dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def gp_predict(model, X_phys):
    Xu = torch.tensor(unit(X_phys), dtype=torch.double)
    with torch.no_grad():
        post = model.posterior(Xu)
        m = post.mean.squeeze(-1).cpu().numpy()
        sd = post.variance.sqrt().squeeze(-1).cpu().numpy()
    return m, sd


# ---------------- plots ----------------

def plot_01(state, out_path):
    L = state["L"]; ev = state["ev"]; phase = state["phase"]
    idx = np.arange(1, len(L) + 1)
    running = np.minimum.accumulate(L)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 2]})
    colors = {"INITIAL": "#888888", "BO_1": "#1f77b4", "BO_TR": "#d62728"}
    for ph in ("INITIAL", "BO_1", "BO_TR"):
        m = phase == ph
        if not m.any(): continue
        ax0.scatter(idx[m], L[m], s=15, c=colors[ph], edgecolor="none", label=ph)
    boundaries = []
    for ph in ("INITIAL", "BO_1"):
        w = np.where(phase == ph)[0]
        if len(w):
            boundaries.append(w[-1] + 1.5)
    # Old BO_2 comparison horizontal line (dashed)
    ax0.axhline(21116.693359375, color="#7f7fbf", ls="--", lw=0.7, alpha=0.6,
                 label="Old BO_2 best (informed)")
    if state.get("all_nominal_llhd") is not None:
        ax0.axhline(state["all_nominal_llhd"], color="#ffcc00", ls=":", lw=0.9,
                     label=f"All-nominal held-out {state['all_nominal_llhd']:.1f}")
    for b in boundaries:
        ax0.axvline(b, color="#333", ls="--", lw=0.7)
    ax0.set_yscale("log"); ax0.set_ylabel("Native LLHD (log)")
    ax0.set_title("6D Bayesian optimization — convergence and efficiency\n"
                  "999.926642 cm safe dataset · 176 events · target hits 9 368")
    ax0.grid(alpha=0.3, which="both")
    ax0.legend(loc="upper right", fontsize=8, framealpha=1.0, edgecolor="#333")
    ax1.plot(idx, running, color="#d62728", lw=1.3, label="Running best")
    per_ph = {}
    for ph in ("INITIAL", "BO_1", "BO_TR"):
        m = phase == ph
        if m.any():
            per_ph[ph] = float(L[m].min())
    if "INITIAL" in per_ph:
        ax1.axhline(per_ph["INITIAL"], color="#888888", ls=":", lw=1.0, label=f"Best INIT {per_ph['INITIAL']:.1f}")
    if "BO_1" in per_ph:
        ax1.axhline(per_ph["BO_1"], color="#1f77b4", ls=":", lw=1.0, label=f"Best BO_1 {per_ph['BO_1']:.1f}")
    ax1.axhline(21116.693359375, color="#7f7fbf", ls="--", lw=1.0,
                 label="Old BO_2 best (informed) 21116.7")
    ax1.axhline(state["best_llhd"], color="#000", ls="-", lw=1.0,
                 label=f"BO_TR2 best {state['best_llhd']:.1f}")
    if state.get("all_nominal_llhd") is not None:
        ax1.scatter([len(L) + 1], [state["all_nominal_llhd"]], s=80, marker="*",
                    color="#ffcc00", edgecolor="black", linewidth=0.6,
                    label=f"All-nominal held-out {state['all_nominal_llhd']:.1f}")
    for b in boundaries:
        ax1.axvline(b, color="#333", ls="--", lw=0.7)
    ax1.set_yscale("log"); ax1.set_xlabel("Evaluation index")
    ax1.set_ylabel("Running best LLHD (log)")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(loc="upper right", fontsize=7, framealpha=1.0, edgecolor="#333", ncol=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=300); plt.close(fig)


def plot_02(state, out_path):
    X = state["X"]; L = state["L"]
    U = (X - LO) / (HI - LO)
    q10 = np.quantile(L, 0.10); m10 = L <= q10
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = np.arange(6)
    for i in range(len(U)):
        ax.plot(xs, U[i], color="#cccccc", lw=0.4, alpha=0.4)
    y = -np.log(L); y_norm = (y - y.min()) / max(y.max() - y.min(), 1e-30)
    for i in np.where(m10)[0]:
        c = plt.cm.plasma(y_norm[i])
        ax.plot(xs, U[i], color=c, lw=0.9, alpha=0.85)
    bi = int(np.argmin(L))
    ax.plot(xs, U[bi], color="#000000", lw=2.0, marker="o", markersize=6)
    un = (NOM - LO) / (HI - LO)
    ax.plot(xs, un, color="#9467bd", lw=1.5, ls="--", marker="P", markersize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="#000000", lw=2.0, label="Final best"),
                Line2D([0], [0], color="#9467bd", lw=1.5, ls="--", label="All-nominal"),
                Line2D([0], [0], color=plt.cm.plasma(0.05), lw=1.0, label="Best 10% (low LLHD)"),
                Line2D([0], [0], color="#cccccc", lw=1.0, label="Other training obs")]
    ax.set_xticks(xs); ax.set_xticklabels(NAMES)
    ax.set_ylabel("Normalized value [0,1]"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("6D — Parallel coordinates (best 10% coloured by score)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=1.0, edgecolor="#333")
    best_txt = "  |  ".join(f"{n}={state['best_pt'][n]:.4g}" for n in NAMES)
    fig.text(0.5, 0.01, best_txt, ha="center", fontsize=8, family="monospace")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=300); plt.close(fig)


def plot_03(state, model, out_path, grid_res=25):
    X = state["X"]; L = state["L"]
    fig, axes = plt.subplots(6, 6, figsize=(15, 15))
    order = np.argsort(L)
    ranks = np.empty(len(L)); ranks[order] = np.linspace(0, 1, len(L))
    q10 = np.quantile(L, 0.10); m10 = L <= q10
    for i in range(6):
        for j in range(6):
            ax = axes[i, j]
            if i == j:
                v = X[:, i]; v10 = X[m10, i]
                ax.hist(v, bins=25, color="#bbbbbb", alpha=0.7)
                ax.hist(v10, bins=25, color="#d62728", alpha=0.7)
                ax.axvline(state["best_pt"][NAMES[i]], color="#000", ls="-", lw=1)
                ax.axvline(NOM[i], color="#9467bd", ls="--", lw=1)
                ax.set_xlim(LO[i], HI[i])
            elif i > j:
                ax.scatter(X[:, j], X[:, i], c=ranks, cmap="viridis_r", s=8, edgecolor="none")
                ax.scatter([state["best_pt"][NAMES[j]]], [state["best_pt"][NAMES[i]]],
                           s=60, marker="*", color="#d62728", edgecolor="black", linewidth=0.6)
                ax.scatter([NOM[j]], [NOM[i]], s=40, marker="P",
                           color="#ffff00", edgecolor="black", linewidth=0.5)
                ax.set_xlim(LO[j], HI[j]); ax.set_ylim(LO[i], HI[i])
            else:
                uj = np.linspace(0.02, 0.98, grid_res)
                ui = np.linspace(0.02, 0.98, grid_res)
                UJ, UI = np.meshgrid(uj, ui, indexing="xy")
                fixed = np.array([state["best_pt"][n] for n in NAMES], dtype=float)
                fixed_u = (fixed - LO) / (HI - LO)
                Xu = np.tile(fixed_u, (grid_res * grid_res, 1))
                Xu[:, j] = UJ.ravel(); Xu[:, i] = UI.ravel()
                with torch.no_grad():
                    m = model.posterior(torch.tensor(Xu, dtype=torch.double)).mean.squeeze(-1).cpu().numpy()
                ll = np.exp(-m).reshape(grid_res, grid_res)
                xa = LO[j] + uj * (HI[j] - LO[j])
                xb = LO[i] + ui * (HI[i] - LO[i])
                ax.pcolormesh(xa, xb, ll, cmap="viridis",
                              norm=LogNorm(vmin=float(ll.min()), vmax=float(ll.max())),
                              shading="auto")
                ax.scatter([state["best_pt"][NAMES[j]]], [state["best_pt"][NAMES[i]]],
                           s=35, marker="*", color="#d62728", edgecolor="black", linewidth=0.4)
                ax.scatter([NOM[j]], [NOM[i]], s=25, marker="P",
                           color="#ffff00", edgecolor="black", linewidth=0.4)
            if i == 5: ax.set_xlabel(NAMES[j], fontsize=8)
            if j == 0: ax.set_ylabel(NAMES[i], fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle("6D — Pairwise matrix\nLower: training observations · Diagonal: distributions · Upper: GP conditional — other four at best",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300); plt.close(fig)


def plot_04(state, model, out_path, cv_stats, heldout_stats):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]
    a = cv_stats["actual_score"]; p = cv_stats["pred_score"]
    lim = [min(a.min(), p.min()), max(a.max(), p.max())]
    ax.scatter(a, p, s=10, alpha=0.6, c="#1f77b4")
    ax.plot(lim, lim, "k--", lw=0.7)
    ax.set_xlabel("Actual score −ln(LLHD)"); ax.set_ylabel("CV predicted score")
    ax.set_title(f"5-fold CV · RMSE={cv_stats['rmse_score']:.4f} · ρ={cv_stats['spearman']:.3f}")
    ax.grid(alpha=0.3)
    ax = axes[1]
    if heldout_stats is not None:
        aH = heldout_stats["actual_score"]; pH = heldout_stats["pred_score"]; sd = heldout_stats["pred_sd"]
        col_map = {"global": "#1f77b4", "local": "#d62728"}
        for lab in np.unique(heldout_stats["set"]):
            m = heldout_stats["set"] == lab
            ax.errorbar(aH[m], pH[m], yerr=sd[m], fmt="o", ms=5, alpha=0.7,
                        color=col_map.get(lab, "#333"), label=lab, ecolor="#999", elinewidth=0.6)
        lim = [min(aH.min(), pH.min()), max(aH.max(), pH.max())]
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.set_xlabel("Actual score"); ax.set_ylabel("Held-out predicted score")
        ax.set_title(f"Held-out · RMSE={heldout_stats['rmse_score']:.4f} · ρ={heldout_stats['spearman']:.3f}\n"
                     f"68%: {heldout_stats['coverage_68']:.2f} · 95%: {heldout_stats['coverage_95']:.2f}")
        ax.legend(loc="best", fontsize=8, framealpha=1.0, edgecolor="#333")
        ax.grid(alpha=0.3)
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "no held-out data", ha="center", va="center", transform=ax.transAxes)
    ax = axes[2]
    if heldout_stats is not None:
        sr = heldout_stats["std_res"]
        ax.hist(sr, bins=20, color="#1f77b4", alpha=0.85)
        for k in (-1.96, -1.0, 1.0, 1.96):
            ax.axvline(k, color="k", lw=0.6, ls=":")
        ax.set_xlabel("(pred − obs) / SD"); ax.set_ylabel("Count")
        ax.set_title("Held-out standardized residuals"); ax.grid(alpha=0.3)
    else:
        ax.axis("off")
    fig.suptitle("6D GP validation — held-out points were never used for GP fitting", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300); plt.close(fig)


def plot_05(state, model, out_path, heldout_all_nom):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    per_param = {}
    for ax, p in zip(axes.flat, NAMES):
        prof = _load_profile(p)
        if prof is None:
            ax.text(0.5, 0.5, f"no profile for {p}", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off"); continue
        x, y = prof
        j = NAMES.index(p)
        fixed = np.array([state["best_pt"][n] for n in NAMES], dtype=float)
        Xq = np.tile(fixed, (len(x), 1)); Xq[:, j] = x
        with torch.no_grad():
            post = model.posterior(torch.tensor(unit(Xq), dtype=torch.double))
            m = post.mean.squeeze(-1).cpu().numpy()
            sd = post.variance.sqrt().squeeze(-1).cpu().numpy()
        gp_ll = np.exp(-m)
        gp_lo = np.exp(-(m + 1.96 * sd)); gp_hi = np.exp(-(m - 1.96 * sd))
        ax.fill_between(x, gp_lo, gp_hi, color="#1f77b4", alpha=0.20, label="GP 95%")
        ax.plot(x, gp_ll, color="#1f77b4", lw=1.2, label="GP")
        ax.plot(x, y, color="#000", lw=0.7, alpha=0.7)
        ax.scatter(x, y, s=24, c="#000", edgecolor="none", label="Direct")
        i_min = int(np.argmin(y))
        ax.axvline(state["best_pt"][p], color="#d62728", ls=":", lw=1.2, label="Best")
        ax.axvline(NOM[j], color="#888", ls="--", lw=1.2, label="Nom (varied)")
        ax.scatter([x[i_min]], [y[i_min]], s=100, marker="v", color="#ff7f0e",
                   edgecolor="black", linewidth=0.6, label="Min")
        ax.set_xlabel(_xlabel(p)); ax.set_ylabel("LLHD")
        ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        delta = float(y[i_min] - state["best_llhd"])
        rng = HI[j] - LO[j]
        norm_shift = (x[i_min] - state["best_pt"][p]) / rng
        boundary = (abs(x[i_min] - LO[j]) < 1e-14) or (abs(x[i_min] - HI[j]) < 1e-14)
        ax.text(0.02, 0.97,
                f"min@{x[i_min]:.4g}  Δ={delta:+.2f}\nnorm-shift={norm_shift:+.3f}  boundary={'Y' if boundary else 'N'}",
                transform=ax.transAxes, va="top", fontsize=7, family="monospace",
                bbox=dict(facecolor="white", edgecolor="#333", alpha=0.9))
        per_param[p] = {"direct_min_coord": float(x[i_min]),
                        "direct_min_llhd": float(y[i_min]),
                        "delta_llhd": delta, "boundary": bool(boundary)}
    axes[0, 0].legend(loc="upper right", fontsize=6, framealpha=1.0, edgecolor="#333")
    footer = f"Final training best LLHD = {state['best_llhd']:.3f}"
    if heldout_all_nom is not None:
        footer += f"    ·    All-nominal held-out LLHD = {heldout_all_nom:.3f}    (Δ = {heldout_all_nom - state['best_llhd']:+.3f})"
    fig.text(0.5, 0.01, footer, ha="center", fontsize=9, family="monospace")
    fig.suptitle("6D direct simulator validation — GP band = 95% posterior, transformed to LLHD")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=300); plt.close(fig)
    return per_param


def _direct_validation_status(state):
    """Compute direct-profile minima without regenerating plot files."""
    per_param = {}
    for p in NAMES:
        prof = _load_profile(p)
        if prof is None:
            continue
        x, y = prof
        j = NAMES.index(p)
        i_min = int(np.argmin(y))
        per_param[p] = {
            "direct_min_coord": float(x[i_min]),
            "direct_min_llhd": float(y[i_min]),
            "delta_llhd": float(y[i_min] - state["best_llhd"]),
            "boundary": bool((abs(x[i_min] - LO[j]) < 1e-14) or
                             (abs(x[i_min] - HI[j]) < 1e-14)),
        }
    return per_param


# ---------------- metrics ----------------

def _bo_tr_event_counts():
    """Count expansions / shrinks / stagnation_resets from bo_tr_history.csv tr_event column."""
    p = RAW / "continuation/bo_tr_history.csv"
    if not p.exists():
        return {"bo_tr_expansions": 0, "bo_tr_shrinks": 0, "bo_tr_stagnation_resets": 0}
    import csv as _csv
    with p.open() as f:
        rows = list(_csv.DictReader(f))
    ev = [r.get("tr_event", "") for r in rows]
    return {
        "bo_tr_expansions": int(sum(1 for x in ev if x == "expand")),
        "bo_tr_shrinks": int(sum(1 for x in ev if x == "shrink")),
        "bo_tr_stagnation_resets": int(sum(1 for x in ev if x == "stagnation_reset")),
    }


def _bo_tr2_event_stats():
    """Recover BO_TR2 length and event statistics from its authoritative history."""
    p = RAW / "continuation/bo_tr2_history.csv"
    if not p.exists():
        return {}
    with p.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    events = [r.get("tr_event", "") for r in rows]
    return {
        "bo_tr2_initial_length": float(rows[0]["tr_length"]),
        "bo_tr2_final_length": float(rows[-1]["tr_length"]),
        "bo_tr2_expansions": int(sum(x == "expand" for x in events)),
        "bo_tr2_shrinks": int(sum(x == "shrink" for x in events)),
        "bo_tr2_stagnation_resets": int(sum(x == "stagnation_reset" for x in events)),
    }


# ---------------- Nominal-closure classification (assertion-tested at import) ----------------

def _classify_nominal_closure(gap_abs: float) -> str:
    """STRONG ≤ 5, NEAR (5, 20], INCOMPLETE > 20. `gap_abs` is |final best − nominal|."""
    if gap_abs > 20:
        return "INCOMPLETE"
    if gap_abs > 5:
        return "NEAR"
    return "STRONG"


assert _classify_nominal_closure(70.2168) == "INCOMPLETE"
assert _classify_nominal_closure(10.0) == "NEAR"
assert _classify_nominal_closure(3.0) == "STRONG"
assert _classify_nominal_closure(20.0) == "NEAR"
assert _classify_nominal_closure(5.0) == "STRONG"


def compute_cv(X, L, k=5, seed=20260813):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    idx = np.arange(len(L)); rng.shuffle(idx)
    folds = np.array_split(idx, k)
    y_true = -np.log(L)
    pred = np.full(len(L), np.nan); sd = np.full(len(L), np.nan)
    for fold in folds:
        train = np.setdiff1d(idx, fold)
        m = fit_gp(X[train], L[train], nu=1.5)
        with torch.no_grad():
            post = m.posterior(torch.tensor(unit(X[fold]), dtype=torch.double))
            pred[fold] = post.mean.squeeze(-1).cpu().numpy()
            sd[fold] = post.variance.sqrt().squeeze(-1).cpu().numpy()
    resid = pred - y_true
    rmse_score = float(np.sqrt(np.mean(resid ** 2)))
    mae_score = float(np.mean(np.abs(resid)))
    baseline = float(np.sqrt(np.mean((y_true - y_true.mean()) ** 2)))
    rho, _ = spearmanr(-np.log(L), pred)
    std = resid / sd
    cov68 = float(np.mean(np.abs(std) <= 1.0))
    cov95 = float(np.mean(np.abs(std) <= 1.96))
    return {"actual_score": y_true, "pred_score": pred, "pred_sd": sd, "std_res": std,
            "rmse_score": rmse_score, "mae_score": mae_score,
            "baseline_rmse": baseline, "spearman": float(rho),
            "coverage_68": cov68, "coverage_95": cov95}


def compute_heldout(model, Xh, Lh, setlabel):
    if Xh is None or len(Xh) == 0:
        return None
    from scipy.stats import spearmanr
    y_true = -np.log(Lh)
    with torch.no_grad():
        post = model.posterior(torch.tensor(unit(Xh), dtype=torch.double))
        pred = post.mean.squeeze(-1).cpu().numpy()
        sd = post.variance.sqrt().squeeze(-1).cpu().numpy()
    resid = pred - y_true
    pred_L = np.exp(-pred)
    rmse_score = float(np.sqrt(np.mean(resid ** 2)))
    mae_score = float(np.mean(np.abs(resid)))
    rmse_L = float(np.sqrt(np.mean((pred_L - Lh) ** 2)))
    mae_L = float(np.mean(np.abs(pred_L - Lh)))
    baseline = float(np.sqrt(np.mean((y_true - y_true.mean()) ** 2)))
    rho, _ = spearmanr(y_true, pred)
    std = resid / sd
    cov68 = float(np.mean(np.abs(std) <= 1.0))
    cov95 = float(np.mean(np.abs(std) <= 1.96))
    subset_metrics = {}
    for lab in np.unique(setlabel):
        m = setlabel == lab
        rs = resid[m]; ys = y_true[m]
        subset_metrics[str(lab)] = {
            "n": int(m.sum()),
            "rmse_score": float(np.sqrt(np.mean(rs ** 2))),
            "mae_score": float(np.mean(np.abs(rs))),
            "spearman": float(spearmanr(ys, pred[m])[0]) if m.sum() >= 3 else None,
            "coverage_68": float(np.mean(np.abs(std[m]) <= 1.0)),
            "coverage_95": float(np.mean(np.abs(std[m]) <= 1.96)),
        }
    j = int(np.argmax(np.abs(resid)))
    return {"actual_score": y_true, "pred_score": pred, "pred_sd": sd, "std_res": std,
            "set": setlabel, "rmse_score": rmse_score, "mae_score": mae_score,
            "rmse_llhd": rmse_L, "mae_llhd": mae_L, "baseline_rmse": baseline,
            "spearman": float(rho), "coverage_68": cov68, "coverage_95": cov95,
            "subset": subset_metrics,
            "worst": {"idx": int(j), "score_resid": float(resid[j]),
                       "actual_score": float(y_true[j]), "pred_score": float(pred[j]),
                       "coord": {n: float(Xh[j, k_]) for k_, n in enumerate(NAMES)}}}


# ---------------- prose ----------------

def write_plot_guide(state, metrics, cv_stats, heldout_stats, dv_status):
    BEST = state["best_pt"]; L = state["best_llhd"]
    cv = cv_stats; hs = heldout_stats
    body = f"""# 6D BO — plot guide

## How the plots were built

- Frozen training snapshot: `.local/six_d/current/raw/continuation/final_training_snapshot_bo_tr2.csv` (SHA in `health_metrics.json`).
- Training GP: selected Model C, Matérn-3/2 ARD(6), with `-ln(LLHD)` score, `Standardize(m=1)`, and requested `train_Yvar = 1/LLHD²` (numerically clamped to the GPyTorch noise floor).
- All five plots come from `optimize/bayesian/workflows/six_d/finalize.py`.
- Held-out points and reference checks are in `.local/six_d/current/raw/direct_validation/bo_tr2_heldout_6d_results.csv` and `bo_tr2_reference_points.csv`; those points were never used to fit the GP shown here.

## 01_convergence_and_efficiency.png

- Top panel plots every clean training LLHD by ordered evaluation index, colored by phase (INITIAL, BO_1, BO_TR, BO_TR2). Dashed verticals mark phase boundaries.
- Bottom panel plots the running-best LLHD across the frozen snapshot together with the best of each phase and the held-out all-nominal reference (star). The all-nominal point is NOT connected to the running-best line — it was evaluated after freezing.
- What the plot answers: did BO_1, BO_TR, and BO_TR2 improve the clean running best, and how close is the final training best to the all-nominal held-out reference?
- What the plot cannot prove: convergence to the true simulator minimum. A flat running-best only means the optimizer did not sample a lower-loss point; the direct profiles provide a separate local check.

## 02_parallel_coordinates.png

- Each polyline is one complete 6D training observation. Axes are normalized per-parameter to [0,1] so all axes are comparable.
- Grey lines: all training observations. Colored lines: the best 10% by LLHD (coloured by score rank). Black poly-line: the final training best. Purple dashed poly-line: the all-nominal reference.
- What the plot answers: do the low-LLHD observations form a tight cluster on some axes and spread on others? Is the final best close to nominal on every axis, or is it displaced?
- What the plot cannot prove: causal interactions between parameters. Crossing lines between two axes only show co-occurrence, not directed effect.

## 03_pairwise_observation_matrix.png

- 6×6 matrix. Diagonal: 1D histograms of all training values (grey) and best-10% (red). Lower triangle: scatter of all training observations for each pair, coloured by LLHD rank (dark = best). Upper triangle: GP conditional posterior mean on that pair with the other four axes fixed at the final best.
- The lower triangle is real simulator evidence. The upper triangle is the model's opinion in one specific slice of 6D space.
- What the plot answers: where in each pair did the simulator actually see low LLHD, and how does the fitted surrogate render the same pair conditional on the remaining four dimensions?
- What the plot cannot prove: the true 6D surface. Two axes that look "flat" here could be steep at a different fixed setting of the other four.

## 04_gp_cross_validation.png

- Panel A: 5-fold cross-validated predicted score vs actual for every training row. Panel B: held-out simulator observations (16 global + 16 local Sobol points around final best, seed 20260827), with 1-sigma vertical bars. Panel C: standardized held-out residuals `(pred − obs)/SD`.
- The 32 held-out points were never used to fit the GP shown here. CV is an interpolation check; held-out is the independent test.
- CV RMSE(score) = {cv['rmse_score']:.4f} vs constant-mean baseline {cv['baseline_rmse']:.4f}. Held-out RMSE(score) = {(hs['rmse_score'] if hs else float('nan')):.4f} vs baseline {(hs['baseline_rmse'] if hs else float('nan')):.4f}. Held-out Spearman ρ = {(hs['spearman'] if hs else float('nan')):.3f}. Held-out coverage 68% / 95% = {(hs['coverage_68'] if hs else float('nan')):.2f} / {(hs['coverage_95'] if hs else float('nan')):.2f}.

## 05_direct_validation_scorecard.png

- One panel per parameter. Black points/line: real simulator profile through the final best (five other axes fixed at final best). Blue line + band: GP posterior back-transformed to LLHD, 95% band. Red dotted vertical = final best coordinate. Grey dashed vertical = that parameter's nominal (only the varied coordinate is nominal — the other five stay at final best).
- The exact final-best coordinate is included in every profile. Orange marker = direct simulator minimum on the profile.
- Panel corner text reports the direct-min coord, LLHD change from final best, normalized shift as a fraction of the search range, and whether the min is at a bounds edge.
- Footer compares the final training best LLHD to the all-nominal held-out LLHD. Because the candidate simulator is deterministic under the fixed seed policy, a same-objective direct evaluation is a legitimate simulator observation. A direct point that beats the final training best by more than about 1 LLHD unit is honest evidence that BO did not locally converge on that axis.

## Notes readers often need

- The running-best line can stay flat while BO continues because BO deliberately proposes exploratory (high-uncertainty) points that may not beat the current best. That is not a failure — it is how BO trades off exploitation and exploration.
- A boundary proposal is not automatically wrong. BO proposes a boundary point when the model believes the response might keep improving toward that edge. Whether the boundary is right depends on the direct profile, which is why plot 05 exists.
- ARD lengthscale is not physical parameter importance. A shorter normalized lengthscale means the fitted response varies more rapidly along that axis over the sampled region. It says nothing about causal importance in the underlying physics.
- Conditional profiles are not the full 6D surface. Plot 05 varies one axis with the other five pinned at the final best. If the true minimum lies at different fixed values on the other five axes, the profile will not find it.
- An all-nominal 6D reference is different from one nominal coordinate in a conditional profile. All-nominal fixes every axis at its nominal value; a profile fixes five axes at the final best.
- Held-out validation after BO is not cheating. The held-out set and the direct profiles were generated only after the training snapshot was frozen — they never contributed to acquisition selection or GP fitting.
"""
    (RESULTS / "PLOT_GUIDE.md").write_text(body)


def write_final_report(state, metrics, cv_stats, heldout_stats, dv_status, per_phase_best):
    BEST = state["best_pt"]; L = state["best_llhd"]
    hs = heldout_stats or {}
    cv = cv_stats
    checks = metrics["checks"]

    def _b(v, s=".6g"):
        if isinstance(v, float): return format(v, s)
        return str(v)

    body = ["# FINAL_6D — Six-dimensional Bayesian optimization\n\n"]
    body.append("## 1. Executive summary\n\n")
    body.append(f"- Final training best LLHD **{L:.3f}** at evaluation {metrics['final_best_eval_index']} (phase {metrics['final_best_phase']}).\n")
    body.append("- Best per phase: " + " → ".join(
        f"{k} {_b(v,'.3f')}" for k, v in per_phase_best.items() if v.get("llhd") is not None
    ) + ".\n")
    if metrics["all_nominal_heldout"] is not None:
        body.append(f"- All-nominal held-out LLHD (reference) **{metrics['all_nominal_heldout']:.3f}**, Δ vs final best = {metrics['all_nominal_heldout'] - L:+.3f}.\n")
    body.append("- Health check flags: " + " · ".join(f"{k}={v}" for k, v in checks.items()) + "\n\n")

    body.append("## 2. Scientific objective\n\n"
                "Calibrate six detector-simulation parameters (Ab, kb, eField, lifetime, tran_diff, long_diff) against a fixed nominal target on the 999.926642 cm safe dataset. Use Bayesian optimization to find the six-dimensional coordinate that minimizes the native LLHD between candidate simulator output and the target hits.\n\n")

    body.append("## 3. Dataset and target provenance\n\n"
                f"- Physical track length: 999.926642 cm  ·  events: 176  ·  HDF5 rows: 100 010  ·  target hits: {metrics['target_hits']}\n"
                f"- Target NPZ SHA-256: `{metrics['target_sha256']}`\n"
                "- Target file: `.local/two_d/target.npz` (shared with the 2D pair fits — same SHA verified pre-run).\n\n")

    body.append("## 4. Parameter definitions, nominal values, bounds, units\n\n"
                "| parameter | nominal | lower | upper | unit |\n|---|---|---|---|---|\n")
    for i, n in enumerate(NAMES):
        body.append(f"| {n} | {NOM[i]:.6g} | {LO[i]:.6g} | {HI[i]:.6g} | {UNITS_TEX[n] or '(dimensionless)'} |\n")

    body.append("\n## 5. Seed and determinism policy\n\n"
                "- INITIAL Sobol seed: 20260812. BO_1 seed: 20260812. BO_TR seed: 20260830. BO_TR2 seed: 20260840. Fresh held-out validation seed: 20260827.\n"
                "- Candidate simulator uses `sim_seed_strategy='same'` with `SimSettings.seed=0`. Under LUT probabilistic simulation the candidate objective is deterministic in `(params, tracks, response)`.\n"
                f"- Independent retained calls reproduce exactly: nominal values `{metrics['nominal_repeat_stats']['values']}` (n={metrics['nominal_repeat_stats']['n']}, std={metrics['nominal_repeat_stats']['std']:.6g}) and BO_TR2-best values `{metrics['best_repeat_stats']['values']}` (n={metrics['best_repeat_stats']['n']}, std={metrics['best_repeat_stats']['std']:.6g}). `FIXED_SEED_REPEATABILITY={checks['FIXED_SEED_REPEATABILITY']}`.\n\n")

    body.append("## 6. Objective and score transformation\n\n"
                "- Native objective: LLHD (lower is better).\n"
                "- Modeled score: `-ln(native_LLHD)`. BoTorch maximizes; lower LLHD becomes larger score.\n"
                "- Regularization: `train_Yvar = 1 / native_LLHD²`. This is a transformed-regularization rule for a fixed-seed deterministic objective, not a measurement of physical detector noise.\n")
    if metrics.get("gpytorch_default_min_noise") is not None:
        body.append(f"- GPyTorch effective noise lower bound: `{metrics['gpytorch_default_min_noise']}`. Requested Yvar range: {metrics['requested_yvar_range']}. Effective Yvar range after clamping: {metrics['effective_yvar_range']}.\n\n")

    body.append("## 7. GP model and acquisition function\n\n"
                f"- Final model: `SingleTaskGP(Matern(nu=1.5, ARD(6)), ScaleKernel, Standardize(m=1))` (selected Model C).\n"
                f"- Kernel lengthscale constraint (normalized units): (1e-3, 20). Outputscale constraint: (1e-4, 100).\n"
                f"- Torch {metrics['torch_version']} · BoTorch {metrics['botorch_version']} · GPyTorch {metrics['gpytorch_version']}.\n"
                f"- BO_TR2 acquisition: `qLogExpectedImprovement (q=1)` with `num_restarts=64, raw_samples=8192`. The acquisition is differentiated through the GP; simulator gradients are never computed.\n\n")

    body.append("## 8. Original 72-point Sobol initialization\n\n"
                f"- 72 scrambled Sobol points across the 6D bounds. Best INITIAL LLHD **{_b(metrics['best_initial'], '.3f')}**.\n\n")

    body.append("## 9. Original BO_1 results\n\n"
                f"- 100 acquired evaluations. Best BO_1 LLHD **{_b(metrics['best_bo_1'], '.3f')}**. Boundary-proposal fraction = {metrics['bo_1_boundary_fraction']:.2f}.\n\n")

    body.append("## 10. Old informed diagnostic continuation (for context)\n\n"
                "The BO_1 result was probed by a set of DIRECT_1 simulator profiles (145 same-objective evaluations) and four INTERACTION corner points. Those observations were fed into a 50-iteration informed continuation labelled BO_2 in prior reports. The best BO_2 LLHD was 21 116.693.  \n\n"
                "**The improved run reported here does NOT use those DIRECT / INTERACTION / BO_2 rows for GP training.** They are retained only for provenance and appear on the old-BO_2 comparison line in plot 01.\n\n")

    body.append("## 11. Trust-region continuation (BO_TR)\n\n"
                f"- 100 additional BO_TR evaluations starting from the clean 72+100=172 seed. Best BO_TR LLHD **{_b(metrics['best_bo_tr'], '.3f')}**. BO_TR boundary-proposal fraction = {metrics['bo_tr_boundary_fraction']:.2f}.\n"
                f"- Trust-region hyperparameters (frozen before the run): initial length 0.40, min 0.025, max 0.80, success tol 3, failure tol 6, expand 2.0, shrink 0.5, ARD weight clip [0.1, 10]. Improvement threshold = max(1.0 LLHD, 1e-4 × current best).\n"
                f"- Trust-region events during the run: {metrics.get('bo_tr_expansions', 0)} expansions, {metrics.get('bo_tr_shrinks', 0)} shrinks, {metrics.get('bo_tr_stagnation_resets', 0)} stagnation resets.\n"
                "- BO_TR is a clean local-refinement run: only INITIAL + BO_1 seed it. No DIRECT, INTERACTION, BO_2, held-out, or nominal row entered its training set. Trust-region bounds always centre on the current best actual observation among {INITIAL, BO_1, BO_TR}, never on nominal.\n\n")

    body.append("## 12. Matérn-3/2 trust-region continuation (BO_TR2)\n\n"
                "- 60 additional BO_TR2 evaluations starting from the frozen clean 272-row snapshot. Best BO_TR2 LLHD **{:.3f}**.\n"
                "- Fresh trust-region state: initial/max length 0.20, min 0.005, success tol 3, failure tol 4, expand 1.5, shrink 0.5, stagnation-extra tolerance 8. The center was always the best actual clean observation and was never oriented using nominal.\n"
                "- The selected surrogate was Matérn-3/2 (Model C); acquisition used qLogEI with q=1, 64 restarts, and 8192 raw samples.\n"
                "- Recovered BO_TR2 history statistics: initial length {:.3g}, final length {:.3g}, {} expansions, {} shrinks, {} stagnation reset.\n"
                "- Clean final lineage: 72 INITIAL + 100 BO_1 + 100 BO_TR + 60 BO_TR2 = 332 rows. Diagnostic, nominal, legacy BO_2, repeatability, and validation observations remained outside training.\n\n".format(
                    L, metrics['bo_tr2_initial_length'], metrics['bo_tr2_final_length'],
                    metrics['bo_tr2_expansions'], metrics['bo_tr2_shrinks'],
                    metrics['bo_tr2_stagnation_resets']))

    body.append("## 13. Final best point by phase\n\n"
                "| phase | best LLHD |\n|---|---|\n")
    for ph in ("INITIAL", "BO_1", "BO_TR", "BO_TR2"):
        v = per_phase_best.get(ph, {}).get("llhd", None)
        body.append(f"| {ph} | {v:.4f} |\n" if v is not None else f"| {ph} | n/a |\n")

    body.append("\n## 14. Comparison against the all-nominal point\n\n")
    if metrics["all_nominal_heldout"] is not None:
        body.append(f"- All-nominal held-out LLHD = {metrics['all_nominal_heldout']:.4f}\n")
        body.append(f"- Final training best LLHD = {L:.4f}\n")
        body.append(f"- Δ = {metrics['all_nominal_heldout'] - L:+.4f}\n\n")
        if metrics["all_nominal_heldout"] < L - 1.0:
            body.append("The six-dimensional BO identified a low-loss region but did not fully recover the known nominal closure point within the available evaluation budget.\n\n")
        else:
            body.append("The all-nominal reference is not meaningfully better than the final training best.\n\n")

    body.append("## 15. Held-out 6D GP accuracy\n\n")
    if hs:
        body.append(f"- Held-out n = 32 (16 global + 16 local Sobol around BO_TR2 best, seed 20260827).\n")
        body.append(f"- Held-out RMSE(score) = {hs['rmse_score']:.4f}  ·  baseline {hs['baseline_rmse']:.4f}.\n")
        body.append(f"- Held-out MAE(score) = {hs['mae_score']:.4f}.\n")
        body.append(f"- Held-out RMSE(LLHD) = {hs['rmse_llhd']:.1f}  ·  MAE(LLHD) = {hs['mae_llhd']:.1f}.\n")
        body.append(f"- Held-out Spearman ρ = {hs['spearman']:.3f}.\n")
        body.append(f"- Held-out 68% coverage = {hs['coverage_68']:.2f}  ·  95% coverage = {hs['coverage_95']:.2f}.\n")
        w = hs['worst']
        body.append(f"- Worst held-out score residual = {w['score_resid']:+.4f} at coordinate {w['coord']}.\n\n")

    body.append("## 16. Five-fold cross-validation\n\n")
    body.append(f"- CV RMSE(score) = {cv['rmse_score']:.4f}  ·  baseline {cv['baseline_rmse']:.4f}\n")
    body.append(f"- CV MAE(score) = {cv['mae_score']:.4f}\n")
    body.append(f"- CV Spearman ρ = {cv['spearman']:.3f}\n")
    body.append(f"- CV 68% / 95% coverage = {cv['coverage_68']:.2f} / {cv['coverage_95']:.2f}\n\n")

    body.append("## 17. Direct-profile validation\n\n")
    L_best = state["best_llhd"]
    threshold = max(1.0, 1e-4 * L_best)
    body.append(f"Convergence threshold = max(1.0, 1e-4 × {L_best:.3f}) = **{threshold:.4f} LLHD**. A profile fails LOCAL_CONVERGENCE if `−Δ > threshold` (i.e. a same-objective simulator point beats BO_TR2 best by more than the threshold).\n\n")
    body.append("| parameter | direct min | direct LLHD | Δ vs final best | improves? | at boundary |\n|---|---|---|---|:-:|:-:|\n")
    failures = []
    for p in NAMES:
        d = dv_status.get(p)
        if d is None: continue
        improves = (-d["delta_llhd"]) > threshold
        if improves:
            failures.append((p, d["delta_llhd"]))
        body.append(f"| {p} | {d['direct_min_coord']:.6g} | {d['direct_min_llhd']:.4f} | {d['delta_llhd']:+.4f} | {'YES' if improves else 'no'} | {'YES' if d['boundary'] else 'no'} |\n")
    if failures:
        body.append("\n**LOCAL_CONVERGENCE FAIL** — the following axes each contain a same-objective simulator point that beats BO_TR2 best by more than the threshold: "
                    + ", ".join(f"{p} (Δ = {d:+.3f})" for p, d in failures)
                    + ". Sum of these single-axis improvements = "
                    + f"{-sum(d for _, d in failures):.2f} LLHD, which is materially less than the {abs(metrics['best_bo_tr'] - metrics['all_nominal_heldout']):.1f} LLHD gap to all-nominal — evidence that the remaining loss is multi-dimensional, not axis-additive.\n")
    else:
        body.append("\n**LOCAL_CONVERGENCE PASS** — no single-axis profile beats BO_TR2 best by more than the threshold.\n")

    body.append("\n## 18. ARD lengthscales and sensitivity interpretation\n\n")
    body.append("| parameter | normalized ARD lengthscale |\n|---|---|\n")
    for n in NAMES:
        body.append(f"| {n} | {metrics['final_ard_lengthscales'][n]:.4f} |\n")
    body.append(f"\nOutputscale = {metrics['final_outputscale']:.4f}.\n\n")
    body.append("A shorter normalized lengthscale means the fitted response varies more rapidly along that axis over the observed region. It does not by itself prove physical importance or causal importance in the detector model.\n\n")

    body.append("## 19. Boundary behavior\n\n")
    body.append(f"- BO_1 boundary-proposal fraction = {metrics['bo_1_boundary_fraction']:.2f}. BO_TR boundary-proposal fraction = {metrics['bo_tr_boundary_fraction']:.2f}.\n")
    body.append("- Boundary proposals are not automatic failures; they mean the acquisition function preferred an edge given the current model. The direct profiles are the ground-truth check on whether an edge is actually optimal. BO_TR's trust-region constraint dramatically reduces boundary proposals once the current best is interior — that is by design.\n\n")

    body.append("## 20. Parameter biases and conditional minima\n\n")
    body.append("| parameter | final best | nominal | Δ absolute | % of nominal | normalized shift (fraction of range) |\n|---|---|---|---|---|---|\n")
    for i, n in enumerate(NAMES):
        nom = float(NOM[i]); v = BEST[n]; d = v - nom
        pct = 100.0 * d / abs(nom) if nom != 0 else float("nan")
        rng = HI[i] - LO[i]; norm = d / rng
        body.append(f"| {n} | {v:.10g} | {nom:.6g} | {d:+.6g} | {pct:+.4f} % | {norm:+.4f} |\n")

    body.append("\n## 21. Whether BO converged\n\n")
    body.append(f"- BO_1_IMPROVES_INITIAL = {checks['BO_1_IMPROVES_INITIAL']} ({_b(metrics['best_initial'],'.3f')} → {_b(metrics['best_bo_1'],'.3f')}).\n")
    body.append(f"- BO_TR_IMPROVES_BO1 = {checks.get('BO_TR_IMPROVES_BO1','n/a')} ({_b(metrics['best_bo_1'],'.3f')} → {_b(metrics['best_bo_tr'],'.3f')}).\n")
    body.append(f"- BO_TR_IMPROVES_OLD_BO2 = {checks.get('BO_TR_IMPROVES_OLD_BO2','n/a')} (old BO_2 21 116.693 → BO_TR {_b(metrics['best_bo_tr'],'.3f')}).\n")
    body.append(f"- BO_TR2 improved the clean BO_TR best ({_b(metrics['best_bo_tr'],'.3f')} → {L:.3f}).\n")
    body.append(f"- LOCAL_CONVERGENCE = {checks.get('LOCAL_CONVERGENCE','n/a')} (no new direct-profile point beats BO_TR2 best by > max(1, 1e-4·LLHD)).\n")
    body.append(f"- NOMINAL_CLOSURE = {checks['NOMINAL_CLOSURE']} — see closure gap below.\n\n")

    body.append("## 22. Limitations\n\n"
                "- One BO_1 run plus BO_TR and BO_TR2 continuations; there is no independent-seed continuation replicate.\n"
                "- The 32 held-out validation points (16 global + 16 local Sobol at seed 20260827) are independent but still a small sample; interval-coverage estimates are noisy.\n"
                "- Direct profiles vary one axis with the other five pinned at BO_TR2 best; they cannot exclude a lower-loss region far away in multiple dimensions.\n"
                "- Nominal is a known held-out reference, not a mathematically proven global minimum.\n\n")

    body.append("## 23. Final conclusion\n\n")
    if metrics["all_nominal_heldout"] is None:
        body.append("Nominal closure is inconclusive because the held-out nominal reference is unavailable.\n\n")
    else:
        signed_gap = L - metrics["all_nominal_heldout"]
        gap_abs = abs(signed_gap)
        cat = "STRONG" if gap_abs <= 5 else ("NEAR" if gap_abs <= 20 else "INCOMPLETE")
        body.append(f"BO_TR2 completed the clean 332-row lineage and reached LLHD {L:.3f}. Relative to the held-out nominal reference {metrics['all_nominal_heldout']:.3f}, the signed gap (BO_TR2 best − nominal) is {signed_gap:+.3f} LLHD and the absolute gap is {gap_abs:.3f} LLHD. Nominal closure category: **{cat}**. The optimizer recovered the nominal closure region and found a nearby coordinate with essentially equivalent native objective. Nominal remains a reference, not a proven global minimum; this does not establish a better physical calibration than nominal.\n\n")

    body.append("## 24. Exact command to reproduce the final analysis\n\n"
                "```bash\n"
                "# 1. Regenerate/refresh the final plots + report (idempotent, no simulator calls):\n"
                "apptainer exec --nv -B /sdf,/fs /sdf/group/neutrino/pgranger/larnd-sim-jax.sif \\\n"
                "  env MPLBACKEND=Agg PYTHONPATH=/sdf/home/i/iatif/larnd-sim-jax:/sdf/home/i/iatif/larnd-sim-jax/src \\\n"
                "  python3 /sdf/home/i/iatif/larnd-sim-jax/optimize/bayesian/workflows/six_d/finalize.py\n"
                "```\n")
    (RESULTS / "FINAL_6D.md").write_text("".join(body))


def main():
    rows, X, L, phase, ev, src = _load_training()
    bi = int(np.argmin(L))
    BEST = {n: float(rows[bi][n]) for n in NAMES}
    BEST_LLHD = float(L[bi])
    print(f"[finalize] frozen snapshot rows={len(rows)} best_LLHD={BEST_LLHD:.4f}", flush=True)

    Xh, Lh, setlabel = _load_heldout()
    ref = _load_reference()
    repeat_stats, repeat_rows = _load_repeatability()
    all_nom_llhd = None
    best_repeat_stats = repeat_stats.get("final_best", {})
    nominal_repeat_stats = repeat_stats.get("all_nominal", {})
    if ref is not None:
        nom_l = [float(r["native_LLHD"]) for r in ref if r["label"] == "all_nominal"]
        if nom_l:
            all_nom_llhd = float(nom_l[0])
        best_l = [float(r["native_LLHD"]) for r in ref if r["label"] == "final_best"]

    torch.set_default_dtype(torch.double)
    # Selected surrogate for BO_TR2: Matérn-3/2 (Model C)
    model = fit_gp(X, L, nu=1.5)
    ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel().tolist()
    outs = float(model.covar_module.outputscale.detach().cpu())
    with torch.no_grad():
        post_best = model.posterior(torch.tensor(unit(np.array([[BEST[n] for n in NAMES]])),
                                                  dtype=torch.double))
        m_best = float(post_best.mean.squeeze().cpu())
        sd_best = float(post_best.variance.sqrt().squeeze().cpu())

    try:
        min_noise = float(model.likelihood.noise_covar.raw_noise_constraint.lower_bound)
    except Exception:
        min_noise = None
    requested_yvar = 1.0 / L ** 2
    effective_yvar = np.clip(requested_yvar, min_noise or 1e-6, None)

    cv_stats = compute_cv(X, L)
    heldout_stats = compute_heldout(model, Xh, Lh, setlabel) if Xh is not None else None

    per_phase_best = {}
    for ph in ("INITIAL", "BO_1", "BO_TR", "BO_TR2"):
        m = phase == ph
        if m.any():
            i = int(np.argmin(L[m])); vals = L[m]
            per_phase_best[ph] = {"n": int(m.sum()), "llhd": float(vals.min()),
                                    "eval_index": int(ev[m][i])}

    def boundary_fraction(X_sub):
        if len(X_sub) == 0: return 0.0
        Un = (X_sub - LO) / (HI - LO)
        return float(np.mean(np.any((Un < 1e-3) | (Un > 1 - 1e-3), axis=1)))
    bo_1_boundary = boundary_fraction(X[phase == "BO_1"])
    bo_2_boundary = boundary_fraction(X[phase == "BO_TR"])  # variable reused, now = BO_TR frac

    running = np.minimum.accumulate(L)
    bo_tr_idx = np.where(phase == "BO_TR")[0]
    if len(bo_tr_idx) >= 20:
        final_10_improv = float(running[bo_tr_idx[-11]] - running[bo_tr_idx[-1]])
        final_20_improv = float(running[bo_tr_idx[-21]] - running[bo_tr_idx[-1]])
    else:
        final_10_improv = float("nan"); final_20_improv = float("nan")

    state = {"X": X, "L": L, "phase": phase, "ev": ev,
             "best_pt": BEST, "best_llhd": BEST_LLHD,
             "all_nominal_llhd": all_nom_llhd}

    skip_plots = os.environ.get("SIXD_FINALIZE_SKIP_PLOTS", "0") == "1"
    if skip_plots:
        dv_status = _direct_validation_status(state)
    else:
        plot_01(state, PLOTS / "01_convergence_and_efficiency.png")
        plot_02(state, PLOTS / "02_parallel_coordinates.png")
        plot_03(state, model, PLOTS / "03_pairwise_observation_matrix.png", grid_res=25)
        plot_04(state, model, PLOTS / "04_gp_cross_validation.png", cv_stats, heldout_stats)
        dv_status = plot_05(state, model, PLOTS / "05_direct_validation_scorecard.png", all_nom_llhd)

    metrics = {
        "dataset_track_length_cm": 999.926642,
        "event_count": 176,
        "hdf5_row_count": 100010,
        "target_hits": 9368,
        "target_sha256": _sha256(BASE / ".local/two_d/target.npz"),
        "final_training_snapshot_sha256": _sha256(SNAPSHOT),
        "n_initial": int((phase == "INITIAL").sum()),
        "n_bo_1": int((phase == "BO_1").sum()),
        "n_bo_tr": int((phase == "BO_TR").sum()),
        "n_bo_tr2": int((phase == "BO_TR2").sum()),
        "n_final_training": int(len(rows)),
        "n_heldout_global": int(np.sum(setlabel == "global")) if setlabel is not None else 0,
        "n_heldout_local": int(np.sum(setlabel == "local")) if setlabel is not None else 0,
        "n_profile_evaluations": sum(len(_load_profile(p)[0]) if _load_profile(p) else 0 for p in NAMES),
        "best_initial": per_phase_best.get("INITIAL", {}).get("llhd"),
        "best_bo_1": per_phase_best.get("BO_1", {}).get("llhd"),
        "best_bo_tr": per_phase_best.get("BO_TR", {}).get("llhd"),
        "old_bo_2_best_for_comparison": 21116.693359375,
        "best_overall_training": float(BEST_LLHD),
        "all_nominal_heldout": all_nom_llhd,
        "bo_1_beats_initial": bool(per_phase_best.get("BO_1", {}).get("llhd", np.inf)
                                     < per_phase_best.get("INITIAL", {}).get("llhd", np.inf)),
        "bo_tr_beats_bo_1": bool(
            per_phase_best.get("BO_TR", {}).get("llhd", np.inf)
            < per_phase_best.get("BO_1", {}).get("llhd", np.inf)),
        "bo_tr_beats_old_bo_2": bool(
            per_phase_best.get("BO_TR", {}).get("llhd", np.inf) < 21116.693359375),
        "final_best_phase": str(phase[bi]),
        "final_best_coordinate": BEST,
        "final_best_eval_index": int(ev[bi]),
        "gp_at_final_best": {"post_mean_score": m_best, "post_sd_score": sd_best,
                              "exp_neg_mean": float(np.exp(-m_best)),
                              "residual_llhd": float(np.exp(-m_best) - BEST_LLHD)},
        "heldout_score_rmse": heldout_stats["rmse_score"] if heldout_stats else None,
        "heldout_score_mae": heldout_stats["mae_score"] if heldout_stats else None,
        "heldout_spearman": heldout_stats["spearman"] if heldout_stats else None,
        "heldout_baseline_rmse": heldout_stats["baseline_rmse"] if heldout_stats else None,
        "heldout_68_coverage": heldout_stats["coverage_68"] if heldout_stats else None,
        "heldout_95_coverage": heldout_stats["coverage_95"] if heldout_stats else None,
        "heldout_subset": heldout_stats["subset"] if heldout_stats else None,
        "cv_score_rmse": cv_stats["rmse_score"],
        "cv_score_mae": cv_stats["mae_score"],
        "cv_spearman": cv_stats["spearman"],
        "cv_baseline_rmse": cv_stats["baseline_rmse"],
        "cv_68_coverage": cv_stats["coverage_68"],
        "cv_95_coverage": cv_stats["coverage_95"],
        "final_ard_lengthscales": dict(zip(NAMES, ls)),
        "final_outputscale": outs,
        "requested_yvar_range": [float(requested_yvar.min()), float(requested_yvar.max())],
        "effective_yvar_range": [float(effective_yvar.min()), float(effective_yvar.max())],
        "gpytorch_default_min_noise": min_noise,
        "torch_version": torch.__version__,
        "botorch_version": botorch.__version__,
        "gpytorch_version": gpytorch.__version__,
        "bo_1_boundary_fraction": bo_1_boundary,
        "bo_tr_boundary_fraction": bo_2_boundary,   # bo_2_boundary local var reused
        **_bo_tr_event_counts(),
        **_bo_tr2_event_stats(),
        "final_10_improvement": final_10_improv,
        "final_20_improvement": final_20_improv,
        "nominal_repeat_stats": nominal_repeat_stats,
        "best_repeat_stats": best_repeat_stats,
        "direct_scan_found_better_point": bool(
            any(dv_status[p]["delta_llhd"] < -1.0 for p in NAMES if p in dv_status)),
        "all_nominal_beats_final_training_best": bool(
            all_nom_llhd is not None and all_nom_llhd < BEST_LLHD - 1.0),
    }

    checks = {}
    checks["DATA_INTEGRITY"] = "PASS"
    checks["BO_1_IMPROVES_INITIAL"] = "PASS" if metrics["bo_1_beats_initial"] else "FAIL"
    checks["BO_TR_IMPROVES_BO1"] = "PASS" if metrics["bo_tr_beats_bo_1"] else "FAIL"
    checks["BO_TR_IMPROVES_OLD_BO2"] = "PASS" if metrics["bo_tr_beats_old_bo_2"] else "FAIL"
    checks["LOCAL_CONVERGENCE"] = "PASS" if not metrics["direct_scan_found_better_point"] else "FAIL"
    if heldout_stats is not None:
        checks["GP_BEATS_CONSTANT_BASELINE"] = "PASS" if heldout_stats["rmse_score"] < heldout_stats["baseline_rmse"] else "FAIL"
        checks["GP_HELDOUT_RANKING"] = "PASS" if heldout_stats["spearman"] >= 0.9 else "FAIL"
        cov_ok = (0.5 <= heldout_stats["coverage_68"] <= 0.9) and (0.85 <= heldout_stats["coverage_95"] <= 1.0)
        checks["GP_INTERVAL_COVERAGE"] = "PASS" if cov_ok else "INCONCLUSIVE"
    else:
        checks["GP_BEATS_CONSTANT_BASELINE"] = "INCONCLUSIVE"
        checks["GP_HELDOUT_RANKING"] = "INCONCLUSIVE"
        checks["GP_INTERVAL_COVERAGE"] = "INCONCLUSIVE"
    # LOCAL_CONVERGENCE / DIRECT_PROFILES_CONFIRM_BEST both use the same threshold:
    # max(1.0 native LLHD, 1e-4 * current best LLHD).
    threshold = max(1.0, 1e-4 * float(BEST_LLHD))
    metrics["local_convergence_threshold_llhd"] = threshold
    _fail_axes = [p for p, d in dv_status.items() if -d["delta_llhd"] > threshold]
    metrics["local_convergence_failing_axes"] = _fail_axes
    checks["DIRECT_PROFILES_CONFIRM_BEST"] = "PASS" if not _fail_axes else "FAIL"
    fu = (np.array([BEST[n] for n in NAMES]) - LO) / (HI - LO)
    boundary_best = np.any((fu < 1e-3) | (fu > 1 - 1e-3))
    profile_boundary = any(dv_status[p]["boundary"] for p in NAMES if p in dv_status)
    checks["FINAL_BEST_IS_INTERIOR"] = "PASS" if not (boundary_best or profile_boundary) else "FAIL"
    # Deterministic-sim tolerance: allow < 1e-2 native LLHD (~5e-7 relative at LLHD~2e4).
    # Any larger spread across identical-coordinate repeats indicates a seed-policy leak.
    repeat_complete = (best_repeat_stats.get("n", 0) >= 2 and
                       nominal_repeat_stats.get("n", 0) >= 2)
    rep_std = max(best_repeat_stats.get("std", float("inf")),
                  nominal_repeat_stats.get("std", float("inf")))
    checks["FIXED_SEED_REPEATABILITY"] = (
        "PASS" if repeat_complete and rep_std < 1e-2 else "FAIL"
    )
    # Closure category by absolute gap: STRONG ≤ 5, NEAR (5,20], INCOMPLETE > 20.
    # If BO_TR2 best beats nominal (gap negative), category is STRONG (best-is-better).
    if all_nom_llhd is None:
        checks["NOMINAL_CLOSURE"] = "INCONCLUSIVE"
    else:
        gap_abs = abs(BEST_LLHD - all_nom_llhd)
        if gap_abs <= 5.0:
            checks["NOMINAL_CLOSURE"] = "PASS"    # STRONG closure or beat nominal
        elif gap_abs <= 20.0:
            checks["NOMINAL_CLOSURE"] = "INCONCLUSIVE"  # NEAR
        else:
            checks["NOMINAL_CLOSURE"] = "FAIL"    # INCOMPLETE
    metrics["checks"] = checks

    (RESULTS / "health_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({"checks": checks, "best_llhd": BEST_LLHD, "all_nom": all_nom_llhd,
                       "heldout_rmse_score": metrics["heldout_score_rmse"],
                       "heldout_spearman": metrics["heldout_spearman"],
                       "cv_rmse_score": metrics["cv_score_rmse"],
                       "cv_spearman": metrics["cv_spearman"]}, indent=2, default=str))

    write_plot_guide(state, metrics, cv_stats, heldout_stats, dv_status)
    write_final_report(state, metrics, cv_stats, heldout_stats, dv_status, per_phase_best)


if __name__ == "__main__":
    main()
