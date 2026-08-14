"""Finalize the 2D pair: fit the final GP on the completed history and produce
the five presentation plots + compact summaries.

Reads `.local/two_d/<pair>/{history.csv, direct_profile_parameter{1,2}.csv}`
and writes `results/two_d/<pair>/{01…05}.png` + `summary.csv` + `summary.json`.

Never runs the simulator; never submits Slurm.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BAY, DTYPE, fit_gp
from botorch.acquisition.logei import qLogExpectedImprovement

LEGEND_KW = dict(frameon=True, framealpha=1.0, facecolor="white",
                 edgecolor="black", fontsize=8)


def _plot(cfg, X, L, phase_arr, direct_scans, outdir: Path, model, U_grid,
          A, B, pred_llhd, alpha, next_phys):
    NAMES = cfg["tunable_params"]
    UNITS = cfg["units"]
    NOM = cfg["nominal"]
    LO = cfg["lower"]; HI = cfg["upper"]
    label_x = f"{NAMES[0]} [{UNITS[0]}]" if UNITS[0] else NAMES[0]
    label_y = f"{NAMES[1]} [{UNITS[1]}]" if UNITS[1] else NAMES[1]
    sobol_n = int(cfg["sobol_n"])
    best_i = int(np.argmin(L))

    # 01 trajectory
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    it = np.arange(1, len(L) + 1)
    for j, ax in enumerate(axs):
        ax.plot(it[:sobol_n], X[:sobol_n, j], "o", ms=4, color="tab:blue", label="Sobol")
        ax.plot(it[sobol_n:], X[sobol_n:, j], "-", lw=1.3, color="tab:orange", label="BO")
        ax.axhline(NOM[j], ls="--", color="gray", lw=1, label="Nominal")
        ax.axvline(sobol_n + 0.5, ls=":", color="black", lw=1, label="Init")
        ax.set_ylabel(f"{NAMES[j]} [{UNITS[j]}]" if UNITS[j] else NAMES[j])
        ax.grid(alpha=0.2)
        if j == 0: ax.legend(loc="upper right", **LEGEND_KW)
    axs[-1].set_xlabel("Iteration")
    fig.suptitle(f"{cfg['pair']} — Parameters")
    fig.tight_layout()
    fig.savefig(outdir / "01_parameter_trajectory.png", dpi=180); plt.close(fig)

    # 02 loss
    fig, ax = plt.subplots(figsize=(9, 4.5))
    running = np.minimum.accumulate(L)
    ax.plot(it, L, "o-", ms=3, lw=0.9, color="tab:orange", alpha=0.55, label="LLHD")
    ax.plot(it, running, "-", lw=1.8, color="tab:red", label="Running best")
    ax.axvline(sobol_n + 0.5, ls=":", color="black", lw=1, label="Init")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration"); ax.set_ylabel("LLHD")
    ax.grid(alpha=0.2, which="both")
    ax.legend(loc="upper right", **LEGEND_KW)
    ax.set_title(f"{cfg['pair']} — Loss")
    fig.tight_layout()
    fig.savefig(outdir / "02_loss_trajectory.png", dpi=180); plt.close(fig)

    gp_min_idx = np.unravel_index(np.argmax(-np.log(pred_llhd)), pred_llhd.shape)
    gp_min = (A[gp_min_idx], B[gp_min_idx])

    # 03 GP posterior
    fig, ax = plt.subplots(figsize=(9.8, 6))
    im = ax.contourf(A, B, pred_llhd, 30, cmap="viridis")
    cb = fig.colorbar(im, ax=ax); cb.set_label("LLHD-like")
    ax.scatter(X[:, 0], X[:, 1], s=14, c="white", edgecolor="black", lw=0.5, label="Observed")
    ax.scatter(NOM[0], NOM[1], marker="*", s=200, c="gold", edgecolor="black", label="Nominal")
    ax.scatter(X[best_i, 0], X[best_i, 1], marker="D", s=80, c="red", edgecolor="black", label="Best")
    ax.scatter(gp_min[0], gp_min[1], marker="X", s=80, c="cyan", edgecolor="black", label="GP min")
    ax.set_xlabel(label_x); ax.set_ylabel(label_y)
    ax.set_title(f"{cfg['pair']} — GP")
    ax.legend(loc="upper left", bbox_to_anchor=(1.18, 1.0), borderaxespad=0., **LEGEND_KW)
    fig.tight_layout()
    fig.savefig(outdir / "03_final_gp_posterior.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    # 04 acquisition
    fig, ax = plt.subplots(figsize=(9.8, 6))
    im = ax.contourf(A, B, alpha, 30, cmap="magma")
    cb = fig.colorbar(im, ax=ax); cb.set_label("qLogEI")
    ax.scatter(X[:, 0], X[:, 1], s=14, c="white", edgecolor="black", lw=0.5, label="Observed")
    ax.scatter(X[best_i, 0], X[best_i, 1], marker="D", s=80, c="red", edgecolor="black", label="Best")
    ax.scatter(next_phys[0], next_phys[1], marker="^", s=100, c="cyan", edgecolor="black", label="Next")
    ax.set_xlabel(label_x); ax.set_ylabel(label_y)
    ax.set_title(f"{cfg['pair']} — qLogEI")
    ax.legend(loc="upper left", bbox_to_anchor=(1.18, 1.0), borderaxespad=0., **LEGEND_KW)
    fig.tight_layout()
    fig.savefig(outdir / "04_final_acquisition.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    # 05 direct validation
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, (ax, scan, name, unit, nom) in enumerate(zip(
            axs, direct_scans, NAMES, UNITS, NOM)):
        xv = np.asarray(scan["x"]); yv = np.asarray(scan["y"])
        order = np.argsort(xv); xv, yv = xv[order], yv[order]
        ax.plot(xv, yv, "o-", color="black", ms=4, lw=1, label="Direct")
        ax.axvline(nom, ls="--", color="gray", lw=1, label="Nominal")
        ax.axvline(X[best_i, j], ls=":", color="red", lw=1, label="Best")
        ax.set_xlabel(f"{name} [{unit}]" if unit else name)
        ax.set_ylabel("LLHD")
        ax.set_yscale("log")
        ax.grid(alpha=0.2, which="both")
        if j == 0: ax.legend(loc="upper right", **LEGEND_KW)
    fig.suptitle(f"{cfg['pair']} — Validation")
    fig.tight_layout()
    fig.savefig(outdir / "05_direct_validation.png", dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    NAMES = cfg["tunable_params"]
    LO = np.array(cfg["lower"]); HI = np.array(cfg["upper"])

    src = BAY / f".local/two_d/{cfg['pair']}"
    dst = BAY / f"results/two_d/{cfg['pair']}"
    dst.mkdir(parents=True, exist_ok=True)

    hist = list(csv.DictReader((src / "history.csv").open()))
    X = np.array([[float(r[NAMES[0]]), float(r[NAMES[1]])] for r in hist])
    L = np.array([float(r["native_LLHD"]) for r in hist])
    phase_arr = np.array([r["phase"] for r in hist])

    Xu = (X - LO) / (HI - LO)
    model = fit_gp(Xu, L, seed=cfg["bo_seed"], ard_num_dims=2)

    G = 121
    a = np.linspace(0, 1, G); b = np.linspace(0, 1, G)
    aa, bb = np.meshgrid(a, b)
    U_grid = np.column_stack([aa.ravel(), bb.ravel()])
    with torch.no_grad():
        p = model.posterior(torch.tensor(U_grid, dtype=DTYPE))
        mu = p.mean.squeeze(-1).cpu().numpy().reshape(G, G)
    A = LO[0] + aa * (HI[0] - LO[0])
    B = LO[1] + bb * (HI[1] - LO[1])
    pred_llhd = np.exp(-mu)

    y = -torch.log(torch.tensor(L, dtype=DTYPE)).view(-1, 1)
    acq = qLogExpectedImprovement(model, best_f=y.max())
    Xacq = torch.tensor(U_grid, dtype=DTYPE).unsqueeze(1)
    with torch.no_grad():
        alpha = acq(Xacq).cpu().numpy().reshape(G, G)
    ai = np.unravel_index(np.argmax(alpha), alpha.shape)
    next_phys = (A[ai], B[ai])

    scans = []
    for j in (1, 2):
        rows = list(csv.DictReader((src / f"direct_profile_parameter{j}.csv").open()))
        scans.append({"x": [float(r[NAMES[j-1]]) for r in rows],
                      "y": [float(r["llhd"]) for r in rows]})

    _plot(cfg, X, L, phase_arr, scans, dst, model, U_grid, A, B, pred_llhd, alpha, next_phys)

    best_i = int(np.argmin(L))
    summary = {"pair": cfg["pair"],
               "best_point": {NAMES[j]: float(X[best_i, j]) for j in (0, 1)},
               "best_native_LLHD": float(L[best_i]),
               "n_training": int(len(L)),
               "sobol_n": int(cfg["sobol_n"]),
               "bo_iters": int(cfg["bo_iters"])}
    (dst / "summary.json").write_text(json.dumps(summary, indent=2))
    with (dst / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["best_" + NAMES[0], summary["best_point"][NAMES[0]]])
        w.writerow(["best_" + NAMES[1], summary["best_point"][NAMES[1]]])
        w.writerow(["best_native_LLHD", summary["best_native_LLHD"]])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
