"""Regenerate controlled-study plots from frozen optimizer histories only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from scipy.stats import spearmanr


ROOT = Path("/sdf/home/i/iatif/REAL_BEACON/bayesian")
STUDY = ROOT / "batch_study"
WORKFLOW = ROOT / "workflows/six_d"
sys.path.insert(0, str(ROOT / "workflows"))
sys.path.insert(0, str(WORKFLOW))
from build_6d import fit_gp, unit  # noqa: E402

NAMES = ("Ab", "kb", "eField", "lifetime", "tran_diff", "long_diff")
LABELS = ("Ab", "kb", "E field", "Lifetime", "Transverse diffusion", "Longitudinal diffusion")
UNITS = ("", "kV g / (MeV cm³)", "kV/cm", "µs", "cm²/µs", "cm²/µs")
LO = np.array([0.75, 0.03, 0.49, 400.0, 3.0e-6, 1.0e-6])
HI = np.array([0.90, 0.08, 0.51, 6000.0, 15.0e-6, 10.0e-6])
NOM = np.array([0.80, 0.0486, 0.50, 2200.0, 8.8e-6, 4.0e-6])
PHASES = ("INITIAL", "BO", "BO_TR", "BO_TR2")
PHASE_LABELS = {"INITIAL": "INITIAL / Sobol", "BO": "BO / global",
                "BO_TR": "BO_TR", "BO_TR2": "BO_TR2"}
COLORS = {"INITIAL": "#8C8C8C", "BO": "#3B82C4", "BO_TR": "#E69F00", "BO_TR2": "#8E5AA9"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_run(experiment: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = experiment / "history/raw"
    main = read_rows(raw / "traces/bo_6d_history.csv")
    tr = read_rows(raw / "continuation/bo_tr_history.csv")
    tr2 = read_rows(raw / "continuation/bo_tr2_history.csv")
    rows = main + tr + tr2
    phases = np.array([({"initial": "INITIAL", "BO": "BO"}.get(row["phase"], row["phase"]))
                       for row in rows])
    x = np.array([[float(row[name]) for name in NAMES] for row in rows])
    loss = np.array([float(row["native_LLHD"]) for row in rows])
    expected = {"INITIAL": 72, "BO": 100, "BO_TR": 100, "BO_TR2": 60}
    counts = {phase: int(np.sum(phases == phase)) for phase in PHASES}
    if len(rows) != 332 or counts != expected:
        raise RuntimeError(f"{experiment.name}: invalid history lineage {counts}")
    return x, loss, phases


def cross_validate(x: np.ndarray, loss: np.ndarray) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(20260813)
    order = rng.permutation(len(loss))
    prediction = np.full(len(loss), np.nan)
    sigma = np.full(len(loss), np.nan)
    fold_id = np.full(len(loss), -1)
    for number, fold in enumerate(np.array_split(order, 5), start=1):
        train = np.setdiff1d(order, fold)
        model = fit_gp(x[train], loss[train], seed=20260813 + number, nu=1.5)
        with torch.no_grad():
            posterior = model.posterior(torch.tensor(unit(x[fold]), dtype=torch.double))
            prediction[fold] = posterior.mean.squeeze(-1).cpu().numpy()
            sigma[fold] = posterior.variance.sqrt().squeeze(-1).cpu().numpy()
        fold_id[fold] = number
    actual = -np.log(loss)
    residual = prediction - actual
    return {
        "actual": actual, "prediction": prediction, "sigma": sigma, "fold": fold_id,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "baseline_rmse": float(np.sqrt(np.mean((actual - actual.mean())**2))),
        "spearman": float(spearmanr(actual, prediction)[0]),
    }


def nominal_value(seed: int) -> float:
    record = json.loads((STUDY / "shared" / f"seed_validation_{seed}.json").read_text())
    return float(np.mean(record["nominal_native_LLHD_repeats"]))


def convergence(out: Path, x: np.ndarray, loss: np.ndarray, phases: np.ndarray,
                repeat: int, nominal: float, data_label: str,
                poster: bool = False) -> None:
    size = (11, 6.3) if not poster else (10, 5.5)
    fig, ax = plt.subplots(figsize=size)
    index = np.arange(1, len(loss) + 1)
    for phase in PHASES:
        mask = phases == phase
        ax.scatter(index[mask], loss[mask], s=18 if not poster else 14,
                   alpha=0.55, color=COLORS[phase], label=PHASE_LABELS[phase], zorder=2)
    running = np.minimum.accumulate(loss)
    ax.plot(index, running, color="#111111", linewidth=2.4, label="Running best", zorder=4)
    ax.axhline(nominal, color="#009E73", linestyle="--", linewidth=1.8,
               label="Nominal reference")
    best_idx = int(np.argmin(loss))
    ax.scatter(best_idx + 1, loss[best_idx], marker="*", s=190, color="#D55E00",
               edgecolor="black", linewidth=0.7, label="Final best", zorder=6)
    for boundary in (72.5, 172.5, 272.5):
        ax.axvline(boundary, color="0.65", linestyle=":", linewidth=1.1)
    ax.set_xlabel("Simulator evaluation")
    ax.set_ylabel("Native LLHD")
    ax.set_yscale("log")
    ax.set_title(f"{data_label} calibration convergence — optimizer repeat {repeat}")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=260)
    plt.close(fig)


def parallel_coordinates(out: Path, x: np.ndarray, loss: np.ndarray) -> None:
    u = (x - LO) / (HI - LO)
    nominal_u = (NOM - LO) / (HI - LO)
    best_idx = int(np.argmin(loss))
    cutoff = np.quantile(loss, 0.10)
    top = loss <= cutoff
    fig, ax = plt.subplots(figsize=(11, 6.5))
    axis = np.arange(len(NAMES))
    for row in u[~top]:
        ax.plot(axis, row, color="0.70", alpha=0.12, linewidth=0.7)
    for row in u[top]:
        ax.plot(axis, row, color="#3B82C4", alpha=0.32, linewidth=1.0)
    ax.plot(axis, u[best_idx], color="#D55E00", marker="o", linewidth=3.0, markersize=6)
    ax.plot(axis, nominal_u, color="#009E73", marker="s", linestyle="--", linewidth=2.2, markersize=5)
    ax.set_xticks(axis, LABELS, rotation=18, ha="right")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Normalized coordinate within allowed range")
    ax.set_title("Six-dimensional optimizer observations")
    ax.grid(axis="y", alpha=0.25)
    handles = [
        Line2D([0], [0], color="0.65", lw=2, label="Other training observations"),
        Line2D([0], [0], color="#3B82C4", lw=2, label="Best 10%"),
        Line2D([0], [0], color="#D55E00", marker="o", lw=3, label="Final best"),
        Line2D([0], [0], color="#009E73", marker="s", ls="--", lw=2, label="Nominal"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=4, frameon=True)
    fig.subplots_adjust(bottom=0.28, left=0.10, right=0.98, top=0.91)
    fig.savefig(out, dpi=260)
    plt.close(fig)


def pairwise_matrix(out: Path, x: np.ndarray, loss: np.ndarray) -> None:
    u = (x - LO) / (HI - LO)
    score = -np.log(loss)
    best_idx = int(np.argmin(loss))
    best_u = u[best_idx]
    nominal_u = (NOM - LO) / (HI - LO)
    model = fit_gp(x, loss, seed=20260840, nu=1.5)
    vmin, vmax = float(score.min()), float(score.max())
    fig, axes = plt.subplots(6, 6, figsize=(14.5, 13.5))
    grid = np.linspace(0, 1, 31)
    gx, gy = np.meshgrid(grid, grid)
    image = None
    for row in range(6):
        for col in range(6):
            ax = axes[row, col]
            if row == col:
                ax.hist(u[:, col], bins=18, color="#829AB1", alpha=0.85)
                ax.axvline(best_u[col], color="#D55E00", lw=2)
                ax.axvline(nominal_u[col], color="#009E73", lw=1.7, ls="--")
                ax.set_yticks([])
            elif row > col:
                image = ax.scatter(u[:, col], u[:, row], c=score, cmap="viridis",
                                   vmin=vmin, vmax=vmax, s=8, alpha=0.60, rasterized=True)
                ax.scatter(best_u[col], best_u[row], marker="*", s=85, color="#D55E00",
                           edgecolor="black", linewidth=0.4)
                ax.scatter(nominal_u[col], nominal_u[row], marker="x", s=38,
                           color="#009E73", linewidth=1.7)
            else:
                points = np.repeat(best_u[None, :], gx.size, axis=0)
                points[:, col] = gx.ravel()
                points[:, row] = gy.ravel()
                with torch.no_grad():
                    mean = model.posterior(torch.tensor(points, dtype=torch.double)).mean
                surface = mean.squeeze(-1).cpu().numpy().reshape(gx.shape)
                image = ax.pcolormesh(grid, grid, surface, shading="auto", cmap="viridis",
                                      vmin=vmin, vmax=vmax, rasterized=True)
                ax.scatter(best_u[col], best_u[row], marker="*", s=85, color="#D55E00",
                           edgecolor="black", linewidth=0.4)
                ax.scatter(nominal_u[col], nominal_u[row], marker="x", s=38,
                           color="#009E73", linewidth=1.7)
            ax.set_xlim(0, 1)
            if row != col:
                ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)
            if row == 5:
                ax.set_xlabel(LABELS[col], fontsize=8)
            if col == 0 and row > 0:
                ax.set_ylabel(LABELS[row], fontsize=8)
    cbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.015)
    cbar.set_label("Score −ln(LLHD): observations / conditional GP mean")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.45", markersize=5,
               label="Real 6D observations (lower triangle)"),
        Line2D([0], [0], marker="*", color="#D55E00", markeredgecolor="black", markersize=10,
               label="Final best"),
        Line2D([0], [0], marker="x", color="#009E73", markersize=7, label="Nominal"),
        Line2D([0], [0], color="#5B8E7D", lw=5,
               label="Conditional final-GP mean (upper; other four fixed at best)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True,
               bbox_to_anchor=(0.48, 0.006), fontsize=9)
    fig.suptitle("Observation projections and conditional final-GP surfaces", y=0.995)
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.10, top=0.965, wspace=0.08, hspace=0.08)
    fig.savefig(out, dpi=240)
    plt.close(fig)


def cv_plot(out: Path, cv: dict[str, np.ndarray | float]) -> None:
    actual = np.asarray(cv["actual"])
    prediction = np.asarray(cv["prediction"])
    sigma = np.asarray(cv["sigma"])
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    ax.errorbar(actual, prediction, yerr=1.96 * sigma, fmt="o", ms=4, alpha=0.42,
                color="#3B82C4", ecolor="#8DB9D8", elinewidth=0.7, capsize=0,
                label="Five-fold GP predictions (95% interval)")
    low = float(min(actual.min(), prediction.min()))
    high = float(max(actual.max(), prediction.max()))
    ax.plot([low, high], [low, high], "--", color="#222222", lw=1.8, label="Ideal y = x")
    stats = (f"RMSE = {cv['rmse']:.4f}\nSpearman ρ = {cv['spearman']:.4f}\n"
             f"Constant baseline RMSE = {cv['baseline_rmse']:.4f}")
    ax.text(0.04, 0.96, stats, transform=ax.transAxes, va="top", ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92})
    ax.set_xlabel("Actual score −ln(LLHD)")
    ax.set_ylabel("Predicted score −ln(LLHD)")
    ax.set_title("Five-fold GP cross-validation")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=260)
    plt.close(fig)


def parameter_recovery(out: Path, x: np.ndarray, loss: np.ndarray, repeat: int) -> None:
    best = x[int(np.argmin(loss))]
    deviation = 100.0 * (best - NOM) / (HI - LO)
    axis = np.arange(6)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(axis, deviation, color="#3B82C4", width=0.65, label="Final best")
    ax.axhline(0, color="#009E73", linestyle="--", linewidth=2, label="Nominal")
    ax.bar_label(bars, labels=[f"{value:+.1f}%" for value in deviation], padding=3, fontsize=9)
    ax.set_xticks(axis, LABELS, rotation=16, ha="right")
    ax.set_ylabel("Deviation from nominal (% of allowed search range)")
    ax.set_title(f"Recovered parameter coordinate — optimizer repeat {repeat}")
    ax.grid(axis="y", alpha=0.23)
    ax.legend(loc="best", frameon=True)
    pad = max(4.0, float(np.max(np.abs(deviation))) * 0.18)
    ax.set_ylim(float(min(deviation.min() - pad, -pad)), float(max(deviation.max() + pad, pad)))
    fig.subplots_adjust(bottom=0.24, left=0.11, right=0.98, top=0.90)
    fig.savefig(out, dpi=260)
    plt.close(fig)


class ResultLoader:
    """Load and validate one completed 332-observation history."""

    def __init__(self, experiment: Path):
        self.experiment = experiment

    def load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return load_run(self.experiment)


class AnalysisPlotter:
    """Generate the supported plot family from frozen observations only."""

    def __init__(self, experiment: Path, repeat: int, nominal: float, data_label: str):
        self.experiment = experiment
        self.repeat = repeat
        self.nominal = nominal
        self.data_label = data_label
        self.plots = experiment / "plots"

    def render(self) -> dict[str, object]:
        self.plots.mkdir(exist_ok=True)
        x, loss, phases = ResultLoader(self.experiment).load()
        convergence(self.plots / "01_convergence_and_efficiency.png", x, loss, phases,
                    self.repeat, self.nominal, self.data_label)
        parallel_coordinates(self.plots / "02_parallel_coordinates.png", x, loss)
        pairwise_matrix(self.plots / "03_pairwise_observation_matrix.png", x, loss)
        cv = cross_validate(x, loss)
        cv_plot(self.plots / "04_gp_cross_validation.png", cv)
        convergence(self.plots / "poster_02_convergence.png", x, loss, phases,
                    self.repeat, self.nominal, self.data_label, poster=True)
        parameter_recovery(self.plots / "poster_03_parameter_recovery.png", x, loss, self.repeat)
        return {"experiment": self.experiment.name, "n": len(loss),
                "best_LLHD": float(loss.min()),
                "plots": sorted(path.name for path in self.plots.glob("*.png"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--nominal-llhd", type=float)
    parser.add_argument("--data-label")
    args = parser.parse_args()
    if args.experiment_dir is not None:
        if args.repeat is None or args.nominal_llhd is None or args.data_label is None:
            parser.error("single-experiment mode requires --repeat, --nominal-llhd, and --data-label")
        print(json.dumps(AnalysisPlotter(args.experiment_dir.resolve(), args.repeat,
                                         args.nominal_llhd, args.data_label).render(), indent=2))
        return
    for dirname, repeat in (("200cm_seed0", 0), ("200cm_seed1", 1)):
        experiment = STUDY / dirname
        plots = experiment / "plots"
        plots.mkdir(exist_ok=True)
        print(json.dumps(AnalysisPlotter(experiment, repeat, nominal_value(repeat),
                                         "200 cm").render(), indent=2))


if __name__ == "__main__":
    main()
