"""Regenerate controlled-study plots from frozen optimizer histories only."""
from __future__ import annotations

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
PHASE_ENDS = (72, 172, 272, 332)


def format_llhd(value: float) -> str:
    return f"{value:,.1f}" if value >= 10000 else f"{value:.1f}"


def annotate_running_best(ax: plt.Axes, loss: np.ndarray, nominal: float) -> None:
    """Label the four cumulative phase-end minima and nominal reference."""
    running = np.minimum.accumulate(loss)
    offsets = ((6, 8), (6, 10), (-10, 30), (-72, 8))
    for end, offset in zip(PHASE_ENDS, offsets):
        value = float(running[end - 1])
        ax.annotate(
            f"{end}: {format_llhd(value)}", xy=(end, value), xytext=offset,
            textcoords="offset points", fontsize=7.8, color="#222222",
            ha="left", va="center",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.82}, zorder=7)
    ax.annotate(
        f"Nominal LLHD = {format_llhd(nominal)}",
        xy=(8, nominal), xytext=(0, 7), textcoords="offset points",
        fontsize=8.0, color="#007A58", ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
              "edgecolor": "none", "alpha": 0.82}, zorder=7)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_run(raw: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        raise RuntimeError(f"{raw}: invalid history lineage {counts}")
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


def convergence_and_efficiency(out: Path, loss: np.ndarray, phases: np.ndarray,
                               nominal: float, title: str) -> None:
    fig, (observed, best_ax) = plt.subplots(
        2, 1, figsize=(11, 7.2), sharex=True,
        gridspec_kw={"height_ratios": (1.45, 1.0)})
    index = np.arange(1, len(loss) + 1)
    for phase in PHASES:
        mask = phases == phase
        observed.scatter(index[mask], loss[mask], s=17, alpha=0.55,
                         color=COLORS[phase], label=PHASE_LABELS[phase], zorder=2)
    running = np.minimum.accumulate(loss)
    best_ax.plot(index, running, color="#111111", linewidth=2.4,
                 label="Running best", zorder=4)
    for axis in (observed, best_ax):
        axis.axhline(nominal, color="#009E73", linestyle="--", linewidth=1.8,
                     label="Nominal reference")
        for boundary in (72.5, 172.5, 272.5):
            axis.axvline(boundary, color="0.65", linestyle=":", linewidth=1.1)
        axis.set_yscale("log")
        axis.grid(alpha=0.22)
    best_idx = int(np.argmin(loss))
    best_ax.scatter(best_idx + 1, loss[best_idx], marker="*", s=170,
                    color="#D55E00", edgecolor="black", linewidth=0.7,
                    label="Final best", zorder=6)
    annotate_running_best(best_ax, loss, nominal)
    observed.set_ylabel("Observed native LLHD")
    best_ax.set_ylabel("Running-best native LLHD")
    best_ax.set_xlabel("Simulator evaluation")
    observed.set_title(f"{title} — Convergence and efficiency")
    observed.legend(loc="upper right", ncol=3, frameon=True, framealpha=0.95,
                    fontsize=8.5)
    best_ax.legend(loc="upper right", ncol=3, frameon=True, framealpha=0.95,
                   fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=260)
    plt.close(fig)


def presentation_convergence(out: Path, loss: np.ndarray, phases: np.ndarray,
                             nominal: float, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    index = np.arange(1, len(loss) + 1)
    running = np.minimum.accumulate(loss)
    ax.plot(index, running, color="#174A7E", linewidth=2.6, label="Running best")
    ax.axhline(nominal, color="#009E73", linestyle="--", linewidth=1.8,
               label="Nominal reference")
    for boundary, label, left, right in zip(
            (72.5, 172.5, 272.5), ("Global BO", "BO_TR", "BO_TR2"),
            (1, 72.5, 172.5), (72.5, 172.5, 272.5)):
        ax.axvline(boundary, color="0.65", linestyle=":", linewidth=1.1)
    for center, label in zip((36.5, 122.5, 222.5, 302.5),
                             ("Sobol", "Global BO", "BO_TR", "BO_TR2")):
        ax.text(center, 0.97, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color="#555555")
    best_idx = int(np.argmin(loss))
    ax.scatter(best_idx + 1, loss[best_idx], marker="*", s=170,
               color="#D55E00", edgecolor="black", linewidth=0.7,
               label="Final best", zorder=5)
    annotate_running_best(ax, loss, nominal)
    ax.set_yscale("log")
    ax.set_xlabel("Simulator evaluation")
    ax.set_ylabel("Running-best native LLHD")
    ax.set_title(f"{title} — Convergence")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=260)
    plt.close(fig)


def parallel_coordinates(out: Path, x: np.ndarray, loss: np.ndarray,
                         title: str) -> None:
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
    recovered = x[best_idx]
    rows = [
        ("Ab", f"{recovered[0]:.5f}", f"{NOM[0]:.5g}"),
        ("kb", f"{recovered[1]:.5f}", f"{NOM[1]:.5g}"),
        ("eField", f"{recovered[2]:.6f}", f"{NOM[2]:.5g}"),
        ("lifetime", f"{recovered[3]:.1f}", f"{NOM[3]:.0f}"),
        (r"$D_T$", f"{recovered[4]:.3e}", f"{NOM[4]:.3e}"),
        (r"$D_L$", f"{recovered[5]:.3e}", f"{NOM[5]:.3e}"),
    ]
    value_text = "Parameter    Recovered       Nominal\n" + "\n".join(
        f"{name:<10} {value:>11}  {nominal:>11}" for name, value, nominal in rows)
    ax.text(0.985, 0.975, value_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=7.7, family="monospace",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "0.75", "alpha": 0.92}, zorder=8)
    ax.set_xticks(axis, LABELS, rotation=18, ha="right")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Normalized coordinate within allowed range")
    ax.set_title(f"{title} — Six-dimensional observations")
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


def pairwise_matrix(out: Path, x: np.ndarray, loss: np.ndarray,
                    title: str) -> None:
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
    fig.suptitle(f"{title} — Observation projections and conditional final-GP surfaces",
                 y=0.995)
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.10, top=0.965, wspace=0.08, hspace=0.08)
    fig.savefig(out, dpi=240)
    plt.close(fig)


def cv_plot(out: Path, cv: dict[str, np.ndarray | float], title: str) -> None:
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
    ax.set_title(f"{title} — Five-fold GP cross-validation")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=260)
    plt.close(fig)


def parameter_recovery(out: Path, x: np.ndarray, loss: np.ndarray,
                       title: str) -> None:
    best = x[int(np.argmin(loss))]
    deviation = 100.0 * (best - NOM) / (HI - LO)
    axis = np.arange(6)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(axis, deviation, color="#3B82C4", width=0.65, label="Final best")
    ax.axhline(0, color="#009E73", linestyle="--", linewidth=2, label="Nominal")
    ax.set_xticks(axis, LABELS, rotation=16, ha="right")
    ax.set_ylabel("Deviation from nominal (% of allowed search range)")
    ax.set_title(f"{title} — Parameter recovery")
    ax.grid(axis="y", alpha=0.23)
    ax.legend(loc="best", frameon=True)
    for xpos, value in zip(axis, deviation):
        label = "0.000%" if abs(value) < 0.0005 else f"{value:+.3f}%"
        offset = 4 if value >= 0 else -5
        ax.annotate(label, xy=(xpos, value), xytext=(0, offset),
                    textcoords="offset points", ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=8.2,
                    color="#173B5F")
    pad = max(4.0, float(np.max(np.abs(deviation))) * 0.18)
    ax.set_ylim(float(min(deviation.min() - pad, -pad)), float(max(deviation.max() + pad, pad)))
    fig.subplots_adjust(bottom=0.24, left=0.11, right=0.98, top=0.90)
    fig.savefig(out, dpi=260)
    plt.close(fig)


class ResultLoader:
    """Load and validate one completed 332-observation history."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.metadata = json.loads((run_dir / "run.json").read_text())
        self.raw = Path(self.metadata["authoritative_history"])

    def load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return load_run(self.raw)


class AnalysisPlotter:
    """Generate the supported plot family from frozen observations only."""

    def __init__(self, run_dir: Path, summary: dict[str, str]):
        self.run_dir = run_dir
        self.loader = ResultLoader(run_dir)
        self.metadata = self.loader.metadata
        self.repeat = int(self.metadata["repeat"])
        self.nominal = float(summary["nominal_llhd"])
        self.title = f"~{self.metadata['data_size_label']} — Repeat {self.repeat}"
        self.plots = run_dir / "plots"

    def render(self) -> dict[str, object]:
        self.plots.mkdir(exist_ok=True)
        x, loss, phases = self.loader.load()
        convergence_and_efficiency(
            self.plots / "01_convergence_and_efficiency.png", loss, phases,
            self.nominal, self.title)
        parallel_coordinates(
            self.plots / "02_parallel_coordinates.png", x, loss, self.title)
        presentation_convergence(
            self.plots / "05_convergence_presentation.png", loss, phases,
            self.nominal, self.title)
        parameter_recovery(
            self.plots / "06_parameter_recovery.png", x, loss, self.title)
        return {"run": str(self.run_dir), "n": len(loss),
                "best_LLHD": float(loss.min()),
                "plots": sorted(path.name for path in self.plots.glob("*.png"))}


def main() -> None:
    summary_path = STUDY / "comparison/optimizer_repeats.csv"
    summary_rows = read_rows(summary_path)
    by_run = {(row["data_size_label"].replace(" ", ""), int(row["repeat"])): row
              for row in summary_rows}
    run_dirs = sorted(STUDY.glob("*cm/repeat_*/run.json"))
    if len(run_dirs) != 8:
        raise RuntimeError(f"expected eight run metadata files, found {len(run_dirs)}")
    rendered = []
    for metadata_path in run_dirs:
        run_dir = metadata_path.parent
        metadata = json.loads(metadata_path.read_text())
        key = (metadata["data_size_label"], int(metadata["repeat"]))
        rendered.append(AnalysisPlotter(run_dir, by_run[key]).render())
    print(json.dumps(rendered, indent=2))


if __name__ == "__main__":
    main()
