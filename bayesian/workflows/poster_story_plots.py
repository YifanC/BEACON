"""Render the retained 1D/2D development-study poster figures.

Plotting only: this module reads retained raw histories.  It never imports or
calls the detector simulator and it never performs Bayesian optimization.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch
import yaml


BAY = Path(__file__).resolve().parents[1]
ONE_D_RAW = BAY / ".local/one_d/eField_scan_51.pkl"
ONE_D_OUTPUT = BAY / "results/one_d/poster_01_efield_scan.png"
TWO_D_CONFIG = BAY / "workflows/two_d/configs/Ab_kb.yaml"
TWO_D_HISTORY = BAY / ".local/two_d/Ab_kb/history.csv"
TWO_D_OUTPUT = BAY / "results/two_d/poster_03_multidimensional_scaling.png"

COLORS = {
    "navy": "#174A7E",
    "orange": "#E07A1F",
    "red": "#B33B3B",
    "dark": "#26323D",
    "gray": "#666666",
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 17,
        "axes.titlesize": 25,
        "axes.titleweight": "bold",
        "axes.labelsize": 21,
        "axes.labelweight": "bold",
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 17,
        "axes.linewidth": 1.6,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, path: Path) -> None:
    if not path.parent.is_dir():
        raise RuntimeError(f"existing result directory is missing: {path.parent}")
    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.16,
                facecolor="white")
    plt.close(fig)


def load_efield_scan(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with path.open("rb") as stream:
        history = pickle.load(stream)
    cfg = vars(history["config"])
    n_grid = int(cfg["iterations"])
    n_batches = int(cfg["max_nbatch"])
    values = np.asarray(history["eField_iter"][1:], dtype=float)
    losses = np.asarray(history["losses_iter"], dtype=float)
    expected = n_grid * n_batches
    if n_grid != 51 or n_batches != 50:
        raise RuntimeError(f"unexpected scan design: {n_grid} points, {n_batches} batches")
    if values.size != expected or losses.size != expected:
        raise RuntimeError(
            f"scan shape mismatch: values={values.size}, losses={losses.size}, expected={expected}"
        )
    x_by_batch = values.reshape(n_batches, n_grid)
    loss_by_batch = losses.reshape(n_batches, n_grid)
    grid = x_by_batch[0]
    if not np.allclose(x_by_batch, grid[None, :], rtol=0.0, atol=1e-14):
        raise RuntimeError("eField coordinates differ across batches")
    expected_grid = np.linspace(0.45, 0.55, 51)
    if not np.allclose(grid, expected_grid, rtol=0.0, atol=1e-14):
        raise RuntimeError("scan grid is not the authoritative 0.45--0.55, 51-point grid")
    nominal_hits = np.flatnonzero(np.isclose(grid, 0.5, rtol=0.0, atol=1e-14))
    if nominal_hits.tolist() != [25]:
        raise RuntimeError(f"nominal eField is not exactly the central sample: {nominal_hits}")
    mean_loss = np.mean(loss_by_batch, axis=0)
    return grid, mean_loss, cfg


def figure_efield_scan() -> dict:
    grid, mean_loss, cfg = load_efield_scan(ONE_D_RAW)
    best_i = int(np.argmin(mean_loss))
    best_x = float(grid[best_i])
    best_loss = float(mean_loss[best_i])
    nominal_i = int(np.flatnonzero(grid == 0.5)[0])
    nominal_loss = float(mean_loss[nominal_i])

    fig, ax = plt.subplots(figsize=(11.8, 7.8), constrained_layout=True)
    ax.plot(grid, mean_loss, color=COLORS["orange"], lw=4.2,
            label="Mean loss", zorder=3)
    ax.axvline(0.5, color=COLORS["red"], ls="--", lw=3.0,
               label="Nominal ($E=0.5$ kV/cm)", zorder=2)
    ax.axvline(best_x, color=COLORS["dark"], ls=":", lw=3.2,
               label="Lowest sampled mean", zorder=2)
    ax.set_title("One-dimensional electric-field scan", pad=16)
    ax.set_xlabel("eField [kV/cm]")
    ax.set_ylabel("Mean LLHD loss")
    ax.set_xlim(float(grid[0]), float(grid[-1]))
    ax.grid(color="#D9DEE4", lw=1.0, alpha=0.9)
    ax.legend(loc="upper right", frameon=False)
    save(fig, ONE_D_OUTPUT)
    return {
        "source": str(ONE_D_RAW),
        "output": str(ONE_D_OUTPUT),
        "bounds": [float(grid[0]), float(grid[-1])],
        "coordinates": grid.tolist(),
        "n_scan_positions": int(grid.size),
        "n_batches": int(cfg["max_nbatch"]),
        "nominal_included_exactly": bool(grid[nominal_i] == 0.5),
        "sampled_minimum_efield": best_x,
        "sampled_minimum_mean_llhd": best_loss,
        "nominal_mean_llhd": nominal_loss,
        "sampled_minimum_minus_nominal_mean_llhd": best_loss - nominal_loss,
        "objective": str(cfg["loss_fn"]),
        "sim_seed_strategy": str(cfg["sim_seed_strategy"]),
        "target_seed": int(cfg["seed"]),
        "probabilistic_sim": bool(cfg["probabilistic_sim"]),
        "input_file_target": str(cfg["input_file_tgt"]),
    }


def figure_two_d() -> dict:
    cfg = yaml.safe_load(TWO_D_CONFIG.read_text())
    rows = list(csv.DictReader(TWO_D_HISTORY.open(newline="")))
    names = cfg["tunable_params"]
    x = np.array([[float(row[names[0]]), float(row[names[1]])] for row in rows])
    loss = np.array([float(row["llhd"]) for row in rows])
    lo = np.asarray(cfg["lower"], dtype=float)
    hi = np.asarray(cfg["upper"], dtype=float)
    nominal = np.asarray(cfg["nominal"], dtype=float)
    if len(rows) != 130 or {row["phase"] for row in rows} != {"structured_seed", "BO"}:
        raise RuntimeError("unexpected historical Ab+kb history lineage")

    sys.path.insert(0, str(BAY / "workflows"))
    from common import DTYPE, fit_gp  # imported only for retained GP analysis

    unit_x = (x - lo) / (hi - lo)
    model = fit_gp(unit_x, loss, seed=int(cfg["bo_seed"]), ard_num_dims=2)
    grid_n = 181
    ua = np.linspace(0.0, 1.0, grid_n)
    ub = np.linspace(0.0, 1.0, grid_n)
    aa, bb = np.meshgrid(ua, ub)
    unit_grid = np.column_stack([aa.ravel(), bb.ravel()])
    with torch.no_grad():
        posterior = model.posterior(torch.tensor(unit_grid, dtype=DTYPE))
        score_mean = posterior.mean.squeeze(-1).cpu().numpy().reshape(grid_n, grid_n)
    predicted_llhd = np.exp(-score_mean)
    axis_ab = lo[0] + aa * (hi[0] - lo[0])
    axis_kb = lo[1] + bb * (hi[1] - lo[1])
    best_i = int(np.argmin(loss))

    fig, ax = plt.subplots(figsize=(11.2, 8.2), constrained_layout=True)
    positive = predicted_llhd[predicted_llhd > 0]
    vmin = max(float(np.percentile(positive, 1.0)), float(positive.min()))
    vmax = float(np.percentile(positive, 99.0))
    levels = np.geomspace(vmin, vmax, 32)
    field = ax.contourf(axis_ab, axis_kb, predicted_llhd, levels=levels,
                        cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax), extend="both")
    low_levels = np.geomspace(vmin, min(vmax, vmin * 1.8), 4)[1:]
    ax.contour(axis_ab, axis_kb, predicted_llhd, levels=low_levels,
               colors="white", linewidths=1.6, alpha=0.9)
    colorbar = fig.colorbar(field, ax=ax, pad=0.02)
    colorbar.set_label("GP posterior mean LLHD")
    colorbar_ticks = np.geomspace(vmin, vmax, 4)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    ax.scatter(nominal[0], nominal[1], marker="*", s=330, color="#F2C14E",
               edgecolor="black", linewidth=1.5, label="Nominal", zorder=4)
    ax.scatter(x[best_i, 0], x[best_i, 1], marker="D", s=130,
               color=COLORS["red"], edgecolor="white", linewidth=1.4,
               label="Best observed", zorder=4)
    ax.set_title("Parameter correlations in two dimensions", pad=16)
    ax.set_xlabel(r"$A_b$")
    ax.set_ylabel(r"$k_b$ [kV g / (MeV cm$^3$)]")
    ax.legend(loc="upper right", frameon=False)
    save(fig, TWO_D_OUTPUT)
    return {
        "source": str(TWO_D_HISTORY),
        "output": str(TWO_D_OUTPUT),
        "representation": "GP posterior mean fitted to retained historical Ab+kb observations",
        "n_observations": len(rows),
        "phase_counts": {
            "structured_seed": sum(row["phase"] == "structured_seed" for row in rows),
            "BO": sum(row["phase"] == "BO" for row in rows),
        },
        "best_observed": {
            "Ab": float(x[best_i, 0]),
            "kb": float(x[best_i, 1]),
            "LLHD": float(loss[best_i]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("all", "one-d", "two-d"), default="all")
    args = parser.parse_args()
    apply_style()
    result = {}
    if args.only in ("all", "one-d"):
        result["one_d"] = figure_efield_scan()
    if args.only in ("all", "two-d"):
        result["two_d"] = figure_two_d()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
