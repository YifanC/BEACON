"""Render four poster-specific figures for the accepted final 6D BO result.

Plotting only: reads frozen training and retained validation/diagnostic data.
It never imports or calls the detector objective.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr


REPO = Path("/sdf/home/i/iatif/larnd-sim-jax").resolve()
BAY = Path(__file__).resolve().parents[2]
RAW = BAY / ".local/six_d/current/raw"
WORKFLOW = BAY / "workflows/six_d"
RESULTS = BAY / "results/six_d"
PLOTS = RESULTS / "plots"

SNAPSHOT = RAW / "continuation/final_training_snapshot_bo_tr2.csv"
BRIDGE = RAW / "direct_validation/surrogate_bridge.csv"
HELDOUT = RAW / "direct_validation/bo_tr2_heldout_6d_results.csv"
PROVENANCE = RAW / "direct_validation/bo_tr2_heldout_provenance.json"
HEALTH = RESULTS / "health_metrics.json"
FINAL_REPORT = RESULTS / "FINAL_6D.md"

OUTPUTS = [
    PLOTS / "poster_01_bridge_diagnostic.png",
    PLOTS / "poster_02_convergence.png",
    PLOTS / "poster_03_parameter_recovery.png",
    PLOTS / "poster_04_heldout_validation.png",
]

EXPECTED_SNAPSHOT_SHA = "3599ee904c69654e8ec9da5404669db32ba13345b4205b1be1d16d9236a1fafd"
EXPECTED_PHASES = {"INITIAL": 72, "BO_1": 100, "BO_TR": 100, "BO_TR2": 60}
EXPECTED_BEST = 21026.755859375
EXPECTED_NOMINAL = 21028.986328125
EXPECTED_VALIDATION_SEED = 20260827

sys.path.insert(0, str(WORKFLOW))
from build_6d import HI, LO, NAMES, NOM, fit_gp, unit  # noqa: E402


COLORS = {
    "navy": "#174A7E",
    "blue": "#2878B5",
    "orange": "#E07A1F",
    "red": "#B33B3B",
    "green": "#2A7F62",
    "gray": "#666666",
    "light_gray": "#E9EDF2",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def close(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(float(a) - float(b)) <= tol


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
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.16, facecolor="white")
    plt.close(fig)


def load_and_verify():
    for path in (SNAPSHOT, BRIDGE, HELDOUT, PROVENANCE, HEALTH, FINAL_REPORT):
        if not path.is_file():
            raise FileNotFoundError(f"required input missing: {path}")
    if not PLOTS.is_dir():
        raise RuntimeError(f"existing output directory missing: {PLOTS}")

    health = json.loads(HEALTH.read_text())
    final_text = FINAL_REPORT.read_text()
    training = read_csv(SNAPSHOT)
    bridge = read_csv(BRIDGE)
    heldout = read_csv(HELDOUT)
    provenance = json.loads(PROVENANCE.read_text())

    digest = sha256(SNAPSHOT)
    if digest != EXPECTED_SNAPSHOT_SHA or health["final_training_snapshot_sha256"] != digest:
        raise RuntimeError(f"snapshot SHA mismatch: file={digest}, health={health.get('final_training_snapshot_sha256')}")
    phases = Counter(r["phase"] for r in training)
    if dict(phases) != EXPECTED_PHASES or len(training) != 332:
        raise RuntimeError(f"clean training lineage mismatch: n={len(training)}, phases={dict(phases)}")
    if set(phases) != set(EXPECTED_PHASES):
        raise RuntimeError(f"forbidden phase in clean training: {sorted(set(phases) - set(EXPECTED_PHASES))}")

    losses = np.array([float(r["native_LLHD"]) for r in training])
    if not close(losses.min(), EXPECTED_BEST) or not close(health["best_overall_training"], EXPECTED_BEST):
        raise RuntimeError("final best disagrees between snapshot and health metrics")
    if not close(health["all_nominal_heldout"], EXPECTED_NOMINAL):
        raise RuntimeError("nominal reference disagrees with accepted value")
    if "21026.7559" not in final_text or "21028.986" not in final_text or "332" not in final_text:
        raise RuntimeError("FINAL_6D.md does not contain the accepted final values")

    held_counts = Counter(r["set"] for r in heldout)
    if held_counts != {"global": 16, "local": 16} or len(heldout) != 32:
        raise RuntimeError(f"fresh held-out composition mismatch: {dict(held_counts)}")
    if provenance["validation_seed"] != EXPECTED_VALIDATION_SEED:
        raise RuntimeError(f"held-out seed mismatch: {provenance['validation_seed']}")
    if provenance["final_training_snapshot_sha256"] != digest or provenance["n_training_rows"] != 332:
        raise RuntimeError("held-out provenance does not point to the accepted frozen snapshot")

    coord = lambda r: tuple(float(r[n]) for n in NAMES)
    train_coords = {coord(r) for r in training}
    heldout_overlap = [i for i, r in enumerate(heldout, 1) if coord(r) in train_coords]
    if heldout_overlap:
        raise RuntimeError(f"held-out coordinates overlap clean training rows: {heldout_overlap}")
    if any(r["phase"].startswith("DIAG") for r in training):
        raise RuntimeError("diagnostic bridge/shell phase entered clean training")

    if len(bridge) != 9 or [float(r["t"]) for r in bridge] != list(np.linspace(0, 1, 9)):
        raise RuntimeError("diagnostic bridge does not contain the expected nine t values")
    if any(r["phase"] != "DIAG_BRIDGE" for r in bridge):
        raise RuntimeError("unexpected phase in bridge data")

    best_row = training[int(np.argmin(losses))]
    best = np.array([float(best_row[n]) for n in NAMES])
    health_best = np.array([float(health["final_best_coordinate"][n]) for n in NAMES])
    if not np.array_equal(best, health_best):
        raise RuntimeError("final-best coordinates disagree between snapshot and health metrics")

    stage_bests = {p: min(float(r["native_LLHD"]) for r in training if r["phase"] == p)
                   for p in EXPECTED_PHASES}
    for p, key in (("INITIAL", "best_initial"), ("BO_1", "best_bo_1"), ("BO_TR", "best_bo_tr")):
        if not close(stage_bests[p], health[key]):
            raise RuntimeError(f"stage best mismatch for {p}")
    if not close(stage_bests["BO_TR2"], EXPECTED_BEST):
        raise RuntimeError("BO_TR2 stage best mismatch")
    return health, training, bridge, heldout, best, stage_bests


def figure_bridge(bridge: list[dict[str, str]]) -> None:
    t = np.array([float(r["t"]) for r in bridge])
    actual = np.array([float(r["actual_native_LLHD"]) for r in bridge])
    original_gp = np.array([float(r["gp_llhd_like"]) for r in bridge])

    fig, ax = plt.subplots(figsize=(11.8, 7.8), constrained_layout=True)
    ax.plot(t, actual, color=COLORS["navy"], lw=4.2, marker="o", ms=11,
            markeredgecolor="white", markeredgewidth=1.5, label="Simulator")
    ax.plot(t, original_gp, color=COLORS["orange"], lw=4.0, ls="--", marker="s", ms=9,
            markerfacecolor="white", markeredgewidth=2.0, label="Original Matérn-5/2 GP")
    ax.set_title("Local surrogate diagnostic", pad=16)
    ax.set_xlabel("Interpolation toward nominal, $t$")
    ax.set_ylabel("LLHD (lower is better)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.grid(axis="y", color="#D6DCE3", lw=1.2)
    ax.legend(loc="upper left", frameon=False)
    ax.annotate("BO_TR best", xy=(0, actual[0]), xytext=(0.07, actual[0] + 48),
                fontsize=17, fontweight="bold", color=COLORS["navy"],
                arrowprops=dict(arrowstyle="->", lw=2, color=COLORS["navy"]))
    ax.annotate("Nominal reference", xy=(1, actual[-1]), xytext=(0.63, actual[-1] + 65),
                fontsize=17, fontweight="bold", color=COLORS["navy"],
                arrowprops=dict(arrowstyle="->", lw=2, color=COLORS["navy"]))
    ax.text(0.61, 0.70, "Assessment only\nBridge points were never added to BO training",
            transform=ax.transAxes, ha="center", va="center", fontsize=16, color=COLORS["gray"])
    save(fig, OUTPUTS[0])


def figure_convergence(training: list[dict[str, str]], stage_bests: dict[str, float], nominal: float) -> None:
    loss = np.array([float(r["native_LLHD"]) for r in training])
    evaluation = np.array([int(r["evaluation_index"]) for r in training])
    running = np.minimum.accumulate(loss)
    spans = [
        (1, 72, "Sobol initialization", "#E9F1F8"),
        (73, 172, "Global BO", "#F3F5F7"),
        (173, 272, "Trust-region BO", "#EAF4EF"),
        (273, 332, "Final BO refinement", "#FFF1E3"),
    ]
    endpoints = [72, 172, 272, 332]
    phases = ["INITIAL", "BO_1", "BO_TR", "BO_TR2"]

    fig, ax = plt.subplots(figsize=(13.2, 7.8), constrained_layout=True)
    for lo, hi, label, color in spans:
        ax.axvspan(lo, hi, color=color, zorder=0)
        ax.text((lo + hi) / 2, 0.965, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=16, fontweight="bold", color="#3E4A55")
    for x in endpoints[:-1]:
        ax.axvline(x + 0.5, color="#A9B1BA", lw=1.6)
    ax.plot(evaluation, running, color=COLORS["navy"], lw=4.2, zorder=3)
    ax.scatter(endpoints, [stage_bests[p] for p in phases], s=120, color=COLORS["navy"],
               edgecolor="white", linewidth=1.5, zorder=4)
    label_positions = [(82, 46300), (182, 25200), (230, 24200), (287, 22900)]
    for x, p, (tx, ty) in zip(endpoints, phases, label_positions):
        y = stage_bests[p]
        ax.annotate(f"{y:,.1f}", xy=(x, y), xytext=(tx, ty),
                    fontsize=17, fontweight="bold", color=COLORS["navy"],
                    arrowprops=dict(arrowstyle="-", lw=1.6, color=COLORS["navy"]))
    ax.axhline(nominal, color=COLORS["red"], lw=3.0, ls="--", zorder=2)
    ax.text(24, nominal + 650, f"Held-out nominal reference  {nominal:,.1f}",
            ha="left", va="bottom", fontsize=17, fontweight="bold", color=COLORS["red"])
    ax.set_title("Six-dimensional optimization convergence", pad=16)
    ax.set_xlabel("Simulator evaluation")
    ax.set_ylabel("Best LLHD observed so far")
    ax.set_xlim(1, 332)
    ax.set_ylim(19500, 47500)
    ax.grid(axis="y", color="#D6DCE3", lw=1.1)
    ax.text(0.70, 0.50, "332 clean simulator evaluations\nNominal is held out from training",
            transform=ax.transAxes, ha="center", va="center", fontsize=16, color=COLORS["gray"])
    save(fig, OUTPUTS[1])


def figure_parameter_recovery(best: np.ndarray) -> None:
    deviation = 100.0 * (best - NOM) / (HI - LO)
    labels = [r"$A_b$", r"$k_b$", r"$E$", r"$\tau$", r"$D_T$", r"$D_L$"]
    y = np.arange(len(labels))
    colors = [COLORS["blue"] if d >= 0 else COLORS["orange"] for d in deviation]

    fig, ax = plt.subplots(figsize=(11.8, 7.8), constrained_layout=True)
    ax.axvline(0, color="#343A40", lw=2.2)
    ax.hlines(y, 0, deviation, color=colors, lw=5, alpha=0.9)
    ax.scatter(deviation, y, s=220, color=colors, edgecolor="white", linewidth=1.7, zorder=3)
    for yi, d in zip(y, deviation):
        offset = 0.045 if d >= 0 else -0.045
        ax.text(d + offset, yi, f"{d:+.2f}%", ha="left" if d >= 0 else "right",
                va="center", fontsize=18, fontweight="bold", color="#26323D")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.17, 0.94)
    ax.set_xlabel("Deviation from nominal (% of allowed range)")
    ax.set_title("Recovery of nominal calibration", pad=16)
    ax.grid(axis="x", color="#D6DCE3", lw=1.1)
    ax.text(0, -0.72, "Nominal", ha="center", va="center", fontsize=16,
            fontweight="bold", color="#343A40")
    ax.text(0.98, 0.57, "All six parameters within 0.75%\nof their allowed range from nominal",
            transform=ax.transAxes, ha="right", va="center", fontsize=17,
            fontweight="bold", color=COLORS["navy"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEF4F9", edgecolor="none"))
    ax.text(0.99, 0.03, "Normalized proximity only — not parameter importance",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=16, color=COLORS["gray"])
    save(fig, OUTPUTS[2])


def figure_heldout(health: dict, training: list[dict[str, str]], heldout: list[dict[str, str]]) -> dict:
    X = np.array([[float(r[n]) for n in NAMES] for r in training])
    L = np.array([float(r["native_LLHD"]) for r in training])
    Xh = np.array([[float(r[n]) for n in NAMES] for r in heldout])
    actual = np.array([float(r["score_log"]) for r in heldout])
    groups = np.array([r["set"] for r in heldout])

    torch.set_default_dtype(torch.double)
    model = fit_gp(X, L, nu=1.5)
    with torch.no_grad():
        predicted = model.posterior(torch.tensor(unit(Xh), dtype=torch.double)).mean.squeeze(-1).cpu().numpy()
    residual = predicted - actual
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    rho = float(spearmanr(actual, predicted)[0])
    baseline = float(np.sqrt(np.mean((actual - actual.mean()) ** 2)))
    for name, value, expected in (
        ("held-out RMSE", rmse, health["heldout_score_rmse"]),
        ("held-out Spearman", rho, health["heldout_spearman"]),
        ("held-out baseline", baseline, health["heldout_baseline_rmse"]),
    ):
        if not close(value, expected, 1e-9):
            raise RuntimeError(f"{name} mismatch: computed={value}, health={expected}")

    fig, ax = plt.subplots(figsize=(10.2, 8.4), constrained_layout=True)
    styles = {
        "global": ("o", COLORS["blue"], "Global"),
        "local": ("s", COLORS["orange"], "Local"),
    }
    for group in ("global", "local"):
        m = groups == group
        marker, color, label = styles[group]
        ax.scatter(actual[m], predicted[m], s=145, marker=marker, color=color,
                   edgecolor="white", linewidth=1.3, alpha=0.95, label=label, zorder=3)
    lo = float(min(actual.min(), predicted.min())) - 0.12
    hi = float(max(actual.max(), predicted.max())) + 0.12
    ax.plot([lo, hi], [lo, hi], color="#4D5359", lw=2.2, ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Independent GP validation", pad=16)
    ax.set_xlabel(r"Actual score, $-\ln(\mathrm{LLHD})$")
    ax.set_ylabel(r"GP-predicted score, $-\ln(\mathrm{LLHD})$")
    ax.grid(color="#D9DEE4", lw=1.0)
    ax.legend(loc="lower right", frameon=False)
    metrics = (f"Spearman $\\rho$ = {rho:.3f}\n"
               f"RMSE = {rmse:.3f}\n"
               f"Baseline RMSE = {baseline:.3f}\n"
               f"95% coverage = {100 * health['heldout_95_coverage']:.1f}%\n"
               "n = 32, independently held out")
    ax.text(0.04, 0.96, metrics, transform=ax.transAxes, ha="left", va="top",
            fontsize=17, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#AAB2BA", alpha=0.95))
    save(fig, OUTPUTS[3])
    return {"rmse": rmse, "rho": rho, "baseline": baseline}


def main() -> None:
    apply_style()
    health, training, bridge, heldout, best, stage_bests = load_and_verify()
    figure_bridge(bridge)
    figure_convergence(training, stage_bests, float(health["all_nominal_heldout"]))
    figure_parameter_recovery(best)
    validation_metrics = figure_heldout(health, training, heldout)
    missing = [str(p) for p in OUTPUTS if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"poster outputs missing or empty: {missing}")
    deviations = 100.0 * (best - NOM) / (HI - LO)
    print(json.dumps({
        "status": "complete",
        "outputs": [str(p) for p in OUTPUTS],
        "snapshot_sha256": sha256(SNAPSHOT),
        "phase_counts": EXPECTED_PHASES,
        "best_llhd": EXPECTED_BEST,
        "nominal_llhd": EXPECTED_NOMINAL,
        "normalized_deviation_percent": dict(zip(NAMES, deviations.tolist())),
        "heldout": {"n": len(heldout), "global": 16, "local": 16,
                    "seed": EXPECTED_VALIDATION_SEED, **validation_metrics},
    }, indent=2))


if __name__ == "__main__":
    main()
