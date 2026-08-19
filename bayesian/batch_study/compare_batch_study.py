"""Compare completed optimizer repeats using frozen histories only."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BAYESIAN = Path("/sdf/home/i/iatif/REAL_BEACON/bayesian")
STUDY = BAYESIAN / "batch_study"
NAMES = ("Ab", "kb", "eField", "lifetime", "tran_diff", "long_diff")
LO = np.array([0.75, 0.03, 0.49, 400.0, 3.0e-6, 1.0e-6])
HI = np.array([0.90, 0.08, 0.51, 6000.0, 15.0e-6, 10.0e-6])
NOM = np.array([0.80, 0.0486, 0.50, 2200.0, 8.8e-6, 4.0e-6])
SCALE_COLORS = {"200 cm": "#3B6FB6", "1000 cm": "#D07A24", "2000 cm": "#5B8F69"}
LINESTYLES = ("-", "--", ":")
DISPLAY_NAMES = (r"$A_b$", r"$k_b$", "eField", "lifetime", r"$D_T$", r"$D_L$")


@dataclass(frozen=True)
class RunSpec:
    data_size_label: str
    requested_length_cm: float
    actual_length_cm: float
    repeat: int
    optimizer_seed: int
    target: Path
    n_events: int
    n_tracks: int
    experiment: Path
    nominal_llhd: float
    raw_direct: bool = False

    @property
    def raw(self) -> Path:
        return self.experiment / "raw" if self.raw_direct else self.experiment / "history/raw"

    @property
    def complete(self) -> bool:
        return all(path.exists() for path in history_paths(self))


TARGET_200 = STUDY / "shared/target_200cm.npz"
TARGET_1000 = BAYESIAN / ".local/two_d/target.npz"
TARGET_2000 = STUDY / "2000cm/target_2000cm.npz"
NOMINAL_200 = float(np.mean([4077.037353515625, 4077.039306640625]))
NOMINAL_1000 = 21028.986328125
NOMINAL_2000 = 43462.88671875

RUNS = [
    RunSpec("200 cm", 200, 195.33575677871704, 0, 20260812, TARGET_200, 33, 19535,
            STUDY / "200cm_seed0", NOMINAL_200),
    RunSpec("200 cm", 200, 195.33575677871704, 1, 20260812, TARGET_200, 33, 19535,
            STUDY / "200cm_seed1", NOMINAL_200),
    RunSpec("1000 cm", 1000, 999.926641702652, 0, 20260812, TARGET_1000, 176, 100010,
            BAYESIAN / ".local/six_d/current", NOMINAL_1000, raw_direct=True),
    RunSpec("1000 cm", 1000, 999.926636338234, 1, 20260813, TARGET_1000, 176, 100010,
            STUDY / "1000cm/repeat_1", NOMINAL_1000),
    RunSpec("1000 cm", 1000, 999.926636338234, 2, 20260814, TARGET_1000, 176, 100010,
            STUDY / "1000cm/repeat_2", NOMINAL_1000),
    RunSpec("2000 cm", 2000, 1998.4871374368668, 0, 20260812, TARGET_2000, 374, 199877,
            STUDY / "2000cm/run", NOMINAL_2000),
    RunSpec("2000 cm", 2000, 1998.4871374368668, 1, 20260813, TARGET_2000, 374, 199877,
            STUDY / "2000cm/repeat_1", NOMINAL_2000),
    RunSpec("2000 cm", 2000, 1998.4871374368668, 2, 20260814, TARGET_2000, 374, 199877,
            STUDY / "2000cm/repeat_2", NOMINAL_2000),
]


def history_paths(spec: RunSpec) -> tuple[Path, Path, Path]:
    return (spec.raw / "traces/bo_6d_history.csv",
            spec.raw / "continuation/bo_tr_history.csv",
            spec.raw / "continuation/bo_tr2_history.csv")


def load_history(spec: RunSpec) -> tuple[np.ndarray, np.ndarray]:
    rows: list[dict[str, str]] = []
    for path in history_paths(spec):
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    if len(rows) != 332:
        raise RuntimeError(f"{spec.experiment}: expected 332 observations, got {len(rows)}")
    x = np.array([[float(row[name]) for name in NAMES] for row in rows])
    loss = np.array([float(row["native_LLHD"]) for row in rows])
    return x, loss


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(spec: RunSpec) -> tuple[float, float, float]:
    if spec.data_size_label == "1000 cm" and spec.repeat == 0:
        health = json.loads((BAYESIAN / "results/six_d/health_metrics.json").read_text())
        return (float(health["cv_score_rmse"]), float(health["cv_spearman"]),
                float(health["cv_baseline_rmse"]))
    summary = json.loads((spec.experiment / "results/summary.json").read_text())
    return (float(summary["gp_cv"]["rmse"]), float(summary["gp_cv"]["spearman"]),
            float(summary["gp_cv"]["baseline_rmse"]))


def summary_row(spec: RunSpec) -> dict[str, object]:
    x, loss = load_history(spec)
    index = int(np.argmin(loss))
    best = x[index]
    deviation = 100.0 * (best - NOM) / (HI - LO)
    rmse, spearman, baseline_rmse = metrics(spec)
    first = int(np.flatnonzero(loss == loss[index])[0])
    stage = ("INITIAL" if first < 72 else "BO_GLOBAL" if first < 172
             else "BO_TR" if first < 272 else "BO_TR2")
    row: dict[str, object] = {
        "data_size_label": spec.data_size_label,
        "requested_length_cm": spec.requested_length_cm,
        "actual_length_cm": spec.actual_length_cm,
        "repeat": spec.repeat,
        "repeat_type": ("repeated_execution_same_optimizer_seed"
                        if spec.data_size_label == "200 cm" else "optimizer_seed_repeat"),
        "optimizer_seed": spec.optimizer_seed,
        "target_hash": sha256(spec.target),
        "n_events": spec.n_events,
        "n_tracks": spec.n_tracks,
        "n_evaluations": len(loss),
        "best_llhd": float(loss[index]),
        "nominal_llhd": spec.nominal_llhd,
        "best_minus_nominal": float(loss[index] - spec.nominal_llhd),
        "relative_best_minus_nominal": float(
            (loss[index] - spec.nominal_llhd) / spec.nominal_llhd),
        "gp_cv_rmse": rmse,
        "gp_cv_spearman": spearman,
        "gp_cv_baseline_rmse": baseline_rmse,
        "best_first_found_stage": stage,
        "best_first_found_evaluation": first + 1,
    }
    row.update({name: float(best[i]) for i, name in enumerate(NAMES)})
    row.update({f"{name}_nominal": float(NOM[i]) for i, name in enumerate(NAMES)})
    row.update({f"{name}_deviation_pct_range": float(deviation[i])
                for i, name in enumerate(NAMES)})
    return row


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 8:
        raise RuntimeError(f"expected eight completed runs in {path}, found {len(rows)}")
    return rows


def plot_cross_scale_convergence(
        specs: list[RunSpec], rows: list[dict[str, str]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    for spec in specs:
        _, loss = load_history(spec)
        running = np.minimum.accumulate(loss)
        y = np.log(running / spec.nominal_llhd)
        ax.plot(np.arange(1, 333), y, lw=1.9,
                label=f"{spec.data_size_label} — repeat {spec.repeat}",
                color=SCALE_COLORS[spec.data_size_label],
                linestyle=LINESTYLES[spec.repeat], alpha=0.88)
    for boundary in (72.5, 172.5, 272.5):
        ax.axvline(boundary, color="#8A8A8A", ls="--", lw=0.8, alpha=0.65)
    for boundary, evaluation in zip((72.5, 172.5, 272.5), (72, 172, 272)):
        ax.text(boundary, 0.91, str(evaluation), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, color="#666666",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8,
                      "pad": 1.0})
    ax.axhline(0, color="#333333", ls="-.", lw=1.5,
               label="Nominal reference (held out)")
    ax.annotate("0 = nominal reference", xy=(210, 0), xytext=(0, 7),
                textcoords="offset points", fontsize=8.5, color="#444444",
                ha="center", va="bottom")
    for x, label in zip((36.5, 122.5, 222.5, 302.5),
                        ("Sobol", "Global BO", "BO_TR", "BO_TR2")):
        ax.text(x, 0.975, label, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=10, color="#444444",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72,
                      "pad": 1.5})
    ax.set_xlabel("Simulator evaluation")
    ax.set_ylabel("ln(running-best LLHD / nominal LLHD)")
    ax.set_xlim(1, 332)
    ax.set_title("Cross-scale normalized convergence", pad=12)
    ax.grid(axis="y", alpha=0.20)
    ax.legend(frameon=True, ncol=2, loc="upper right", fontsize=9.5,
              bbox_to_anchor=(0.99, 0.86))
    summary_offsets = {"200 cm": 58, "1000 cm": 37, "2000 cm": 17}
    for scale in ("200 cm", "1000 cm", "2000 cm"):
        scale_rows = [row for row in rows if row["data_size_label"] == scale]
        normalized = np.array([
            np.log(float(row["best_llhd"]) / float(row["nominal_llhd"]))
            for row in scale_rows])
        mean_normalized = float(normalized.mean())
        ax.annotate(f"{scale}: {mean_normalized:+.3g}",
                    xy=(332, mean_normalized),
                    xytext=(-105, summary_offsets[scale]),
                    textcoords="offset points", fontsize=8.5,
                    color=SCALE_COLORS[scale], ha="left",
                    arrowprops={"arrowstyle": "-", "lw": 0.7,
                                "color": SCALE_COLORS[scale]})
    fig.text(0.5, 0.018,
             "200 cm curves are same-seed repeated executions; 1000 and 2000 cm curves are optimizer-seed repeats.",
             ha="center", fontsize=9.5, color="#444444")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(output, dpi=320)
    plt.close(fig)


def plot_cross_scale_parameter_recovery(rows: list[dict[str, str]], output: Path) -> None:
    axis = np.arange(len(NAMES))
    fig, ax = plt.subplots(figsize=(13.8, 7.4))
    offsets = {"200 cm": -0.22, "1000 cm": 0.0, "2000 cm": 0.22}
    labels = {"200 cm": "~200 cm", "1000 cm": "~1000 cm", "2000 cm": "~2000 cm"}
    for scale in ("200 cm", "1000 cm", "2000 cm"):
        scale_rows = [row for row in rows if row["data_size_label"] == scale]
        values = np.array([[float(row[f"{name}_deviation_pct_range"])
                            for name in NAMES] for row in scale_rows])
        center = axis + offsets[scale]
        mean = values.mean(axis=0)
        spread = values.std(axis=0)
        ax.errorbar(center, mean, yerr=spread, fmt="D", ms=6.5, capsize=4,
                    elinewidth=1.5, capthick=1.2, color=SCALE_COLORS[scale],
                    label=labels[scale], zorder=4)
        run_offsets = np.linspace(-0.035, 0.035, len(scale_rows))
        for row, run_offset, run_values in zip(scale_rows, run_offsets, values):
            ax.scatter(center + run_offset, run_values, s=27,
                       facecolor="white", edgecolor=SCALE_COLORS[scale],
                       linewidth=1.1, alpha=0.9, zorder=3)
            if scale == "2000 cm" and int(row["repeat"]) == 1:
                point_x = center[0] + run_offset
                point_y = run_values[0]
                ax.annotate(f"R1: {point_y:+.2f}%", xy=(point_x, point_y),
                            xytext=(9, 9), textcoords="offset points",
                            fontsize=8.0, color=SCALE_COLORS[scale],
                            arrowprops={"arrowstyle": "-", "lw": 0.6,
                                        "color": SCALE_COLORS[scale]})
        for parameter_index in (0, 1):
            precision = 2
            ax.annotate(f"{mean[parameter_index]:+.{precision}f}%",
                        xy=(center[parameter_index],
                            mean[parameter_index] + spread[parameter_index]),
                        xytext=(0, 5), textcoords="offset points",
                        fontsize=8.0, color=SCALE_COLORS[scale], ha="center",
                        va="bottom")
    efield_values = [abs(float(row["eField_deviation_pct_range"])) for row in rows]
    ax.annotate(f"|deviation| < {max(efield_values):.2f}% at all scales",
                xy=(2, 0), xytext=(0, 34), textcoords="offset points",
                fontsize=8.5, color="#444444", ha="center",
                arrowprops={"arrowstyle": "-", "lw": 0.6,
                            "color": "#777777"})
    ax.axhline(0, color="#333333", ls="--", lw=1.4, label="Nominal")
    ax.set_xticks(axis, DISPLAY_NAMES)
    ax.set_ylabel("Deviation from nominal (% of allowed range)")
    ax.set_title("Cross-scale six-parameter closure recovery", pad=12)
    ax.grid(axis="y", alpha=0.20)
    ax.legend(frameon=True, ncol=4, loc="upper right")
    fig.text(0.5, 0.018,
             "Diamonds show means; error bars show population spread; open circles are individual runs. "
             "The 200 cm spread is across same-seed executions.",
             ha="center", fontsize=9.5, color="#444444")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output, dpi=320)
    plt.close(fig)


def main() -> None:
    complete = [spec for spec in RUNS if spec.complete]
    comparison = STUDY / "comparison"
    comparison.mkdir(exist_ok=True)
    rows = load_summary(comparison / "optimizer_repeats.csv")
    plot_cross_scale_convergence(
        complete, rows, comparison / "01_cross_scale_convergence.png")
    plot_cross_scale_parameter_recovery(
        rows, comparison / "02_cross_scale_parameter_recovery.png")
    print(json.dumps({"completed_runs": len(complete),
                      "figures": ["01_cross_scale_convergence.png",
                                  "02_cross_scale_parameter_recovery.png"]}, indent=2))


if __name__ == "__main__":
    main()
