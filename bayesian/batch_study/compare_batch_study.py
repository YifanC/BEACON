"""Compare completed optimizer repeats using frozen histories only."""
from __future__ import annotations

import csv
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
    repeat: int
    raw: Path
    nominal_llhd: float

    @property
    def complete(self) -> bool:
        return all(path.exists() for path in history_paths(self))


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
        raise RuntimeError(f"{spec.raw}: expected 332 observations, got {len(rows)}")
    x = np.array([[float(row[name]) for name in NAMES] for row in rows])
    loss = np.array([float(row["native_LLHD"]) for row in rows])
    return x, loss


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 8:
        raise RuntimeError(f"expected eight completed runs in {path}, found {len(rows)}")
    return rows


def load_specs(rows: list[dict[str, str]]) -> list[RunSpec]:
    summary = {(row["data_size_label"].replace(" ", ""), int(row["repeat"])): row
               for row in rows}
    specs = []
    for metadata_path in sorted(STUDY.glob("*cm/repeat_*/run.json")):
        metadata = json.loads(metadata_path.read_text())
        key = (metadata["data_size_label"], int(metadata["repeat"]))
        row = summary[key]
        specs.append(RunSpec(row["data_size_label"], key[1],
                             Path(metadata["authoritative_history"]),
                             float(row["nominal_llhd"])))
    if len(specs) != 8:
        raise RuntimeError(f"expected eight run metadata files, found {len(specs)}")
    return specs


def plot_within_scale(specs: list[RunSpec], rows: list[dict[str, str]],
                      output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for spec in specs:
        _, loss = load_history(spec)
        ax.plot(np.arange(1, 333), np.minimum.accumulate(loss),
                color="#3B6FB6", linestyle=LINESTYLES[spec.repeat], lw=2.0,
                label=f"Repeat {spec.repeat}")
    ax.axhline(specs[0].nominal_llhd, color="#333333", linestyle="--", lw=1.5,
               label="Nominal reference")
    for boundary in (72.5, 172.5, 272.5):
        ax.axvline(boundary, color="#888888", linestyle=":", lw=0.9)
    ax.set(xlabel="Simulator evaluation", ylabel="Running-best native LLHD",
           title=f"~{specs[0].data_size_label} — Repeat convergence")
    ax.grid(alpha=0.22)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_convergence.png", dpi=260)
    plt.close(fig)

    scale_rows = sorted(
        (row for row in rows if row["data_size_label"] == specs[0].data_size_label),
        key=lambda row: int(row["repeat"]))
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    axis = np.arange(len(NAMES))
    offsets = np.linspace(-0.16, 0.16, len(scale_rows))
    for offset, row in zip(offsets, scale_rows):
        values = [float(row[f"{name}_deviation_pct_range"]) for name in NAMES]
        ax.plot(axis + offset, values, marker="o", lw=1.4,
                label=f"Repeat {row['repeat']}")
    ax.axhline(0, color="#333333", linestyle="--", lw=1.5, label="Nominal")
    ax.set_xticks(axis, DISPLAY_NAMES)
    ax.set_ylabel("Deviation from nominal (% of allowed range)")
    ax.set_title(f"~{specs[0].data_size_label} — Final parameter coordinates")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_final_parameters.png", dpi=260)
    plt.close(fig)


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
    comparison = STUDY / "comparison"
    comparison.mkdir(exist_ok=True)
    rows = load_summary(comparison / "optimizer_repeats.csv")
    complete = [spec for spec in load_specs(rows) if spec.complete]
    if len(complete) != 8:
        raise RuntimeError(f"expected eight complete runs, found {len(complete)}")
    plot_cross_scale_convergence(
        complete, rows, comparison / "01_cross_scale_convergence.png")
    plot_cross_scale_parameter_recovery(
        rows, comparison / "02_cross_scale_parameter_recovery.png")
    for scale in ("200 cm", "1000 cm", "2000 cm"):
        scale_specs = [spec for spec in complete if spec.data_size_label == scale]
        plot_within_scale(scale_specs, rows, STUDY / scale.replace(" ", "") / "comparison")
    print(json.dumps({"completed_runs": len(complete),
                      "figures": ["01_cross_scale_convergence.png",
                                  "02_cross_scale_parameter_recovery.png"]}, indent=2))


if __name__ == "__main__":
    main()
