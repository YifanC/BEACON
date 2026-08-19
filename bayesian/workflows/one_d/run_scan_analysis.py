"""One-dimensional eField scan analysis.

Reads the single authoritative scan PKL under .local/one_d/, computes the
native-LLHD loss curve and finite-difference gradient at nominal, and writes:

  results/one_d/scan.png
  results/one_d/scan_summary.csv
  results/one_d/scan_summary.json

No simulator is run. No BO is run. Analysis-only.

Usage:
  python workflows/one_d/run_scan_analysis.py \
      --config workflows/one_d/config.yaml

Or:
  sbatch workflows/one_d/run_scan_analysis.sbatch
"""
from __future__ import annotations
import argparse, csv, json, pickle, sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAY = Path(__file__).resolve().parents[2]


def load_pkl(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def summarize(history: dict, cfg: dict) -> dict:
    """Reduce iteration-level trajectories to per-scan-value best losses.

    Expected schema: `history` is a dict whose top-level keys index scan values
    (either integer indices or physical parameter values). Each entry contains
    a `loss` array and either a `physical` scalar or `params[cfg.parameter]`.
    """
    param = cfg["parameter"]
    xs, ys = [], []
    if isinstance(history, dict) and any(k in history for k in ("scans", "history")):
        obj = history.get("scans") or history.get("history")
    else:
        obj = history
    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, list):
        items = list(enumerate(obj))
    else:
        raise RuntimeError(f"Unsupported history structure: {type(obj)}")
    for _, entry in items:
        if not isinstance(entry, dict):
            continue
        # Physical scan value
        phys = None
        for k in ("physical", "x", "value", param):
            if k in entry:
                try:
                    phys = float(entry[k])
                    break
                except (TypeError, ValueError):
                    pass
        if phys is None and "params" in entry:
            phys = float(entry["params"].get(param, np.nan))
        if phys is None or not np.isfinite(phys):
            continue
        losses = np.asarray(entry.get("loss", []), dtype=float)
        if losses.size == 0:
            continue
        finite = losses[np.isfinite(losses)]
        if finite.size == 0:
            continue
        xs.append(phys); ys.append(float(finite.min()))
    order = np.argsort(xs)
    xs = np.asarray(xs)[order]
    ys = np.asarray(ys)[order]
    return {"physical": xs.tolist(), "best_loss": ys.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(BAY / "workflows/one_d/config.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pkl_path = BAY / cfg["pkl_path"]
    if not pkl_path.is_file():
        raise SystemExit(f"PKL not found: {pkl_path}")

    history = load_pkl(pkl_path)
    s = summarize(history, cfg)

    xs = np.asarray(s["physical"])
    ys = np.asarray(s["best_loss"])
    if xs.size < 2:
        raise SystemExit(f"Not enough scan points to plot (found {xs.size}).")

    idx_min = int(np.argmin(ys))
    x_min, y_min = float(xs[idx_min]), float(ys[idx_min])

    # FD gradient at nominal (central difference to nearest grid neighbors)
    nom = float(cfg["nominal"])
    i_nom = int(np.argmin(np.abs(xs - nom)))
    grad = None
    if 0 < i_nom < len(xs) - 1:
        grad = float((ys[i_nom + 1] - ys[i_nom - 1]) / (xs[i_nom + 1] - xs[i_nom - 1]))

    out_dir = BAY / "results/one_d"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, "o-", color="tab:orange", lw=1.3, ms=4, label="LLHD")
    ax.axvline(nom, ls="--", color="gray", lw=1, label="Nominal")
    ax.axvline(x_min, ls=":", color="red", lw=1, label="Best")
    ax.set_yscale("log")
    ax.set_xlabel(f"{cfg['parameter']}")
    ax.set_ylabel("LLHD")
    ax.grid(alpha=0.2, which="both")
    ax.legend(loc="upper right", frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="black", fontsize=8)
    ax.set_title(f"1D scan — {cfg['parameter']}")
    fig.tight_layout()
    fig.savefig(out_dir / "scan.png", dpi=180); plt.close(fig)

    # --- Compact CSV ---
    with (out_dir / "scan_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([cfg["parameter"], "best_loss"])
        for x, y in zip(xs, ys):
            w.writerow([float(x), float(y)])

    # --- Compact JSON ---
    summary = {
        "parameter": cfg["parameter"],
        "n_scan_values": int(xs.size),
        "n_scan_values_expected": int(cfg["n_scan_values_expected"]),
        "nominal": nom,
        "lower": float(cfg["lower"]),
        "upper": float(cfg["upper"]),
        "best_x": x_min,
        "best_loss": y_min,
        "gradient_at_nominal_central_diff": grad,
        "source_pkl_relative": cfg["pkl_path"],
    }
    (out_dir / "scan_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
