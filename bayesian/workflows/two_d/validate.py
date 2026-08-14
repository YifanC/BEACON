"""Post-BO direct simulator validation for a 2D pair.

Reads the finished BO history under `.local/two_d/<pair>/history.csv`, selects
the best actual observed 2D point, and runs two 1D direct scans (one per
parameter, other parameter fixed at BO best).

Writes:
  .local/two_d/<pair>/direct_profile_parameter1.csv
  .local/two_d/<pair>/direct_profile_parameter2.csv
  .local/two_d/<pair>/direct_profile_summary.json
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BAY, make_llhd_objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-per-axis", type=int, default=21)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    NAMES = list(cfg["tunable_params"])
    LO = list(cfg["lower"]); HI = list(cfg["upper"])
    outdir = BAY / f".local/two_d/{cfg['pair']}"
    hist_path = outdir / "history.csv"
    if not hist_path.is_file():
        raise SystemExit(f"missing history: {hist_path}")

    rows = list(csv.DictReader(hist_path.open()))
    best = min(rows, key=lambda r: float(r["native_LLHD"]))
    best_point = {n: float(best[n]) for n in NAMES}
    print(f"best point: {best_point}, LLHD={best['native_LLHD']}")

    obj, load_target = make_llhd_objective(
        n_events=cfg["n_events"], tunable_params=NAMES,
    )
    target_npz = BAY / ".local/two_d/target.npz"
    targets = load_target(str(target_npz))

    for j, name in enumerate(NAMES):
        grid = np.linspace(LO[j], HI[j], args.n_per_axis)
        out = outdir / f"direct_profile_parameter{j+1}.csv"
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([name, "llhd", "elapsed_seconds"])
            for v in grid:
                params = dict(best_point); params[name] = float(v)
                t = time.perf_counter()
                llhd = float(obj.evaluate(params, targets))
                elapsed = time.perf_counter() - t
                w.writerow([float(v), llhd, elapsed]); f.flush(); os.fsync(f.fileno())
                print(json.dumps({"param": name, "value": float(v),
                                  "llhd": llhd, "elapsed_seconds": elapsed}), flush=True)

    summary = {"pair": cfg["pair"], "best_point": best_point,
               "best_native_LLHD": float(best["native_LLHD"]),
               "n_per_axis": args.n_per_axis,
               "completed_at": datetime.now(timezone.utc).isoformat()}
    (outdir / "direct_profile_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
