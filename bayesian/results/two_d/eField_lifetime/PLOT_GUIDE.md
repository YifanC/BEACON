# PLOT_GUIDE — eField + lifetime pair

Full dataset description: **999.93-cm safe dataset** — 176 ordered events, 100 010 HDF5 rows, 999.926642 cm physical track, one simulator batch, 9 368 target hits at target seed 0. Plots display only the short label `999.93 cm`.

## Provenance and single-run rule

Three independent 100-iteration BO runs were carried out for this pair (BO-A, BO-B, BO-C). They are **repeated executions of the same Bayesian-optimization method with different acquisition seeds**, not different algorithms. The **original per-iteration traces were deleted** before recovery was attempted. A representative reproduction of BO-B was run (Slurm job 34091445, COMPLETED 2026-08-03T15:06:12) with the same seed and the same repair configuration; it produced 35 structured + 100 BO-reproduction rows.

**All plots on this page are based on the recovered BO-B trace, and are a representative reproduction because the original detailed trace was deleted.** The reproduction is not the exact deleted history; the exact deleted history is not recoverable. The reproduction is a faithful re-execution of the same method under the same seed and configuration; it converges to the same physical basin.

`BO-B` is used because the predeclared project rule uses `BO-B` for every 2D pair (and here BO-B is also the only surviving reproduction).

Parameters:
- **eField** (kV/cm): nominal 0.5. Search range [0.49, 0.51].
- **lifetime** (μs): nominal 2200. Search range [400, 6000].

Method: score = -ln(native_LLHD); observation variance = 1 / native_LLHD²; Matérn-5/2 ARD kernel; Standardize(m=1); qLogExpectedImprovement (q=1).

Exact numerical result (reproduction): 135-row history, best evaluated native LLHD **21024.98046875** at `eField=0.49999216895208287`, `lifetime=2209.32369635 μs` — from `bo_2d_1000cm_efield_lifetime_trace_recovery/rerun_BO-B/history.csv` row `evaluation_index=114`.

---

## 01_parameter_trajectory.png

- **Purpose**: show BO-B reproduction's parameter evolution.
- **Source**: `bo_2d_1000cm_efield_lifetime_trace_recovery/rerun_BO-B/history.csv`.
- **Rows used**: all 135 rows (35 `structured` + 100 `BO-reproduction`).
- **Columns used**: `evaluation_index`, `eField`, `lifetime`, `phase`.
- **Values shown**: real simulator inputs.
- **X-axis**: `Iteration` (mapped from `evaluation_index`).
- **Y-axis top**: `eField [kV/cm]`.
- **Y-axis bottom**: `Lifetime [us]`.
- **Legend labels**: **BO** (reproduction trajectory), **Nominal** (dashed gray), **Init** (structured/BO separator at index 35.5).
- **How to read**: trace the orange line; the reproduction converges into eField ≈ 0.5 and lifetime ≈ 2200 μs.
- **Conclusion supported**: the reproduction reaches the correct physical basin.
- **Conclusion not supported**: this is not the exact deleted trace and does not prove per-iteration identity with the deleted BO-B.

---

## 02_loss_trajectory.png

- **Purpose**: how fast the BO-B reproduction drives native LLHD toward the minimum.
- **Source**: same recovery `history.csv`.
- **Columns**: `evaluation_index`, `native_LLHD`, `phase`.
- **Values shown**: real simulator native LLHD; no `-ln(LLHD)` displayed.
- **X-axis**: `Iteration`. **Y-axis**: `LLHD` (log scale).
- **Legend labels**: **LLHD**, **Running best**, **Init**.
- **How to read**: watch the running best step down; it plateaus at LLHD ≈ 21025.
- **Conclusion supported**: the reproduction achieves the same LLHD basin the repaired method achieves.
- **Conclusion not supported**: exact per-iteration match with the deleted original.

---

## 03_final_gp_posterior.png

- **Purpose**: what the GP has learned about the (eField, lifetime) surface after the reproduction.
- **Source**: same recovery `history.csv`. The GP is refit using the validated recipe.
- **X-axis**: `eField [kV/cm]`. **Y-axis**: `Lifetime [us]`.
- **Colorbar**: `LLHD-like` — `exp(-posterior mean score)`. Not the exact posterior mean of native LLHD; a monotonic transformed visualization.
- **Legend labels**: **Observed**, **Nominal**, **Best**, **GP min**.
- **How to read**:
  1. Locate the darkest region.
  2. Verify `Best`, `Nominal`, `GP min` sit inside it.
  3. Confirm there is **no** false low-LLHD region above lifetime ≈ 4000 μs — that was the historical failure the repair fixed.
- **Conclusion supported**: the repaired GP places its minimum near eField=0.5, lifetime≈2200 μs; the pathological high-lifetime false minimum is absent.
- **Conclusion not supported**: exact numerical LLHD at grid points.

---

## 04_final_acquisition.png

- **Purpose**: where the BO would query next.
- **Source**: same recovery `history.csv` and the same refit GP.
- **Colorbar**: `qLogEI` (next-sampling score, not LLHD).
- **Legend labels**: **Observed**, **Best**, **Next**.
- **How to read**: the acquisition surface should be small near the observed minimum and peak somewhere else — a well-converged run "wants" to explore, not re-sample.
- **Conclusion supported**: the reproduction is essentially converged.
- **Conclusion not supported**: nothing about native LLHD.

---

## 05_direct_validation.png

- **Purpose**: confirm the BO-B best point is a real simulator basin using independent 1D scans.
- **Sources**:
  - `bo_2d_2000cm/eField_lifetime/direct_parameter1_scan.csv` (33 rows varying eField; the surviving 999.93-cm direct scan).
  - `bo_2d_2000cm/eField_lifetime/direct_parameter2_scan.csv` (24 rows varying lifetime).
- **Provenance note**: no BO-B-specific post-BO direct scan survived the trace deletion. The pre-BO 999.93-cm direct scans held under `bo_2d_2000cm/eField_lifetime/` are used because they cover the correct axis ranges, use the same common target, and were performed with the same simulator configuration. These are **real simulator points**, not GP predictions.
- **X-axis left**: `eField [kV/cm]`. **X-axis right**: `Lifetime [us]`.
- **Y-axis**: `LLHD` (log scale).
- **Legend labels**: **Direct**, **GP**, **Nominal**, **Best**.
- **How to read**: locate the direct-scan minimum; verify `Best` sits on it, and the `GP` curve tracks the `Direct` dots without inventing a lifetime minimum near 4000-6000 μs.
- **Conclusion supported**: the repaired BO-B minimum is confirmed; the historical false-lifetime region is absent.
- **Conclusion not supported**: joint (eField, lifetime) shape; use plot 3.
