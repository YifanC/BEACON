# PLOT_GUIDE — transverse + longitudinal diffusion pair (Diffusion)

Full dataset description: **999.93-cm safe dataset** — 176 ordered events, 100 010 HDF5 rows, 999.926642 cm physical track, one simulator batch, 9 368 target hits at target seed 0. Plots display only the short label `999.93 cm`.

## Why BO-B was selected

Three independent 100-iteration BO runs were carried out — BO-A, BO-B, BO-C — as **repeated executions of the same Bayesian-optimization method with different acquisition seeds**, not different algorithms. Under the new project policy the final presentation plots must display **one** run. `BO-B` is used by a **predeclared** rule fixed before inspecting any final metric. BO-A and BO-C remain preserved under `bo_2d_2000cm/tran_diff_long_diff/runs/BO-{A,C}/` and are unchanged.

Parameters:
- **Transverse** (`tran_diff`, cm²/μs): nominal 8.8 × 10⁻⁶. Search range [3 × 10⁻⁶, 1.5 × 10⁻⁵].
- **Longitudinal** (`long_diff`, cm²/μs): nominal 4.0 × 10⁻⁶. Search range [1 × 10⁻⁶, 1.0 × 10⁻⁵].

Method: score = -ln(native_LLHD); observation variance = 1 / native_LLHD²; Matérn-5/2 ARD; Standardize(m=1); qLogExpectedImprovement (q=1).

Exact numerical result (BO-B): 130-row history, best native LLHD **21025.76171875** at `tran_diff ≈ 8.85e-6` cm²/μs, `long_diff ≈ 3.91e-6` cm²/μs — from `bo_2d_2000cm/tran_diff_long_diff/runs/BO-B/history.csv` and `summary.json`.

---

## 01_parameter_trajectory.png

- **Purpose**: show BO-B's diffusion parameter evolution.
- **Source**: `bo_2d_2000cm/tran_diff_long_diff/runs/BO-B/history.csv`.
- **Rows used**: 130 (30 `structured_seed` + 100 `BO`).
- **Columns used**: `iteration`, `tran_diff`, `long_diff`, `phase`.
- **Values shown**: real simulator inputs.
- **X-axis**: `Iteration`.
- **Y-axis top**: `Transverse [cm2/us]`.
- **Y-axis bottom**: `Longitudinal [cm2/us]`.
- **Legend labels**: **BO**, **Nominal**, **Init**.
- **How to read**: same as the other trajectory guides. The orange line settles near nominal.
- **Conclusion supported**: BO-B converges to the expected diffusion basin.
- **Conclusion not supported**: seed robustness (BO-A/C are the evidence for that).

---

## 02_loss_trajectory.png

- **Purpose**: how fast BO-B drives native LLHD toward the minimum.
- **Source**: same `history.csv`.
- **Columns**: `iteration`, `llhd`, `phase`.
- **X-axis**: `Iteration`. **Y-axis**: `LLHD` (log scale).
- **Legend labels**: **LLHD**, **Running best**, **Init**.
- **How to read**: watch the running best step down and plateau at LLHD ≈ 21025.76.
- **Conclusion supported**: BO-B reaches the same basin fast, within the 100 BO iterations.
- **Conclusion not supported**: joint parameter shape — plot 3 for that.

---

## 03_final_gp_posterior.png

- **Purpose**: what the GP has learned about the (Transverse, Longitudinal) surface after BO-B finished.
- **Source**: same `history.csv`. GP is refit using the validated recipe.
- **X-axis**: `Transverse [cm2/us]`. **Y-axis**: `Longitudinal [cm2/us]`.
- **Colorbar**: `LLHD-like` — monotonic transformed visualization, not exact native LLHD.
- **Legend labels**: **Observed**, **Nominal**, **Best**, **GP min**.
- **How to read**:
  1. Find the darkest region.
  2. Check that `Best`, `Nominal`, `GP min` all sit inside it.
- **Conclusion supported**: BO-B's GP concentrates around the physical diffusion basin.
- **Conclusion not supported**: exact numerical native LLHD at grid points.

---

## 04_final_acquisition.png

- **Purpose**: where the BO would query next.
- **Source**: same `history.csv` and the same refit GP.
- **Colorbar**: `qLogEI` (next-sampling score).
- **Legend labels**: **Observed**, **Best**, **Next**.
- **How to read**: after 100 iterations the acquisition prefers regions away from the tight training-point cluster near the basin — the BO wants to explore, not resample.
- **Conclusion supported**: BO-B is well-converged.
- **Conclusion not supported**: `qLogEI` values are not LLHDs.

---

## 05_direct_validation.png

- **Purpose**: verify BO-B best is a real simulator basin.
- **Sources**:
  - `bo_2d_2000cm/tran_diff_long_diff/runs/BO-B/direct_profile_parameter1.csv` — 22 rows varying `Transverse` with `Longitudinal` fixed at BO best.
  - `bo_2d_2000cm/tran_diff_long_diff/runs/BO-B/direct_profile_parameter2.csv` — 22 rows varying `Longitudinal` with `Transverse` fixed at BO best.
- **Values shown**: `Direct` = real simulator points; `GP` = transformed prediction.
- **X-axis left**: `Transverse [cm2/us]`. **X-axis right**: `Longitudinal [cm2/us]`. Scientific notation.
- **Y-axis**: `LLHD` (log scale).
- **Legend labels**: **Direct**, **GP**, **Nominal**, **Best**.
- **How to read**: locate the minimum of the `Direct` points; verify `Best` sits at or next to it; check `GP` tracks the `Direct` shape.
- **Conclusion supported**: BO-B best is confirmed by independent simulator scans.
- **Conclusion not supported**: joint 2D minimum — plot 3 for that.
