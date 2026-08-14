# PLOT_GUIDE — Ab + kb pair

Full dataset description: **999.93-cm safe dataset** — 176 ordered events, 100 010 HDF5 rows, 999.926642 cm physical track, one simulator batch, 9 368 target hits at target seed 0. Plots display only the short label `999.93 cm`.

## Why BO-B was selected

Three independent 100-iteration BO runs were carried out — BO-A, BO-B, BO-C. They are **repeated executions of the same Bayesian-optimization method with different acquisition seeds**, not different algorithms. They exist only as historical seed-robustness evidence. Under the new project policy the final presentation plots must display **one** run.

`BO-B` is used for **all** Ab+kb plots by a **predeclared** rule (single choice fixed for every pair before looking at the numbers). Predeclaring the choice avoids cherry-picking the best-looking run. The BO-A and BO-C histories remain preserved under `bo_2d_2000cm/Ab_kb/runs/BO-A/` and `.../BO-C/` and are unchanged.

Parameters:
- **Ab** (dimensionless): recombination coefficient. Nominal 0.8. Search range [0.75, 0.9].
- **kb** (kV·g·MeV⁻¹·cm⁻³): recombination coefficient. Nominal 0.0486. Search range [0.03, 0.08].

Method (all plots): score = -ln(native_LLHD); observation variance = 1 / native_LLHD²; Matérn-5/2 ARD kernel; Standardize(m=1); acquisition = qLogExpectedImprovement (q=1).

Exact numerical result (BO-B): 130-row history, best native LLHD **21029.005859375** at (`Ab=0.8`, `kb=0.0486`) — from `bo_2d_2000cm/Ab_kb/runs/BO-B/history.csv` and `summary.json`.

---

## 01_parameter_trajectory.png

- **Purpose**: show how the two parameters evolve across the BO-B run.
- **Source**: `bo_2d_2000cm/Ab_kb/runs/BO-B/history.csv`.
- **Rows used**: all 130 rows (30 `structured_seed` + 100 `BO`).
- **Columns used**: `iteration`, `Ab`, `kb`, `phase`.
- **Values shown**: real simulator inputs. No GP predictions.
- **X-axis**: `Iteration`.
- **Y-axis top**: `Ab` (dimensionless).
- **Y-axis bottom**: `kb` (kV·g·MeV⁻¹·cm⁻³).
- **Legend labels**:
  - **BO** — the single BO-B parameter trajectory.
  - **Nominal** — the generating value used to build the target (dashed gray horizontal).
  - **Init** — vertical line at iteration 29.5 marking the boundary between the 30 structured initial rows and the 100 BO rows.
- **How to read**:
  1. Trace the orange line from left to right.
  2. Compare its settled value at the right of the panel with the dashed gray horizontal.
- **Conclusion supported**: BO-B settles into the nominal band well before iteration 100.
- **Conclusion not supported**: this plot does not prove seed robustness; that is what BO-A/C would show, and they are preserved separately.

---

## 02_loss_trajectory.png

- **Purpose**: how fast BO-B drives native LLHD toward the minimum.
- **Source**: same `history.csv`.
- **Columns used**: `iteration`, `llhd`, `phase`.
- **Values shown**: real simulator native LLHD (not `-ln(LLHD)`).
- **X-axis**: `Iteration`.
- **Y-axis**: `LLHD` (log scale, lower is better).
- **Legend labels**:
  - **LLHD** — the raw evaluated native LLHD at each iteration.
  - **Running best** — cumulative minimum-so-far.
  - **Init** — vertical structured/BO separator.
- **How to read**: follow the thick red curve (`Running best`) — it steps down and plateaus once BO enters the basin.
- **Conclusion supported**: BO-B reaches the plateau at native LLHD ≈ 21029 with room to spare inside the 100 BO iterations.
- **Conclusion not supported**: nothing about the joint (Ab, kb) location — see plot 3.

---

## 03_final_gp_posterior.png

- **Purpose**: what the GP learned about the (Ab, kb) surface after the BO-B run finished.
- **Source**: same `history.csv`. The GP is refit here using exactly the validated recipe.
- **Values shown**: colored surface = **GP prediction** on an LLHD-like scale (`exp(-posterior mean score)`). Markers = real simulator inputs.
- **X-axis**: `Ab`.
- **Y-axis**: `kb [kV g / (MeV cm3)]`.
- **Colorbar**: `LLHD-like`. This is a monotonic transformed visualization; it is **not** the exact posterior mean of native LLHD. The transformation from the GP's score = -ln(LLHD) to `exp(-score)` is nonlinear, but darker/blue = smaller predicted LLHD-like value.
- **Legend labels**:
  - **Observed** — the 130 training points (white circles).
  - **Nominal** — the generating value (gold star at Ab=0.8, kb=0.0486).
  - **Best** — the best actual observed BO-B point (red diamond).
  - **GP min** — the point where the GP predicts the smallest LLHD-like value on the grid (cyan X).
- **How to read**:
  1. Find the darkest region.
  2. Check that `Best`, `Nominal`, and `GP min` all sit inside that region.
- **Conclusion supported**: the BO-B GP places its minimum at essentially the same location as the best actual observation and the nominal.
- **Conclusion not supported**: exact numerical native-LLHD values at grid points.

---

## 04_final_acquisition.png

- **Purpose**: show where the next simulation would be queried, given the final BO-B GP.
- **Source**: same `history.csv` and the same refit GP as plot 3.
- **X-axis**, **Y-axis**: same as plot 3.
- **Colorbar**: `qLogEI`. Higher = better next-query. qLogEI is a **next-sampling score**, not native LLHD.
- **Legend labels**:
  - **Observed** — training points.
  - **Best** — best actual observed BO-B point.
  - **Next** — the acquisition maximum on the grid (cyan triangle).
- **How to read**: after 100 rounds the acquisition is small; `Next` sits away from the tightest cluster of training points.
- **Conclusion supported**: BO-B is essentially converged.
- **Conclusion not supported**: `qLogEI` values are not LLHDs and are not comparable across pairs.

---

## 05_direct_validation.png

- **Purpose**: verify the BO-B best point is a real simulator basin (independent 1D scans).
- **Sources**:
  - `bo_2d_2000cm/Ab_kb/runs/BO-B/direct_profile_parameter1.csv` — 22 rows varying Ab with kb fixed at the BO best.
  - `bo_2d_2000cm/Ab_kb/runs/BO-B/direct_profile_parameter2.csv` — 22 rows varying kb with Ab fixed at the BO best.
- **Values shown**: black dots = real simulator points; blue curve = **GP** (transformed LLHD-like prediction).
- **X-axis left**: `Ab`. **X-axis right**: `kb`.
- **Y-axis**: `LLHD` (log scale).
- **Legend labels**:
  - **Direct** — simulator points.
  - **GP** — the transformed GP prediction along the same 1D slice.
  - **Nominal** — dashed gray vertical.
  - **Best** — dotted red vertical (BO-B best value on this axis).
- **How to read**:
  1. Locate the minimum of the `Direct` points.
  2. Check that `Best` and (approximately) `Nominal` fall at or next to it.
  3. Check the `GP` overlay tracks the `Direct` points.
- **Conclusion supported**: the BO-B minimum is confirmed by an independent simulator scan.
- **Conclusion not supported**: nothing about the joint 2D minimum — plot 3 is for that.
