# Final 2D plot status (single-seed presentation)

Dataset shown on plots: small corner label `999.93 cm`. Full description: **999.93-cm safe dataset** — 176 events, 100 010 HDF5 rows, 999.926642 cm physical track, one simulator batch, 9 368 target hits, target seed 0.

**Presentation policy**: no seed name (`BO-A`, `BO-B`, `BO-C`) appears in any visible plot title, subtitle, annotation, caption, or legend. Only `BO` is used in the plot artwork. Which surviving run backs each plot is documented in this table and in each pair's `PLOT_GUIDE.md` — the presentation is single-run by policy, the on-disk source path retains its historical seed suffix.

**Predeclared consistency rule** (project policy): the single representative run for every 2D pair is the surviving `BO-B` trace. This is fixed before inspecting any final metric so no cherry-picking is possible. Multiple BO seeds are optional robustness tests, not a required part of Bayesian optimization; future production runs use one seed unless the user explicitly requests repeated runs. `BO-A` and `BO-C` raw histories under `bo_2d_2000cm/{Ab_kb,tran_diff_long_diff}/runs/BO-A/` and `.../BO-C/` remain preserved as historical robustness evidence.

The previous multi-seed plots (with 3-run trajectories and long titles) are kept once, byte-for-byte, under `bo_2d_final_plots/_previous_multi_seed/` as plotting history. They are not final results.

| pair | plot | final_path | source_data (on-disk path retains seed suffix) | source_row_count | representative run | existing_or_regenerated | exact_or_reproduction | validation_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ab+kb | 01 | Ab_kb/01_parameter_trajectory.png | bo_2d_2000cm/Ab_kb/runs/BO-B/history.csv | 130 (30 structured + 100 BO) | BO-B | regenerated | exact | complete | single trajectory shown; visible label is `BO` |
| Ab+kb | 02 | Ab_kb/02_loss_trajectory.png | same history.csv | 130 | BO-B | regenerated | exact | complete | native LLHD + running best only |
| Ab+kb | 03 | Ab_kb/03_final_gp_posterior.png | same history.csv | 130 | BO-B | regenerated | exact | complete | 121x121 grid; markers Observed / Nominal / Best / GP min |
| Ab+kb | 04 | Ab_kb/04_final_acquisition.png | same history.csv | 130 | BO-B | regenerated | exact | complete | qLogEI on final GP; markers Observed / Best / Next |
| Ab+kb | 05 | Ab_kb/05_direct_validation.png | bo_2d_2000cm/Ab_kb/runs/BO-B/direct_profile_parameter{1,2}.csv | 22 + 22 direct simulator points | BO-B | regenerated | exact | complete | Direct + GP + Nominal + Best |
| eField+lifetime | 01 | eField_lifetime/01_parameter_trajectory.png | bo_2d_1000cm_efield_lifetime_trace_recovery/rerun_BO-B/history.csv | 135 (35 structured + 100 BO-reproduction) | BO-B reproduction | regenerated | representative reproduction | complete | original three traces were deleted; reproduction uses original seed 20260804 + original repair configuration |
| eField+lifetime | 02 | eField_lifetime/02_loss_trajectory.png | same recovery history.csv | 135 | BO-B reproduction | regenerated | representative reproduction | complete | native LLHD + running best only |
| eField+lifetime | 03 | eField_lifetime/03_final_gp_posterior.png | same recovery history.csv | 135 | BO-B reproduction | regenerated | representative reproduction | complete | GP fit on the 135-row history |
| eField+lifetime | 04 | eField_lifetime/04_final_acquisition.png | same recovery history.csv | 135 | BO-B reproduction | regenerated | representative reproduction | complete | qLogEI on same reproduction GP |
| eField+lifetime | 05 | eField_lifetime/05_direct_validation.png | bo_2d_2000cm/eField_lifetime/direct_parameter{1,2}_scan.csv | 33 + 24 direct simulator points | pre-BO direct scans (no per-seed post-BO scan survived) | regenerated | representative | complete | the surviving 999.93-cm direct scans are used because they cover the correct axis ranges and were performed with the same target |
| Diffusion | 01 | tran_diff_long_diff/01_parameter_trajectory.png | bo_2d_2000cm/tran_diff_long_diff/runs/BO-B/history.csv | 130 (30 structured + 100 BO) | BO-B | regenerated | exact | complete | scientific notation on axes |
| Diffusion | 02 | tran_diff_long_diff/02_loss_trajectory.png | same history.csv | 130 | BO-B | regenerated | exact | complete | |
| Diffusion | 03 | tran_diff_long_diff/03_final_gp_posterior.png | same history.csv | 130 | BO-B | regenerated | exact | complete | |
| Diffusion | 04 | tran_diff_long_diff/04_final_acquisition.png | same history.csv | 130 | BO-B | regenerated | exact | complete | |
| Diffusion | 05 | tran_diff_long_diff/05_direct_validation.png | bo_2d_2000cm/tran_diff_long_diff/runs/BO-B/direct_profile_parameter{1,2}.csv | 22 + 22 direct simulator points | BO-B | regenerated | exact | complete | |

**Titles (visible on plot artwork).** `Ab + kb — Parameters`, `Ab + kb — Loss`, `Ab + kb — GP`, `Ab + kb — qLogEI`, `Ab + kb — Validation`. Same pattern for `eField + lifetime` and `Diffusion`. No seed name in any title.

**Subtitle.** A small `999.93 cm` label sits in the upper-right corner of each figure, rendered at 7-point size in muted gray (`color=0.35`). It does not dominate the title.

**Legend inventory (short, no sentences).**
- Trajectory: `BO`, `Nominal`, `Init`.
- Loss: `LLHD`, `Running best`, `Init`.
- Posterior: `Observed`, `Nominal`, `Best`, `GP min`.
- Acquisition: `Observed`, `Best`, `Next`.
- Validation: `Direct`, `GP`, `Nominal`, `Best`.
- Colorbars: `LLHD-like`, `qLogEI`.
- Max legend-label word count: **2** (`Running best`, `GP min`).

**Legend styling.**
- Trajectory / Loss / Validation: legend inside axes at upper right with `frameon=True`, `framealpha=1.0`, `facecolor="white"`, `edgecolor="black"`.
- GP posterior / qLogEI (heatmaps): legend placed **outside** the axes, to the right of the colorbar, so it never blends into the colormap. Same opaque white frame styling.

Plots regenerated this task: **15**. Plots copied (byte-for-byte) this task: **0**. Previous plots preserved: **15** under `_previous_multi_seed/`.
