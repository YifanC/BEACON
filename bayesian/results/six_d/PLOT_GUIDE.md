# 6D BO — plot guide

## How the plots were built

- Frozen training snapshot: `.local/six_d/current/raw/continuation/final_training_snapshot_bo_tr2.csv` (SHA in `health_metrics.json`).
- Training GP: selected Model C, Matérn-3/2 ARD(6), with `-ln(LLHD)` score, `Standardize(m=1)`, and requested `train_Yvar = 1/LLHD²` (numerically clamped to the GPyTorch noise floor).
- All five plots come from `optimize/bayesian/workflows/six_d/finalize.py`.
- Held-out points and reference checks are in `.local/six_d/current/raw/direct_validation/bo_tr2_heldout_6d_results.csv` and `bo_tr2_reference_points.csv`; those points were never used to fit the GP shown here.

## 01_convergence_and_efficiency.png

- Top panel plots every clean training LLHD by ordered evaluation index, colored by phase (INITIAL, BO_1, BO_TR, BO_TR2). Dashed verticals mark phase boundaries.
- Bottom panel plots the running-best LLHD across the frozen snapshot together with the best of each phase and the held-out all-nominal reference (star). The all-nominal point is NOT connected to the running-best line — it was evaluated after freezing.
- What the plot answers: did BO_1, BO_TR, and BO_TR2 improve the clean running best, and how close is the final training best to the all-nominal held-out reference?
- What the plot cannot prove: convergence to the true simulator minimum. A flat running-best only means the optimizer did not sample a lower-loss point; the direct profiles provide a separate local check.

## 02_parallel_coordinates.png

- Each polyline is one complete 6D training observation. Axes are normalized per-parameter to [0,1] so all axes are comparable.
- Grey lines: all training observations. Colored lines: the best 10% by LLHD (coloured by score rank). Black poly-line: the final training best. Purple dashed poly-line: the all-nominal reference.
- What the plot answers: do the low-LLHD observations form a tight cluster on some axes and spread on others? Is the final best close to nominal on every axis, or is it displaced?
- What the plot cannot prove: causal interactions between parameters. Crossing lines between two axes only show co-occurrence, not directed effect.

## 03_pairwise_observation_matrix.png

- 6×6 matrix. Diagonal: 1D histograms of all training values (grey) and best-10% (red). Lower triangle: scatter of all training observations for each pair, coloured by LLHD rank (dark = best). Upper triangle: GP conditional posterior mean on that pair with the other four axes fixed at the final best.
- The lower triangle is real simulator evidence. The upper triangle is the model's opinion in one specific slice of 6D space.
- What the plot answers: where in each pair did the simulator actually see low LLHD, and how does the fitted surrogate render the same pair conditional on the remaining four dimensions?
- What the plot cannot prove: the true 6D surface. Two axes that look "flat" here could be steep at a different fixed setting of the other four.

## 04_gp_cross_validation.png

- Panel A: 5-fold cross-validated predicted score vs actual for every training row. Panel B: held-out simulator observations (16 global + 16 local Sobol points around final best, seed 20260827), with 1-sigma vertical bars. Panel C: standardized held-out residuals `(pred − obs)/SD`.
- The 32 held-out points were never used to fit the GP shown here. CV is an interpolation check; held-out is the independent test.
- CV RMSE(score) = 0.1454 vs constant-mean baseline 1.0483. Held-out RMSE(score) = 0.1470 vs baseline 0.9040. Held-out Spearman ρ = 0.987. Held-out coverage 68% / 95% = 0.84 / 0.97.

## 05_direct_validation_scorecard.png

- One panel per parameter. Black points/line: real simulator profile through the final best (five other axes fixed at final best). Blue line + band: GP posterior back-transformed to LLHD, 95% band. Red dotted vertical = final best coordinate. Grey dashed vertical = that parameter's nominal (only the varied coordinate is nominal — the other five stay at final best).
- The exact final-best coordinate is included in every profile. Orange marker = direct simulator minimum on the profile.
- Panel corner text reports the direct-min coord, LLHD change from final best, normalized shift as a fraction of the search range, and whether the min is at a bounds edge.
- Footer compares the final training best LLHD to the all-nominal held-out LLHD. Because the candidate simulator is deterministic under the fixed seed policy, a same-objective direct evaluation is a legitimate simulator observation. A direct point that beats the final training best by more than about 1 LLHD unit is honest evidence that BO did not locally converge on that axis.

## Notes readers often need

- The running-best line can stay flat while BO continues because BO deliberately proposes exploratory (high-uncertainty) points that may not beat the current best. That is not a failure — it is how BO trades off exploitation and exploration.
- A boundary proposal is not automatically wrong. BO proposes a boundary point when the model believes the response might keep improving toward that edge. Whether the boundary is right depends on the direct profile, which is why plot 05 exists.
- ARD lengthscale is not physical parameter importance. A shorter normalized lengthscale means the fitted response varies more rapidly along that axis over the sampled region. It says nothing about causal importance in the underlying physics.
- Conditional profiles are not the full 6D surface. Plot 05 varies one axis with the other five pinned at the final best. If the true minimum lies at different fixed values on the other five axes, the profile will not find it.
- An all-nominal 6D reference is different from one nominal coordinate in a conditional profile. All-nominal fixes every axis at its nominal value; a profile fixes five axes at the final best.
- Held-out validation after BO is not cheating. The held-out set and the direct profiles were generated only after the training snapshot was frozen — they never contributed to acquisition selection or GP fitting.
