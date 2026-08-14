# FINAL_6D — Six-dimensional Bayesian optimization

## 1. Executive summary

- Final training best LLHD **21116.693** at evaluation 368 (phase BO_2).
- Best per phase: INIT 44702.832 → BO_1 21484.777 → pre-BO_2 21339.062 → BO_2 21116.693.
- All-nominal held-out LLHD (reference) **21028.988**, Δ vs final best = -87.705.
- Health check flags: DATA_INTEGRITY=PASS · BO_1_IMPROVES_INITIAL=PASS · BO_2_IMPROVES_PRECONTINUATION=PASS · GP_BEATS_CONSTANT_BASELINE=PASS · GP_HELDOUT_RANKING=PASS · GP_INTERVAL_COVERAGE=INCONCLUSIVE · DIRECT_PROFILES_CONFIRM_BEST=FAIL · FINAL_BEST_IS_INTERIOR=PASS · FIXED_SEED_REPEATABILITY=PASS · NOMINAL_CLOSURE=FAIL

## 2. Scientific objective

Calibrate six detector-simulation parameters (Ab, kb, eField, lifetime, tran_diff, long_diff) against a fixed nominal target on the 999.926642 cm safe dataset. Use Bayesian optimization to find the six-dimensional coordinate that minimizes the native LLHD between candidate simulator output and the target hits.

## 3. Dataset and target provenance

- Physical track length: 999.926642 cm  ·  events: 176  ·  HDF5 rows: 100 010  ·  target hits: 9368
- Target NPZ SHA-256: `ed72e051a6111cfee218e0e009a9ebd78ae1167ea7225a58f2ac4b32f6edd34f`
- Target file: `.local/two_d/target.npz` (shared with the 2D pair fits — same SHA verified pre-run).

## 4. Parameter definitions, nominal values, bounds, units

| parameter | nominal | lower | upper | unit |
|---|---|---|---|---|
| Ab | 0.8 | 0.75 | 0.9 | (dimensionless) |
| kb | 0.0486 | 0.03 | 0.08 | kV·g/(MeV·cm³) |
| eField | 0.5 | 0.49 | 0.51 | kV/cm |
| lifetime | 2200 | 400 | 6000 | μs |
| tran_diff | 8.8e-06 | 3e-06 | 1.5e-05 | cm²/μs |
| long_diff | 4e-06 | 1e-06 | 1e-05 | cm²/μs |

## 5. Seed and determinism policy

- INITIAL Sobol seed: 20 260 812. BO_1 seed: 20 260 812. Continuation BO_2 seed: 20 260 815. Held-out validation seed: 20 260 823.
- Candidate simulator uses `sim_seed_strategy='same'` with `SimSettings.seed=0`. Under LUT probabilistic simulation the candidate objective is deterministic in `(params, tracks, response)` — two calls at the same 6D coordinate return the same LLHD.

## 6. Objective and score transformation

- Native objective: LLHD (lower is better).
- Modeled score: `-ln(native_LLHD)`. BoTorch maximizes; lower LLHD becomes larger score.
- Regularization: `train_Yvar = 1 / native_LLHD²`. This is a transformed-regularization rule for a fixed-seed deterministic objective, not a measurement of physical detector noise.
## 7. GP model and acquisition function

- Model: `SingleTaskGP(Matern(nu=2.5, ARD(6)), ScaleKernel, Standardize(m=1))`.
- Kernel lengthscale constraint (normalized units): (1e-3, 20). Outputscale constraint: (1e-4, 100).
- Torch 2.6.0+cu124 · BoTorch 0.16.1 · GPyTorch 1.15.2.
- Acquisition function: `qLogExpectedImprovement (q=1)` with `num_restarts=24, raw_samples=2048`. The acquisition is differentiated through the GP; simulator gradients are never computed.

## 8. Original 72-point Sobol initialization

- 72 scrambled Sobol points across the 6D bounds. Best INITIAL LLHD **44702.832**.

## 9. Original BO_1 results

- 100 acquired evaluations. Best BO_1 LLHD **21484.777**. Boundary-proposal fraction = 0.74.

## 10. Compatible direct observations and interaction design

- DIRECT_1 simulator profiles produced 145 additional same-objective evaluations by varying one axis at a time with the other five fixed at the BO_1 best. Every DIRECT row uses the frozen target SHA and the deterministic candidate simulator, so it counts as a legitimate simulator observation at that 6D coordinate.
- The one-dimensional profiles independently found improvements on Ab (0.7875), lifetime (2 700 μs), and long_diff (4.0e-6). Because these are one-at-a-time moves, an interaction design was needed to check whether the three improvements combine. Four missing corners of the (Ab, lifetime, long_diff) cube were simulated; the fixed axes (kb, eField, tran_diff) were held at the BO_1 best.
- Interaction result summary: the single move on long_diff (LLHD 21 339.06) was the winner; combining Ab+lifetime (LLHD 21 690.72) and Ab+lifetime+long_diff (21 813.46) actually made LLHD worse. Strong non-additivity — the four INTERACTION points fed into BO_2 gave the model this signal directly.

## 11. BO_2 continuation

- 50 additional BO evaluations using the augmented training set (172 + 145 + 4 = 321 seed rows). Best BO_2 LLHD **21116.693**. Boundary-proposal fraction = 0.24.
- BO_2 is an informed continuation, not an independent seed. It uses direct-profile and interaction knowledge in the training set.

## 12. Final best point by phase

| phase | best LLHD |
|---|---|
| INITIAL | 44702.8320 |
| BO_1 | 21484.7773 |
| DIRECT | 21339.0625 |
| INTERACTION | 21398.7891 |
| BO_2 | 21116.6934 |

## 13. Comparison against the all-nominal point

- All-nominal held-out LLHD = 21028.9883
- Final training best LLHD = 21116.6934
- Δ = -87.7051

The six-dimensional BO identified a low-loss region but did not fully recover the known nominal closure point within the available evaluation budget.

## 14. Held-out 6D GP accuracy

- Held-out n = 24 (12 global + 12 local Sobol around final best, seed 20 260 823).
- Held-out RMSE(score) = 0.1145  ·  baseline 0.8912.
- Held-out MAE(score) = 0.0856.
- Held-out RMSE(LLHD) = 31636.5  ·  MAE(LLHD) = 17144.9.
- Held-out Spearman ρ = 0.988.
- Held-out 68% coverage = 0.92  ·  95% coverage = 0.96.
- Worst held-out score residual = +0.3077 at coordinate {'Ab': 0.837244713306427, 'kb': 0.03087893074378371, 'eField': 0.49644455790519715, 'lifetime': 5145.321226119995, 'tran_diff': 1.4633254051208497e-05, 'long_diff': 7.506099224090576e-06}.

## 15. Five-fold cross-validation

- CV RMSE(score) = 0.1129  ·  baseline 1.0475
- CV MAE(score) = 0.0483
- CV Spearman ρ = 0.991
- CV 68% / 95% coverage = 0.87 / 0.96

## 16. Direct-profile validation

| parameter | direct min | direct LLHD | Δ vs final best | at boundary |
|---|---|---|---|---|
| Ab | 0.796053 | 21046.3926 | -70.3008 | no |
| kb | 0.0482322 | 21116.6934 | +0.0000 | no |
| eField | 0.499992 | 21116.6934 | +0.0000 | no |
| lifetime | 2400.93 | 21075.3262 | -41.3672 | no |
| tran_diff | 8.58126e-06 | 21096.6680 | -20.0254 | no |
| long_diff | 3.84674e-06 | 21084.6543 | -32.0391 | no |

## 17. ARD lengthscales and sensitivity interpretation

| parameter | normalized ARD lengthscale |
|---|---|
| Ab | 0.5585 |
| kb | 0.2034 |
| eField | 0.0772 |
| lifetime | 0.4217 |
| tran_diff | 1.2658 |
| long_diff | 1.9032 |

Outputscale = 0.3318.

A shorter normalized lengthscale means the fitted response varies more rapidly along that axis over the observed region. It does not by itself prove physical importance or causal importance in the detector model.

## 18. Boundary behavior

- BO_1 boundary-proposal fraction = 0.74. BO_2 boundary-proposal fraction = 0.24.
- Boundary proposals are not automatic failures; they mean the acquisition function preferred an edge given the current model. The direct profiles are the ground-truth check on whether an edge is actually optimal.

## 19. Parameter biases and conditional minima

| parameter | final best | nominal | Δ absolute | % of nominal | normalized shift (fraction of range) |
|---|---|---|---|---|---|
| Ab | 0.7945534307 | 0.8 | -0.00544657 | -0.6808 % | -0.0363 |
| kb | 0.04823220565 | 0.0486 | -0.000367794 | -0.7568 % | -0.0074 |
| eField | 0.4999916641 | 0.5 | -8.33588e-06 | -0.0017 % | -0.0004 |
| lifetime | 2260.929938 | 2200 | +60.9299 | +2.7695 % | +0.0109 |
| tran_diff | 8.881255988e-06 | 8.8e-06 | +8.1256e-08 | +0.9234 % | +0.0068 |
| long_diff | 4.071744357e-06 | 4e-06 | +7.17444e-08 | +1.7936 % | +0.0080 |

## 20. Whether BO converged

- BO_1_IMPROVES_INITIAL = PASS (best INITIAL 44702.832 → best BO_1 21484.777).
- BO_2_IMPROVES_PRECONTINUATION = PASS (best pre-BO_2 21339.062 → best BO_2 21116.693).
- DIRECT_PROFILES_CONFIRM_BEST = FAIL.
- NOMINAL_CLOSURE = FAIL.

## 21. Limitations

- One 6D BO run (72 INITIAL + 100 BO_1 + 50 BO_2, plus 145 DIRECT + 4 INTERACTION rows brought in as informed prior). No independent-seed replicate.
- 24 held-out validation points is a small sample; interval-coverage measurements are noisy at this budget.
- The final direct profiles vary one axis with the other five pinned at the final training best; they cannot exclude a lower-loss region far from the final best in the other five dimensions.

## 22. Final conclusion

The 6D BO identified a low-loss region but did not fully recover the known all-nominal reference point within the available evaluation budget. The final training best is a defensible six-dimensional local basin, but the BO run is not fully converged. Any downstream analysis should treat the final six-parameter coordinate as a first-generation multi-dimensional fit rather than a final result, and should be paired with the all-nominal reference LLHD when reporting fit quality.

## 23. Exact commands needed to reproduce or resume

```bash
# 1. Regenerate/refresh the final plots + report (idempotent, no simulator calls):
apptainer exec --nv -B /sdf,/fs /sdf/group/neutrino/pgranger/larnd-sim-jax.sif \
  env MPLBACKEND=Agg PYTHONPATH=/sdf/home/i/iatif/larnd-sim-jax:/sdf/home/i/iatif/larnd-sim-jax/src \
  python3 /sdf/home/i/iatif/larnd-sim-jax/optimize/bayesian/workflows/six_d/finalize.py

# 2. If new held-out or profile evaluations are needed:
sbatch /sdf/home/i/iatif/larnd-sim-jax/optimize/bayesian/workflows/six_d/validate.sbatch

# 3. If BO_2 needs to continue from the checkpoint (rare — snapshot already reflects 50 BO_2 evals):
sbatch /sdf/home/i/iatif/larnd-sim-jax/optimize/bayesian/workflows/six_d/continue_bo.sbatch
```
