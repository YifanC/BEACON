# FINAL_6D — Six-dimensional Bayesian optimization

## 1. Executive summary

- Final training best LLHD **21026.756** at evaluation 331 (phase BO_TR2).
- Best per phase: INITIAL {'n': 72, 'llhd': 44702.83203125, 'eval_index': 23} → BO_1 {'n': 100, 'llhd': 21484.77734375, 'eval_index': 170} → BO_TR {'n': 100, 'llhd': 21099.203125, 'eval_index': 260} → BO_TR2 {'n': 60, 'llhd': 21026.755859375, 'eval_index': 331}.
- All-nominal held-out LLHD (reference) **21028.986**, Δ vs final best = +2.230.
- Health check flags: DATA_INTEGRITY=PASS · BO_1_IMPROVES_INITIAL=PASS · BO_TR_IMPROVES_BO1=PASS · BO_TR_IMPROVES_OLD_BO2=PASS · LOCAL_CONVERGENCE=PASS · GP_BEATS_CONSTANT_BASELINE=PASS · GP_HELDOUT_RANKING=PASS · GP_INTERVAL_COVERAGE=PASS · DIRECT_PROFILES_CONFIRM_BEST=PASS · FINAL_BEST_IS_INTERIOR=PASS · FIXED_SEED_REPEATABILITY=PASS · NOMINAL_CLOSURE=PASS

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

- INITIAL Sobol seed: 20260812. BO_1 seed: 20260812. BO_TR seed: 20260830. BO_TR2 seed: 20260840. Fresh held-out validation seed: 20260827.
- Candidate simulator uses `sim_seed_strategy='same'` with `SimSettings.seed=0`. Under LUT probabilistic simulation the candidate objective is deterministic in `(params, tracks, response)`.
- Independent retained calls reproduce exactly: nominal values `[21028.986328125, 21028.986328125]` (n=2, std=0) and BO_TR2-best values `[21026.755859375, 21026.755859375]` (n=2, std=0). `FIXED_SEED_REPEATABILITY=PASS`.

## 6. Objective and score transformation

- Native objective: LLHD (lower is better).
- Modeled score: `-ln(native_LLHD)`. BoTorch maximizes; lower LLHD becomes larger score.
- Regularization: `train_Yvar = 1 / native_LLHD²`. This is a transformed-regularization rule for a fixed-seed deterministic objective, not a measurement of physical detector noise.
## 7. GP model and acquisition function

- Final model: `SingleTaskGP(Matern(nu=1.5, ARD(6)), ScaleKernel, Standardize(m=1))` (selected Model C).
- Kernel lengthscale constraint (normalized units): (1e-3, 20). Outputscale constraint: (1e-4, 100).
- Torch 2.6.0+cu124 · BoTorch 0.16.1 · GPyTorch 1.15.2.
- BO_TR2 acquisition: `qLogExpectedImprovement (q=1)` with `num_restarts=64, raw_samples=8192`. The acquisition is differentiated through the GP; simulator gradients are never computed.

## 8. Original 72-point Sobol initialization

- 72 scrambled Sobol points across the 6D bounds. Best INITIAL LLHD **44702.832**.

## 9. Original BO_1 results

- 100 acquired evaluations. Best BO_1 LLHD **21484.777**. Boundary-proposal fraction = 0.74.

## 10. Old informed diagnostic continuation (for context)

The BO_1 result was probed by a set of DIRECT_1 simulator profiles (145 same-objective evaluations) and four INTERACTION corner points. Those observations were fed into a 50-iteration informed continuation labelled BO_2 in prior reports. The best BO_2 LLHD was 21 116.693.  

**The improved run reported here does NOT use those DIRECT / INTERACTION / BO_2 rows for GP training.** They are retained only for provenance and appear on the old-BO_2 comparison line in plot 01.

## 11. Trust-region continuation (BO_TR)

- 100 additional BO_TR evaluations starting from the clean 72+100=172 seed. Best BO_TR LLHD **21099.203**. BO_TR boundary-proposal fraction = 0.01.
- Trust-region hyperparameters (frozen before the run): initial length 0.40, min 0.025, max 0.80, success tol 3, failure tol 6, expand 2.0, shrink 0.5, ARD weight clip [0.1, 10]. Improvement threshold = max(1.0 LLHD, 1e-4 × current best).
- Trust-region events during the run: 0 expansions, 5 shrinks, 1 stagnation resets.
- BO_TR is a clean local-refinement run: only INITIAL + BO_1 seed it. No DIRECT, INTERACTION, BO_2, held-out, or nominal row entered its training set. Trust-region bounds always centre on the current best actual observation among {INITIAL, BO_1, BO_TR}, never on nominal.

## 12. Matérn-3/2 trust-region continuation (BO_TR2)

- 60 additional BO_TR2 evaluations starting from the frozen clean 272-row snapshot. Best BO_TR2 LLHD **21026.756**.
- Fresh trust-region state: initial/max length 0.20, min 0.005, success tol 3, failure tol 4, expand 1.5, shrink 0.5, stagnation-extra tolerance 8. The center was always the best actual clean observation and was never oriented using nominal.
- The selected surrogate was Matérn-3/2 (Model C); acquisition used qLogEI with q=1, 64 restarts, and 8192 raw samples.
- Recovered BO_TR2 history statistics: initial length 0.2, final length 0.05, 0 expansions, 8 shrinks, 1 stagnation reset.
- Clean final lineage: 72 INITIAL + 100 BO_1 + 100 BO_TR + 60 BO_TR2 = 332 rows. Diagnostic, nominal, legacy BO_2, repeatability, and validation observations remained outside training.

## 13. Final best point by phase

| phase | best LLHD |
|---|---|
| INITIAL | 44702.8320 |
| BO_1 | 21484.7773 |
| BO_TR | 21099.2031 |
| BO_TR2 | 21026.7559 |

## 14. Comparison against the all-nominal point

- All-nominal held-out LLHD = 21028.9863
- Final training best LLHD = 21026.7559
- Δ = +2.2305

The all-nominal reference is not meaningfully better than the final training best.

## 15. Held-out 6D GP accuracy

- Held-out n = 32 (16 global + 16 local Sobol around BO_TR2 best, seed 20260827).
- Held-out RMSE(score) = 0.1470  ·  baseline 0.9040.
- Held-out MAE(score) = 0.1079.
- Held-out RMSE(LLHD) = 23828.5  ·  MAE(LLHD) = 16029.9.
- Held-out Spearman ρ = 0.987.
- Held-out 68% coverage = 0.84  ·  95% coverage = 0.97.
- Worst held-out score residual = +0.4811 at coordinate {'Ab': 0.8056892878856775, 'kb': 0.04435319772156436, 'eField': 0.4997768175043614, 'lifetime': 2676.9151384465613, 'tran_diff': 9.746960306811794e-06, 'long_diff': 3.969455679041201e-06}.

## 16. Five-fold cross-validation

- CV RMSE(score) = 0.1454  ·  baseline 1.0483
- CV MAE(score) = 0.0464
- CV Spearman ρ = 0.995
- CV 68% / 95% coverage = 0.84 / 0.95

## 17. Direct-profile validation

Convergence threshold = max(1.0, 1e-4 × 21026.756) = **2.1027 LLHD**. A profile fails LOCAL_CONVERGENCE if `−Δ > threshold` (i.e. a same-objective simulator point beats BO_TR2 best by more than the threshold).

| parameter | direct min | direct LLHD | Δ vs final best | improves? | at boundary |
|---|---|---|---|:-:|:-:|
| Ab | 0.800693 | 21026.7559 | +0.0000 | no | no |
| kb | 0.0487162 | 21026.7559 | +0.0000 | no | no |
| eField | 0.499992 | 21026.7559 | +0.0000 | no | no |
| lifetime | 2241.73 | 21026.7559 | +0.0000 | no | no |
| tran_diff | 8.82719e-06 | 21026.7559 | +0.0000 | no | no |
| long_diff | 3.9126e-06 | 21026.4414 | -0.3145 | no | no |

**LOCAL_CONVERGENCE PASS** — no single-axis profile beats BO_TR2 best by more than the threshold.

## 18. ARD lengthscales and sensitivity interpretation

| parameter | normalized ARD lengthscale |
|---|---|
| Ab | 1.4334 |
| kb | 0.4906 |
| eField | 0.1079 |
| lifetime | 1.2291 |
| tran_diff | 2.1651 |
| long_diff | 2.3920 |

Outputscale = 0.2870.

A shorter normalized lengthscale means the fitted response varies more rapidly along that axis over the observed region. It does not by itself prove physical importance or causal importance in the detector model.

## 19. Boundary behavior

- BO_1 boundary-proposal fraction = 0.74. BO_TR boundary-proposal fraction = 0.01.
- Boundary proposals are not automatic failures; they mean the acquisition function preferred an edge given the current model. The direct profiles are the ground-truth check on whether an edge is actually optimal. BO_TR's trust-region constraint dramatically reduces boundary proposals once the current best is interior — that is by design.

## 20. Parameter biases and conditional minima

| parameter | final best | nominal | Δ absolute | % of nominal | normalized shift (fraction of range) |
|---|---|---|---|---|---|
| Ab | 0.800693493 | 0.8 | +0.000693493 | +0.0867 % | +0.0046 |
| kb | 0.0487161774 | 0.0486 | +0.000116177 | +0.2390 % | +0.0023 |
| eField | 0.4999923871 | 0.5 | -7.61288e-06 | -0.0015 % | -0.0004 |
| lifetime | 2241.726206 | 2200 | +41.7262 | +1.8966 % | +0.0075 |
| tran_diff | 8.827187634e-06 | 8.8e-06 | +2.71876e-08 | +0.3090 % | +0.0023 |
| long_diff | 4.002600755e-06 | 4e-06 | +2.60076e-09 | +0.0650 % | +0.0003 |

## 21. Whether BO converged

- BO_1_IMPROVES_INITIAL = PASS (44702.832 → 21484.777).
- BO_TR_IMPROVES_BO1 = PASS (21484.777 → 21099.203).
- BO_TR_IMPROVES_OLD_BO2 = PASS (old BO_2 21 116.693 → BO_TR 21099.203).
- BO_TR2 improved the clean BO_TR best (21099.203 → 21026.756).
- LOCAL_CONVERGENCE = PASS (no new direct-profile point beats BO_TR2 best by > max(1, 1e-4·LLHD)).
- NOMINAL_CLOSURE = PASS — see closure gap below.

## 22. Limitations

- One BO_1 run plus BO_TR and BO_TR2 continuations; there is no independent-seed continuation replicate.
- The 32 held-out validation points (16 global + 16 local Sobol at seed 20260827) are independent but still a small sample; interval-coverage estimates are noisy.
- Direct profiles vary one axis with the other five pinned at BO_TR2 best; they cannot exclude a lower-loss region far away in multiple dimensions.
- Nominal is a known held-out reference, not a mathematically proven global minimum.

## 23. Final conclusion

BO_TR2 completed the clean 332-row lineage and reached LLHD 21026.756. Relative to the held-out nominal reference 21028.986, the signed gap (BO_TR2 best − nominal) is -2.230 LLHD and the absolute gap is 2.230 LLHD. Nominal closure category: **STRONG**. The optimizer recovered the nominal closure region and found a nearby coordinate with essentially equivalent native objective. Nominal remains a reference, not a proven global minimum; this does not establish a better physical calibration than nominal.

## 24. Exact command to reproduce the final analysis

```bash
# 1. Regenerate/refresh the final plots + report (idempotent, no simulator calls):
apptainer exec --nv -B /sdf,/fs /sdf/group/neutrino/pgranger/larnd-sim-jax.sif \
  env MPLBACKEND=Agg PYTHONPATH=/sdf/home/i/iatif/larnd-sim-jax:/sdf/home/i/iatif/larnd-sim-jax/src \
  python3 /sdf/home/i/iatif/larnd-sim-jax/optimize/bayesian/workflows/six_d/finalize.py
```
